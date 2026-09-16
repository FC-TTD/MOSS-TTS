"""One native batch lane per resident model; never retry failed inference."""
from concurrent.futures import Future
from dataclasses import dataclass
import logging
import threading
import time


@dataclass(frozen=True)
class InferenceRequest:
    text: str
    reference_audio: str | None
    mode_with_reference: str
    duration_control_enabled: bool
    duration_tokens: int
    language_tag: str | None
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_new_tokens: int

    def group_key(self):
        # Native repetition penalty flattens history across batch rows. Preserve
        # the existing per-request semantics by making non-default penalty solo.
        if self.repetition_penalty != 1.0:return None
        reference_mode = self.mode_with_reference if self.reference_audio else 'generation'
        return (reference_mode, self.temperature, self.top_p, self.top_k,
                self.repetition_penalty, self.max_new_tokens)


class RequestBatcher:
    def __init__(self, run_batch, *, max_batch_size=4, window_seconds=.002):
        if max_batch_size < 1 or window_seconds < 0:raise ValueError('invalid batch bounds')
        self.run_batch = run_batch
        self.max_batch_size, self.window_seconds = max_batch_size, window_seconds
        self.condition = threading.Condition()
        self.pending = []
        self.closing = False
        self.running = False
        self.completed_batches = self.completed_requests = self.peak_batch_size = 0
        self.last_batch_size = 0
        self.thread = threading.Thread(target=self._run, name='moss-native-batches', daemon=True)
        self.thread.start()

    def submit(self, item):
        future = Future()
        with self.condition:
            if self.closing:raise RuntimeError('MOSS residency is closing')
            self.pending.append((item, future, time.monotonic()))
            self.condition.notify_all()
        # Each engine RPC (and therefore Hub activity) remains live until its
        # own result/exception is delivered. There is no detached completion.
        return future.result()

    def snapshot(self):
        with self.condition:
            return dict(queued=len(self.pending), running=self.running,
                        completed_batches=self.completed_batches,
                        completed_requests=self.completed_requests,
                        last_batch_size=self.last_batch_size,
                        peak_batch_size=self.peak_batch_size)

    def close(self):
        if threading.current_thread() is self.thread:
            raise RuntimeError('cannot release MOSS from its batch worker')
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        # Do not time out, cancel native work, or release shared weights early.
        self.thread.join()

    def _take(self):
        with self.condition:
            while not self.pending:
                if self.closing:return None
                self.condition.wait()
            first = self.pending[0]
            key = first[0].group_key()
            deadline = first[2] + self.window_seconds
            while key is not None:
                compatible = sum(item[0].group_key() == key for item in self.pending)
                remaining = deadline - time.monotonic()
                if compatible >= self.max_batch_size or remaining <= 0 or self.closing:break
                self.condition.wait(remaining)
            selected = [self.pending.pop(0)]
            if key is not None:
                remaining = []
                for item in self.pending:
                    if len(selected) < self.max_batch_size and item[0].group_key() == key:selected.append(item)
                    else:remaining.append(item)
                self.pending = remaining
            self.running = True
            self.last_batch_size = len(selected)
            self.peak_batch_size = max(self.peak_batch_size, len(selected))
            return selected

    def _run(self):
        while True:
            selected = self._take()
            if selected is None:return
            began = time.monotonic()
            logging.getLogger(__name__).info('moss_native_batch_started size=%d', len(selected))
            try:
                results = self.run_batch([item for item, _, _ in selected])
                if len(results) != len(selected):raise RuntimeError('native batch result count mismatch')
            except BaseException as exc:
                # Every member fails once. Preserve native error class when
                # constructible (notably CUDA OOM); no per-item retry fallback.
                for _, future, _ in selected:
                    try:failure = type(exc)(*exc.args)
                    except Exception:failure = RuntimeError(type(exc).__name__ + ': ' + str(exc))
                    future.set_exception(failure)
            else:
                for (_, future, _), result in zip(selected, results):future.set_result(result)
            finally:
                with self.condition:
                    self.running = False
                    self.completed_batches += 1
                    self.completed_requests += len(selected)
                    self.condition.notify_all()
                logging.getLogger(__name__).info('moss_native_batch_finished size=%d elapsed_seconds=%.3f',len(selected),time.monotonic()-began)

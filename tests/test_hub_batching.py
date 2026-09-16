"""CPU proofs of request ownership, native batching and natural shutdown."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
import importlib
import sys
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hub_runtime.batching import InferenceRequest, RequestBatcher


def item(text='text',**values):
    return replace(InferenceRequest(text,None,'克隆音色',False,1,'中文',1.7,.8,25,1.,4096),**values)


class BatchOwnership(unittest.TestCase):
    def test_four_simultaneous_requests_use_one_native_call(self):
        calls=[];gate=threading.Barrier(4)
        def run(rows):calls.append(rows);return [row.text for row in rows]
        batcher=RequestBatcher(run,window_seconds=.05)
        try:
            def submit(i):gate.wait(2);return batcher.submit(item(str(i)))
            with ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(submit,range(4)))
            self.assertEqual(result,['0','1','2','3']);self.assertEqual(len(calls),1);self.assertEqual(len(calls[0]),4)
            self.assertEqual(batcher.snapshot()['peak_batch_size'],4)
        finally:batcher.close()

    def test_incompatible_sampling_modes_and_penalty_use_separate_batches(self):
        # Hold a first group in flight so the full remaining queue is visible;
        # this tests grouping deterministically without wall-clock assumptions.
        entered=threading.Event();release=threading.Event();calls=[];active=peak=0;lock=threading.Lock()
        def run(rows):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            try:
                calls.append(rows)
                if rows[0].text=='hold':entered.set();self.assertTrue(release.wait(3))
                return [row.text for row in rows]
            finally:
                with lock:active-=1
        batcher=RequestBatcher(run,window_seconds=.002)
        cases=[item('a'),item('b'),item('temperature',temperature=.9),item('limit',max_new_tokens=64),item('continue',reference_audio='a.wav',mode_with_reference='续写'),item('penalty-a',repetition_penalty=1.1),item('penalty-b',repetition_penalty=1.1)]
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                held=pool.submit(batcher.submit,item('hold'));self.assertTrue(entered.wait(2))
                futures=[pool.submit(batcher.submit,row) for row in cases]
                deadline=time.monotonic()+2
                while batcher.snapshot()['queued']!=len(cases):self.assertLess(time.monotonic(),deadline);time.sleep(.005)
                release.set();self.assertEqual(held.result(2),'hold');self.assertEqual([f.result(2) for f in futures],[row.text for row in cases])
            self.assertEqual(peak,1)
            self.assertIn({'a','b'},[{r.text for r in rows} for rows in calls])
            for rows in calls:
                self.assertLessEqual(len(rows),4)
                if rows[0].repetition_penalty!=1:self.assertEqual(len(rows),1)
                else:self.assertEqual(len({r.group_key() for r in rows}),1)
        finally:release.set();batcher.close()

    def test_native_batches_reuse_one_thread_across_idle_and_never_exceed_four(self):
        native_threads=[];sizes=[];gate=threading.Barrier(8)
        def run(rows):native_threads.append(threading.get_ident());sizes.append(len(rows));return [r.text for r in rows]
        batcher=RequestBatcher(run,window_seconds=.03)
        try:
            def submit(i):gate.wait(2);return batcher.submit(item(str(i)))
            with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(submit,range(8)))
            self.assertEqual(results,[str(i) for i in range(8)])
            time.sleep(.01)
            self.assertEqual(batcher.submit(item('after idle')),'after idle')
            self.assertEqual(len(set(native_threads)),1)
            self.assertNotEqual(native_threads[0],threading.get_ident())
            self.assertTrue(all(1<=size<=4 for size in sizes));self.assertEqual(sum(sizes),9)
        finally:batcher.close()

    def test_batch_failure_fails_all_once_without_single_request_retry(self):
        calls=[];gate=threading.Barrier(4)
        def failed(rows):calls.append(rows);raise ValueError('native batch failed')
        batcher=RequestBatcher(failed,window_seconds=.05)
        try:
            def submit(i):gate.wait(2);return batcher.submit(item(str(i)))
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(submit,i) for i in range(4)]
                for future in futures:
                    with self.assertRaisesRegex(ValueError,'native batch failed'):future.result(2)
            self.assertEqual(len(calls),1);self.assertEqual(len(calls[0]),4)
        finally:batcher.close()

    def test_release_waits_for_active_batch_before_clearing_weights(self):
        from hub_runtime.adapter import release as release_model
        entered=threading.Event();finish=threading.Event()
        def run(rows):entered.set();self.assertTrue(finish.wait(3));return [r.text for r in rows]
        batcher=RequestBatcher(run)
        native=types.SimpleNamespace(load_backend=Mock())
        model=types.SimpleNamespace(_batcher=batcher,native=native,backend=('weights','processor'))
        with ThreadPoolExecutor(max_workers=2) as pool:
            call=pool.submit(batcher.submit,item('real activity'));self.assertTrue(entered.wait(2))
            closing=pool.submit(release_model,model)
            try:
                time.sleep(.03);self.assertFalse(closing.done());self.assertIsNotNone(model.backend);native.load_backend.cache_clear.assert_not_called()
                with self.assertRaises(RuntimeError):batcher.submit(item('late'))
            finally:finish.set()
            self.assertEqual(call.result(2),'real activity');closing.result(2)
        native.load_backend.cache_clear.assert_called_once();self.assertIsNone(model.backend)


class NativeBatchShape(unittest.TestCase):
    def test_variable_prompts_duration_and_language_keep_ordered_audio_rows(self):
        class Input:
            def __init__(self,values):self.values=values
            def to(self,device):self.device=device;return self
        class Processor:
            def __init__(self):self.events=[];self.conversations=[]
            def __call__(self,conversations,mode):
                self.events.append('encode');self.conversations=conversations
                longest=max(len(c[0]['text']) for c in conversations)
                # Native processor performs this left-padding itself.
                values=np.array([[0]*(longest-len(c[0]['text']))+[ord(x) for x in c[0]['text']] for c in conversations])
                return {'input_ids':Input(values),'attention_mask':Input(values!=0)}
            def decode(self,outputs):
                self.events.append('decode')
                return [types.SimpleNamespace(audio_codes_list=[np.array(row,dtype=np.float32)]) for row in outputs]
        class Model:
            def __init__(self):self.calls=[]
            def generate(self,**kwargs):
                self.calls.append(kwargs)
                ids=kwargs['input_ids'].values;mask=kwargs['attention_mask'].values
                return [row[row_mask] for row,row_mask in zip(ids,mask)]
        fake_torch=types.SimpleNamespace(Tensor=type('Tensor',(),{}),no_grad=nullcontext)
        native=types.SimpleNamespace(supports_duration_control=lambda mode:mode!='续写',build_conversation=lambda **kw:([[kw]],'generation',kw['mode_with_reference']),normalize_language_tag=lambda x:x,display_language_tag=lambda x:x)
        requests=[item('a',duration_control_enabled=True,duration_tokens=11),item('longer',language_tag='English'),item('xy',duration_control_enabled=True,duration_tokens=99),item('last')]
        with patch.dict(sys.modules,{'torch':fake_torch}):
            module=importlib.import_module('hub_runtime.inference')
            try:
                model=Model();processor=Processor();results=module.run_batch(requests,backend=(model,processor,'cuda:0',24000),codec_lock=threading.RLock(),native=native)
                self.assertEqual(len(model.calls),1);self.assertEqual(processor.events,['encode','decode']);self.assertEqual(model.calls[0]['input_ids'].values.shape,(4,6))
                for req,result in zip(requests,results):self.assertEqual(result[0][0],24000);self.assertEqual(result[0][1].tolist(),[ord(c) for c in req.text])
                self.assertEqual([c[0]['expected_tokens'] for c in processor.conversations],[11,None,99,None]);self.assertIn('expected_tokens=99',results[2][1]);self.assertIn('English',results[1][1])
                self.assertEqual(model.calls[0]['audio_temperature'],1.7);self.assertEqual(model.calls[0]['audio_top_p'],.8);self.assertEqual(model.calls[0]['audio_top_k'],25)
                # A lone non-default penalty still uses the unchanged native call.
                solo=module.run_batch([item('solo',repetition_penalty=1.15)],backend=(model,processor,'cuda:0',24000),codec_lock=threading.RLock(),native=native)
                self.assertEqual(len(solo),1);self.assertEqual(model.calls[-1]['audio_repetition_penalty'],1.15)
                with self.assertRaises(ValueError):module.run_batch([item('x',repetition_penalty=1.15),item('y',repetition_penalty=1.15)],backend=(model,processor,'cuda:0',24000),codec_lock=threading.RLock(),native=native)
            finally:sys.modules.pop('hub_runtime.inference',None)

if __name__=='__main__':unittest.main()

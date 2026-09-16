# syntax=docker/dockerfile:1
ARG BASE_IMAGE
ARG HUB_IMAGE
FROM ${HUB_IMAGE} AS runtime_package
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG HUB_IMAGE
ARG SOURCE_REV
WORKDIR /app
RUN /opt/venv/bin/python -c 'import re,sys; assert all(re.fullmatch(r".+@sha256:[a-f0-9]{64}", v) for v in sys.argv[1:])' "${BASE_IMAGE}" "${HUB_IMAGE}"
COPY --from=runtime_package /opt/hub-runtime-wheel /tmp/hub-runtime-wheel
RUN uv pip install --python /opt/venv/bin/python --no-deps --no-cache /tmp/hub-runtime-wheel/*.whl
COPY hub_runtime /app/hub_runtime
# Preserve the source-controlled original overlay used by the current deployment.
# This is provenance/reference; the new entry never imports api.fusion.
COPY deploy/formal-worker/runtime /opt/moss-formal-baseline
COPY deploy/formal-worker/runtime/clis/moss_tts_app.py /app/clis/moss_tts_app.py
ENV TTD_WEBHOOK_URL="" GRADIO_ANALYTICS_ENABLED="False" MOSS_PRELOAD="0"
LABEL org.opencontainers.image.source="https://github.com/FC-TTD/MOSS-TTS" org.opencontainers.image.revision="${SOURCE_REV}"
RUN /opt/venv/bin/python -m hub_runtime describe > /app/hub_runtime/contract.json
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/venv/bin/python", "-m", "hub_runtime"]

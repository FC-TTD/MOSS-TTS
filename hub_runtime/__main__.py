"""Stable formal API + original root UI; one managed GPU engine per residency."""
import json
import sys
from .adapter import load_model, completion, cleanup, release
from .api import build_api

UI_PATH = "/__gradio__"


class NativeUIRoutes:
    prefixes = ("/assets/", "/favicon", "/manifest.json", "/gradio_api/",
                "/theme.css", "/robots.txt", "/config", "/info")

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] in ('http', 'websocket'):
            path = scope.get('path', '/')
            if path == '/' or any(path == p.rstrip('/') or path.startswith(p) for p in self.prefixes):
                scope = dict(scope)
                scope['path'] = UI_PATH + path
                scope['raw_path'] = UI_PATH.encode() + scope.get('raw_path', path.encode())
            if scope.get('path', '').startswith(UI_PATH + '/'):
                scope = dict(scope)
                scope['headers'] = [(k, v) for k, v in scope.get('headers', [])
                                    if k.lower() != b'x-forwarded-host']
        await self.app(scope, receive, send)


def create_app(runtime=None):
    from ttd_model_runtime import Runtime
    from ttd_model_runtime.integrations.fastapi import attach
    runtime = runtime or Runtime(load_model, completion=completion, cleanup=cleanup,
                                 release=release)
    app = build_api(runtime)
    def ui():
        from .ui import build_ui
        return build_ui(runtime)
    app.add_middleware(NativeUIRoutes)
    return attach(app, runtime=runtime, ui_factory=ui, ui_path=UI_PATH)


def describe():
    class Contract:
        def task(self, function):return function
    return build_api(Contract()).openapi()


def main():
    if sys.argv[1:] == ['describe']:
        print(json.dumps(describe(), ensure_ascii=False));return
    if sys.argv[1:]:raise SystemExit('Use python -m hub_runtime [describe]')
    import uvicorn
    uvicorn.run(create_app(), host='0.0.0.0', port=8000)


if __name__ == '__main__':main()

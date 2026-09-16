"""Formal API and copied native UI against CPU fixtures; no real model loads."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hub_runtime import api,parameters
from hub_runtime.__main__ import create_app,describe
import hub_runtime_fixture as fixture


class Contract:
    def task(self,fn):return fn
    def __init__(self):self.model=fixture.Model()
    def get(self):return self.model
    def status(self):return dict(residency='unloaded',healthy=True)


def native_api_module():
    module=types.ModuleType('clis.moss_tts_app')
    for name in ['DEFAULT_ATTN_IMPLEMENTATION','DEFAULT_MAX_NEW_TOKENS','LANGUAGE_TAG_MAP','MODE_CLONE','MODE_CONTINUE','MODE_CONTINUE_CLONE','estimate_duration_tokens']:
        setattr(module,name,getattr(parameters,name))
    module.load_backend=lambda **kwargs:None;module.run_inference=lambda *args:None
    spec=importlib.util.spec_from_file_location('native_api_baseline',ROOT/'deploy/formal-worker/runtime/api/moss_formal.py')
    native=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'clis.moss_tts_app':module}):spec.loader.exec_module(native)
    return native


def original_cpu_ui():
    # Execute the original UI portion unchanged; omit only GPU/CLI definitions
    # and GPU imports, so collecting controls cannot instantiate native weights.
    text=(ROOT/'deploy/formal-worker/runtime/clis/moss_tts_app.py').read_text();tree=ast.parse(text)
    excluded={'load_backend','resolve_attn_implementation','run_inference','main'}
    body=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)) and any(x in ast.get_source_segment(text,node) for x in ['import torch','from transformers']):continue
        if isinstance(node,ast.Expr) and ast.get_source_segment(text,node).startswith('torch.backends.'):continue
        if isinstance(node,ast.FunctionDef) and node.name in excluded:continue
        if isinstance(node,ast.If):continue
        body.append(node)
    tree.body=body
    module=types.ModuleType('native_ui_baseline');module.__file__=str(ROOT/'clis/moss_tts_app.py')
    module.run_inference=lambda **kwargs:((24000,np.zeros(24000,dtype=np.float32)),'fixture')
    exec(compile(tree,'native_ui_baseline','exec'),module.__dict__)
    return module


@contextmanager
def served(app):
    import httpx,uvicorn
    sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen()
    server=uvicorn.Server(uvicorn.Config(app,log_level='error',timeout_graceful_shutdown=2))
    thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
    try:
        deadline=time.monotonic()+10
        while not server.started:
            if not thread.is_alive() or time.monotonic()>deadline:raise RuntimeError('fixture HTTP startup failed')
            time.sleep(.01)
        with httpx.Client(base_url=f'http://127.0.0.1:{sock.getsockname()[1]}',timeout=20) as client:yield client
    finally:
        server.should_exit=True;thread.join(10);sock.close()
        if thread.is_alive():raise RuntimeError('fixture HTTP server did not stop')


class MOSSAdoption(unittest.TestCase):
    def test_form_contract_and_tuned_defaults_match_formal_api(self):
        before=native_api_module().app.openapi();after=describe()
        for path in ['/generate','/api/tts','/v1/audio/speech']:
            self.assertEqual(before['paths'][path],after['paths'][path])
        for key,value in before['components']['schemas'].items():self.assertEqual(value,after['components']['schemas'][key])
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('transformers',sys.modules)

    def test_api_upload_aliases_modes_parameters_and_timing_preserved(self):
        runtime=Contract();client=TestClient(api.build_api(runtime));buf=io.BytesIO();sf.write(buf,np.zeros(2400),24000,format='WAV')
        for index,alias in enumerate(['prompt_wav','reference_audio','prompt_speech','ref_audio']):
            response=client.post(['/generate','/api/tts','/v1/audio/speech'][index%3],data={'text':'原声文本','mode':'continue-clone','prompt_text':'accepted as before','language':'zh','temperature':'1.6','top_p':'.75','top_k':'30','repetition_penalty':'1.1','max_new_tokens':'1024','duration_tokens':'45'},files={alias:('ref.wav',buf.getvalue(),'audio/wav')})
            self.assertEqual(response.status_code,200,response.text);last=runtime.model.last_call();self.assertTrue(last['reference_exists']);self.assertFalse(Path(last['reference_path']).exists());self.assertEqual(last['mode'],parameters.MODE_CONTINUE_CLONE)
            self.assertEqual((last['temperature'],last['top_p'],last['top_k'],last['repetition_penalty'],last['max_new_tokens']),(1.6,.75,30,1.1,1024));self.assertEqual(last['tokens'],45);self.assertEqual(last['language'],'中文')
            wave,rate=sf.read(io.BytesIO(response.content));self.assertEqual(rate,24000);self.assertAlmostEqual(len(wave)/rate,1);self.assertEqual(response.headers['x-duration-tokens'],'45')
        response=client.post('/generate',data={'text':'timing','speed':'较快','expected_duration':'.5','duration_control_enabled':'true'})
        self.assertEqual(response.status_code,200,response.text);last=runtime.model.last_call();self.assertFalse(last['enabled']);self.assertEqual(last['tokens'],1);self.assertEqual((last['temperature'],last['top_p'],last['top_k']),(1.7,.8,25));self.assertEqual(response.headers['x-final-speed-factor'],'2.000000');self.assertAlmostEqual(float(response.headers['x-measured-duration']),.5,delta=.03)
        self.assertEqual(client.post('/model/unload').status_code,409)
        self.assertEqual(client.post('/generate',data={'text':'fail'}).status_code,400)

    def test_original_ui_controls_and_asr_callbacks_preserved(self):
        from hub_runtime import ui
        baseline=original_cpu_ui();args=types.SimpleNamespace(model_path='fixed',device='cuda:0',attn_implementation='auto')
        original=baseline.build_demo(args);adopted=ui.build_ui(Contract())
        def controls(demo):
            keys=['label','value','minimum','maximum','step','choices','info','visible','interactive','lines','placeholder']
            return [(c['type'],{k:c['props'][k] for k in keys if k in c.get('props',{})}) for c in demo.config['components']]
        self.assertEqual(controls(original),controls(adopted))
        self.assertEqual([d['api_name'] for d in original.config['dependencies']],[d['api_name'] for d in adopted.config['dependencies']])
        with patch.object(ui,'get_asr_client') as asr:
            asr.return_value.predict.return_value='参考原文'
            with patch.object(ui,'handle_file',return_value='file'):
                self.assertEqual(ui.auto_fill_reference_transcript('ref.wav','后续文本',ui.MODE_CONTINUE),'参考原文\n后续文本')
                self.assertEqual(ui.auto_fill_reference_transcript('ref.wav','原文',ui.MODE_CLONE),'原文')
            self.assertEqual(asr.return_value.predict.call_count,1)

    def test_loader_preserves_formal_split_and_release_clears_native_lru(self):
        from hub_runtime import adapter
        from unittest.mock import Mock
        native=types.SimpleNamespace(DEFAULT_ATTN_IMPLEMENTATION='auto',load_backend=Mock(),run_inference=Mock(return_value=((24000,np.zeros(4,dtype=np.float32)),'native')))
        main_model=types.SimpleNamespace(parameters=lambda:iter([types.SimpleNamespace(device='cuda:0',dtype='torch.bfloat16')]))
        processor=types.SimpleNamespace(audio_tokenizer=types.SimpleNamespace(parameters=lambda:iter([types.SimpleNamespace(device='cpu',dtype='torch.float32')])))
        backend=(main_model,processor,'cuda:0',24000);native.load_backend.return_value=backend
        fake_torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True))
        clis=types.ModuleType('clis');clis.moss_tts_app=native
        with patch.dict(sys.modules,{'torch':fake_torch,'clis':clis}),patch.dict(os.environ,{'DEVICE':'cuda:0','MODEL_PATH':'fixed-model','MOSS_AUDIO_TOKENIZER_DEVICE':'cpu','ATTN_IMPLEMENTATION':'auto'}):
            model=adapter.load_model();self.assertIs(model.backend,backend)
            native.load_backend.assert_called_once_with(model_path='fixed-model',device_str='cuda:0',attn_implementation='auto')
            inference=types.ModuleType('hub_runtime.inference');inference.run_batch=Mock(return_value=[((24000,np.zeros(4)),'native')])
            with patch.dict(sys.modules,{'hub_runtime.inference':inference}):
                model.infer('text',None,parameters.MODE_CLONE,False,1,'中文',1.7,.8,25,1.,4096)
            from hub_runtime.batching import InferenceRequest
            inference.run_batch.assert_called_once_with([InferenceRequest('text',None,parameters.MODE_CLONE,False,1,'中文',1.7,.8,25,1.,4096)],backend=backend,codec_lock=model._codec_lock,native=native)
            self.assertEqual(model.__hub_device_summary__,{'main_model':'cuda:0','main_dtype':'torch.bfloat16','audio_tokenizer':'cpu','audio_tokenizer_dtype':'torch.float32'})
            adapter.release(model);native.load_backend.cache_clear.assert_called_once();self.assertIsNone(model.backend);self.assertIsNone(model.native)
        self.assertNotIn('torch',sys.modules)
        self.assertEqual(adapter.actual_tensor_placement(object()),('unknown','unknown'))

    def test_root_ui_queue_and_api_use_live_process_without_parent_weights(self):
        from ttd_model_runtime import Runtime
        from gradio_client import Client
        lease=dict(id='moss-fixture',service='moss-tts',generation=1,revision=1,gpu='GPU-658f2323-d990-2adf-d488-d5dd945a27d5',fingerprint='a'*64)
        class Node:
            service='moss-tts'
            def __init__(self):self.ends={};self.begins=[]
            def begin(self,key,grant=None):
                activity=dict(id=uuid4().hex,key=key,service=self.service,lease_id=lease['id'],generation=lease['generation']);self.begins.append(activity);return activity
            def finish(self,activity,state):self.ends[activity['id']]=state
        node=Node()
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'','GRADIO_ANALYTICS_ENABLED':'False'}):
            runtime=Runtime(fixture.load_model,completion=fixture.completion,cleanup=fixture.cleanup,node=node,token='t'*32,version='cpu-fixture',state_dir=directory,gpu_process=True)
            with served(create_app(runtime)) as client:
                self.assertEqual(client.get('/').status_code,200);self.assertEqual(client.get('/health').status_code,200);self.assertEqual(client.get('/model/status').json()['loaded'],False);self.assertEqual(client.get('/capabilities').status_code,200);self.assertIsNone(runtime._engine);self.assertEqual(node.begins,[])
                from urllib import request
                req=request.Request(f'http://127.0.0.1:{runtime.control_port}/hub-runtime/v1/load',data=json.dumps(lease).encode(),headers={'Authorization':'Bearer '+'t'*32,'Content-Type':'application/json'})
                with request.build_opener(request.ProxyHandler({})).open(req,timeout=10) as response:self.assertEqual(response.status,200)
                self.assertEqual(client.post('/generate',data={'text':'managed API'}).status_code,200)
                self.assertNotEqual(runtime._model.last_call()['pid'],os.getpid())
                ui=Client(str(client.base_url),headers={'X-Forwarded-Host':'moss-tts'},verbose=False)
                try:
                    result=ui.predict('managed UI',None,parameters.MODE_CLONE,False,1,parameters.LANGUAGE_TAG_AUTO,1.7,.8,25,1.0,4096,api_name='/lambda')
                    audio=Path(result[0]);self.assertTrue(audio.is_file());wave,rate=sf.read(audio);self.assertEqual(rate,24000);self.assertTrue(np.any(wave));self.assertEqual(result[1],'native fixture completed')
                finally:ui.close()
                self.assertEqual(len(node.begins),2);self.assertEqual(set(node.ends.values()),{'succeeded'});self.assertEqual(runtime.status()['active'],0)
                runtime.unload(dict(lease,revision=2));self.assertTrue(runtime.status()['engine']['group_empty'])
            self.assertFalse(runtime._started)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()

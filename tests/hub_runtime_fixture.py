"""CPU-only native-shaped fixture. No weights, Torch or external ASR calls."""
import gc
import os
from pathlib import Path
import numpy as np


class Model:
    def __init__(self):self.history=[]
    def infer(self,*args):
        text,reference,mode,enabled,tokens,language,temp,top_p,top_k,penalty,max_tokens=args
        self.history.append(dict(text=text,has_reference=bool(reference),reference_exists=bool(reference and Path(reference).is_file()),reference_path=reference,mode=mode,enabled=enabled,tokens=tokens,language=language,temperature=temp,top_p=top_p,top_k=top_k,repetition_penalty=penalty,max_new_tokens=max_tokens,pid=os.getpid()))
        if text=='fail':raise ValueError('native failure')
        rate=24000;wave=(.15*np.sin(2*np.pi*220*np.arange(rate)/rate)).astype(np.float32)
        return (rate,wave),'native fixture completed'
    def last_call(self):return self.history[-1]
    def call_count(self):return len(self.history)


def load_model():return Model()
def completion(model):pass
def cleanup():gc.collect()

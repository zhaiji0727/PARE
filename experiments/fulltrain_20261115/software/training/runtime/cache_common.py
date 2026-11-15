"""Shared non-executing helpers for independently routed training-cache readers."""
import hashlib
import json
from pathlib import Path
DIMENSIONS={'vitb32':512,'vitb16':512,'remoteclip':512,'siglip':768,'georsclip':512}
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def objsha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

class BoundReader:
    def __init__(self):self.files={}
    def bind(self,row):
        p=Path(row['path']).resolve();require(p.is_file() and sha(p)==row['sha256'],'Bound file changed: '+str(p))
        if 'bytes' in row:require(p.stat().st_size==row['bytes'],'Bound size changed')
        if str(p) in self.files:require(self.files[str(p)]==row['sha256'],'Conflicting binding')
        self.files[str(p)]=row['sha256'];return p
    def json(self,row):return read(self.bind(row))
    def finish(self):
        for p,d in self.files.items():require(sha(p)==d,'Input changed during run: '+p)

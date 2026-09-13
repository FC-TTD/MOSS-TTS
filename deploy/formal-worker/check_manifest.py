"""Read-only identity check for the preserved formal runtime overlay."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'source-manifest.json').read_text())
for name, digest in manifest['files'].items():
    actual = hashlib.sha256((root / 'runtime' / name).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f'Runtime source differs from the reviewed existing formal file: {name}')
print(f"Verified {len(manifest['files'])} unchanged runtime source files")

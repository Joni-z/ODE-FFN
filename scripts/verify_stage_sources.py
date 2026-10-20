"""Verify an immutable stage snapshot before any queued work starts."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'source-manifest.json').read_text())
for name,expected in manifest['files'].items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=expected:raise SystemExit(f'Stage source differs: {name}')
data_root=root/'data/synthetic/v1'
actual_data=hashlib.sha256((data_root/'manifest.json').read_bytes()).hexdigest()
if actual_data!=manifest['data_manifest_sha256']:
    raise SystemExit('Stage data manifest differs: freeze the actual execution-host manifest, not a local reconstruction')
data_manifest=json.loads((data_root/'manifest.json').read_text())
for name,entry in data_manifest['files'].items():
    if hashlib.sha256((data_root/name).read_bytes()).hexdigest()!=entry['sha256']:
        raise SystemExit(f'Stage data file differs: {name}')
print(f"Stage snapshot verified: {len(manifest['files'])} files",flush=True)

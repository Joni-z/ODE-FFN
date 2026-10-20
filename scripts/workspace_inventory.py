"""Read-only inventory of the three known JiT workspaces and legacy checkpoints."""
import csv
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def git(path, *args):
    result = subprocess.run(['git','-C',str(path),*args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    repos = []
    for path in (ROOT, Path('/scratch/zz5070/jit-codebase'), Path('/scratch/zz5070/jit-codebase-tmp')):
        repos.append({'path':str(path), 'exists':path.exists(), 'head':git(path,'rev-parse','HEAD'),
                      'status':git(path,'status','--short')})
    (WORKSPACE/'workspace-inventory.json').write_text(json.dumps({'repositories':repos},indent=2)+'\n')
    with (ROOT/'docs/checkpoints.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path','bytes','mtime_ns'])
        for path in sorted((ROOT/'results').rglob('*.pt')):
            stat = path.stat()
            writer.writerow([str(path.relative_to(ROOT)), stat.st_size, stat.st_mtime_ns])
    print('Wrote workspace-inventory.json and docs/checkpoints.csv')


if __name__ == '__main__':
    main()

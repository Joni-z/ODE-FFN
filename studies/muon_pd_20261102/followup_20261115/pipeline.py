"""Sequential native training and registered frozen evaluation for one queue cell."""
import json
from pathlib import Path
import subprocess
import sys
from train import HERE, TAG, base


def main():
    args=base.argument_parser().parse_args()
    assert args.steps is None and args.resume is None,'Readiness uses check.py, not this formal pipeline'
    cell=next(c for c in json.loads((HERE/'grid.json').read_text())['cells'] if c['id']==args.cell)
    status=Path(args.output)/'pipeline'/f'{args.cell}.json'
    def run(phase,cmd):
        base.dump(status,dict(cell=args.cell,phase=phase,command=cmd))
        subprocess.run(cmd,check=True)
    run('training',[sys.executable,'-u',str(HERE/'train.py'),*sys.argv[1:]])
    folder=Path(args.output)/TAG/cell['dataset']/cell['id']/'seed000'
    evaluation=Path(args.output)/'evaluations'/cell['id']
    cmd=[sys.executable,'-u',str(HERE/'evaluate.py'),'--folder',str(folder),
         '--data-root',args.data_root,'--output',str(evaluation),'--device',args.device]
    if cell['dataset']=='gmm_ring8_2d':cmd+=['--c8']
    run('frozen_evaluation',cmd)
    assert json.loads((evaluation/'COMPLETE.json').read_text())['passed']
    base.dump(status,dict(cell=args.cell,phase='complete',training_folder=str(folder),evaluation=str(evaluation)))

if __name__=='__main__':main()

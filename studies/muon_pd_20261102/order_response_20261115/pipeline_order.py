"""Native training, common-bank sampling and registered truncation diagnosis."""
import json
from pathlib import Path
import subprocess
import sys
from order_trial import HERE,TAG,base


def main():
    args=base.argument_parser().parse_args();assert args.steps is None and args.resume is None
    cell=next(c for c in json.loads((HERE/'grid.json').read_text())['cells'] if c['id']==args.cell)
    root=Path(args.output);status=root/'pipeline'/f'{args.cell}.json'
    def run(phase,command):
        base.dump(status,dict(cell=args.cell,phase=phase,command=command));subprocess.run(command,check=True)
    run('training',[sys.executable,'-u',str(HERE/'order_trial.py'),*sys.argv[1:]])
    folder=root/TAG/cell['dataset']/cell['id']/'seed000';evaluation=root/'evaluations'/cell['id'];diagnosis=root/'diagnostics'/cell['id']
    cmd=[sys.executable,'-u',str(HERE/'evaluate_order.py'),'--folder',str(folder),'--data-root',args.data_root,'--output',str(evaluation),'--device',args.device]
    if cell['dataset']=='gmm_ring8_2d':cmd+=['--c8']
    run('frozen_evaluation',cmd)
    run('term_diagnostics',[sys.executable,'-u',str(HERE/'diagnostics.py'),'--folder',str(folder),'--evaluation',str(evaluation),'--data-root',args.data_root,'--output',str(diagnosis),'--device',args.device])
    assert all(json.loads((p/'COMPLETE.json').read_text())['passed'] for p in (evaluation,diagnosis))
    base.dump(status,dict(cell=args.cell,phase='complete',training_folder=str(folder),evaluation=str(evaluation),diagnostics=str(diagnosis)))

if __name__=='__main__':main()

"""Native training followed by source-bound selection evaluation only."""
import json,os,subprocess,sys
from pathlib import Path
from screen_trial import HERE,TAG,base

def main():
 args=base.argument_parser().parse_args();assert args.steps is None and args.resume is None
 cell=next(c for c in json.loads((HERE/'grid.json').read_text())['cells'] if c['id']==args.cell)
 root=Path(args.output);phase=root/'pipeline'/(cell['id']+'.json')
 def run(name,cmd):base.dump(phase,dict(cell=args.cell,phase=name,command=cmd));subprocess.run(cmd,check=True)
 run('training',[sys.executable,'-u',str(HERE/'screen_trial.py'),*sys.argv[1:]])
 folder=root/TAG/cell['dataset']/cell['id']/'seed000';out=root/'selection'/cell['id']
 run('selection',[sys.executable,'-u',str(HERE/'evaluate_screen.py'),'--folder',str(folder),'--data-root',args.data_root,
       '--assets',os.environ['RESPONSE_SCREEN_ASSETS'],'--output',str(out),'--device',args.device])
 assert json.loads((out/'COMPLETE.json').read_text())['passed']
 base.dump(phase,dict(cell=args.cell,phase='complete',folder=str(folder),selection=str(out)))

if __name__=='__main__':main()

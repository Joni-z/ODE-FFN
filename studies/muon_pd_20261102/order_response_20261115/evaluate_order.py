"""Use the unchanged common-bank evaluator with the order-aware model builder."""
import json
import sys
from pathlib import Path
from order_trial import HERE,base,build_model
import importlib.util
_spec=importlib.util.spec_from_file_location('response_evaluation_parent',HERE.parent/'followup_20261115/evaluate.py')
parent=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent)

parent.build_model=build_model

if __name__=='__main__':
    parent.main()
    out=Path(sys.argv[sys.argv.index('--output')+1])
    base.dump(out/'order-adapter.json',dict(builder_sha256=base.digest(HERE/'order_trial.py'),
              evaluator_sha256=base.digest(parent.__file__),adapter_sha256=base.digest(__file__)))

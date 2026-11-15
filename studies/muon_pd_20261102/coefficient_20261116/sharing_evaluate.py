"""Native metric/sampler with the registered stage-operator builder."""
from sharing_trial import build_model
import evaluate_screen as evaluator
evaluator.ev.build_model=build_model
if __name__=='__main__':evaluator.main()

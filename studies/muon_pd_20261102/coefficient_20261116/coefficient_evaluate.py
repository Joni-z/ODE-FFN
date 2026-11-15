"""Use the existing native sampling/metric evaluation with this model builder."""
from coefficient_trial import build_model
import evaluate_screen as evaluator
evaluator.ev.build_model=build_model

if __name__=='__main__':evaluator.main()

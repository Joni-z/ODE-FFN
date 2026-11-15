"""Original common-bank evaluator with the state-evolving builder."""
from state_trial import build_model
import coefficient_evaluate as parent
parent.evaluator.ev.build_model=build_model

if __name__=='__main__':parent.evaluator.main()

"""Unchanged common-bank evaluator with only the RK4 model-builder callback."""
from rk4_trial import build_model
import coefficient_evaluate as parent
parent.evaluator.ev.build_model = build_model

if __name__ == '__main__':
    parent.evaluator.main()

"""Native checkpoint evaluator with the fixed mean-preserving-stage builder."""
from contrast_trial import build_model
import evaluate_screen as evaluator

evaluator.ev.build_model = build_model

if __name__ == '__main__':
    evaluator.main()

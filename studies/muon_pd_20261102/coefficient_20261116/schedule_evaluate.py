"""Existing common-bank evaluator with the matched chronological-stage builder."""
from schedule_trial import build_model
import evaluate_screen as evaluator

evaluator.ev.build_model = build_model

if __name__ == '__main__':
    evaluator.main()

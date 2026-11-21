"""Bind the archived fixed-bank evaluator to the solver under test."""
from efficient_trial import build_model
from evaluate_screen import ev, main

if __name__ == "__main__":
    ev.build_model = build_model
    main()

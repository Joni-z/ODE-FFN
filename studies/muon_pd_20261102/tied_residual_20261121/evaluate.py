"""Apply the existing eight-bank evaluator to a trained tied residual model."""
from tied_residual import build_model
import evaluate_screen


if __name__ == "__main__":
    evaluate_screen.ev.build_model = build_model
    evaluate_screen.main()

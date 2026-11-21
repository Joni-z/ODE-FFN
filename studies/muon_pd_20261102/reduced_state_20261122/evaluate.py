"""Apply the unchanged confirmation protocol to reduced-state study checkpoints."""
from reduced_trial import build_model
from evaluate_screen import ev, main


if __name__ == "__main__":
    ev.build_model = build_model
    main()

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v tectonic >/dev/null 2>&1; then
    tectonic report1.tex
elif command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error report1.tex
else
    pdflatex -interaction=nonstopmode -halt-on-error report1.tex
    pdflatex -interaction=nonstopmode -halt-on-error report1.tex
fi

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error report2.tex
elif command -v tectonic >/dev/null 2>&1; then
    tectonic --keep-logs report2.tex
else
    pdflatex -interaction=nonstopmode -halt-on-error report2.tex
    pdflatex -interaction=nonstopmode -halt-on-error report2.tex
fi

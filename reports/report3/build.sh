#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${REPORT_PYTHON:-python3}" make_artifacts.py
if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error report3.tex
elif command -v tectonic >/dev/null 2>&1; then
    tectonic --keep-logs report3.tex
else
    pdflatex -interaction=nonstopmode -halt-on-error report3.tex
    pdflatex -interaction=nonstopmode -halt-on-error report3.tex
fi

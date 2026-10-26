"""Independent QR/coordinate-energy check of the SVD projection analysis."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT, DOCS = map(Path, sys.argv[1:3])
rows = list(csv.DictReader((DOCS/'error-components.csv').open()))
teachers = list(csv.DictReader((DOCS/'teacher-components.csv').open()))
maximum = 0.
for j in range(3):
    z = np.load(ROOT/'output/cache'/f'validation-z{j}.npy').astype(np.float64)
    a = np.column_stack((np.ones(len(z)), z))
    q, r = np.linalg.qr(a, mode='reduced')
    assert np.linalg.matrix_rank(r, tol=np.linalg.norm(r, 2)*1e-12) == 65
    def energy(value):
        coefficient = q.T@value
        affine = float(np.sum(coefficient*coefficient)/value.size)
        total = float(np.sum(value*value)/value.size)
        return affine, total-affine
    for teacher in ('ode_k2', 'swiglu128'):
        target = np.load(ROOT/'output/cache'/f'validation-{teacher}-y{j}.npy').astype(np.float64)
        wanted = next(row for row in teachers if row['teacher']==teacher and int(row['position'])==j)
        affine, remaining = energy(target)
        for actual, key in [(affine, 'affine_energy'), (remaining, 'remaining_energy')]:
            maximum = max(maximum, abs(actual-float(wanted[key])))
            np.testing.assert_allclose(actual, float(wanted[key]), rtol=2e-11, atol=2e-13)
        for family in ('ode_k2', 'swiglu128', 'swiglu192'):
            prediction = np.load(ROOT/'output/runs'/f'{teacher}--{family}--{j}'/'validation_0005000.npy').astype(np.float64)
            affine, remaining = energy(prediction-target)
            wanted = next(row for row in rows if row['teacher']==teacher and int(row['position'])==j and row['student']==family)
            for actual, key in [(affine,'affine_error'),(remaining,'remaining_error')]:
                maximum = max(maximum, abs(actual-float(wanted[key])))
                np.testing.assert_allclose(actual, float(wanted[key]), rtol=2e-11, atol=2e-13)
for contrast in csv.DictReader((DOCS/'component-contrasts.csv').open()):
    def find(family):
        return next(row for row in rows if row['teacher']==contrast['teacher'] and row['position']==contrast['position'] and row['student']==family)
    ode, other = find('ode_k2'), find(contrast['generic'])
    for key, destination in [('affine_error','affine_excess'),('remaining_error','remaining_excess'),('total_error','total_ODE_excess')]:
        np.testing.assert_allclose(float(ode[key])-float(other[key]),float(contrast[destination]),rtol=2e-11,atol=2e-13)
receipt = dict(status='PASS',decompositions=18,teacher_decompositions=6,contrasts=12,
               independent_method='unpivoted full-rank QR and coefficient energy, versus analysis SVD and explicit residual',
               maximum_absolute_difference=maximum,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(DOCS/'projection-audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

"""Independent raw-endpoint screen and immutable-source/process receipt."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import psutil

HERE = Path(__file__).resolve().parents[1]
ROOT = Path('/Users/mr.z/ode-workspace-prep/architecture-search/sigmoid_glu_20261025_m1')
PRIOR = ROOT.parent/'state_clock_20261025_m1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
raw, statuses, recorded_pids, telemetry = [], {}, set(), []
for suffix in ('v1', 'product16'):
    study = ROOT/'output'/f'sigmoid_glu_{suffix}_20261025_m1'
    status = json.loads((study/'status.json').read_text())
    assert status['state'] == 'COMPLETE' and status['completed'] == status['total']
    statuses[suffix] = status
    raw += json.loads((study/'results.json').read_text())
    history = [json.loads(line) for line in (study/'telemetry.jsonl').read_text().splitlines()]
    for row in history:
        recorded_pids.add(row['pid'])
        recorded_pids.update(p['pid'] for p in row.get('processes', []))
    active = [row for row in history if row.get('processes')]
    telemetry.append(dict(study=study.name, samples=len(history), first_active=active[0],
                          minimum_available_gib=min(row['memory_available_gib'] for row in history)))
queue = json.loads((ROOT/'output/sigmoid_glu_queue_20261025_m1/status.json').read_text())
assert queue['state'] == 'COMPLETE' and queue['completed'] == queue['total'] == 2
assert len(raw) == 9
lookup = {(row['dataset'], row['label']): row for row in raw}
dec = json.loads((HERE/'screen-decision.json').read_text())
primary_ok, secondary_ok, qualifying, details = True, True, [], {}
datasets = ('gaussian_aniso_2d', 'gmm_ring8_2d', 'rotated_product_bimodal_16d')
for dataset in datasets:
    ratios = {}
    for metric in ('validation_oracle_mse', 'jacobian_error', 'validation_sw2'):
        numerator = lookup[dataset, 'sigmoid_glu'][metric]
        denominator = min(lookup[dataset, label][metric] for label in ('swiglu_budget', 'swiglu'))
        ratios[metric] = numerator/denominator
        assert ratios[metric] == dec['details'][dataset]['ratio_to_better_swiglu'][metric]
    primary_ok &= ratios['validation_oracle_mse'] <= 1.05
    if dataset != datasets[0]:
        secondary_ok &= ratios['jacobian_error'] <= 1.1 and ratios['validation_sw2'] <= 1.1
        if ratios['validation_oracle_mse'] <= .9: qualifying.append(dataset)
    details[dataset] = ratios
assert primary_ok == dec['all_primary_within_5pct']
assert secondary_ok == dec['nonlinear_Jacobian_SW2_within_10pct']
assert qualifying == dec['qualifying_nonlinear_leads']
assert bool(qualifying and primary_ok and secondary_ok) == dec['generic_baseline_lead'] == False
csv_rows = list(csv.DictReader((HERE/'comparison.csv').open()))
for row in csv_rows:
    for metric in ('validation_oracle_mse', 'field_mse', 'jacobian_error', 'validation_sw2', 'parameters', 'forward_macs'):
        assert float(row[metric]) == lookup[row['dataset'], row['label']][metric]
source_checks = {}
for root, expected in ((ROOT, 587), (PRIOR, 580)):
    manifest = json.loads((root/'source-manifest.json').read_text())
    names = {str(p.relative_to(root/'source')) for p in (root/'source').rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    assert names == set(manifest['all_files']) and len(names) == expected
    for name, digest in manifest['all_files'].items(): assert sha(root/'source'/name) == digest
    source_checks[str(root)] = dict(files=expected, exact_tree=True, manifest_sha256=sha(root/'source-manifest.json'))
launch = json.loads((ROOT/'launch-review.json').read_text())
assert launch['ready_sha256'] == sha(ROOT/'ready.json')
assert launch['preflight_summary_sha256'] == sha(ROOT/'preflight/summary.json')
assert launch['source_manifest_sha256'] == sha(ROOT/'source-manifest.json')
provenance = json.loads((HERE/'provenance.json').read_text())
for path, digest in provenance['input_sha256'].items(): assert sha(path) == digest, path
live = []
for process in psutil.process_iter(['pid', 'cmdline'], ad_value=None):
    command = process.info['cmdline']
    if command and process.pid != os.getpid() and str(ROOT/'source/scripts/run_architecture_study.py') in command:
        live.append(process.info)
assert not live
pid_state = {str(pid): psutil.pid_exists(pid) for pid in sorted(recorded_pids)}
assert not any(pid_state.values())
result = dict(state='PASS', checked_at=datetime.now(timezone.utc).isoformat(),
              independent_source_sha256=sha(__file__), terminal_states=statuses, queue_terminal=queue,
              recorded_pids_exist=pid_state, live_formal_runner_matches=live, telemetry=telemetry,
              source_checks=source_checks, analysis_input_hashes_verified=len(provenance['input_sha256']),
              raw_endpoint_ratios=details, screen='STOP_GENERIC_SETTING', architecture_goal_complete=False,
              figures_visually_inspected=['sigmoid-glu-endpoints', 'sigmoid-glu-learning', 'sigmoid-glu-posterior-regions'],
              observation_limits='Final monitor training subrows may lag checkpoints; terminal statuses and native audits are authoritative. Queue status PID equals the final child supervisor and is not asserted to be the actual queue PID.')
(HERE/'independent-check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'state': result['state'], 'source_files': [587, 580], 'recorded_exited_pids': len(pid_state),
                  'input_hashes_verified': result['analysis_input_hashes_verified'], 'screen': result['screen']}))

"""Render a one-page audit summary without modifying numerical environments."""
from pathlib import Path
import argparse,json,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();root=a.root
proof=json.loads((root/'proof-review/coverage-and-verdict.json').read_text());assert proof['open_fatal']==proof['open_critical']==0 and proof['acceptance_status']=='provisional'
out=root/'curation';f=plt.figure(figsize=(8.27,11.69));ax=f.add_axes([.09,.07,.82,.86]);ax.axis('off')
ax.text(0,1.0,'Proof Audit: Supervised Denoiser Dynamics',fontsize=16,weight='bold',va='top')
ax.text(0,.95,'Zhizhe Zhang | 28 October 2026',fontsize=10,va='top')
paragraphs=[
('Assurance','Native same-family review: provisional. No cross-family semantic acceptance, theorem novelty or architecture improvement is claimed. Actual returned model identity was unavailable; the requested reviewer route is recorded separately.'),
('Scope','All nine source units U0-U8 and 35 extracted microclaims were checked, then rechecked in the same session. Source-unit execution coverage is complete; coverage does not prove that all possible obligations were discovered.'),
('Before and after','R1: an implicit conditional-mean version was replaced by an explicit smooth Bayes-ratio representative. R2: density and vector-domain notation were clarified. The U6-U4 dependency was recorded. No mathematical formula, prior class or numerical tolerance changed.'),
('Reviewed mathematical relations','The supervised conditional mean obeys a necessary heat/denoiser PDE. It need not be invariant along the marginal flow. A wrong-prior denoiser can satisfy the PDE. A precision derivative generally includes a contracted third posterior moment. These are bounded mathematical statements, not a new model result.'),
('Separate numerical verification','All 120 fixed existing-law probes and 240 central-difference checks pass their registered gates. Complete derivative arrays and both step sizes remain available. Numerical reproduction and prior-file application were outside the semantic reviewer scope.'),
('Remaining scope','No new architecture, PDE regularizer, target change or covariance rescue is selected. Report1 and all previous data, training checkpoints and P histories are unchanged. The useful-ODE-successor objective remains open.'),
('Build provenance','This PDF is rendered with Matplotlib. A separate LaTeX summary source is supplied, but no local TeX compiler was found and no TeX compilation is claimed.')]
y=.89
for title,body in paragraphs:
 ax.text(0,y,title,fontsize=11,weight='bold',va='top');y-=.029
 lines=textwrap.wrap(body,91);ax.text(0,y,'\n'.join(lines),fontsize=9.3,va='top',linespacing=1.4);y-=.021*len(lines)+.022
assert y>-.01,y
for ext in ('pdf','png'):f.savefig(out/f'proof-audit-summary.{ext}',dpi=170)
plt.close(f)
(out/'proof-audit-build.json').write_text(json.dumps(dict(engine='Matplotlib PDF backend',pages=1,TeX_compiled=False,TeX_source='proof_audit_report.tex',visual_review_required=True),indent=2)+'\n')
print('AUDIT_SUMMARY_RENDERED',flush=True)

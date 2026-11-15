"""Check fixed response normalizations; this script does not train a model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import erf, expit, ndtr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = np.sqrt(1.2)
L = 1 / C
NAMES = ("ISRU", "tanh", "arctan", "Softsign", "erf", "SiLU x2", "GELU x2", "Linear")
BOUNDED = NAMES[:5]

def response(name, h):
    if name == "ISRU": return h / np.sqrt(1 + (C*h)**2)
    if name == "tanh": return L * np.tanh(C*h)
    if name == "arctan": return 2*L/np.pi * np.arctan(np.pi*C*h/2)
    if name == "Softsign": return h / (1 + np.abs(C*h))
    if name == "erf": return L * erf(np.sqrt(np.pi)*C*h/2)
    if name == "SiLU x2": return 2*h*expit(h)
    if name == "GELU x2": return 2*h*ndtr(h)
    if name == "Linear": return h.copy()
    raise ValueError(name)

def diagonal(name, h):
    if name == "ISRU": return 1 / np.sqrt(1 + (C*h)**2)
    if name == "Softsign": return 1 / (1 + np.abs(C*h))
    if name == "SiLU x2": return 2*expit(h)
    if name == "GELU x2": return 2*ndtr(h)
    if name == "Linear": return np.ones_like(h)
    return np.divide(response(name, h), h, out=np.ones_like(h), where=h != 0)

def derivative(name, h):
    if name == "ISRU": return (1 + (C*h)**2)**(-1.5)
    if name == "tanh": return 1 - np.tanh(C*h)**2
    if name == "arctan": return 1 / (1 + (np.pi*C*h/2)**2)
    if name == "Softsign": return (1 + np.abs(C*h))**(-2)
    if name == "erf": return np.exp(-(np.sqrt(np.pi)*C*h/2)**2)
    if name == "SiLU x2":
        s=expit(h)
        return 2*s + 2*h*s*(1-s)
    if name == "GELU x2":
        return 2*ndtr(h) + 2*h*np.exp(-h*h/2)/np.sqrt(2*np.pi)
    if name == "Linear": return np.ones_like(h)
    raise ValueError(name)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output", default=str(Path(__file__).resolve().parent))
    out=Path(p.parse_args().output)
    out.mkdir(parents=True,exist_ok=True)
    h=np.r_[np.linspace(-8,-.001,2000),0.,np.linspace(.001,8,2000)]
    probe=h[h!=0];eps=1e-5;checks=[]
    for name in NAMES:
        q=response(name,h);d=diagonal(name,h)
        assert np.isfinite(q).all() and np.isfinite(d).all()
        error=float(np.max(np.abs(q-d*h)))
        assert error < 1e-14
        slope=float((response(name,np.array([1e-7]))-response(name,np.array([-1e-7])))[0]/2e-7)
        assert abs(slope-1)<2e-6 and diagonal(name,np.array([0.]))[0]==1
        fd=(response(name,probe+eps)-response(name,probe-eps))/(2*eps)
        de=float(np.max(np.abs(fd-derivative(name,probe))))
        assert de<1e-8
        row=dict(family=name,q_equals_d_times_h_max_error=error,origin_slope=slope,
                 derivative_finite_difference_max_error=de,bounded_odd=name in BOUNDED)
        if name in BOUNDED:
            odd=float(np.max(np.abs(q+response(name,-h))))
            tail=float(response(name,np.array([1e8/C]))[0])
            assert odd<1e-14 and np.max(np.abs(q))<=L
            assert abs(tail-L)<1e-7 and np.min(d)>0 and np.max(d)<=1
            row.update(oddness_max_error=odd,positive_tail_at_scaled_1e8=tail,saturation_limit=float(L))
        else:
            assert np.min(d)>=0 and np.max(d)<=2
        checks.append(row)
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout="constrained")
    x=np.linspace(-4,4,1601)
    for name in NAMES:
        col=0 if name in BOUNDED else 1
        axes[0,col].plot(x,response(name,x),label=name)
        axes[1,col].plot(x,derivative(name,x),label=name)
    for y in (-L,L):axes[0,0].axhline(y,color="gray",lw=.6)
    axes[0,0].set(xlabel="Feature h",ylabel="q(h)",title="Bounded responses: matched slopes and limits")
    axes[0,1].set(xlabel="Feature h",ylabel="q(h)",title="Unbounded standard responses and linear control")
    for col in range(2):
        axes[1,col].set(xlabel="Feature h",ylabel="q'(h)",title="Derivatives; all origin slopes equal 1")
        axes[0,col].legend(fontsize=8)
    for ax in axes.flat:ax.spines[["top","right"]].set_visible(False)
    fig.savefig(out/"response-screen-curves.png",dpi=180);plt.close(fig)
    record=dict(passed=True,scope="Float64 normalization and scalar derivative checks; no autograd or model training",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy=np.__version__,
        c=float(C),saturation_limit=float(L),checks=checks,
        grid=dict(responses=8,orders=[1,8],datasets=2,total=32,existing_response_cases=8,existing_linear_k1_cases=2,new_cases_if_anchors_available=22),
        nonsmooth_case="Softsign diagonal at zero: full K8 map may have coordinate-hyperplane cusps")
    (out/"response-screen-formulas.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(passed=True,families=len(checks),output=str(out))))

if __name__=="__main__":
    main()


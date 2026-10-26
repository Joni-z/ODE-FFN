"""Numerical witnesses for the post-result parity derivation, not model results."""
import json
import numpy as np

def main():
    sigmoid=lambda x:1/(1+np.exp(-x));rho=lambda x:x*sigmoid(x)
    x=np.linspace(-30,30,1001);np.testing.assert_allclose(rho(x)-rho(-x),x,atol=1e-13)
    rng=np.random.default_rng(0);h=rng.normal(size=(128,5));a=rng.normal(size=(7,5));b=rng.normal(size=(3,7))
    plus=rho(h@a.T)@b.T;minus=rho(-h@a.T)@b.T
    np.testing.assert_allclose((plus-minus)/2,(h@a.T@b.T)/2,atol=1e-12)
    u=rng.normal(size=(128,7));v=rng.normal(size=(128,7))
    np.testing.assert_allclose((sigmoid(u)*v+sigmoid(-u)*v)/2,v/2,atol=1e-13)
    np.testing.assert_allclose((rho(u)*v+rho(-u)*v)/2,u*v*np.tanh(u/2)/2,atol=1e-13)
    m=np.linspace(-20,20,257)[:,None];d=np.linspace(-20,20,257)[None,:]
    remainder=(rho(m+d)-rho(m-d))/2-d/2;bound=.75*np.abs(m)*np.minimum(np.abs(d),1)
    assert np.max(np.abs(remainder)-bound)<1e-12
    target=rng.normal(size=(128,3));f=rng.normal(size=(128,3));reverse=rng.normal(size=(128,3))
    lhs=(((f-target)**2+(reverse+target)**2)/2).sum(-1)
    rhs=(((f-reverse)/2-target)**2+((f+reverse)/2)**2).sum(-1)
    np.testing.assert_allclose(lhs,rhs,atol=1e-12)
    return dict(status='PASS',checks=['SiLU identity','bias-free ridge odd part','GLU odd part','SwiGLU odd part','offset bound witness','paired risk identity'],
        raw_model_evaluations=0,new_training_runs=0)

if __name__=='__main__':print(json.dumps(main()))

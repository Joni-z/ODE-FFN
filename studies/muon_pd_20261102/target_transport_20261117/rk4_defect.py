"""Finite Taylor-jet construction of midpoint/RK4 local defects, no fitting."""
import math
import numpy as np


def multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros_like(left)
    for degree in range(len(result)):
        for index in range(degree+1):
            result[degree] += left[index]*right[degree-index]
    return result


def response_jet(y: np.ndarray, a: float = 1.2) -> np.ndarray:
    u = 1+a*y*y
    return np.stack((y/np.sqrt(u),u**(-1.5),-3*a*y*u**(-2.5),
        (12*a*a*y*y-3*a)*u**(-3.5),(45*a*a*y-60*a**3*y**3)*u**(-4.5)))


def field_series(coefficients: np.ndarray, matrix: np.ndarray,
                 linear: bool = False) -> np.ndarray:
    if linear:
        return coefficients@matrix.T
    jet = response_jet(coefficients[0])
    delta = coefficients.copy();delta[0] = 0
    power = np.zeros_like(delta);power[0] = 1
    value = np.zeros_like(delta)
    for order in range(5):
        value += jet[order]*power/math.factorial(order)
        power = multiply(power,delta)
    return value@matrix.T


def local_defects(entrance: np.ndarray, matrix: np.ndarray,
                  linear: bool = False) -> dict[str,np.ndarray]:
    y = np.asarray(entrance,dtype=np.float64)
    if y.ndim != 2 or matrix.shape != (y.shape[1],y.shape[1]):
        raise ValueError('Expected batch-by-width states and square shared P')
    exact = np.zeros((6,*y.shape));exact[0] = y
    for degree in range(5):
        exact[degree+1] = field_series(exact[:5],matrix,linear)[degree]/(degree+1)
    start = np.zeros_like(exact[:5]);start[0] = y
    def stage(previous: np.ndarray, scale: float) -> np.ndarray:
        point = start.copy();point[1:] += scale*previous[:-1]
        return field_series(point,matrix,linear)
    a = field_series(start,matrix,linear)
    b = stage(a,.5)
    c = stage(b,.5)
    d = stage(c,1.)
    rk4 = (a[4]+2*b[4]+2*c[4]+d[4])/6-exact[5]
    midpoint = b[2]-exact[3]
    return dict(midpoint=midpoint,rk4=rk4,exact_coefficients=exact)


def explicit_defects(entrance: np.ndarray, matrix: np.ndarray) -> dict[str,np.ndarray]:
    """Equivalent nine-tree formula; avoid polynomial work in future forcing."""
    jet = response_jet(np.asarray(entrance,dtype=np.float64))
    def j(v):return (jet[1]*v)@matrix.T
    def h(v,w):return (jet[2]*v*w)@matrix.T
    def t(v,w,z):return (jet[3]*v*w*z)@matrix.T
    f=jet[0]@matrix.T
    jf=j(f);jjf=j(jf);jjjjf=j(j(jjf))
    hff=h(f,f);jjhff=j(j(hff));jhf_jf=j(h(f,jf))
    jtfff=j(t(f,f,f))
    rk4=(-jjjjf/120+jjhff/480-jhf_jf/240-jtfff/720+
        h(f,jjf)/120-h(f,hff)/480+h(jf,jf)/160+
        t(f,f,jf)/480+(jet[4]*f**4)@matrix.T/2880)
    return dict(midpoint=-hff/24-jjf/6,rk4=rk4)

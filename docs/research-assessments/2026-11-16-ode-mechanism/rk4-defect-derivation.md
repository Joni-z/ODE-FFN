# Fifth-degree defect of the actual classical RK4 hidden solve

This deterministic construction prepares the conditional ODE-specific transport
stage. It does not inspect learned checkpoints or claim target-quality success.
The full measured-delta transport study must qualify before learned application.

For f(y)=P q(y), q_i(y)=y_i/sqrt(1+a*y_i^2),a=1.2, use ordinary Taylor coefficient
arrays (derivative divided by factorial). For u=1+a*y^2 the coordinate derivatives
through order4 are:

    q0=y/sqrt(u)
    q1=u^(-3/2)
    q2=-3*a*y*u^(-5/2)
    q3=(12*a^2*y^2-3*a)*u^(-7/2)
    q4=(45*a^2*y-60*a^3*y^3)*u^(-9/2)

Compose q(c0+delta)=sum(q_n(c0)*delta^n/n!) by truncated polynomial multiplication,
then multiply by P. For the exact ODE series c(s), recursively set
c_(n+1)=[s^n]f(c(s))/(n+1),n=0..4. Compute the four actual RK4 stage series using
the copied classical algebra, and let b5 be its endpoint coefficient at degree5.
The leading local defect (method minus exact flow) is

    g_RK4=b5-c5
    local defect=eta^5*g_RK4+O(eta^6)
    leading modified vector field=f+eta^4*g_RK4

This recursive expression is an explicit finite algebraic construction, with
no fitted coefficient. It avoids omitting any mixed derivative term from a
hand-written fifth-order expression. The four stage coefficients through degree4
are sufficient because the endpoint multiplies their weighted sum by eta.

For midpoint, the same construction at degree3 must reproduce the earlier
formula g_M=-H[f,f]/24-J(Jf)/6. For a linear field f=Ay, RK4 must give
g_RK4=-A^5*y/120. Zero P gives exactly zero in both methods.

An equivalent explicit formula reduces the future forcing cost. Let J,H,T,Q
denote the first through fourth derivatives of f at the same state. Products
J^k are repeated Jacobian applications; H/T/Q are multilinear maps:

    g_RK4 = -J^4 f/120 + J^2 H(f,f)/480
            -J H(f,Jf)/240 -J T(f,f,f)/720
            +H(f,J^2 f)/120 -H(f,H(f,f))/480
            +H(Jf,Jf)/160 +T(f,f,Jf)/480 +Q(f,f,f,f)/2880

For the coordinatewise response, each multilinear map is just a diagonal
derivative product followed by P, so no dense Jacobian/Hessian is formed.
Independently compare this expression with the polynomial-jet construction
at every deterministic random control point (absolute gate1e-12). This is
standard local-error algebra specialized to the recorded field, not a new
numerical theorem or evidence of a generation benefit.

Independent scalar check: for y>0 and scalar p, an antiderivative of1/f is

    G(y)=sqrt(1+a*y^2)+log(y/(1+sqrt(1+a*y^2)))
    G(y(eta))-G(y0)=p*eta

Solve this implicit equation at80 decimal digits, using only Decimal sqrt/log
and Newton's method, then independently evaluate classical RK4 and midpoint in
Decimal arithmetic. Fixed controls:y0=.7,p=-.4,a=1.2,eta=.1/2^j,j=0..6. Before
observing the controls, require final scaled-defect relative error<.02 and
strict decrease over the last four refinements. Compare every earlier scale too.
No learned-state application or new numerical gate follows from this check alone.

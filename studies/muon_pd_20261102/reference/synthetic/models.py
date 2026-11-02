"""Matched time-conditioned vector-field backbones for synthetic FM.

The ODEFFN below reproduces the professor's two ODELayer solves separated by SiLU.
The external residual/time conditioning is shared across all compared FFNs.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from synthetic.covariance import CovarianceField
from synthetic.diagonal import DiagonalFlow,DiagonalODEFFN
from synthetic.forced_gate import ForcedGateResponse
from synthetic.state_clock import StateClockLayer
from synthetic.augmented import AugmentedODEFFN, PrunedLowRankLayer


class OriginalODELayer(nn.Module):
    def __init__(self, width, gate_hidden, orders=2, tau=10.0, scale=0.8,
                 shift=0.2, spectral_cap=None, gate_source='input',
                 clock_curvature=1.0, gate_center=0.6,gate_transform='sigmoid',
                 rate_placement='after_projection',solve_return='map',operator_mode='multiplicative',
                 forcing_mode='none'):
        super().__init__()
        if orders < 1 or tau <= 0 or scale <= 0 or not 0 <= shift < shift + scale:
            raise ValueError('invalid ODELayer settings')
        if spectral_cap is not None and spectral_cap <= 0:
            raise ValueError('spectral_cap must be positive')
        self.orders, self.tau, self.scale, self.shift = orders, tau, scale, shift
        self.spectral_cap = spectral_cap
        if not math.isfinite(clock_curvature) or not math.isfinite(gate_center):
            raise ValueError('clock curvature and center must be finite')
        if clock_curvature != 1.0 and orders != 2:
            raise ValueError('centered clock control requires orders=2')
        self.clock_curvature, self.gate_center = clock_curvature, gate_center
        if rate_placement not in ('after_projection','before_projection'):
            raise ValueError('unsupported rate placement')
        if rate_placement=='before_projection' and (orders!=1 or gate_source!='input' or
                spectral_cap is not None or clock_curvature!=1. or gate_transform!='sigmoid'):
            raise ValueError('rate-order study requires the plain input-dependent sigmoid K1 layer')
        self.rate_placement=rate_placement
        if solve_return not in ('map','increment'):
            raise ValueError('solve return must be map or increment')
        if solve_return=='increment' and (orders!=1 or gate_source!='input' or
                spectral_cap is not None or clock_curvature!=1. or gate_transform!='sigmoid' or
                rate_placement!='after_projection'):
            raise ValueError('zero-order study requires the plain after-projection sigmoid K1 layer')
        self.solve_return=solve_return
        if operator_mode not in ('multiplicative','additive'):
            raise ValueError('unsupported operator mode')
        if operator_mode=='additive' and (orders!=1 or gate_source!='input' or spectral_cap is not None or
                clock_curvature!=1. or gate_transform!='sigmoid' or rate_placement!='after_projection' or
                solve_return!='map'):
            raise ValueError('additive-generator study requires the plain K1 map with input sigmoid controls')
        self.operator_mode=operator_mode
        if forcing_mode not in ('none','static','conditional'):
            raise ValueError('forcing mode must be none, static or conditional')
        if forcing_mode!='none' and (orders!=1 or gate_source!='input' or
                spectral_cap is not None or clock_curvature!=1. or gate_transform!='sigmoid' or
                rate_placement!='after_projection' or solve_return!='map' or
                operator_mode!='multiplicative'):
            raise ValueError('bias completeness retains the original input-clock K1 map')
        self.forcing_mode=forcing_mode
        if gate_transform not in ('sigmoid','affine'):
            raise ValueError('gate transform must be sigmoid or affine')
        if gate_transform=='affine' and clock_curvature!=1.:
            raise ValueError('affine durations are not supported by the centered-curvature control')
        self.gate_transform=gate_transform
        if gate_source not in ('input', 'time', 'fixed', 'scalar', 'provided'):
            raise ValueError('unsupported gate source')
        self.gate_source = gate_source
        self.t_func = nn.Sequential(nn.Linear(width, gate_hidden), nn.SiLU(),
                                    nn.Linear(gate_hidden, 1))
        self.proj = nn.Linear(width, width, bias=False)
        if forcing_mode!='none':
            # Add no random draws; all inherited tensors and surrounding draws
            # remain paired with the homogeneous reference.
            self.forcing=nn.Parameter(torch.zeros(width))
        # Consume the legacy initialization draws so all common weights stay paired.
        # Remove the unused network rather than counting dormant parameters.
        if gate_source in ('fixed', 'scalar', 'provided'):
            del self.t_func
        if gate_source == 'scalar':
            self.duration_logit = nn.Parameter(torch.zeros(()))

    def operator(self):
        weight = self.proj.weight
        if self.spectral_cap is None:
            return weight
        # Exact differentiable spectral-norm cap. Small (64x64) experiment only.
        sigma = torch.linalg.matrix_norm(weight.float(), ord=2)
        factor = torch.clamp(self.spectral_cap / sigma.clamp_min(1e-12), max=1.0)
        return weight * factor.to(weight.dtype)

    def duration_from_logit(self,g):
        if self.gate_transform=='affine':
            return g*(self.scale/(4*self.tau))+(self.shift+self.scale/2)
        return torch.sigmoid(g/self.tau)*self.scale+self.shift

    def step_size(self, x, context=None):
        if self.gate_source == 'provided':
            if (context is None or context.shape[:-1] != x.shape[:-1]
                    or context.shape[-1] not in (1,x.shape[-1])):
                raise ValueError('provided durations must match the input batch and channels')
            return context
        if self.gate_source == 'fixed':
            return x.new_full((*x.shape[:-1], 1), self.shift + self.scale / 2)
        if self.gate_source == 'scalar':
            s = self.duration_from_logit(self.duration_logit)
            return s.expand(*x.shape[:-1], 1)
        gate_input = x
        if self.gate_source == 'time':
            if context is None or context.shape != x.shape:
                raise ValueError('time gate requires a matching time embedding')
            gate_input = context
        return self.duration_from_logit(self.t_func(gate_input))

    def forward(self, x, context=None, *, return_duration=False):
        s = self.step_size(x, context)
        p = self.operator()
        total, term = (x if self.solve_return=='map' else None), x
        for order in range(1, self.orders + 1):
            if self.operator_mode=='additive':
                center=self.shift+self.scale/2
                term=(center*F.linear(term,p)+(s-center)*term)/order
            elif self.rate_placement=='before_projection':
                term = F.linear(term * s, p) / order
            else:
                term = F.linear(term, p) * s / order
            total = term if total is None else total + term
        if self.forcing_mode!='none':
            coefficient=s if self.forcing_mode=='conditional' else self.shift+self.scale/2
            total=total+coefficient*self.forcing
        if self.clock_curvature != 1.0:
            # term = s^2 P^2 x/2. Reuse it to keep the matrix MAC budget unchanged.
            if s.shape[-1]==1:
                correction=((s-self.gate_center)/s).square()*term
            else:
                delta=s-self.gate_center
                correction=0.5*delta*F.linear(delta*F.linear(x,p),p)
            total = total - (1.0-self.clock_curvature)*correction
        return (total,s) if return_duration else total

    @torch.no_grad()
    def activation_stats(self, x, context=None):
        s = self.step_size(x, context)
        p = self.operator()
        rms = lambda a: a.square().mean().sqrt().item()
        denominator = max(rms(x), 1e-12)
        term, out = x, x
        stats = {'s_mean':s.mean().item(), 's_std':s.std(unbiased=False).item(),
                 's_min':s.min().item(), 's_max':s.max().item(),
                 'gate_channels':s.shape[-1],
                 's_sample_std':s.std(dim=0,unbiased=False).mean().item(),
                 's_channel_std':s.std(dim=-1,unbiased=False).mean().item(),
                 'gate_transform':self.gate_transform,
                 'rate_placement':self.rate_placement,
                 'solve_return':self.solve_return,
                 'operator_mode':self.operator_mode,
                 'solve_initialization':getattr(self,'solve_initialization','legacy'),
                 'controller_tau':self.tau,
                 'gate_parameter_scale':getattr(self,'gate_parameter_scale',1.),
                 's_negative_fraction':float((s<0).float().mean()),
                 's_below_original_floor_fraction':float((s<.2).float().mean()),
                 's_above_original_ceiling_fraction':float((s>1.).float().mean()),
                 'input_rms':denominator}
        if self.gate_source in ('input','time','scalar'):
            slope=self.scale/(4*self.tau)
            if self.gate_transform=='sigmoid':
                probability=(s-self.shift)/self.scale
                sensitivity=self.scale/self.tau*probability*(1-probability)
                stats['clock_outer_saturated_fraction']=float(((probability<.01)|(probability>.99)).float().mean())
            else:
                sensitivity=torch.full_like(s,slope)
                stats['clock_outer_saturated_fraction']=0.
            stats['clock_logit_sensitivity_over_initial']=float(sensitivity.mean()/slope)
            stats['clock_raw_logit_sensitivity_mean']=float(sensitivity.mean())
        if hasattr(self,'t_func'):
            stats['gate_output_weight_over_tau_frobenius']=float(self.t_func[-1].weight.norm()/self.tau)
            stats['gate_output_bias_over_tau_norm']=float(self.t_func[-1].bias.norm()/self.tau)
        if s.shape[-1]>1 and self.operator_mode=='multiplicative':
            commutator=s*F.linear(x,p)-F.linear(s*x,p)
            stats['rate_commutator_rms_over_input']=rms(commutator)/denominator
        for order in range(1,self.orders+1):
            if self.operator_mode=='additive':
                center=self.shift+self.scale/2
                term=(center*F.linear(term,p)+(s-center)*term)/order
            else:
                term = (F.linear(term*s,p) if self.rate_placement=='before_projection'
                        else F.linear(term,p)*s)/order
            stats[f'term{order}_rms_over_input'] = rms(term)/denominator
            out = out+term
        stats['output_rms_over_input'] = rms(self(x, context))/denominator
        if self.forcing_mode!='none':
            center=self.shift+self.scale/2
            static=center*self.forcing
            conditional=((s-center)*self.forcing if self.forcing_mode=='conditional'
                         else torch.zeros_like(s)*self.forcing)
            actual=s*self.forcing if self.forcing_mode=='conditional' else static
            # The logged first term must describe the actual biased update.
            stats['homogeneous_term1_rms_over_input']=stats['term1_rms_over_input']
            stats['term1_rms_over_input']=rms(s*F.linear(x,p)+actual)/denominator
            mean=conditional.reshape(-1,x.shape[-1]).mean(dim=0)
            stats.update(forcing_mode=self.forcing_mode,
                forcing_parameter_norm=float(self.forcing.norm()),
                forcing_rms_over_input=rms(actual)/denominator,
                static_bias_rms_over_input=rms(static)/denominator,
                conditional_bias_rms_over_input=rms(conditional)/denominator,
                conditional_bias_mean_rms_over_input=rms(mean)/denominator,
                conditional_bias_centered_rms_over_input=rms(conditional-mean)/denominator)
            local_sensitivity=0.
            if self.forcing_mode=='conditional':
                hidden=self.t_func[0](x)
                probability=torch.sigmoid(hidden)
                derivative=probability*(1+hidden*(1-probability))
                logit_jacobian=(self.t_func[-1].weight*derivative.unsqueeze(-2))@self.t_func[0].weight
                clock_probability=(s-self.shift)/self.scale
                row_scale=self.forcing*(self.scale/self.tau)*clock_probability*(1-clock_probability)
                local_sensitivity=float(torch.linalg.matrix_norm(row_scale.unsqueeze(-1)*logit_jacobian,ord='fro').mean())
            stats['conditional_bias_local_jacobian_frobenius_mean']=local_sensitivity
        if self.orders == 2:
            delta=s-self.gate_center
            curvature = 0.5*delta*F.linear(delta*F.linear(x,p),p)
            stats['centered_curvature_rms_over_input'] = rms(curvature)/denominator
            stats['clock_curvature'] = self.clock_curvature
        return stats


class OriginalODEFFN(nn.Module):
    def __init__(self, width, gate_hidden, middle_activation='silu', solve_returns=None, **ode_options):
        super().__init__()
        if middle_activation not in ('silu','identity'):
            raise ValueError('middle activation must be silu or identity')
        self.gate_source = ode_options.get('gate_source', 'input')
        if solve_returns is not None and (len(solve_returns)!=2 or any(v not in ('map','increment') for v in solve_returns)):
            raise ValueError('two solve return modes required')
        first=ode_options if solve_returns is None else dict(ode_options,solve_return=solve_returns[0])
        second=ode_options if solve_returns is None else dict(ode_options,solve_return=solve_returns[1])
        self.ode1 = OriginalODELayer(width,gate_hidden,**first)
        self.act = nn.SiLU() if middle_activation=='silu' else nn.Identity()
        self.ode2 = OriginalODELayer(width,gate_hidden,**second)

    def forward(self, x, context=None):
        return self.ode2(self.act(self.ode1(x, context)), context)


class VectorODELayer(OriginalODELayer):
    """Channel rates with matched scalar-reference P/RNG draws; default D(h)P."""
    def __init__(self,width,gate_hidden,reference_gate_hidden=64,feature_mode='independent',**options):
        if feature_mode not in ('independent','shared_projection','paired_untied'):
            raise ValueError('unsupported ODE feature mode')
        if feature_mode!='independent' and (gate_hidden!=width or options.get('orders',2)!=1 or
                options.get('gate_source','input')!='input' or options.get('spectral_cap') is not None or
                options.get('clock_curvature',1.)!=1. or options.get('gate_transform','sigmoid')!='sigmoid' or
                options.get('rate_placement','after_projection')!='after_projection' or
                options.get('solve_return','map')!='map' or options.get('operator_mode','multiplicative')!='multiplicative' or
                options.get('forcing_mode','none')!='none' or
                (options.get('tau',10.),options.get('scale',.8),options.get('shift',.2))!=(10.,.8,.2)):
            raise ValueError('feature sharing requires width-sized original sigmoid K1 map controllers')
        gate_state=torch.get_rng_state()
        super().__init__(width,reference_gate_hidden,**options)
        if self.gate_source!='input':
            raise ValueError('vector-rate study requires input-dependent gates')
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(gate_state)
            self.t_func=nn.Sequential(nn.Linear(width,gate_hidden),nn.SiLU(),
                nn.Linear(gate_hidden,width))
        self.feature_mode=feature_mode
        if feature_mode!='independent':
            with torch.no_grad():
                self.t_func[0].bias.zero_()
                self.t_func[0].weight.copy_(self.proj.weight)
            if feature_mode=='shared_projection':
                self.t_func[0].weight=self.proj.weight
                # assign=True loading replaces Parameter objects. Restore the
                # identity after all child modules have loaded their tensors.
                self.register_load_state_dict_post_hook(self._retie_loaded_projection)

    @staticmethod
    def _retie_loaded_projection(module,incompatible_keys):
        module.t_func[0].weight=module.proj.weight

    def _apply(self,fn,recurse=True):
        result=super()._apply(fn,recurse=recurse)
        if self.feature_mode=='shared_projection':
            self.t_func[0].weight=self.proj.weight
        return result

    def _feature_duration(self,x,projected=None):
        if self.feature_mode=='shared_projection':
            if projected is None:projected=F.linear(x,self.proj.weight)
            hidden=projected+self.t_func[0].bias
        else:
            hidden=self.t_func[0](x)
        return self.duration_from_logit(self.t_func[2](self.t_func[1](hidden)))

    def step_size(self,x,context=None):
        if self.feature_mode=='independent':return super().step_size(x,context)
        return self._feature_duration(x)

    def forward(self,x,context=None,*,return_duration=False):
        if self.feature_mode=='independent':
            return super().forward(x,context,return_duration=return_duration)
        projected=F.linear(x,self.proj.weight)
        # Gate interventions replace step_size on the instance. Keep that
        # diagnostic interface while caching P*x on the ordinary training path.
        duration=(self.step_size(x,context) if 'step_size' in self.__dict__
                  else self._feature_duration(x,projected))
        total=x+projected*duration
        return (total,duration) if return_duration else total

    @torch.no_grad()
    def activation_stats(self,x,context=None):
        stats=super().activation_stats(x,context)
        if self.feature_mode!='independent':
            hidden=F.linear(x,self.t_func[0].weight)+self.t_func[0].bias
            stats.update(feature_mode=self.feature_mode,
                controller_feature_rms=float(hidden.square().mean().sqrt()),
                controller_feature_bias_norm=float(self.t_func[0].bias.norm()),
                projection_shared_with_controller=self.t_func[0].weight is self.proj.weight)
        return stats


class VectorODEFFN(OriginalODEFFN):
    def __init__(self,width,gate_hidden,reference_gate_hidden=64,
                 middle_activation='silu',clock_mode='independent',solve_returns=None,
                 feature_mode='independent',**options):
        nn.Module.__init__(self)
        if middle_activation not in ('silu','identity'):
            raise ValueError('middle activation must be silu or identity')
        if clock_mode not in ('independent','tied','reuse'):
            raise ValueError('clock mode must be independent, tied or reuse')
        if feature_mode!='independent' and (clock_mode!='independent' or middle_activation!='silu' or
                solve_returns not in (None,['map','map'])):
            raise ValueError('feature sharing retains two independent K1 maps and middle SiLU')
        self.clock_mode=clock_mode
        self.gate_source='input'
        if solve_returns is not None and (len(solve_returns)!=2 or any(v not in ('map','increment') for v in solve_returns)):
            raise ValueError('two solve return modes required')
        first=options if solve_returns is None else dict(options,solve_return=solve_returns[0])
        second=options if solve_returns is None else dict(options,solve_return=solve_returns[1])
        self.ode1=VectorODELayer(width,gate_hidden,reference_gate_hidden,feature_mode=feature_mode,**first)
        self.act=nn.SiLU() if middle_activation=='silu' else nn.Identity()
        self.ode2=VectorODELayer(width,gate_hidden,reference_gate_hidden,feature_mode=feature_mode,**second)
        # Construct the full reference first, preserving all P/surrounding RNG draws.
        if clock_mode=='tied':
            self.ode2.t_func=self.ode1.t_func
        elif clock_mode=='reuse':
            del self.ode2.t_func
            self.ode2.gate_source='provided'

    def forward(self,x,context=None):
        if self.clock_mode!='reuse':
            return super().forward(x,context)
        first,duration=self.ode1(x,context,return_duration=True)
        return self.ode2(self.act(first),duration)


class CovarianceODELayer(nn.Module):
    """K1 update x+D(x)f(x), with a covariance-shaped internal field."""
    def __init__(self, width, gate_hidden, reference_gate_hidden=64, *,
                 heads=4, atoms=8, temperature=1., mode='tied',
                 tau=10., scale=.8, shift=.2):
        super().__init__()
        if heads*atoms > width:
            raise ValueError('paired initialization requires heads*atoms <= width')
        reference=VectorODELayer(width,gate_hidden,reference_gate_hidden,
            orders=1,tau=tau,scale=scale,shift=shift)
        self.t_func=reference.t_func
        # Reuse a recorded subset of the reference P draws, preserving every
        # surrounding tensor and RNG draw without allocating dormant weights.
        self.field=CovarianceField(width,heads,atoms,temperature,mode,
            initial_weight=reference.proj.weight[:heads*atoms])
        self.tau,self.scale,self.shift=tau,scale,shift
        self.gate_source='input'

    def step_size(self,x):
        return torch.sigmoid(self.t_func(x)/self.tau)*self.scale+self.shift

    def forward(self,x):
        return x+self.step_size(x)*self.field(x)

    @torch.no_grad()
    def activation_stats(self,x):
        duration=self.step_size(x)
        increment=duration*self.field(x)
        rms=lambda value:float(value.square().mean().sqrt())
        denominator=max(rms(x),1e-12)
        return dict(s_mean=float(duration.mean()),s_std=float(duration.std(unbiased=False)),
            s_min=float(duration.min()),s_max=float(duration.max()),
            input_rms=denominator,increment_rms_over_input=rms(increment)/denominator,
            output_rms_over_input=rms(x+increment)/denominator)


class CovarianceODEFFN(nn.Module):
    def __init__(self,width,gate_hidden,reference_gate_hidden=64,**options):
        super().__init__()
        self.ode1=CovarianceODELayer(width,gate_hidden,reference_gate_hidden,**options)
        self.act=nn.SiLU()
        self.ode2=CovarianceODELayer(width,gate_hidden,reference_gate_hidden,**options)

    def forward(self,x):
        return self.ode2(self.act(self.ode1(x)))


class StandardFFN(nn.Module):
    def __init__(self,width,hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width,hidden),nn.SiLU(),nn.Linear(hidden,width))

    def forward(self,x):
        return self.net(x)


class NonlinearDriftFFN(nn.Module):
    """Taylor increment of f_D(y)=D0 P2 SiLU(P1 y), with D0 frozen per solve.

    The K1 map/increment pair differs only by returning the internal state x.
    A full professor-reference construction preserves common initialization draws.
    """
    def __init__(self,width,gate_hidden,reference_gate_hidden=64,orders=1,
                 tau=10.,scale=.8,shift=.2,return_mode='increment',gate_source='input'):
        super().__init__()
        if orders not in (1,2) or return_mode not in ('increment','map') or gate_source not in ('input','fixed'):
            raise ValueError('invalid nonlinear drift specification')
        reference=VectorODEFFN(width,gate_hidden,reference_gate_hidden,orders=orders,
                              tau=tau,scale=scale,shift=shift)
        self.proj1=reference.ode1.proj
        self.proj2=reference.ode2.proj
        if gate_source=='input':self.t_func=reference.ode1.t_func
        self.orders,self.return_mode,self.gate_source=orders,return_mode,gate_source
        self.tau,self.scale,self.shift=tau,scale,shift

    def step_size(self,x):
        if self.gate_source=='fixed':return x.new_full(x.shape,self.shift+self.scale/2)
        return torch.sigmoid(self.t_func(x)/self.tau)*self.scale+self.shift

    def terms(self,x):
        s=self.step_size(x)
        z=self.proj1(x)
        drift=self.proj2(F.silu(z))
        first=s*drift
        second=None
        if self.orders==2:
            sigmoid=torch.sigmoid(z)
            derivative=sigmoid*(1+z*(1-sigmoid))
            second=.5*s*self.proj2(derivative*self.proj1(first))
        return s,first,second

    def forward(self,x):
        _,first,second=self.terms(x)
        delta=first if second is None else first+second
        return delta if self.return_mode=='increment' else x+delta

    @torch.no_grad()
    def activation_stats(self,x):
        s,first,second=self.terms(x)
        denominator=max(float(x.square().mean().sqrt()),1e-12)
        record=dict(s_mean=float(s.mean()),s_std=float(s.std(unbiased=False)),
            s_min=float(s.min()),s_max=float(s.max()),gate_channels=s.shape[-1],
            input_rms=denominator,term1_rms_over_input=float(first.square().mean().sqrt())/denominator,
            return_mode=self.return_mode,orders=self.orders)
        if second is not None:record['term2_rms_over_input']=float(second.square().mean().sqrt())/denominator
        return record


class CollapsedODEFFN(nn.Module):
    """Direct matrices initialized to a fixed-duration professor block.

    No new random draws: the reference construction consumes exactly the draws
    of ode_fixed. The global feasible matrix sets subsequently differ.
    """
    def __init__(self, reference):
        super().__init__()
        matrices=[]
        for layer in (reference.ode1,reference.ode2):
            if layer.gate_source!='fixed':
                raise ValueError('collapse requires fixed gates')
            p=layer.proj.weight.detach()
            a=(layer.shift+layer.scale/2)*p
            total=torch.eye(len(p),dtype=p.dtype,device=p.device)
            term=total
            for order in range(1,layer.orders+1):
                term=term@a/order
                total=total+term
            matrices.append(nn.Parameter(total))
        self.matrix1,self.matrix2=matrices

    def forward(self,x):
        return F.linear(F.silu(F.linear(x,self.matrix1)),self.matrix2)


class SwiGLUFFN(nn.Module):
    """JiT-style SwiGLU, retaining the nominal 2/3 hidden-width convention.

    Only the FFN is ported; the toy backbone and initialization stay unchanged.
    Source: https://github.com/LTH14/JiT/blob/main/model_jit.py
    """
    def __init__(self, width, hidden):
        super().__init__()
        hidden = int(hidden * 2 / 3)
        if hidden < 1:
            raise ValueError('SwiGLU hidden width must be positive')
        self.w12 = nn.Linear(width, 2 * hidden)
        self.w3 = nn.Linear(hidden, width)

    def forward(self, x):
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)


class SigmoidGLUFFN(SwiGLUFFN):
    """Known sigmoid GLU control with exactly the native SwiGLU projections."""
    def forward(self,x):
        gate,value=self.w12(x).chunk(2,dim=-1)
        return self.w3(torch.sigmoid(gate)*value)


class ForcedGateSwiGLUFFN(SwiGLUFFN):
    """Preserve SwiGLU projections and perturb only its gate response."""
    def __init__(self,width,hidden,solver='exact',clock='adaptive'):
        super().__init__(width,hidden)
        self.flow=ForcedGateResponse(self.w3.in_features,solver=solver,clock=clock)

    def forward(self,x):
        gate,value=self.w12(x).chunk(2,dim=-1)
        return self.w3(self.flow(gate)*value)


class StateClockFFN(nn.Module):
    """Original vector fields with a fixed within-solve clock convention."""
    def __init__(self,width,gate_hidden,reference_gate_hidden=64,solver='midpoint'):
        super().__init__()
        reference=VectorODEFFN(width,gate_hidden,reference_gate_hidden,orders=2)
        self.ode1=StateClockLayer(reference.ode1,solver=solver)
        self.act=reference.act
        self.ode2=StateClockLayer(reference.ode2,solver=solver)

    def forward(self,x):
        return self.ode2(self.act(self.ode1(x)))


class TimeEmbedding(nn.Module):
    def __init__(self,width):
        super().__init__()
        frequencies = torch.exp(torch.linspace(math.log(1),math.log(1000),16))
        self.register_buffer('frequencies',frequencies,persistent=False)
        self.net = nn.Sequential(nn.Linear(33,width),nn.SiLU(),nn.Linear(width,width))

    def forward(self,t):
        phases = t*self.frequencies[None]*2*math.pi
        return self.net(torch.cat((t,torch.sin(phases),torch.cos(phases)),dim=1))


class TimeConditionedBlock(nn.Module):
    def __init__(self,width,time_width,ffn,residual_scale):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.condition=nn.Linear(time_width,width*2)
        self.ffn=ffn
        self.residual_scale=residual_scale

    def forward(self,x,temb):
        scale,shift=self.condition(temb).chunk(2,dim=-1)
        z=self.norm(x)*(1+scale)+shift
        if isinstance(self.ffn, OriginalODEFFN) and self.ffn.gate_source == 'time':
            return x+self.residual_scale*self.ffn(z, temb)
        return x+self.residual_scale*self.ffn(z)


class VelocityNet(nn.Module):
    def __init__(self,dim,kind,width=64,depth=3,gate_hidden=64,
                 ffn_hidden=128,orders=2,tau=10.0,scale=0.8,shift=0.2,
                 spectral_cap=1.0, gate_center=0.6,gate_reference_hidden=64,
                 ode_middle_activation='silu',ode_clock_mode='independent',
                 ode_drift_return='increment',ode_drift_gate='input',ode_gate_transform='sigmoid',
                 ode_covariance_mode='tied',ode_covariance_heads=4,
                 ode_covariance_atoms=8,ode_covariance_temperature=1.,ode_gate_reparam=1.,
                 ode_rate_placement='after_projection',ode_solve_return='map',ode_solve_init='legacy',
                 ode_solve_pattern=None,ode_operator_mode='multiplicative',ode_feature_mode='independent',
                 diagonal_solver='k2',diagonal_clock='adaptive',diagonal_beta=1.,
                 forced_gate_solver='exact',forced_gate_clock='adaptive',
                 state_clock_solver='midpoint',ode_forcing_mode='none',
                 lowrank_state_width=None,lowrank_rank=None,lowrank_gate_hidden=None):
        super().__init__()
        if kind not in ('ffn','swiglu','glu','ode_original','ode_spectral_cap','ode_time',
                        'ode_fixed','ode_scalar','ode_gate_linear','ffn_poly_init','ode_vector','ode_nonlinear',
                        'ode_covariance','ode_diagonal','ode_forced_gate','ode_state_clock','ode_lowrank'):
            raise ValueError(f'unsupported model kind: {kind}')
        self.dim,self.kind,self.width,self.depth=dim,kind,width,depth
        lowrank_options=(lowrank_state_width,lowrank_rank,lowrank_gate_hidden)
        if kind!='ode_lowrank' and any(value is not None for value in lowrank_options):
            raise ValueError('low-rank dimensions require the low-rank ODE family')
        if kind=='ode_lowrank' and (any(value is None for value in lowrank_options) or orders!=1 or
                ode_solve_init!='legacy' or ode_solve_pattern is not None or
                ode_feature_mode!='independent' or ode_operator_mode!='multiplicative' or
                ode_middle_activation!='silu' or ode_clock_mode!='independent' or
                ode_gate_transform!='sigmoid' or ode_gate_reparam!=1. or
                ode_rate_placement!='after_projection' or ode_solve_return!='map' or
                ode_forcing_mode!='none' or spectral_cap!=1. or (tau,scale,shift)!=(10.,.8,.2)):
            raise ValueError('low-rank allocation retains original K1 maps, clocks and SiLU')
        if ode_forcing_mode not in ('none','static','conditional'):
            raise ValueError('unsupported forcing mode')
        if ode_forcing_mode!='none' and (kind!='ode_vector' or orders!=1 or
                ode_solve_init!='constant_paired' or ode_feature_mode!='independent' or
                ode_operator_mode!='multiplicative' or ode_middle_activation!='silu' or
                ode_clock_mode!='independent' or ode_gate_transform!='sigmoid' or
                ode_gate_reparam!=1. or ode_rate_placement!='after_projection' or
                ode_solve_return!='map' or ode_solve_pattern not in (None,'map_map') or
                (tau,scale,shift)!=(10.,.8,.2) or spectral_cap!=1.):
            raise ValueError('bias completeness retains the paired original vector K1 composition')
        if kind!='ode_state_clock' and state_clock_solver!='midpoint':
            raise ValueError('state-clock solver requires its own vector-field family')
        if kind=='ode_state_clock' and (ode_feature_mode!='independent' or ode_operator_mode!='multiplicative' or
                ode_middle_activation!='silu' or ode_clock_mode!='independent' or ode_gate_transform!='sigmoid' or
                ode_gate_reparam!=1. or ode_rate_placement!='after_projection' or ode_solve_return!='map' or
                ode_solve_pattern is not None or ode_solve_init!='legacy' or
                (tau,scale,shift)!=(10.,.8,.2) or spectral_cap!=1. or orders!=2):
            raise ValueError('state-clock family retains the original vector field, two maps and SiLU')
        if kind!='ode_forced_gate' and (forced_gate_solver!='exact' or forced_gate_clock!='adaptive'):
            raise ValueError('forced-gate settings require the forced-gate family')
        if kind=='ode_forced_gate' and (ode_feature_mode!='independent' or ode_operator_mode!='multiplicative' or
                ode_middle_activation!='silu' or ode_clock_mode!='independent' or ode_gate_transform!='sigmoid' or
                ode_gate_reparam!=1. or ode_rate_placement!='after_projection' or ode_solve_return!='map' or
                ode_solve_pattern is not None or ode_solve_init!='legacy' or (tau,scale,shift)!=(10.,.8,.2)):
            raise ValueError('forced gate preserves native SwiGLU without legacy ODE interventions')
        if kind!='ode_diagonal' and (diagonal_solver!='k2' or diagonal_clock!='adaptive' or diagonal_beta!=1.):
            raise ValueError('diagonal settings require the diagonal-ODE family')
        if kind=='ode_diagonal' and (ode_feature_mode!='independent' or ode_operator_mode!='multiplicative' or
                ode_middle_activation!='silu' or ode_clock_mode!='independent' or ode_gate_transform!='sigmoid' or
                ode_gate_reparam!=1. or ode_rate_placement!='after_projection' or ode_solve_return!='map' or
                ode_solve_pattern is not None or ode_solve_init!='legacy' or (tau,scale,shift)!=(10.,.8,.2)):
            raise ValueError('diagonal family retains two maps and SiLU without legacy interventions')
        if ode_feature_mode not in ('independent','shared_projection','paired_untied'):
            raise ValueError('unsupported ODE feature mode')
        if ode_feature_mode!='independent' and (kind!='ode_vector' or orders!=1 or gate_hidden!=width or
                ode_clock_mode!='independent' or ode_middle_activation!='silu' or
                ode_gate_transform!='sigmoid' or ode_gate_reparam!=1. or
                ode_rate_placement!='after_projection' or ode_solve_return!='map' or
                ode_solve_pattern not in (None,'map_map') or ode_operator_mode!='multiplicative' or
                (tau,scale,shift)!=(10.,.8,.2)):
            raise ValueError('feature sharing retains original width-sized independent vector K1 map controllers')
        if ode_operator_mode not in ('multiplicative','additive'):
            raise ValueError('unsupported ODE operator mode')
        if ode_operator_mode=='additive' and (kind not in ('ode_original','ode_vector') or orders!=1 or
                ode_clock_mode!='independent' or ode_middle_activation!='silu' or
                ode_gate_transform!='sigmoid' or ode_gate_reparam!=1. or
                ode_rate_placement!='after_projection' or ode_solve_return!='map' or
                ode_solve_pattern not in (None,'map_map')):
            raise ValueError('additive-generator study retains independent original K1 maps and coordinates')
        if ode_solve_return not in ('map','increment') or ode_solve_init not in ('legacy','constant_paired'):
            raise ValueError('unsupported solve return or initialization')
        patterns=('map_map','map_increment','increment_map','increment_increment')
        if ode_solve_pattern is not None and (ode_solve_pattern not in patterns or ode_solve_return!='map'):
            raise ValueError('solve pattern must specify both modes without a uniform-return override')
        solve_returns=ode_solve_pattern.split('_') if ode_solve_pattern is not None else None
        if (ode_solve_return!='map' or ode_solve_init!='legacy' or ode_solve_pattern is not None) and (
                kind not in ('ode_original','ode_vector') or orders!=1 or
                ode_clock_mode!='independent' or ode_middle_activation!='silu' or
                ode_gate_transform!='sigmoid' or ode_gate_reparam!=1. or
                ode_rate_placement!='after_projection'):
            raise ValueError('zero-order study retains plain independent K1 clocks and original coordinates')
        if ode_rate_placement not in ('after_projection','before_projection'):
            raise ValueError('unsupported rate placement')
        if ode_rate_placement!='after_projection' and (kind not in ('ode_original','ode_vector') or
                orders!=1 or ode_clock_mode!='independent' or ode_middle_activation!='silu' or
                ode_gate_transform!='sigmoid' or ode_gate_reparam!=1.):
            raise ValueError('rate-order study retains plain independent K1 solves and original clock coordinates')
        if not math.isfinite(ode_gate_reparam) or ode_gate_reparam<=0:
            raise ValueError('gate reparameterization factor must be finite and positive')
        if ode_gate_reparam!=1. and (kind!='ode_vector' or orders!=1 or
                ode_clock_mode!='independent' or ode_middle_activation!='silu' or ode_gate_transform!='sigmoid'):
            raise ValueError('gate-coordinate study requires the unchanged independent vector K1 composition')
        if kind=='ode_nonlinear' and ode_gate_transform!='sigmoid':
            raise ValueError('gate-range study does not combine with the nonlinear-drift family')
        if kind=='ode_covariance' and (orders!=1 or ode_gate_transform!='sigmoid'
                or ode_clock_mode!='independent' or ode_middle_activation!='silu'):
            raise ValueError('covariance study retains original clocks, SiLU, and independent K1 solves')
        self.input=nn.Linear(dim,width)
        self.time=TimeEmbedding(width)
        blocks=[]
        for _ in range(depth):
            if kind=='ffn':
                ffn=StandardFFN(width,ffn_hidden)
            elif kind=='ode_diagonal':
                ffn=DiagonalODEFFN(width,ffn_hidden,solver=diagonal_solver,
                    clock=diagonal_clock,beta=diagonal_beta)
            elif kind=='swiglu':
                ffn=SwiGLUFFN(width,ffn_hidden)
            elif kind=='glu':
                ffn=SigmoidGLUFFN(width,ffn_hidden)
            elif kind=='ode_forced_gate':
                ffn=ForcedGateSwiGLUFFN(width,ffn_hidden,
                    solver=forced_gate_solver,clock=forced_gate_clock)
            elif kind=='ode_state_clock':
                ffn=StateClockFFN(width,gate_hidden,gate_reference_hidden,solver=state_clock_solver)
            elif kind=='ode_lowrank':
                reference=VectorODEFFN(width,gate_hidden,gate_reference_hidden,orders=1)
                ffn=AugmentedODEFFN(reference,width=width,state_width=lowrank_state_width,
                    rank=lowrank_rank,gate_hidden=lowrank_gate_hidden)
            elif kind=='ffn_poly_init':
                ffn=CollapsedODEFFN(OriginalODEFFN(width,gate_hidden,orders=orders,
                    tau=tau,scale=scale,shift=shift,gate_source='fixed'))
            elif kind=='ode_vector':
                ffn=VectorODEFFN(width,gate_hidden,reference_gate_hidden=gate_reference_hidden,
                    orders=orders,tau=tau,scale=scale,shift=shift,gate_center=gate_center,
                    middle_activation=ode_middle_activation,clock_mode=ode_clock_mode,
                    feature_mode=ode_feature_mode,
                    gate_transform=ode_gate_transform,rate_placement=ode_rate_placement,
                    solve_return=ode_solve_return,solve_returns=solve_returns,operator_mode=ode_operator_mode,
                    forcing_mode=ode_forcing_mode)
            elif kind=='ode_nonlinear':
                ffn=NonlinearDriftFFN(width,gate_hidden,gate_reference_hidden,orders=orders,
                    tau=tau,scale=scale,shift=shift,return_mode=ode_drift_return,gate_source=ode_drift_gate)
            elif kind=='ode_covariance':
                ffn=CovarianceODEFFN(width,gate_hidden,gate_reference_hidden,
                    heads=ode_covariance_heads,atoms=ode_covariance_atoms,
                    temperature=ode_covariance_temperature,mode=ode_covariance_mode,
                    tau=tau,scale=scale,shift=shift)
            else:
                ffn=OriginalODEFFN(width,gate_hidden,orders=orders,tau=tau,
                                   scale=scale,shift=shift,
                                   middle_activation=ode_middle_activation,
                                   gate_transform=ode_gate_transform,
                                   rate_placement=ode_rate_placement,
                                   solve_return=ode_solve_return,
                                   solve_returns=solve_returns,
                                   operator_mode=ode_operator_mode,
                                   gate_center=gate_center,
                                   clock_curvature=0.0 if kind=='ode_gate_linear' else 1.0,
                                   gate_source={'ode_time':'time', 'ode_fixed':'fixed',
                                                'ode_scalar':'scalar'}.get(kind, 'input'),
                                   spectral_cap=spectral_cap if kind=='ode_spectral_cap' else None)
            blocks.append(TimeConditionedBlock(width,width,ffn,1/math.sqrt(depth)))
        self.blocks=nn.ModuleList(blocks)
        self.norm=nn.LayerNorm(width)
        self.output=nn.Linear(width,dim)
        # Exactly zero initial predicted velocity for every architecture.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if ode_gate_reparam!=1.:
            with torch.no_grad():
                for _,layer in self.ode_layers():
                    layer.tau/=ode_gate_reparam
                    layer.t_func[-1].weight.div_(ode_gate_reparam)
                    layer.t_func[-1].bias.div_(ode_gate_reparam)
                    layer.gate_parameter_scale=ode_gate_reparam
        if ode_solve_init=='constant_paired':
            with torch.no_grad():
                for _,layer in self.ode_layers():
                    nn.init.zeros_(layer.t_func[-1].weight)
                    nn.init.zeros_(layer.t_func[-1].bias)
                    if layer.solve_return=='increment':
                        center=layer.shift+layer.scale/2
                        layer.proj.weight.add_(torch.eye(width,dtype=layer.proj.weight.dtype,
                                                        device=layer.proj.weight.device)/center)
                    layer.solve_initialization=ode_solve_init

    def forward(self,x,t):
        if x.ndim != 2 or x.shape[1] != self.dim or t.shape != (len(x),1):
            raise ValueError('expected x=[batch,dim], t=[batch,1]')
        temb=self.time(t)
        h=self.input(x)
        for block in self.blocks:
            h=block(h,temb)
        return self.output(self.norm(h))

    def ode_layers(self):
        for name,module in self.named_modules():
            if isinstance(module,OriginalODELayer):
                yield name,module

    def nonlinear_layers(self):
        for name,module in self.named_modules():
            if isinstance(module,NonlinearDriftFFN):yield name,module

    def diagonal_layers(self):
        for name,module in self.named_modules():
            if isinstance(module,DiagonalFlow):yield name,module

    def forced_gate_layers(self):
        for name,module in self.named_modules():
            if isinstance(module,ForcedGateResponse):yield name,module

    def state_clock_layers(self):
        for name,module in self.named_modules():
            if isinstance(module,StateClockLayer):yield name,module

    def lowrank_layers(self):
        for name,module in self.named_modules():
            if isinstance(module,PrunedLowRankLayer):yield name,module

    def covariance_layers(self):
        for name,module in self.named_modules():
            if isinstance(module,CovarianceODELayer):yield name,module

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

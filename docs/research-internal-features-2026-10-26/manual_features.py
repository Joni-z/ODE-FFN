"""Independent explicit intermediate-state traversal; no forward hooks."""
import numpy as np
import torch

def manual(model,x,t,arm):
    collected={};outputs=[]
    def add(key,value):collected.setdefault(key,[]).append(value.numpy().copy())
    with torch.no_grad():
        for start in range(0,len(x),256):
            xx=torch.from_numpy(np.ascontiguousarray(x[start:start+256]));tt=torch.full((len(xx),1),t,dtype=xx.dtype)
            context=model.raw.time(tt);h=model.raw.input(xx);add('input',h)
            for index,block in enumerate(model.raw.blocks):
                prefix=f'b{index}.';scale,shift=block.condition(context).chunk(2,-1)
                z=block.norm(h)*(1+scale)+shift;add(prefix+'z',z)
                if arm=='ode_k2_direct':
                    first=block.ffn.ode1;second=block.ffn.ode2
                    add(prefix+'clock1',first.t_func[1](first.t_func[0](z)))
                    one=first(z);add(prefix+'solve1',one);add(prefix+'increment1',one-z)
                    middle=block.ffn.act(one);add(prefix+'middle',middle)
                    add(prefix+'clock2',second.t_func[1](second.t_func[0](middle)))
                    update=second(middle);add(prefix+'increment2',update-middle)
                else:update=block.ffn(z)
                add(prefix+'ffn',update);h=h+block.residual_scale*update;add(prefix+'residual',h)
            normalized=model.raw.norm(h);add('post_norm',normalized);outputs.append(model.raw.output(normalized).numpy())
    return {key:np.concatenate(value) for key,value in collected.items()},np.concatenate(outputs)

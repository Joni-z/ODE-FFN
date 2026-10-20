"""Build an auditable stage report. Missing experiments remain explicitly pending."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

LABELS={'gate_k1_input':'ODE K1 input','gate_k1_time':'ODE K1 time',
        'gate_k2_input':'ODE K2 input','gate_k2_time':'ODE K2 time',
        'order_k4_input':'ODE K4 input','order_k8_input':'ODE K8 input',
        'silu_5000':'SiLU','swiglu_5000':'SwiGLU','silu_budget':'SiLU budget','swiglu_budget':'SwiGLU budget'}
DATA_LABEL={'gaussian_aniso_2d':'Gaussian','gmm_ring8_2d':'Ring8','two_moons_2d':'Moons'}

def number(x):return '待结果' if x is None else f'{x:.6f}'
def mean_layer(r,key):return mean(l[key]['mean'] for l in r['layers']) if r['layers'] else None

def read_rows(path):
    if not path.exists():return []
    text=path.read_text();parts=text.splitlines()
    if text and not text.endswith('\n'):parts=parts[:-1]
    return [json.loads(s) for s in parts if s.strip()]


def figures(index,manifest,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for j,ds in enumerate(manifest['datasets']):
        for k,col in [(1,'#0072B2'),(2,'#D55E00')]:
            rows=[index[(ds,f'gate_k{k}_{g}',5000)] for g in ('input','time')]
            axes[0,j].plot([0,1],[r['sampling']['32']['sw2'] for r in rows],marker='o',color=col,label=f'K={k}')
            axes[1,j].plot([0,1],[mean_layer(r,'gate_correction_norm') for r in rows],marker='o',color=col,label=f'K={k}')
        for label,col,style in [('silu_5000','#555555',':'),('swiglu_5000','#009E73','--')]:
            axes[0,j].axhline(index[(ds,label,5000)]['sampling']['32']['sw2'],color=col,ls=style,label=LABELS[label])
        axes[0,j].set_title(DATA_LABEL[ds]);axes[0,j].set_ylabel('SW2 (lower better)')
        axes[1,j].set_ylabel('Mean gate correction norm')
        for ax in axes[:,j]:ax.set_xticks([0,1],['Input gate','Time gate']);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Gate source ablation | seed 0 | 5000 updates | Heun32')
    save(fig,'gate-source')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,ds in zip(axes,manifest['datasets']):
        order_labels=['gate_k1_input','gate_k2_input','order_k4_input','order_k8_input']
        ax.plot([1,2,4,8],[index[(ds,l,5000)]['sampling']['32']['sw2'] for l in order_labels],'-o',label='ODE (5000 updates)',color='#0072B2')
        for base,col in [('silu','#555555'),('swiglu','#009E73')]:
            budgets=manifest['baseline_budgets'][base]['matched_steps']
            ax.plot([1,2,4,8],[index[(ds,base+'_budget',budgets[str(k)])]['sampling']['32']['sw2'] for k in (1,2,4,8)],'--s',label=base+' (MAC matched)',color=col)
        ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8],[1,2,4,8]);ax.set_xlabel('ODE Taylor order / budget');ax.set_ylabel('SW2 (lower better)');ax.set_title(DATA_LABEL[ds]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Arithmetic-budget comparison | seed 0 | forward MAC proxy, not measured GPU time')
    save(fig,'order-and-budget')
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for j,ds in enumerate(manifest['datasets']):
        labels=['gate_k1_input','gate_k2_input','order_k4_input','order_k8_input']
        rows=[index[(ds,l,5000)] for l in labels]
        axes[0,j].plot([1,2,4,8],[mean_layer(r,'relative_action_truncation_error') for r in rows],'-o',color='#0072B2')
        axes[1,j].plot([1,2,4,8],[mean_layer(r,'full_layer_jacobian_norm') for r in rows],'-o',color='#D55E00')
        axes[0,j].set_yscale('log');axes[0,j].set_title(DATA_LABEL[ds]);axes[0,j].set_ylabel('Relative action difference to exp(sP)')
        axes[1,j].set_ylabel('Mean local layer Jacobian norm')
        for ax in axes[:,j]:ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8],[1,2,4,8]);ax.set_xlabel('Taylor order');ax.grid(alpha=.2)
    fig.suptitle('Mechanism probes | seed 0 | 16 inputs/layer, averaged over 6 layers; no global bound')
    save(fig,'operator-mechanisms')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root',type=Path,default=Path('docs/research-2026-10-20'))
    parser.add_argument('--prior-root',type=Path,default=Path('docs/research-2026-10-19'))
    parser.add_argument('--output-root',type=Path,default=Path('../runs/synthetic'))
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'configs/synthetic/stage_20261020/manifest.json').read_text())
    index={}; progress=[]; curves=[]; failures=[]
    for ds in manifest['datasets']:
        for entry in manifest['entries']:
            cfg=json.loads((root/entry['config']).read_text())
            folder=args.output_root/cfg['experiment_id']/ds/entry['model']/'seed000'
            rows=read_rows(folder/'metrics.jsonl')
            failure=args.report_root/'failures'/ds/(entry['label']+'.json')
            if failure.exists():failures.append(json.loads(failure.read_text()))
            complete=((folder/'COMPLETE').exists() and (folder/'COMPLETE').read_text().strip()==str(entry['steps'])
                      and bool(rows) and rows[-1]['step']==entry['steps']
                      and (folder/'checkpoints'/f'step_{entry["steps"]:07d}.pt').exists())
            progress.append({'dataset':ds,'label':entry['label'],'complete':complete,'last_step':rows[-1]['step'] if rows else None,'run_dir':str(folder)})
            if rows:curves.append({'dataset':ds,'label':entry['label'],'records':rows})
            for step in entry['evaluate_steps']:
                path=args.report_root/'diagnostics'/ds/f'{entry["label"]}-step{step:07d}.json'
                if path.exists():
                    r=json.loads(path.read_text())
                    assert (r['dataset'],r['label'],r['step'],r['seed'])==(ds,entry['label'],step,0)
                    assert r['config_sha256']==entry['sha256']
                    index[(ds,entry['label'],step)]=r
    completed=sum(p['complete'] for p in progress)
    full=completed==manifest['run_count'] and len(index)==manifest['eval_checkpoint_count']
    if args.require_complete and not full:raise SystemExit(f'Report incomplete: {completed}/30 training, {len(index)}/54 diagnostics')
    args.report_root.mkdir(parents=True,exist_ok=True)
    (args.report_root/'stage-summary.json').write_text(json.dumps({'complete':full,'progress':progress,'diagnostics':list(index.values()),'curves':curves},indent=2,allow_nan=False)+'\n')
    lines=['# ODELayer 项目阶段性报告','',f'状态：{"实验与诊断齐全；以下为描述性分析，方向由用户决定" if full else "阶段草稿：新实验未齐全，待结果项不作结论"}。训练 {completed}/30；预定checkpoint诊断 {len(index)}/54。',
    '', '## 研究问题与原始设计',
    '项目考察：把普通FFN替换为教授提出的ODELayer组合，是否更适合flow matching学习的速度场。原版单层 F(h)=T_K(s(h)P)h，s=0.2+0.8*sigmoid(g(h)/10)，T_K(A)=I+A+...+A^K/K!；完整FFN为两段ODELayer，中间SiLU。P为自由、无bias的方阵。隐藏演化时间s与外部生成时间t不同，P不等于图像速度场的Jacobian。',
    '本阶段使用2D向量FM，直接预测velocity；JiT图像主干以SwiGLU为baseline、直接预测clean image后换算velocity。toy模型没有attention，当前证据不支持直接声称图像收益。',
    '', '## 数学依据及其边界',
    '冻结s时，指数exp(sP)是dy/du=Py的精确演化。指数级数对任意有限P收敛；本项目研究的是有限阶结构与学习，而不是通过限制eigenvalue保证级数存在。',
    '令A=T_K(sP)，b=P*T_(K-1)(sP)*h，则完整导数J_F=A+b*grad(s)^T。限制P没有自动控制门控项。P=S+R中S对称、R反对称；真实连续系统的长度增长由S决定，但低阶Taylor不会自动继承性质。例如纯二维旋转、s*w=1：精确解增益1，K1为1.4142，K2为1.1180。数值核验见上一阶段math-verification.json。',
    '这些是标准数学恒等式及本项目的核验，不是新定理。更接近exp也未必更好：网络可能利用有限多项式的表达能力。实际FM终点误差还依赖速度拟合误差的时间位置、完整网络的误差传播和外部solver误差。当前没有证明内部P约束能直接控制完整生成场。',
    '', '## 既有18run证据（seed0，5000步）','',
    '|模型|Gaussian SW2|Ring8 SW2|Moons SW2|','|---|---:|---:|---:|']
    prior=json.loads((args.prior_root/'comparison-summary.json').read_text())
    for kind,k,label in [('ffn',None,'SiLU'),('swiglu',None,'SwiGLU'),('ode_original',1,'ODE K1'),('ode_spectral_cap',1,'cap K1'),('ode_original',2,'ODE K2'),('ode_spectral_cap',2,'cap K2')]:
        vals=[]
        for ds in manifest['datasets']:
            r=next(r for r in prior if r['dataset']==ds and r['model']==kind and r['orders']==k)
            vals.append(number(r['last_sample']['sliced_w2']))
        lines.append('|'+label+'|'+'|'.join(vals)+'|')
    lines+=['', '以上SW2来自固定Heun32。原版K2本次在三个数据集优于SwiGLU，但Ring8上SiLU更好；K2比SwiGLU更贵，且只有一个训练seed。谱cap减小内部层Jacobian但无一致生成收益。增大Heun到256未消除主要排序；不能把差距主要归因于32步不足。',
    'Ring8已有分时诊断：SwiGLU在t=0–0.1的oracle MSE约0.0250，SiLU约0.0141；t=0.7–0.8时分别0.0256和0.0690。早期偏差与模态分配的联系仅为假设，尚未通过轨迹干预或mode mass证明。Gaussian上原版K1比K2更偏离指数，但生成更好，不支持“越像精确ODE越好”的简单叙事。完整出处为上一阶段results.md及diagnostics/*.json。',
    '', '## 新实验协议',
    '第一波：gate来源input/time × K1/K2，加SiLU和SwiGLU同步baseline，三个数据集共18run。第二波：原版input gate的K4/K8及两种baseline长轨迹，共12run。没有叠加P约束、新loss、归一化、optimizer或数据改变。所有训练seed0。配置和预算节点在训练前冻结；见protocol.md与configs/synthetic/stage_20261020/manifest.json。',
    '每个数据集的两波实验在同一GPU顺序运行；不同数据集可用不同GPU。输入与时间gate同参数、同初始权重，但gate输入分布、初始s及梯度路径不同，不能把所有性能差异归因于单独移除了rank1项。K8仍是截断多项式，不称精确指数。',
    '计算预算使用整个网络的前向线性MAC代理，含time embedding、FiLM等，不含非线性、反向、优化器与评估成本，因此不声称严格相同GPU时长。额外checkpoint已测试不改变训练权重/RNG轨迹。',
    '', '## 新实验：同5000步对照','',
    '|模型|Gaussian SW2|Ring8 SW2|Moons SW2|','|---|---:|---:|---:|']
    mainlabels=['silu_5000','swiglu_5000','gate_k1_input','gate_k1_time','gate_k2_input','gate_k2_time','order_k4_input','order_k8_input']
    for label in mainlabels:
        vals=[number(index.get((ds,label,5000),{}).get('sampling',{}).get('32',{}).get('sw2')) for ds in manifest['datasets']]
        lines.append('|'+LABELS[label]+'|'+'|'.join(vals)+'|')
    lines+=['','## 配对门控差值','', '差值=time gate减input gate，负数代表time gate的SW2更低；没有跨seed置信区间。','',
            '|数据|K|SW2差值|input gate项范数|time gate项范数|','|---|---:|---:|---:|---:|']
    for ds in manifest['datasets']:
        for k in (1,2):
            a=index.get((ds,f'gate_k{k}_input',5000));b=index.get((ds,f'gate_k{k}_time',5000))
            vals=[b['sampling']['32']['sw2']-a['sampling']['32']['sw2'],mean_layer(a,'gate_correction_norm'),mean_layer(b,'gate_correction_norm')] if a and b else [None]*3
            lines.append(f'|{DATA_LABEL[ds]}|{k}|'+ '|'.join(number(v) for v in vals)+'|')
    lines+=['','## 计算预算对照','', '|数据|K|ODE步数|ODE SW2|SiLU步数|SiLU SW2|SwiGLU步数|SwiGLU SW2|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for ds in manifest['datasets']:
        for k,label in zip((1,2,4,8),('gate_k1_input','gate_k2_input','order_k4_input','order_k8_input')):
            ode=index.get((ds,label,5000));cells=[DATA_LABEL[ds],str(k),'5000',number(ode['sampling']['32']['sw2'] if ode else None)]
            for base in ('silu','swiglu'):
                n=manifest['baseline_budgets'][base]['matched_steps'][str(k)];r=index.get((ds,base+'_budget',n))
                cells += [str(n),number(r['sampling']['32']['sw2'] if r else None)]
            lines.append('|'+'|'.join(cells)+'|')
    lines+=['','## 新实验机制证据','', '|数据|模型|指数作用相对偏差|完整层Jacobian范数|Heun128对256端点RMS|','|---|---|---:|---:|---:|']
    for ds in manifest['datasets']:
        for label in mainlabels[2:]:
            r=index.get((ds,label,5000))
            vals=[mean_layer(r,'relative_action_truncation_error'),mean_layer(r,'full_layer_jacobian_norm'),r['sampling']['128']['endpoint_rms_to_heun256']] if r else [None]*3
            lines.append(f'|{DATA_LABEL[ds]}|{LABELS[label]}|'+ '|'.join(number(v) for v in vals)+'|')
    lines+=['','这些Jacobian是固定probe的局部描述，非全局Lipschitz界；六层和16个输入不算独立训练重复。模型训练后P和gate会共适应，阶数与指数偏差之间的关系不等于性能因果机制。完整时间段误差、P轨迹与NFE扫描保存在stage-summary.json及各checkpoint诊断文件。',
            '', '## 技术复跑与数据完整性']
    for ds in manifest['datasets']:
        path=args.report_root/f'prefix-check-{ds}.json'
        if path.exists():
            checks=json.loads(path.read_text())
            values=[f"{b}={v['model_max_abs_difference_at5000']:.3g}" if 'model_max_abs_difference_at5000' in v else f'{b}=无法核验' for b,v in checks.items()]
            lines.append(f'- {DATA_LABEL[ds]} baseline前5000步权重最大差：'+', '.join(values)+'。这些是技术复跑，不增加独立seed数。')
        else:lines.append(f'- {DATA_LABEL[ds]}：baseline前缀核验待完成。')
    if failures:
        lines+=['','## 已记录的训练失败（不是待运行）']
        for f in failures:lines.append(f"- {f['dataset']} / {f['label']}: {f['reason']}。该配置没有有效终点分数，不参与胜负比较。")
    if full:
        figures(index,manifest,args.report_root/'figures')
        lines+=['','## 图','',
                '![门控比较](figures/gate-source.png)','固定5000步的门控对照与实际门控修正项，baseline为水平参考线。',
                '![阶数与计算预算](figures/order-and-budget.png)','ODE固定5000步，对照baseline使用预定MAC匹配步数；没有挑选最佳checkpoint。',
                '![内部机制](figures/operator-mechanisms.png)','指数作用偏差和局部层Jacobian分开呈现；不暗示二者对生成质量有单调因果关系。']
    lines+=['','## 决策依据与尚未解决的问题',
    '本报告供用户选择继续方向，不自动决定保留哪一结构。判断时应同时读质量、拟合、内部机制和预算：time gate即使降低J也可能损失表达力；高K即使更接近exp也可能不值额外计算；ODE若只在同step优于baseline、在匹配预算后优势消失，应明确降低对架构优势的判断。',
    '只有一个seed，当前是机制探索，不能作统计显著性或普适性结论。既有测试点曾多次用于阶段性观察与假设形成，不是全新的独立确认集。不同分布的SW2不直接平均。Moons无解析oracle，pair MSE不冒充真实场误差。Heun256仅为数值参考；若128→256差异仍大，不宣称收敛。尚未证明隐藏P性质传递到完整速度网络，也尚无图像迁移证据。',
    '待用户决策：①输入依赖是否值得保留；②高阶是否具有预算优势；③是否值得另做P约束/结构化生成元；④何时迁移回JiT。任何新方向应单独立对照，不把多个trick合并后再解释。',
    '', '## 证据与复现入口',
    '- protocol.md：提前确定的比较、预算、固定条件与决策规则。',
    '- stage-summary.json：所有训练曲线、进度与诊断；diagnostics/：逐checkpoint结果及哈希。',
    '- job-*.json / prefix-check-*.json：执行顺序、GPU、jobID和baseline技术复跑核验。',
    '- source-manifest.json / submission.json（如已生成）：发布的代码哈希和真实Slurm作业号。',
    '- 数学背景：[Lipschitz regularity in Flow Matching and Diffusion Models (2026预印本)](https://arxiv.org/html/2604.06065v1)；baseline定义：[JiT官方代码](https://github.com/LTH14/JiT/blob/main/model_jit.py)。这些文献不证明本项目方法有效。',
    '', f'报告生成完整性标记：complete={full}。未齐全时所有新结论保持待定。']
    (args.report_root/'stage-report.md').write_text('\n'.join(lines)+'\n')
    print(f'Report: {completed}/30 training; {len(index)}/54 diagnostics; complete={full}')

if __name__=='__main__':main()

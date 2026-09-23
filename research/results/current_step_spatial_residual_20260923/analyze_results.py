"""Aggregate current-step spatial residual LOPO evaluation."""
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
from statistics import mean,stdev

HERE=Path(__file__).resolve().parent
MODELS=('cached_lowrank','conv_delta','shallow_delta','combined','combined_spatial_shuffle','oracle')
DEPLOYABLE=MODELS[:-1]

def stats(values):
    return {'n':len(values),'mean':mean(values),'std':stdev(values) if len(values)>1 else 0.0,
            'min':min(values),'max':max(values)}

def main():
    out=HERE/'outputs'; manifest=json.loads((out/'manifest.json').read_text())
    files=sorted((out/'evaluation').glob('*.json'))
    if len(files)!=20: raise RuntimeError(f'Expected 20 evaluation files, found {len(files)}')
    values={metric:defaultdict(list) for metric in ('residual_relative_l1','corrected_feature_relative_l1','guided_noise_error','recovery_fraction','milliseconds')}
    by_step=defaultdict(lambda:defaultdict(list)); by_prompt=defaultdict(lambda:defaultdict(list)); rows=[]
    baseline=[]
    for path in files:
        prompt=path.stem.rsplit('_seed',1)[0]; seed=int(path.stem.rsplit('seed',1)[1])
        trace=json.loads(path.read_text());
        if len(trace)!=20: raise RuntimeError(f'{path.name}: expected 20 steps')
        for row in trace:
            if row['refresh']: continue
            baseline.append(row['baseline_guided_noise_error'])
            for model,result in row['models'].items():
                flat={'case':path.stem,'prompt_id':prompt,'seed':seed,'step':row['step'],'age':row['age'],
                      'model':model,'baseline_guided_noise_error':row['baseline_guided_noise_error'],**result}
                rows.append(flat)
                for metric in values: values[metric][model].append(result[metric])
                by_step[row['step']][model].append(result['recovery_fraction'])
                by_prompt[prompt][model].append(result['recovery_fraction'])
    if len(rows)!=600: raise RuntimeError(f'Expected 600 probe rows, found {len(rows)}')
    summary={'cases':len(files),'probe_rows':len(rows),'baseline_reuse_guided_noise_error':stats(baseline),'models':{},
             'by_step_recovery':{},'by_prompt_recovery':{}}
    oracle_mean=mean(values['recovery_fraction']['oracle'])
    for model in MODELS:
        summary['models'][model]={metric:stats(values[metric][model]) for metric in values}
        summary['models'][model]['oracle_gain_fraction']=mean(values['recovery_fraction'][model])/oracle_mean
        summary['models'][model]['positive_prompts']=sum(mean(group[model])>0 for group in by_prompt.values())
    summary['by_step_recovery']={str(step):{model:mean(group[model]) for model in MODELS}
                                 for step,group in sorted(by_step.items())}
    summary['by_prompt_recovery']={prompt:{model:mean(group[model]) for model in MODELS}
                                   for prompt,group in sorted(by_prompt.items())}
    best=max(DEPLOYABLE,key=lambda m:mean(values['recovery_fraction'][m]))
    gates={'mean_recovery_at_least_20pct':mean(values['recovery_fraction'][best])>=0.20,
           'at_least_7_positive_prompts':summary['models'][best]['positive_prompts']>=7,
           'step4_nonnegative':summary['by_step_recovery']['4'][best]>=0,
           'step9_nonnegative':summary['by_step_recovery']['9'][best]>=0,
           'combined_beats_spatial_shuffle':mean(values['recovery_fraction']['combined'])>mean(values['recovery_fraction']['combined_spatial_shuffle'])}
    summary['decision']={'best_model':best,'gates':gates,'pass':all(gates.values()),
      'paired_mean_differences':{
       'conv_delta_minus_cached':mean(values['recovery_fraction']['conv_delta'])-mean(values['recovery_fraction']['cached_lowrank']),
       'shallow_delta_minus_cached':mean(values['recovery_fraction']['shallow_delta'])-mean(values['recovery_fraction']['cached_lowrank']),
       'combined_minus_cached':mean(values['recovery_fraction']['combined'])-mean(values['recovery_fraction']['cached_lowrank']),
       'combined_minus_spatial_shuffle':mean(values['recovery_fraction']['combined'])-mean(values['recovery_fraction']['combined_spatial_shuffle'])}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'rows.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report=['# Current-Step Spatial Feature Conditioned Residual Predictability','',
      f"LOPO cases: {len(files)}; probe rows: {len(rows)}. Best deployable model: `{best}`.",'',
      '| Model | Residual rel-L1 | Corrected feature rel-L1 | Guided error | Recovery | Oracle gain | Positive prompts | Probe ms |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        m=summary['models'][model]
        report.append(f"| {model} | {m['residual_relative_l1']['mean']:.4f} | {m['corrected_feature_relative_l1']['mean']:.4f} | {m['guided_noise_error']['mean']:.6f} | {m['recovery_fraction']['mean']:.4f} | {m['oracle_gain_fraction']:.3f} | {m['positive_prompts']}/10 | {m['milliseconds']['mean']:.1f} |")
    report += ['', '## Recovery by step','',
      '| Step | Cached only | Conv delta | Shallow delta | Combined | Spatial shuffle | Oracle |',
      '|---:|---:|---:|---:|---:|---:|---:|']
    for step,m in summary['by_step_recovery'].items():
        report.append(f"| {step} | {m['cached_lowrank']:.4f} | {m['conv_delta']:.4f} | {m['shallow_delta']:.4f} | {m['combined']:.4f} | {m['combined_spatial_shuffle']:.4f} | {m['oracle']:.4f} |")
    report += ['', '## Predeclared decision','', '| Gate | Pass |','|---|---:|']
    for gate,passed in gates.items(): report.append(f"| {gate} | {'yes' if passed else 'no'} |")
    report += ['', '**FAIL.** No current-conditioned rank-16 model reaches the predeclared 20% recovery threshold, and the best model still worsens steps 4 and 9.', '',
      'The aligned combined model exceeds its spatial-shuffle control by only '
      f"{summary['decision']['paired_mean_differences']['combined_minus_spatial_shuffle']*100:.2f} percentage points. This is not strong evidence that the predictor learned a causal spatial correction.", '',
      'The teacher oracle remains high (67.35%), so the repair location still has theoretical value. The failure is the generalization of the tested inference-available predictors, not the absence of a correct activation at this location.', '',
      'Following the predeclared protocol, do not proceed to a closed-loop SSIM experiment or increase predictor depth. Stop this `up_block_1` residual-predictor branch and move to local selective recomputation or a cache representation that retains current spatial information.', '',
      'All measurements are teacher-forced single-step probes. They do not directly measure final-image SSIM.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(summary['decision'],indent=2))
if __name__=='__main__': main()

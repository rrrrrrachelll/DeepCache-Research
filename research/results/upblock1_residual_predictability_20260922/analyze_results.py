from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
from statistics import mean,stdev

MODELS=('channel_mean','channel_affine','oracle')
HERE=Path(__file__).resolve().parent

def stats(x): return {'n':len(x),'mean':mean(x),'std':stdev(x) if len(x)>1 else 0.0,'min':min(x),'max':max(x)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=HERE/'outputs'); args=ap.parse_args()
    manifest=json.loads((args.output_dir/'manifest.json').read_text()); values=defaultdict(lambda:defaultdict(list)); by_prompt=defaultdict(lambda:defaultdict(list)); by_step=defaultdict(lambda:defaultdict(list)); rows=[]
    for case in manifest['evaluation']:
        trace=json.loads((args.output_dir/case['file']).read_text())
        for row in trace:
            for model,m in row['models'].items():
                flat={'case':case['case'],'prompt_id':case['prompt_id'],'seed':case['seed'],'step':row['step'],'model':model,
                      'baseline_guided_noise_error':row['baseline_guided_noise_error'],**m}; rows.append(flat)
                for k,v in m.items(): values[model][k].append(v)
                by_prompt[case['prompt_id']][model].append(m['recovery_fraction']); by_step[row['step']][model].append(m['recovery_fraction'])
    summary={'cases':len(manifest['evaluation']),'rows':len(rows),'models':{},'by_prompt_recovery':{},'by_step_recovery':{}}
    oracle_mean=mean(values['oracle']['recovery_fraction'])
    for model in MODELS:
        summary['models'][model]={k:stats(v) for k,v in values[model].items()}
        summary['models'][model]['oracle_gain_fraction']=mean(values[model]['recovery_fraction'])/oracle_mean
    for prompt,models in by_prompt.items(): summary['by_prompt_recovery'][prompt]={m:mean(v) for m,v in models.items()}
    for step,models in sorted(by_step.items()): summary['by_step_recovery'][str(step)]={m:mean(v) for m,v in models.items()}
    with (args.output_dir/'rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    report=['# up_block_1 Residual Predictability','',f"LOPO cases: {summary['cases']}; probe rows: {summary['rows']}.",'',
            '| Model | Residual rel-L1 | Corrected feature rel-L1 | Guided error | Recovery | Oracle gain | Probe ms |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for m in MODELS:
        q=summary['models'][m]; report.append(f"| {m} | {q['residual_relative_l1']['mean']:.4f} | {q['corrected_feature_relative_l1']['mean']:.4f} | {q['guided_noise_error']['mean']:.6f} | {q['recovery_fraction']['mean']:.4f} | {q['oracle_gain_fraction']:.3f} | {q['milliseconds']['mean']:.1f} |")
    best=max(MODELS[:-1],key=lambda m:summary['models'][m]['recovery_fraction']['mean']); recovery=summary['models'][best]['recovery_fraction']['mean']; positive=sum(summary['by_prompt_recovery'][p][best]>0 for p in summary['by_prompt_recovery'])
    decision=('PASS: proceed to a lightweight convolutional predictor and closed-loop validation.' if recovery>=0.30 and positive>=8 else
              'PARTIAL: add a cheap current-step shallow feature before deciding.' if recovery>=0.20 and positive>=7 else
              'FAIL: cached activation plus step-conditioned channel statistics are not sufficient; do not deploy this corrector.')
    report += ['', '## Recovery by step','', '| Step | Channel mean | Channel affine | Oracle |', '|---:|---:|---:|---:|']
    for step,models in sorted(by_step.items()): report.append(f"| {step} | {mean(models['channel_mean']):.4f} | {mean(models['channel_affine']):.4f} | {mean(models['oracle']):.4f} |")
    report += ['', '## Prompt-grouped result','',f'Best deployable baseline: `{best}`; positive prompts: {positive}/{len(summary["by_prompt_recovery"])}.','', '## Decision','',decision]
    (args.output_dir/'REPORT.md').write_text('\n'.join(report)+'\n'); print('\n'.join(report))
if __name__=='__main__': main()

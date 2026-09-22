"""Collect prompt-grouped sufficient stats and run LOPO residual prediction probes."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from residual_oracle import StatsCollectorHelper, PredictorEvaluationHelper, combine_stats, fit_channel_models

MODEL_ID="runwayml/stable-diffusion-v1-5"
REFRESH=[0,2,5,7,10,13,16,19]
PROBES=[4,9,12,15,18]
PROMPT_FILE=ROOT/'research/results/oracle_20260911/oracle_prompts.json'


def scheduler(config):
    return DPMSolverMultistepScheduler.from_config(config, algorithm_type='dpmsolver++', solver_order=2)


def load_pipe():
    pipe=StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
        local_files_only=os.environ.get('HF_HUB_OFFLINE','0')=='1').to('cuda')
    pipe.set_progress_bar_config(disable=True)
    return pipe


def latent_for(seed):
    g=torch.Generator(device='cuda').manual_seed(seed)
    return torch.randn((1,4,64,64),generator=g,device='cuda',dtype=torch.float16)


def fit_folds(stats_dir, prompts, seeds, output):
    folds={}
    for held in prompts:
        parameters={}
        for step in PROBES:
            items=[]
            for prompt in prompts:
                if prompt['id']==held['id']: continue
                for seed in seeds:
                    data=torch.load(stats_dir/f"{prompt['id']}_seed{seed}.pt", map_location='cpu')
                    items.append(data['stats'][str(step)])
            parameters[str(step)]=fit_channel_models(combine_stats(items))
        folds[held['id']]=parameters
    torch.save(folds, output)
    return folds


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,default=HERE/'outputs')
    ap.add_argument('--limit-prompts',type=int)
    ap.add_argument('--limit-seeds',type=int)
    ap.add_argument('--phase',choices=('all','collect','evaluate'),default='all')
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    stats_dir=args.output_dir/'stats'; eval_dir=args.output_dir/'evaluation'
    stats_dir.mkdir(exist_ok=True); eval_dir.mkdir(exist_ok=True)
    spec=json.loads(PROMPT_FILE.read_text()); prompts=spec['analysis']
    if args.limit_prompts: prompts=prompts[:args.limit_prompts]
    seeds=spec['seeds']
    if args.limit_seeds: seeds=seeds[:args.limit_seeds]
    pipe=load_pipe(); config=dict(pipe.scheduler.config)
    manifest={'model':MODEL_ID,'refresh_steps':REFRESH,'probe_steps':PROBES,
              'split':'analysis prompts with prompt-grouped LOPO; held-out excluded',
              'prompts':prompts,'seeds':seeds,'collection':[],'evaluation':[]}

    if args.phase in ('all','collect'):
        for i,(prompt,seed) in enumerate([(p,s) for p in prompts for s in seeds],1):
            case=f"{prompt['id']}_seed{seed}"; path=stats_dir/f'{case}.pt'
            if path.exists():
                manifest['collection'].append({'case':case,'prompt_id':prompt['id'],'seed':seed,'file':str(path.relative_to(args.output_dir)),'resumed':True}); continue
            latent=latent_for(seed); kwargs=dict(prompt=prompt['prompt'],latents=latent.clone(),num_inference_steps=20,
                guidance_scale=7.5,height=512,width=512,output_type='np')
            pristine=None
            if i==1:
                pipe.scheduler=scheduler(config)
                with torch.inference_mode(): pristine=pipe(**kwargs).images.copy()
            pipe.scheduler=scheduler(config); pipe.scheduler.set_timesteps(20,device='cuda')
            helper=StatsCollectorHelper(pipe,REFRESH,PROBES); helper.enable(); start=time.perf_counter()
            with torch.inference_mode(): result=pipe(**{**kwargs,'latents':latent.clone()})
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start; helper.disable()
            if pristine is not None and not np.array_equal(pristine,result.images): raise RuntimeError('Collector changed teacher trajectory')
            torch.save({'case':case,'prompt_id':prompt['id'],'seed':seed,'stats':helper.sufficient_stats,'trace':helper.trace},path)
            manifest['collection'].append({'case':case,'file':str(path.relative_to(args.output_dir)),'seconds':elapsed})
            (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print(f"collect [{i}/{len(prompts)*len(seeds)}] {case}: {elapsed:.1f}s",flush=True)

    folds_path=args.output_dir/'lopo_parameters.pt'
    folds=fit_folds(stats_dir,prompts,seeds,folds_path)
    if args.phase in ('all','evaluate'):
        cases=[(p,s) for p in prompts for s in seeds]
        for i,(prompt,seed) in enumerate(cases,1):
            case=f"{prompt['id']}_seed{seed}"; path=eval_dir/f'{case}.json'
            if path.exists():
                manifest['evaluation'].append({'case':case,'prompt_id':prompt['id'],'seed':seed,'file':str(path.relative_to(args.output_dir)),'resumed':True}); continue
            latent=latent_for(seed); pipe.scheduler=scheduler(config); pipe.scheduler.set_timesteps(20,device='cuda')
            helper=PredictorEvaluationHelper(pipe,REFRESH,PROBES,parameters=folds[prompt['id']]); helper.enable(); start=time.perf_counter()
            with torch.inference_mode(): pipe(prompt['prompt'],latents=latent.clone(),num_inference_steps=20,
                guidance_scale=7.5,height=512,width=512,output_type='np')
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start; helper.disable()
            path.write_text(json.dumps(helper.trace,indent=2))
            manifest['evaluation'].append({'case':case,'prompt_id':prompt['id'],'seed':seed,
                'file':str(path.relative_to(args.output_dir)),'seconds':elapsed})
            (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print(f"eval [{i}/{len(cases)}] {case}: {elapsed:.1f}s",flush=True)

if __name__=='__main__': main()

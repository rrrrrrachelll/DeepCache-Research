"""Collect spatial features, fit LOPO low-rank models, and evaluate causal probes."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from spatial_residual import (SpatialCollectorHelper, SpatialEvaluationHelper,
    BASE_MODELS, feature_tensor, fit_lowrank)

MODEL_ID='runwayml/stable-diffusion-v1-5'; REFRESH=[0,2,5,7,10,13,16,19]; PROBES=[4,9,12,15,18]
PROMPT_FILE=ROOT/'research/results/oracle_20260911/oracle_prompts.json'

def scheduler(config):
    return DPMSolverMultistepScheduler.from_config(config,algorithm_type='dpmsolver++',solver_order=2)

def load_pipe():
    pipe=StableDiffusionPipeline.from_pretrained(MODEL_ID,torch_dtype=torch.float16,
        safety_checker=None,requires_safety_checker=False,
        local_files_only=os.environ.get('HF_HUB_OFFLINE','0')=='1').to('cuda')
    pipe.set_progress_bar_config(disable=True); return pipe

def latent_for(seed):
    g=torch.Generator(device='cuda').manual_seed(seed)
    return torch.randn((1,4,64,64),generator=g,device='cuda',dtype=torch.float16)

def sample_features(sample,model):
    parts=[sample['cached'].float()]
    if model in ('conv_delta','combined'): parts.append(sample['conv_delta'].float())
    if model in ('shallow_delta','combined'): parts.append(sample['shallow_delta'].float())
    return torch.cat(parts,dim=1)

def fit_folds(data_dir,prompts,seeds,output):
    loaded={}
    for prompt in prompts:
        for seed in seeds:
            loaded[(prompt['id'],seed)]=torch.load(data_dir/f"{prompt['id']}_seed{seed}.pt",map_location='cpu')
    folds={}
    for fold_index,held in enumerate(prompts,1):
        folds[held['id']]={}
        for model in BASE_MODELS:
            train={}
            for step in PROBES:
                samples=[loaded[(p['id'],s)]['samples'][str(step)] for p in prompts if p['id']!=held['id'] for s in seeds]
                train[str(step)]={'x':torch.cat([sample_features(item,model) for item in samples]),
                                  'y':torch.cat([item['residual'].float() for item in samples])}
            folds[held['id']][model]=fit_lowrank(train,rank=16,ridge=1e-3)
        print(f"fit [{fold_index}/{len(prompts)}] held={held['id']}",flush=True)
    torch.save(folds,output); return folds

def rebuild_manifest(output,prompts,seeds):
    manifest={'model':MODEL_ID,'refresh_steps':REFRESH,'probe_steps':PROBES,
      'split':'10 analysis prompts x seeds 101,202; prompt-grouped LOPO; held-out excluded',
      'rank':16,'ridge':1e-3,'samples_per_cfg_branch':256,'prompts':prompts,'seeds':seeds,
      'collection':[],'evaluation':[]}
    for p in prompts:
        for seed in seeds:
            case=f"{p['id']}_seed{seed}"; path=output/'data'/f'{case}.pt'
            if path.exists(): manifest['collection'].append({'case':case,'prompt_id':p['id'],'seed':seed,'file':str(path.relative_to(output)),'resumed':True})
            ep=output/'evaluation'/f'{case}.json'
            if ep.exists(): manifest['evaluation'].append({'case':case,'prompt_id':p['id'],'seed':seed,'file':str(ep.relative_to(output)),'resumed':True})
    return manifest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=HERE/'outputs')
    ap.add_argument('--phase',choices=('all','collect','fit','evaluate'),default='all')
    ap.add_argument('--limit-prompts',type=int); ap.add_argument('--limit-seeds',type=int)
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    data_dir=args.output_dir/'data'; eval_dir=args.output_dir/'evaluation'; data_dir.mkdir(exist_ok=True); eval_dir.mkdir(exist_ok=True)
    spec=json.loads(PROMPT_FILE.read_text()); prompts=spec['analysis']; seeds=spec['seeds']
    if args.limit_prompts: prompts=prompts[:args.limit_prompts]
    if args.limit_seeds: seeds=seeds[:args.limit_seeds]
    manifest=rebuild_manifest(args.output_dir,prompts,seeds)
    need_gpu=args.phase in ('all','collect','evaluate')
    pipe=load_pipe() if need_gpu else None; config=dict(pipe.scheduler.config) if pipe else None

    if args.phase in ('all','collect'):
        cases=[(p,s) for p in prompts for s in seeds]
        for i,(prompt,seed) in enumerate(cases,1):
            case=f"{prompt['id']}_seed{seed}"; path=data_dir/f'{case}.pt'
            if path.exists(): print(f"collect [{i}/{len(cases)}] resume {case}",flush=True); continue
            latent=latent_for(seed); kwargs=dict(prompt=prompt['prompt'],latents=latent.clone(),num_inference_steps=20,
                guidance_scale=7.5,height=512,width=512,output_type='np')
            pristine=None
            if i==1:
                pipe.scheduler=scheduler(config)
                with torch.inference_mode(): pristine=pipe(**kwargs).images.copy()
            pipe.scheduler=scheduler(config); pipe.scheduler.set_timesteps(20,device='cuda')
            helper=SpatialCollectorHelper(pipe,REFRESH,PROBES); helper.enable(); start=time.perf_counter()
            with torch.inference_mode(): result=pipe(**{**kwargs,'latents':latent.clone()})
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start; helper.disable()
            if pristine is not None and not np.array_equal(pristine,result.images): raise RuntimeError('Collector changed teacher trajectory')
            torch.save({'case':case,'prompt_id':prompt['id'],'seed':seed,'samples':helper.samples,'trace':helper.trace},path)
            print(f"collect [{i}/{len(cases)}] {case}: {elapsed:.1f}s",flush=True)
            manifest=rebuild_manifest(args.output_dir,prompts,seeds); (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))

    folds_path=args.output_dir/'lopo_parameters.pt'
    if args.phase in ('all','fit'):
        folds=fit_folds(data_dir,prompts,seeds,folds_path)
    elif args.phase=='evaluate':
        folds=torch.load(folds_path,map_location='cpu')
    else: folds=None

    if args.phase in ('all','evaluate'):
        cases=[(p,s) for p in prompts for s in seeds]
        for i,(prompt,seed) in enumerate(cases,1):
            case=f"{prompt['id']}_seed{seed}"; path=eval_dir/f'{case}.json'
            if path.exists(): print(f"eval [{i}/{len(cases)}] resume {case}",flush=True); continue
            latent=latent_for(seed); pipe.scheduler=scheduler(config); pipe.scheduler.set_timesteps(20,device='cuda')
            helper=SpatialEvaluationHelper(pipe,REFRESH,PROBES,parameters=folds[prompt['id']]); helper.enable(); start=time.perf_counter()
            with torch.inference_mode(): pipe(prompt['prompt'],latents=latent.clone(),num_inference_steps=20,
                guidance_scale=7.5,height=512,width=512,output_type='np')
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start; helper.disable()
            path.write_text(json.dumps(helper.trace,indent=2)); print(f"eval [{i}/{len(cases)}] {case}: {elapsed:.1f}s",flush=True)
            manifest=rebuild_manifest(args.output_dir,prompts,seeds); (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

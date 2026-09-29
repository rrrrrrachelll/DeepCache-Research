"""Run symmetric matched encoder-decoder recomputation oracle."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler,StableDiffusionPipeline

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from symmetric_oracle import SymmetricMatchedOracle,PATHS
MODEL_ID='runwayml/stable-diffusion-v1-5'; REFRESH=[0,2,5,7,10,13,16,19]; PROBES=[4,9]
PROMPT_FILE=ROOT/'research/results/oracle_20260911/oracle_prompts.json'

def scheduler(config): return DPMSolverMultistepScheduler.from_config(config,algorithm_type='dpmsolver++',solver_order=2)
def latent_for(seed):
    g=torch.Generator(device='cuda').manual_seed(seed)
    return torch.randn((1,4,64,64),generator=g,device='cuda',dtype=torch.float16)
def rebuild_manifest(out,prompts,seeds):
    manifest={'model':MODEL_ID,'scheduler':'DPM-Solver++ order 2','num_inference_steps':20,'guidance_scale':7.5,
      'cache_branch_id':0,'refresh_steps':REFRESH,'probe_steps':PROBES,'frontiers':PATHS,
      'split':'10 analysis prompts x seeds 101,202; consumed held-out excluded','prompts':prompts,'seeds':seeds,'cases':[]}
    for p in prompts:
      for seed in seeds:
       case=f"{p['id']}_seed{seed}"; path=out/'traces'/f'{case}.json'
       if path.exists(): manifest['cases'].append({'case':case,'prompt_id':p['id'],'seed':seed,'trace':str(path.relative_to(out)),'resumed':True})
    return manifest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=HERE/'symmetric_outputs')
    ap.add_argument('--limit-prompts',type=int); ap.add_argument('--limit-seeds',type=int); args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True); traces=args.output_dir/'traces'; traces.mkdir(exist_ok=True)
    spec=json.loads(PROMPT_FILE.read_text()); prompts=spec['analysis']; seeds=spec['seeds']
    if args.limit_prompts: prompts=prompts[:args.limit_prompts]
    if args.limit_seeds: seeds=seeds[:args.limit_seeds]
    pipe=StableDiffusionPipeline.from_pretrained(MODEL_ID,torch_dtype=torch.float16,safety_checker=None,
       requires_safety_checker=False,local_files_only=os.environ.get('HF_HUB_OFFLINE','0')=='1').to('cuda')
    pipe.set_progress_bar_config(disable=True); config=dict(pipe.scheduler.config); cases=[(p,s) for p in prompts for s in seeds]
    for i,(prompt,seed) in enumerate(cases,1):
        case=f"{prompt['id']}_seed{seed}"; path=traces/f'{case}.json'
        if path.exists(): print(f"[{i}/{len(cases)}] resume {case}",flush=True); continue
        latent=latent_for(seed); kwargs=dict(prompt=prompt['prompt'],latents=latent.clone(),num_inference_steps=20,
            guidance_scale=7.5,height=512,width=512,output_type='np'); pristine=None
        if i==1:
            pipe.scheduler=scheduler(config)
            with torch.inference_mode(): pristine=pipe(**kwargs).images.copy()
        pipe.scheduler=scheduler(config); pipe.scheduler.set_timesteps(20,device='cuda')
        helper=SymmetricMatchedOracle(pipe,REFRESH,PROBES); helper.enable(); start=time.perf_counter()
        with torch.inference_mode(): result=pipe(**{**kwargs,'latents':latent.clone()})
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start; helper.disable()
        if pristine is not None and not np.array_equal(pristine,result.images): raise RuntimeError('Oracle changed teacher trajectory')
        path.write_text(json.dumps(helper.trace,indent=2))
        manifest=rebuild_manifest(args.output_dir,prompts,seeds); (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(f"[{i}/{len(cases)}] {case}: {elapsed:.1f}s",flush=True)
if __name__=='__main__': main()

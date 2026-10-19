"""Opened-development inference-mask ablation with unchanged GraphPPO weights."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np
import torch
from EVRoutingEnv.state.action_mask import policy_action_mask
from EVRoutingEnv.state.feasibility import joint_action_feasibility
from scripts.evaluation import run_canonical_campaign as runner

ROOT=None
BOUNDS=None
LAST_POLICY=None
ORIGINAL_RUNNER=runner.CanonicalPolicyRunner


def read(path):return json.loads(path.read_text())
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


class EnergyBoundRunner(ORIGINAL_RUNNER):
    def __init__(self,*args,**kwargs):
        global LAST_POLICY
        super().__init__(*args,**kwargs)
        self.key=None
        self.bounds=None
        self.diagnostics={}
        LAST_POLICY=self

    def __call__(self,env,observation,info):
        key=(id(env),env.scenario_seed)
        if key!=self.key:
            self.key=key
            self.bounds=BOUNDS.OptimisticCompletionBounds(env)
            override=read(ROOT/'policy_protocol.json').get('energy_factor_override')
            if override is not None:self.bounds.factor=float(override)
            self.diagnostics[str(env.scenario_seed)]={'decisions':0,'pruned_candidates':0,'decisions_with_pruning':0,'empty_restriction_fallbacks':0}
        base=policy_action_mask(env)
        decisions=joint_action_feasibility(env)
        selected=self.bounds.candidate_mask(decisions,base)
        stats=self.diagnostics[str(env.scenario_seed)]
        stats['decisions']+=1
        count=int((base & ~selected).sum())
        stats['pruned_candidates']+=count
        stats['decisions_with_pruning']+=int(count>0)
        if base.any() and not selected.any():
            selected=base
            stats['empty_restriction_fallbacks']+=1
        with torch.no_grad():
            action,_,_=self.policy.act(torch.as_tensor(observation).unsqueeze(0),torch.as_tensor(selected).unsqueeze(0),deterministic=self.deterministic)
        return int(action.item())


def score(job):
    runner.CanonicalPolicyRunner=EnergyBoundRunner
    name,summary=runner.score_method(job)
    if summary is not None:
        directory=ROOT/'policy_evaluation/validation'/name
        dump(directory/'energy_mask_diagnostics.json',LAST_POLICY.diagnostics)
        manifest=read(directory/'manifest.json')
        manifest['inference_mask_ablation']={'kind':read(ROOT/'policy_protocol.json')['restriction_semantics'],
             'energy_factor_override':read(ROOT/'policy_protocol.json').get('energy_factor_override'),
             'bounds_sha256':digest(ROOT/'bounds.py'),'weights_unchanged':True,'training_mask_changed':False,
             'scope':'inference ablation on opened development cases; not a retrained-policy result'}
        dump(directory/'manifest.json',manifest)
    return name


def report():
    evaluation=Path(read(ROOT/'policy_protocol.json')['baseline_evaluation'])
    pipeline=module(evaluation/'evaluation_pipeline.py','energy_report_pipeline')
    reporter=module(evaluation/'paired_report.py','energy_report')
    methods=read(ROOT/'policy_methods.json')
    rows={};manifests={}
    inputs=[(item['method'],Path(item['directory'])) for item in read(evaluation/'reference_inputs.json')]
    for name in methods:
        inputs.append((name,ROOT/'policy_evaluation/validation'/name))
        original=methods[name]['baseline_method']
        for index in range(4):inputs.append((original,evaluation/'shards'/str(index)/'validation'/original))
    for name,directory in inputs:
        values=pipeline.load_rows(directory/'episode_rows.jsonl')
        pipeline.validate_task_rows(values,name)
        target=rows.setdefault(name,{})
        for row in values:
            if row['scenario_seed'] in target:raise ValueError('duplicate case')
            target[row['scenario_seed']]=row
        manifests[str(directory)]=read(directory/'manifest.json')
    expected=set(range(1000002000,1000002128))
    if any(set(values)!=expected for values in rows.values()):raise ValueError('incomplete case coverage')
    pipeline.check_manifests(manifests)
    anchor=next(iter(rows.values()))
    for name,values in rows.items():
        for seed,row in values.items():
            if reporter.physical_scenario(row)!=reporter.physical_scenario(anchor[seed]):raise ValueError('physical scenario changed')
    references=read(evaluation/'manifest.json')['reference_methods']
    result=reporter.summarize(rows,references)
    baseline_methods=read(evaluation/'methods.json')
    combined={**{settings['baseline_method']:baseline_methods[settings['baseline_method']] for settings in methods.values()},**methods}
    arms=sorted({settings['arm'] for settings in combined.values()})
    result['training_seed_uncertainty']={arm:reporter.hierarchical_intervals(rows,references,[name for name,settings in combined.items() if settings['arm']==arm]) for arm in arms}
    result['evaluation_role']='opened development inference-mask ablation; unchanged trained weights; no final validation or test'
    dump(ROOT/'policy_evaluation/independent_reference_report.json',result)
    metrics={}
    for arm in arms:
        values=[result['methods'][name] for name,settings in combined.items() if settings['arm']==arm]
        metrics[arm]={key:sum(row[key] for row in values)/len(values) for key in ('success_rate','mean_operating_hours_successful','mean_paired_excess_percent')}
        metrics[arm]['training_seeds']=len(values)
    dump(ROOT/'policy_summary.json',{'status':'completed','metrics':metrics,'role':result['evaluation_role'],
          'optimizer_updates':0,'goal_achieved':False,'final_validation_opened':False,'test_opened':False})
    print(json.dumps(metrics,indent=2),flush=True)


def main():
    global ROOT,BOUNDS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    ROOT=args.directory.resolve()
    torch.set_num_threads(1)
    for name,expected in read(ROOT/'policy_input_hashes.json').items():
        if digest(Path(name))!=expected:raise ValueError('inference ablation input changed: '+name)
    BOUNDS=module(ROOT/'bounds.py','energy_completion_bounds')
    methods=read(ROOT/'policy_methods.json')
    config=__import__('yaml').safe_load((ROOT/'config.yaml').read_text())
    jobs=[(name,settings,config,list(range(1000002000,1000002128)),'validation',str(ROOT/'policy_evaluation/validation'),10000,tuple(sys.argv)) for name,settings in methods.items()]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for name in pool.map(score,jobs):print('completed inference ablation',name,flush=True)
    for name,expected in read(ROOT/'policy_input_hashes.json').items():
        if digest(Path(name))!=expected:raise ValueError('inference ablation input changed during evaluation: '+name)
    report()

if __name__=='__main__':main()

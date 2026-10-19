"""Score completed three-seed cohorts on opened development cases and screen compute.

The shared final validation and test are untouched. Training can stop only after
both cohort recipes meet the preregistered dominated-cohort condition.
"""
from __future__ import annotations
import argparse
from datetime import datetime, UTC
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import shlex
import subprocess


def read(path):
    return json.loads(path.read_text())


def dump(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def queue():
    output=subprocess.check_output(['squeue','-u','sorfanoudakis','-h','-r','-o','%i|%j|%T'],text=True)
    return [dict(zip(('id','name','state'),line.split('|',2))) for line in output.splitlines() if line.strip()]


def prepare(protocol,path):
    work=Path(protocol['workspace'])
    base=work/'results/california_graphppo'
    family=base/protocol['cohort']
    source=Path(protocol['source'])
    helper=load(source/'scripts/campaigns/intermediate_california_graphppo.py','mature_screen_prepare')
    promotions=[(int(p.stem.split('_',1)[1]),p) for p in family.glob('promotion_*.json') if p.stem.split('_',1)[1].isdigit()]
    manifest=read(Path(read(max(promotions)[1])['wave'])/'manifest.json')
    runs=[run for run in manifest['runs'] if run['seed'] in range(3)]
    if len(runs)!=6:
        raise ValueError('requires exactly two recipes with seeds zero through two')
    for run in runs:
        history=read(family/'training'/run['name']/'training_history.json')
        if not history.get('timesteps') or history['timesteps'][-1]<12000000:
            raise ValueError('required 12M run incomplete: '+run['name'])
    helper.prepare(path,work,cohorts=[family.name],source_override=source,helper_directory=source/'scripts/campaigns')
    prior=Path(protocol['prior_matured_report'])
    cached=Path(protocol['baseline_evaluation'])
    prior_helper=load(prior/'intermediate.py','mature_screen_prior')
    prior_helper.verify(prior)
    baseline_helper=load(cached/'intermediate.py','mature_screen_baseline')
    baseline_helper.verify(cached)
    extra={name:settings for name,settings in read(cached/'methods.json').items() if settings['arm']==protocol['anchor_recipe']}
    if len(extra)!=3 or {row['training_seed'] for row in extra.values()}!={0,1,2}:
        raise ValueError('anchor requires exactly three cached selected actors')
    shutil.copy2(prior/'score_and_report.py',path/'score_and_report.py')
    dump(path/'additional_methods.json',extra)
    dump(path/'maturity_protocol.json',protocol)
    inputs={Path(name) for name in read(path/'input_hashes.json')}
    inputs.update((path/'score_and_report.py',path/'additional_methods.json',path/'maturity_protocol.json',Path(__file__).resolve()))
    inputs.update(Path(name) for name in read(cached/'input_hashes.json'))
    for name in extra:
        for index in range(4):
            folder=cached/'shards'/str(index)/'validation'/name
            inputs.update((folder/'episode_rows.jsonl',folder/'manifest.json'))
    dump(path/'input_hashes.json',{str(p):digest(p) for p in sorted(inputs)})
    helper.verify(path)


def screening_decision(protocol,path):
    base=Path(protocol['workspace'])/'results/california_graphppo'
    family=base/protocol['cohort']
    summary=read(path/'summary.json')
    if summary['status']!='completed':
        raise ValueError('screen requires completed development evaluation')
    metrics=summary['metrics']
    anchor=metrics[protocol['anchor_recipe']]
    arms={name:row for name,row in metrics.items() if name!=protocol['anchor_recipe']}
    if len(arms)!=2 or any(row['training_seeds']!=3 for row in arms.values()):
        raise ValueError('requires two recipes with complete three-seed evaluation')
    checks={name:{'no_higher_completion':row['success_rate']<=anchor['success_rate'],
                  'excess_at_least_ten_points_more':row['mean_paired_excess_percent']>=anchor['mean_paired_excess_percent']+10}
            for name,row in arms.items()}
    dominated=all(all(check.values()) for check in checks.values())
    decision={'recorded_at':datetime.now(UTC).isoformat(),'cohort':family.name,'checks':checks,'metrics':arms,
              'anchor_recipe':protocol['anchor_recipe'],'anchor_metrics':anchor,'dominated':dominated,
              'evaluation':str(path),'goal_achieved':False,'action':'keep training'}
    if not dominated:
        dump(path/'screening_decision.json',decision)
        return
    master=base/'ca10t3d_v1'
    with (master/'joint_selection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (family/'training_complete.json').exists():
            decision['action']='all five seeds already completed; preserve final-selection eligibility'
            dump(path/'screening_decision.json',decision)
            return
        promotion=read(family/'promotion_12000000.json')
        jobs=queue()
        coordinator=next((job for job in jobs if job['id']==promotion['coordinator_job']),None)
        if coordinator and coordinator['state']!='PENDING':
            decision['action']='coordinator already active; no automatic cancellation'
            dump(path/'screening_decision.json',decision)
            return
        training=[job for job in jobs if job['name'].startswith(family.name)]
        if not training:
            decision['action']='no live family training; let completed coordinator process results'
            dump(path/'screening_decision.json',decision)
            return
        for job in training:
            if not job['id'].startswith(promotion['training_job']+'_'):
                raise ValueError('unexpected training array; inspect before stopping')
            if job['state'] not in ('RUNNING','PENDING'):
                raise ValueError('unexpected live training state; inspect before stopping')
        completed=[name for name in read(path/'methods.json')]
        record={'status':'screened_out','reason':'dominated_on_predeclared_opened_development_screen_after_three_completed_12m_seeds',
                'recorded_at':decision['recorded_at'],'evaluation':str(path),'completed_12m_runs':completed,'metrics':arms,
                'anchor_recipe':protocol['anchor_recipe'],'anchor_metrics':anchor,'screening_rule':protocol['screening_rule'],
                'coordinator_cancelled':coordinator['id'] if coordinator else None,
                'graceful_stop_allocations':[job['id'] for job in training if job['state']=='RUNNING'],
                'pending_allocations_cancelled':[job['id'] for job in training if job['state']=='PENDING'],
                'goal_achieved':False,'five_seed_convergence_claimed':False,'partial_checkpoints_retained':True}
        dump(family/'promotion_stopped.json',record)
        if coordinator:
            subprocess.run(['scancel',coordinator['id']],check=True)
        for job in training:
            args=['scancel','--signal=USR1','--batch',job['id']] if job['state']=='RUNNING' else ['scancel',job['id']]
            subprocess.run(args,check=True)
        decision['action']='screened out; graceful stops requested; verify terminal states before slot reuse'
        dump(family/'screening_execution.json',{'requested':record,'stage':decision['action']})
        command='cd '+shlex.quote(protocol['source'])+'\nexport PYTHONPATH='+shlex.quote(protocol['source'])+'\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n'+shlex.join([protocol['python'],str(Path(__file__).resolve()),'--mode','settle','--protocol',protocol['protocol_path']])
        job=subprocess.check_output(['sbatch','--parsable','--account=tdsei17344','--partition=gpu_mig','--gpus=1',
              '--cpus-per-task=1','--mem=16G','--time=00:15:00',
              '--dependency=afterany:'+promotion['training_job'],'--kill-on-invalid-dep=yes',
              '--job-name=ca10t3d_screen_settle','--output='+str(path/'settle-%j.out'),'--wrap',command],text=True).strip()
        decision['settle_job']=job
        dump(path/'screening_decision.json',decision)



def settle(protocol,path):
    from algo.checkpoint_io import load_trusted_training
    base=Path(protocol['workspace'])/'results/california_graphppo'
    family=base/protocol['cohort']
    if any(job['name'].startswith(family.name) for job in queue()):
        raise RuntimeError('family training still active; do not claim released slots')
    record=read(family/'promotion_stopped.json')
    execution=read(family/'screening_execution.json')
    ids=record['graceful_stop_allocations']+record['pending_allocations_cancelled']
    accounting={}
    for job in ids:
        output=subprocess.check_output(['sacct','-X','-j',job,'-n','-P','-o','JobID,State,ExitCode,Elapsed'],text=True)
        rows=[line.split('|') for line in output.splitlines() if line.split('|')[0]==job]
        if len(rows)!=1 or rows[0][1].split()[0].split('+')[0] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'):
            raise RuntimeError('unverified terminal state for '+job)
        accounting[job]=dict(zip(('job','state','exit_code','elapsed'),rows[0]))
    snapshots={}
    promotion=read(family/'promotion_12000000.json')
    manifest=read(Path(promotion['wave'])/'manifest.json')
    completed=set(record['completed_12m_runs'])
    for run in manifest['runs']:
        if run['name'] in completed:
            continue
        folder=family/'training'/run['name']
        history=folder/'training_history.json'
        if not history.exists():
            snapshots[run['name']]={'trained':False}
            continue
        steps=read(history)['timesteps'][-1]
        state=load_trusted_training(folder/'training.pt')
        if state['total_steps']!=steps:
            raise ValueError('partial full checkpoint/history disagree: '+run['name'])
        snapshots[run['name']]={'trained':True,'steps':steps,'full_state_loads':True,'checkpoint_sha256':digest(folder/'training.pt')}
        del state
    execution.update(stage='terminal states and retained partial full checkpoints verified; allocation slots released',
                     terminal_accounting=accounting,partial_checkpoints=snapshots,verified_at=datetime.now(UTC).isoformat())
    dump(family/'screening_execution.json',execution)
    controller=load(family/'controller.py','mature_screen_final_coordination')
    controller.maybe_start_joint_evaluation(family,read(family/'evaluation_hold.json'))
    dump(path/'settle_status.json',{'status':'passed','training_allocations_released':True,
                                  'joint_evaluation_rechecked':True,'goal_achieved':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--mode',choices=('screen','settle'),default='screen')
    args=parser.parse_args()
    protocol=read(args.protocol)
    path=Path(protocol['output'])
    for filename,expected in read(args.protocol.with_name(args.protocol.stem+'_input_hashes.json')).items():
        if digest(Path(filename))!=expected:
            raise ValueError('screen input changed: '+filename)
    if args.mode=='settle':
        settle(protocol,path)
        return
    if (path/'screening_decision.json').exists():
        raise RuntimeError('screen already completed; inspect its decision')
    if not path.exists():
        prepare(protocol,path)
    helper=load(path/'intermediate.py','mature_screen_verify')
    helper.verify(path)
    if not (path/'summary.json').exists():
        subprocess.run([protocol['python'],str(path/'score_and_report.py')],check=True,cwd=protocol['source'])
    helper.verify(path)
    screening_decision(protocol,path)
    print(json.dumps(read(path/'screening_decision.json'),indent=2),flush=True)


if __name__=='__main__':
    main()

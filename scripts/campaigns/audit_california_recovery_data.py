"""Audit collected recovery labels, paired controls, and train-only provenance."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from algo.canonical_policy import CanonicalPolicyConfig
from algo.canonical_state import flat_observation_slices
from EVRoutingEnv.state.features import GLOBAL_FEATURES


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def episode_records(metadata):
    return metadata.get('episode_records', metadata.get('episodes'))


def audit(root, config_path, prior=None):
    config = CanonicalPolicyConfig(**json.loads(config_path.read_text()))
    global_slice = flat_observation_slices(config.shape)['global_features']
    step_column = GLOBAL_FEATURES.index('episode_steps')
    summary = json.loads((root / 'summary.json').read_text())
    reports = {}
    for name, expected in summary['summaries'].items():
        folder = root / name
        path = folder / 'recovery.npz'
        metadata = json.loads((folder / 'metadata.json').read_text())
        records = episode_records(metadata)
        retained = [row for row in records if row['retained']]
        archive_digest = digest(path)
        if archive_digest != expected['archive_sha256'] or archive_digest != metadata['archive_sha256']:
            raise ValueError(f'{name}: archive hash changed')
        with np.load(path, allow_pickle=False) as data:
            count = len(data['actions'])
            if count != expected['transitions'] or len(retained) != expected['retained_episodes']:
                raise ValueError(f'{name}: counts disagree')
            seeds = data['scenario_seeds']
            allowed = set(metadata['scenario_seeds'])
            if not allowed or min(allowed) < 20000 or max(allowed) >= 1_000_000_000 or set(map(int,seeds)) != {row['scenario_seed'] for row in retained}:
                raise ValueError(f'{name}: invalid train provenance')
            masks = data['masks'].astype(bool)
            if masks.shape != (count, config.max_actions):
                raise ValueError(f'{name}: mask shape changed')
            columns = ('actions','behavior_actions','student_actions','nominal_teacher_actions')
            arrays = {key:data[key] for key in columns}
            rows = np.arange(count)
            for key, values in arrays.items():
                if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= config.max_actions):
                    raise ValueError(f'{name}: invalid {key}')
                if key != 'nominal_teacher_actions' and not masks[rows,values].all():
                    raise ValueError(f'{name}: masked {key}')
            selected = data['teacher_selected']
            recovered = data['recovery_used']
            if selected.shape != (count,) or recovered.shape != (count,):
                raise ValueError(f'{name}: invalid indicators')
            if not np.array_equal(arrays['behavior_actions'],np.where(selected,arrays['actions'],arrays['student_actions'])):
                raise ValueError(f'{name}: labels confused with behavior')
            if not np.array_equal(recovered,~masks[rows,arrays['nominal_teacher_actions']]):
                raise ValueError(f'{name}: recovery provenance disagrees with mask')
            if not np.array_equal(arrays['actions'][~recovered],arrays['nominal_teacher_actions'][~recovered]):
                raise ValueError(f'{name}: unrecorded teacher repair')
            if int(recovered.sum()) != expected['recovery_expert_queries_retained']:
                raise ValueError(f'{name}: recovery count differs')
            observations = data['observations']
            if observations.shape != (count, config.shape.flat_size) or not np.isfinite(observations).all():
                raise ValueError(f'{name}: observation shape/finiteness')
            steps = observations[:,global_slice][:,step_column]
            cursor=0
            for record in retained:
                stop=cursor+record['decisions']
                if not np.all(seeds[cursor:stop] == record['scenario_seed']) or not np.array_equal(steps[cursor:stop],np.arange(stop-cursor)):
                    raise ValueError(f'{name}: episode continuity failure')
                cursor=stop
            if cursor != count:
                raise ValueError(f'{name}: episode lengths do not cover data')
            reports[name]={'transitions':count,'retained_episodes':len(retained),'recovery_labels':int(recovered.sum()),'archive_sha256':archive_digest,'all_labels_and_behavior_obey_masks':True,'contiguous_complete_prefixes':True}
            if prior is not None and name == 'teacher_100':
                prior_metadata=json.loads((prior/name/'metadata.json').read_text())
                prior_records=episode_records(prior_metadata)
                fields=('scenario_seed','teacher_probability','success','retained','decisions','executed_student_disagreements','teacher_label_rejection','operating_hours','termination_reason')
                if [{key:row[key] for key in fields} for row in records] != [{key:row[key] for key in fields} for row in prior_records]:
                    raise ValueError('pure teacher physical episode records changed')
                with np.load(prior/name/'recovery.npz',allow_pickle=False) as old:
                    compared=[]
                    for key in ('observations','actions','masks','behavior_actions','scenario_seeds'):
                        if not np.array_equal(data[key],old[key]):
                            raise ValueError(f'pure teacher control changed: {key}')
                        compared.append(key)
                reports[name]['paired_control_equal_arrays']=compared
    return {'status':'passed','role':'collected-data integrity and provenance audit; zero optimizer updates','data_root':str(root),'config_sha256':digest(config_path),'arms':reports}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--policy-config',type=Path,required=True)
    parser.add_argument('--prior-data',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.data,args.policy_config,args.prior_data)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()

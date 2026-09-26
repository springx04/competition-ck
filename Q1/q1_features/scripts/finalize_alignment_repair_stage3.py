from __future__ import annotations
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from q1_features.alignment_repair import sha256_file
from q1_features.auto_resolution import read_csv
from q1_features.extractors.vision import VISION_COLUMNS, read_openface_csv
from q1_features.stage3_repair import (
    assert_candidate_path_safe, assert_semantic_audio_mask, assert_source_traceability,
    assert_status_sets, classify_resolution_status, verify_expected_sha256,
)
from q1_features.storage import load_npz, read_json, read_jsonl, write_csv, write_json, write_jsonl, write_npz
from q1_features.word_aligned_candidate import _aggregate_interval


def as_float(value: Any) -> float | None:
    try: out=float(value)
    except (TypeError,ValueError): return None
    return out if math.isfinite(out) else None


def verify_frozen(project:Path,reports:Path)->dict:
    frozen=read_json(reports/'stage3_baseline_hashes.json'); checks=[]
    for row in frozen['files']:
        path=project/row['path']; actual=sha256_file(path) if path.is_file() else ''
        unchanged=path.is_file() and verify_expected_sha256(path,row['sha256'])
        checks.append({'path':row['path'],'expected_sha256':row['sha256'],'actual_sha256':actual,'unchanged':unchanged})
    result={'ok':all(x['unchanged'] for x in checks),'file_count':len(checks),'changed':[x for x in checks if not x['unchanged']]}
    write_json(reports/'stage3_baseline_verification.json',result); return result


def enriched_series(formal:Path,item:dict,source_rows:list[dict],cells:list[dict]):
    sample_dir=formal/'samples'/f"{int(item['sample_index']):06d}"
    raw=read_openface_csv(sample_dir/'openface/features.csv')
    raw_by={(int(float(x['frame'])),int(float(x['face_id']))):x for x in raw}
    ordered=sorted(source_rows,key=lambda x:int(x['source_id']))
    if [int(x['source_id']) for x in ordered]!=list(range(len(ordered))): raise ValueError(f"{item['sample_id']}: non-contiguous source ids")
    values=[]; intervals=[]; eligible=[]; result=[]
    for row in ordered:
        key=(int(row['image_index']),int(row['face_id'])); rawrow=raw_by.get(key)
        if rawrow is None: raise KeyError(f"{item['sample_id']}: raw row missing {key}")
        vector=np.asarray([float(rawrow[c]) for c in VISION_COLUMNS],np.float32)
        mid=(float(row['start'])+float(row['end']))/2
        cell=next((x for x in cells if float(x['start'])<=mid<float(x['end'])),None)
        status=cell['segment_status'] if cell else 'MISSING'; selected=cell.get('selected_identity','') if cell else ''
        ok=(float(row['confidence'])>=.8 and np.isfinite(vector).all() and status in {'VERIFIED_ACTIVE','VERIFIED_CONTINUITY'} and selected!='' and int(selected)==int(row['cluster_id']))
        values.append(vector); intervals.append([float(row['start']),float(row['end'])]); eligible.append(int(ok))
        result.append({**row,'stage3_eligible':int(ok),'stage3_source_status':status,'stage3_selected_identity':selected})
    return (np.asarray(values,np.float32) if values else np.zeros((0,22),np.float32),
        np.asarray(intervals,np.float64) if intervals else np.zeros((0,2),np.float64),
        np.asarray(eligible,np.uint8),np.arange(len(ordered),dtype=np.int64),result)


def build_series(formal:Path,manifest:list[dict],s2:Path,reports:Path,part:dict[str,dict],cells_by:dict[str,list[dict]]):
    identity={sid for sid,x in part.items() if int(x['identity_issue'])}; src=defaultdict(list)
    for x in read_jsonl(s2/'raw_vision_candidate_sources.jsonl'): src[x['sample_id']].append(x)
    series={}; lookup={}; output=[]
    for item in manifest:
        sid=item['sample_id']
        if sid not in identity: continue
        values,intervals,eligible,ids,rows=enriched_series(formal,item,src[sid],cells_by[sid]); series[sid]=(values,intervals,eligible,ids)
        for row in rows:
            x={'sample_id':sid,'sample_index':int(item['sample_index']),**row}; output.append(x); lookup[(int(item['sample_index']),int(row['source_id']))]=x
    write_jsonl(reports/'raw_vision_stage3_sources.jsonl',output); return series,lookup


def dominant_status(start:float,end:float,cells:list[dict])->str:
    overlap=Counter()
    for cell in cells:
        amount=min(end,float(cell['end']))-max(start,float(cell['start']))
        if amount>0: overlap[cell['segment_status']]+=amount
    if not overlap: return 'MISSING'
    verified=[x for x in overlap if x.startswith('VERIFIED')]
    return max(verified,key=lambda x:overlap[x]) if verified else max(overlap,key=lambda x:overlap[x])


def build_word_candidate(s2:Path,reports:Path,manifest:list[dict],part:dict[str,dict],cells_by:dict[str,list[dict]],series:dict,lookup:dict,resolution:dict[str,str]):
    base=load_npz(s2/'q1_word_aligned_repair_candidate.npz')
    out={k:v.copy() for k,v in base.items() if not k.startswith('vision_') and k not in {'vision','masks','visual_status','reason'}}
    identity={sid for sid,x in part.items() if int(x['identity_issue'])}; noface={sid for sid,x in part.items() if int(x['no_face'])}
    vals=[]; masks=[]; cover=[]; statuses=[]; reasons=[]; indptr=[0]; srcids=[]; srcsample=[]; overlaps=[]; weights=[]
    for pos,sidval in enumerate(base['sample_id']):
        sid=str(sidval); idx=int(base['sample_index'][pos]); start=float(base['start'][pos]); end=float(base['end'][pos])
        if sid in identity and end>start:
            v,it,el,ids=series[sid]; pooled,mask,cov,used,ov,w=_aggregate_interval(v,it,el,ids,start,end); status=dominant_status(start,end,cells_by[sid])
        elif sid in identity:
            pooled,mask,cov,used,ov,w=np.zeros(22,np.float32),0,0.,[],[],[]; status='UNRESOLVED'
        elif sid in noface:
            pooled,mask,cov,used,ov,w=np.zeros(22,np.float32),0,0.,[],[],[]; status='MISSING'
        else:
            pooled=base['vision'][pos].copy(); mask=int(base['vision_mask'][pos]); cov=float(base['vision_coverage'][pos])
            left,right=int(base['vision_indptr'][pos]),int(base['vision_indptr'][pos+1]); used=base['vision_source_ids'][left:right].tolist(); ov=base['vision_overlap_s'][left:right].tolist(); w=base['vision_weights'][left:right].tolist(); status='FORMAL_BASELINE' if mask else 'MISSING'
        reason=str(base['reason'][pos])
        if not mask and sid in identity and not reason: reason='stage3_visual_not_verified_by_anchor_or_continuity'
        vals.append(pooled); masks.append(mask); cover.append(cov); statuses.append(status); reasons.append(reason)
        srcids.extend(used); srcsample.extend([idx]*len(used)); overlaps.extend(ov); weights.extend(w); indptr.append(len(srcids))
    out.update(vision=np.stack(vals).astype(np.float32),vision_mask=np.asarray(masks,np.uint8),vision_coverage=np.asarray(cover,np.float32),
        visual_status=np.asarray(statuses),reason=np.asarray(reasons),resolution_status=np.asarray([resolution[str(x)] for x in base['sample_id']]),
        vision_indptr=np.asarray(indptr,np.int64),vision_source_ids=np.asarray(srcids,np.int64),vision_source_sample_index=np.asarray(srcsample,np.int64),
        vision_overlap_s=np.asarray(overlaps,np.float64),vision_weights=np.asarray(weights,np.float64))
    out['masks']=np.column_stack([out['text_mask'],out['audio_mask'],out['vision_mask']]).astype(np.uint8)
    path=reports/'q1_word_aligned_stage3_candidate.npz'
    assert_candidate_path_safe(path,[s2/'q1_word_aligned_repair_candidate.npz'])
    write_npz(path,**out)
    details=[]
    for i in range(len(out['word_id'])):
        vl,vr=int(out['vision_indptr'][i]),int(out['vision_indptr'][i+1]); al,ar=int(out['audio_indptr'][i]),int(out['audio_indptr'][i+1])
        details.append({'sample_id':str(out['sample_id'][i]),'sample_index':int(out['sample_index'][i]),'word_id':int(out['word_id'][i]),'word':str(out['raw_word'][i]),
            'start':float(out['start'][i]),'end':float(out['end'][i]),'text_mask':int(out['text_mask'][i]),'audio_mask':int(out['audio_mask'][i]),'vision_mask':int(out['vision_mask'][i]),
            'audio_coverage':float(out['audio_coverage'][i]),'vision_coverage':float(out['vision_coverage'][i]),'semantic_status':str(out['semantic_status'][i]),
            'temporal_status':str(out['temporal_status'][i]),'visual_status':str(out['visual_status'][i]),'resolution_status':str(out['resolution_status'][i]),
            'audio_source_ids':out['audio_source_ids'][al:ar].tolist(),'vision_source_ids':out['vision_source_ids'][vl:vr].tolist(),'reason':str(out['reason'][i])})
    write_jsonl(reports/'alignment_stage3_candidate.jsonl',details)
    errors=[]; n=len(out['word_id'])
    if out['text'].shape!=(n,768) or out['audio'].shape!=(n,25) or out['vision'].shape!=(n,22): errors.append('feature_shape')
    if len(out['vision_indptr'])!=n+1 or int(out['vision_indptr'][-1])!=len(out['vision_source_ids']): errors.append('vision_CSR')
    if len(out['audio_indptr'])!=n+1 or int(out['audio_indptr'][-1])!=len(out['audio_source_ids']): errors.append('audio_CSR')
    if not set(np.unique(out['masks']).tolist())<={0,1}: errors.append('nonbinary_masks')
    for key in ('text','audio','vision','audio_coverage','vision_coverage'):
        if not np.isfinite(out[key]).all(): errors.append('nonfinite_'+key)
    conflict=np.isin(out['semantic_status'],['CONFLICT','UNRESOLVED'])
    try: assert_semantic_audio_mask(out['semantic_status'],out['audio_mask'])
    except ValueError: errors.append('semantic_conflict_has_audio_alignment')
    if np.any(out['start'][conflict]!=0): errors.append('semantic_conflict_has_audio_time')
    if not np.array_equal(out['audio'],base['audio']) or not np.array_equal(out['audio_mask'],base['audio_mask']): errors.append('stage2_audio_changed')
    trace=0
    for idx,sid in zip(out['vision_source_sample_index'],out['vision_source_ids'],strict=True):
        pair=(int(idx),int(sid))
        if pair in lookup:
            trace+=1
            if int(lookup[pair]['stage3_eligible'])!=1: errors.append('ineligible_stage3_source'); break
    trace_pairs=[(int(idx),int(sid)) for idx,sid in zip(out['vision_source_sample_index'],out['vision_source_ids'],strict=True) if (int(idx),int(sid)) in lookup]
    try: assert_source_traceability([x[0] for x in trace_pairs],[x[1] for x in trace_pairs],list(lookup.values()))
    except ValueError: errors.append('stage3_source_traceability')
    val={'ok':not errors,'errors':errors,'path':str(path.resolve()),'sample_count':100,'word_count':n,
        'text_observed':int(out['text_mask'].sum()),'audio_observed':int(out['audio_mask'].sum()),'vision_observed':int(out['vision_mask'].sum()),
        'stage3_traceable_visual_references':trace,'stage2_audio_unchanged':'stage2_audio_changed' not in errors,
        'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in out.items()}}
    write_json(reports/'q1_word_aligned_stage3_candidate.validation.json',val)
    if errors: raise ValueError('word candidate: '+';'.join(errors))
    return val



def coverage_rows(packages:dict[str,dict],groups:dict[str,set[str]],manifest:list[dict])->list[dict]:
    id_to_index={x['sample_id']:int(x['sample_index']) for x in manifest}; rows=[]
    for group,ids in groups.items():
        indices=np.asarray(sorted(id_to_index[sid] for sid in ids),dtype=int)
        for version,pack in packages.items():
            valid=pack['valid_mask'][indices].astype(bool); denom=max(1,int(valid.sum()))
            observed=pack['observed_mask'][indices]; coverage=pack['coverage'][indices]
            rows.append({'group':group,'sample_count':len(indices),'version':version,
                'text_observed_fraction':float((observed[:,:,0]*valid).sum()/denom),
                'audio_observed_fraction':float((observed[:,:,1]*valid).sum()/denom),
                'vision_observed_fraction':float((observed[:,:,2]*valid).sum()/denom),
                'trimodal_observed_fraction':float((np.all(observed==1,axis=2)*valid).sum()/denom),
                'text_coverage_mean':float((coverage[:,:,0]*valid).sum()/denom),
                'audio_coverage_mean':float((coverage[:,:,1]*valid).sum()/denom),
                'vision_coverage_mean':float((coverage[:,:,2]*valid).sum()/denom)})
    return rows


def build_compact_candidate(formal:Path,s2:Path,reports:Path,manifest:list[dict],part:dict[str,dict],series:dict,
                            resolution:dict[str,str],semantic:dict[str,str],temporal:dict[str,str],visual:dict[str,str]):
    old=load_npz(formal/'features/q1_compact50.npz'); base=load_npz(s2/'q1_compact50_repair_candidate.npz')
    out={k:v.copy() for k,v in base.items() if k not in {'candidate_paired_use','visual_status'}}
    out['stage2_candidate_paired_use']=base['candidate_paired_use'].copy()
    identity={sid for sid,x in part.items() if int(x['identity_issue'])}; noface={sid for sid,x in part.items() if int(x['no_face'])}
    stage3_paired=np.zeros(100,np.uint8)
    for item in manifest:
        sid=item['sample_id']; i=int(item['sample_index'])
        if sid in noface:
            out['vision'][i]=0; out['observed_mask'][i,:,2]=0; out['coverage'][i,:,2]=0
        elif sid in identity:
            values,intervals,eligible,source_ids=series[sid]
            for b,(start,end) in enumerate(out['time_intervals'][i]):
                pooled,mask,cov,_,_,_=_aggregate_interval(values,intervals,eligible,source_ids,float(start),float(end))
                out['vision'][i,b]=pooled; out['observed_mask'][i,b,2]=mask; out['coverage'][i,b,2]=cov
        valid=out['valid_mask'][i].astype(bool)
        has_all=bool(np.any(np.all(out['observed_mask'][i,valid]==1,axis=1))) if np.any(valid) else False
        stage3_paired[i]=int(resolution[sid] in {'RESOLVED_VALID','RESOLVED_PARTIAL'} and has_all)
    out['stage3_candidate_paired_use']=stage3_paired
    out['semantic_status']=np.asarray([semantic[x['sample_id']] for x in manifest])
    out['temporal_status']=np.asarray([temporal[x['sample_id']] for x in manifest])
    out['visual_status']=np.asarray([visual[x['sample_id']] for x in manifest])
    out['resolution_status']=np.asarray([resolution[x['sample_id']] for x in manifest])
    path=reports/'q1_compact50_stage3_candidate.npz'
    assert_candidate_path_safe(path,[formal/'features/q1_compact50.npz',s2/'q1_compact50_repair_candidate.npz'])
    write_npz(path,**out)
    errors=[]
    if out['text'].shape!=(100,50,768) or out['audio'].shape!=(100,50,25) or out['vision'].shape!=(100,50,22): errors.append('feature_shape')
    if not set(np.unique(out['observed_mask']).tolist())<={0,1}: errors.append('nonbinary_observed_mask')
    for key in ('text','audio','vision','coverage'):
        if not np.isfinite(out[key]).all(): errors.append('nonfinite_'+key)
    if not np.array_equal(out['text'],base['text']): errors.append('stage2_text_changed')
    if not np.array_equal(out['audio'],base['audio']) or not np.array_equal(out['observed_mask'][:,:,1],base['observed_mask'][:,:,1]): errors.append('stage2_audio_changed')
    unsafe=np.isin(out['resolution_status'],['RESOLVED_CONFLICT','RESOLVED_MISSING','UNRESOLVED'])
    if np.any(out['stage3_candidate_paired_use'][unsafe]!=0): errors.append('unsafe_resolution_released')
    original51={sid for sid,x in part.items() if not int(x['current_paired_use'])}
    mapping_ids={x['sample_id'] for x in read_csv(reports/'mapping_validation_stage3.csv')}
    groups={'all100':{x['sample_id'] for x in manifest},'original_51_isolated':original51,
        'original_37_visual_risk':identity,'original_23_text_audio_risk':{sid for sid,x in part.items() if int(x['ta_issue'])},
        'semantic_conflict':{sid for sid,s in semantic.items() if s=='CONFLICT'},
        'semantic_unresolved':{sid for sid,s in semantic.items() if s=='UNRESOLVED'},'mapping_suspected':mapping_ids}
    comparisons=coverage_rows({'formal':old,'stage2':base,'stage3':out},groups,manifest)
    write_csv(reports/'coverage_comparison_stage3.csv',comparisons,list(comparisons[0]))
    val={'ok':not errors,'errors':errors,'path':str(path.resolve()),'sample_count':100,
        'formal_paired_use_count':int(old['paired_use'].sum()),'stage2_candidate_paired_use_count':int(base['candidate_paired_use'].sum()),
        'stage3_candidate_paired_use_count':int(stage3_paired.sum()),
        'text_unchanged_from_stage2':'stage2_text_changed' not in errors,'audio_unchanged_from_stage2':'stage2_audio_changed' not in errors,
        'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in out.items()}}
    write_json(reports/'q1_compact50_stage3_candidate.validation.json',val)
    if errors: raise ValueError('compact candidate: '+';'.join(errors))
    return val,comparisons,out


def final_status_rows(manifest:list[dict],stage2_status:list[dict],part:dict[str,dict],visual_rows:list[dict],mapping_rows:list[dict],reports:Path):
    s2={x['sample_id']:x for x in stage2_status}; vis3={x['sample_id']:x for x in visual_rows}; maps={x['sample_id']:x for x in mapping_rows}
    visual={}; resolution={}; reasons={}; rows=[]
    translate={'fully_verified':'FULLY_VERIFIED','partially_verified':'PARTIALLY_VERIFIED','unresolved':'UNRESOLVED','missing':'MISSING'}
    for item in manifest:
        sid=item['sample_id']
        # Stage3 visual re-evaluation is scoped to the original identity/no-face risk set.
        # Samples outside that set retain the accepted single-identity baseline instead
        # of being re-gated by a diagnostic TalkNet status.
        v=vis3[sid]['stage3_visual_status'] if sid in vis3 else 'FULLY_VERIFIED'
        m=maps[sid]['mapping_status'] if sid in maps else 'NOT_SUSPECTED'
        status,reason=classify_resolution_status(semantic_status=s2[sid]['semantic_pairing_status'],temporal_status=s2[sid]['temporal_alignment_status'],visual_status=v,mapping_status=m)
        assert_status_sets(m,v,status); visual[sid]=v; resolution[sid]=status; reasons[sid]=reason
        partition=part.get(sid)
        rows.append({'sample_id':sid,'sample_index':int(item['sample_index']),
            'original_paired_use':int(partition['current_paired_use']) if partition else 1,
            'original_group':partition['group'] if partition else 'no_alert',
            'semantic_status':s2[sid]['semantic_pairing_status'],'temporal_status':s2[sid]['temporal_alignment_status'],
            'stage2_visual_status':s2[sid]['visual_candidate_status'],'stage3_visual_status':v,'visual_status':v,'mapping_status':m,
            'resolution_status':status,'resolution_reason':reason,'candidate_only':1,'production_written':0})
    write_csv(reports/'final_resolution_status_stage3.csv',rows,list(rows[0]))
    return rows,visual,resolution,reasons


def enrich_final_rows(reports:Path,rows:list[dict],compact:dict)->list[dict]:
    enriched=[]
    for row in rows:
        i=int(row['sample_index']); valid=compact['valid_mask'][i].astype(bool); denom=max(1,int(valid.sum()))
        obs=compact['observed_mask'][i]
        item={**row,
            'usable_text':float((obs[:,0]*valid).sum()/denom),
            'usable_audio_fraction':float((obs[:,1]*valid).sum()/denom),
            'usable_visual_fraction':float((obs[:,2]*valid).sum()/denom),
            'usable_trimodal_fraction':float((np.all(obs==1,axis=1)*valid).sum()/denom),
            'reason':row['resolution_reason']}
        enriched.append(item)
    write_csv(reports/'final_resolution_status_stage3.csv',enriched,list(enriched[0]))
    return enriched


def write_proposals(reports:Path,rows:list[dict],compact:dict,visual_summary:dict[str,dict],mapping:dict[str,dict],cells:dict[str,list[dict]]):
    output=[]
    for row in rows:
        sid=row['sample_id']; i=int(row['sample_index']); r=row['resolution_status']; v=visual_summary.get(sid,{})
        segments=[{'start':float(x['start']),'end':float(x['end']),'status':x['segment_status'],'identity':x.get('selected_identity','')}
            for x in cells.get(sid,[]) if x['segment_status'] in {'VERIFIED_ACTIVE','VERIFIED_CONTINUITY'}]
        masks={'audio_mask_required':row['semantic_status'] in {'CONFLICT','UNRESOLVED'},
            'vision_mask_nonverified_segments':True,'no_face_full_vision_mask':row['stage3_visual_status']=='MISSING'}
        output.append({**row,
            'recommended_semantic_status':row['semantic_status'],'recommended_temporal_status':row['temporal_status'],
            'recommended_visual_segments':json.dumps(segments,separators=(',',':')),
            'recommended_masks':json.dumps(masks,separators=(',',':')),
            'recommended_resolution_status':r,'writeback_recommended':int(r!='UNRESOLVED'),
            'proposed_paired_use':int(compact['stage3_candidate_paired_use'][i]),
            'direct_talknet_fraction':v.get('direct_talknet_verified_fraction',''),
            'continuity_recovered_fraction':v.get('continuity_recovered_fraction',''),
            'visual_usable_fraction':v.get('total_usable_visual_fraction',''),
            'mapping_candidate_sample_id':mapping.get(sid,{}).get('candidate_sample_id',''),
            'requires_explicit_review':int(r=='UNRESOLVED' or row['mapping_status']=='MAPPING_STRONG'),
            'writeback_approved':0,'writeback_executed':0})
    write_csv(reports/'production_writeback_proposal_stage3.csv',output,list(output[0])); return output


def write_report(reports:Path,mapping:list[dict],visual_rows:list[dict],final_rows:list[dict],comparisons:list[dict],word:dict,compact:dict,frozen:dict):
    mc=Counter(x['mapping_status'] for x in mapping); vc=Counter(x['stage3_visual_status'] for x in visual_rows if int(x['original_identity_risk']))
    rc=Counter(x['resolution_status'] for x in final_rows); unresolved=[x for x in final_rows if x['resolution_status']=='UNRESOLVED']
    vrisk=[x for x in visual_rows if int(x['original_identity_risk'])]
    means={k:float(np.mean([float(x[k]) for x in vrisk])) for k in ('stage2_usable_fraction','direct_talknet_verified_fraction','continuity_recovered_fraction','total_usable_visual_fraction','ambiguous_fraction','missing_fraction','unresolved_fraction')}
    cov={(x['group'],x['version']):x for x in comparisons}; gap=read_csv(reports/'visual_gap_sensitivity_stage3.csv')
    input_state=read_json(reports/'stage3_input_state_summary.json')
    risk_ids={x['sample_id'] for x in visual_rows if int(x['original_identity_risk'])}
    visual_seconds=Counter()
    for x in read_csv(reports/'vision_segment_stage3.csv'):
        if x['sample_id'] in risk_ids:
            visual_seconds[x['segment_status']]+=max(0.0,float(x['end'])-float(x['start']))
    eng=read_json(reports/'stage3_engineering_verification.json',{}) if (reports/'stage3_engineering_verification.json').is_file() else {}
    map_lines=[f"- `{x['sample_id']}` → `{x['candidate_sample_id']}`：{x['mapping_status']}；Whisper token {float(x['whisper_token_original_score']):.3f}→{float(x['whisper_token_candidate_score']):.3f}，CTC token {float(x['ctc_token_original_score']):.3f}→{float(x['ctc_token_candidate_score']):.3f}；Hungarian(W-token/C-token/W-char/C-char)=`{x['hungarian_whisper_candidate']}`/`{x['hungarian_ctc_candidate']}`/`{x['hungarian_whisper_char_candidate']}`/`{x['hungarian_ctc_char_candidate']}`，4指标一致率={float(x['cross_metric_agreement']):.2f}，全局rank {x['global_true_rank']}→{x['global_candidate_rank']}。" for x in mapping]
    unresolved_lines=[f"- `{x['sample_id']}`：{x['resolution_reason']}（semantic={x['semantic_status']}，visual={x['stage3_visual_status']}，mapping={x['mapping_status']}）" for x in unresolved]
    cov_lines=['| 分组 | 版本 | Text | Audio | Vision | Trimodal |','|---|---|---:|---:|---:|---:|']
    for group in ('all100','original_51_isolated','original_37_visual_risk','original_23_text_audio_risk','semantic_conflict','semantic_unresolved','mapping_suspected'):
        for version in ('formal','stage2','stage3'):
            x=cov[(group,version)]; cov_lines.append(f"| {group} | {version} | {100*float(x['text_observed_fraction']):.2f}% | {100*float(x['audio_observed_fraction']):.2f}% | {100*float(x['vision_observed_fraction']):.2f}% | {100*float(x['trimodal_observed_fraction']):.2f}% |")
    lines=['# Q1 Stage 3：剩余异常收敛与视觉连续性修复报告','',
        '> 本阶段仅生成自动候选，不使用人工审核或情感标签，不交换 clip，不修改正式配置、正式 NPZ 或 Stage 2 产物。','',
        '## 1. 结论','',
        f"Stage 3 进一步恢复了视觉静默间隔，但仍是**部分解决**而非生产写回结论。原37条视觉身份风险从 Stage 2 的8条 fully verified提升到{vc['FULLY_VERIFIED']}条；Stage 3 分布为 {dict(vc)}。4条映射疑点中仅1条达到 MAPPING_STRONG，仍只形成显式候选。",'',
        '## 2. A：4条 clip 映射疑点','',*map_lines,'',f"汇总：{dict(mc)}。没有自动修改 manifest，也没有交换音视频。",'',
        '## 3. B：视觉连续性修复','',
        f"Stage2 样本状态={input_state['original_37_visual_risk']}；Stage3 样本状态={dict(vc)}。",
        f"原37条样本平均 Stage2 可用视觉比例={means['stage2_usable_fraction']:.4f}；Stage3直接TalkNet={means['direct_talknet_verified_fraction']:.4f}，ArcFace连续性新增={means['continuity_recovered_fraction']:.4f}，总可用={means['total_usable_visual_fraction']:.4f}。",
        f"绝对时间：TalkNet直接确认={visual_seconds['VERIFIED_ACTIVE']:.3f}s，连续性恢复={visual_seconds['VERIFIED_CONTINUITY']:.3f}s，竞争身份/歧义mask={visual_seconds['AMBIGUOUS']:.3f}s，missing={visual_seconds['MISSING']:.3f}s，unresolved={visual_seconds['UNRESOLVED']:.3f}s。",
        f"歧义={means['ambiguous_fraction']:.4f}，缺失={means['missing_fraction']:.4f}，未决={means['unresolved_fraction']:.4f}。连续性只在同一ArcFace组件、1.0秒有界间隔且无竞争活跃身份时传播。",'',
        '敏感性（最大连续间隔0.5/1.0/1.5秒）：','',*[f"- {x['max_identity_gap_s']}s：fully={x['fully_verified']}，partial={x['partially_verified']}，ambiguous={x['ambiguous']}，unresolved={x['unresolved']}，usable={float(x['usable_fraction']):.4f}" for x in gap],'',
        '## 4. C：最终自动候选状态','',f"100条分布：{dict(rc)}。真正未决 {len(unresolved)} 条：",'',*unresolved_lines,'',
        '## 5. D：Formal / Stage2 / Stage3 覆盖率','',*cov_lines,'',
        f"paired-use计数：formal={compact['formal_paired_use_count']}，Stage2候选={compact['stage2_candidate_paired_use_count']}，Stage3候选={compact['stage3_candidate_paired_use_count']}。候选计数仅用于覆盖分析。",
        f"semantic conflict 组的 Stage3 audio observed={100*float(cov[('semantic_conflict','stage3')]['audio_observed_fraction']):.2f}%，冲突词级audio mask保持为0。",'',
        '## 6. 候选特征产物','',
        f"- 词级候选：`q1_word_aligned_stage3_candidate.npz`，{word['word_count']}词，验证 ok={str(word['ok']).lower()}，视觉源引用{word['stage3_traceable_visual_references']}条。",
        f"- 50-bin候选：`q1_compact50_stage3_candidate.npz`，验证 ok={str(compact['ok']).lower()}。",
        '- 无有效时间或不安全证据保持 mask=0；没有伪造视觉或音频时间。','',
        '## 7. E：工程验收与冻结保护','',
        f"- pytest：{eng.get('pytest','待最终运行')}。",f"- pip check：{eng.get('pip_check','待最终运行')}。",
        f"- 正式 validation：ok={str(eng.get('formal_validation_ok',False)).lower()}，errors={eng.get('formal_validation_errors',[])}。",
        f"- 冻结基线 {frozen['file_count']} 项复核：ok={str(frozen['ok']).lower()}，changed={len(frozen['changed'])}。",
        f"- 词级验证={word['ok']}；50-bin验证={compact['ok']}；可追踪Stage3视觉源引用={word['stage3_traceable_visual_references']}。",
        '- `production_writeback_proposal_stage3.csv` 已生成100行；writeback_executed 全部为0。',
        '- candidate_only=true；production_write=false。','',
        '## 8. 停止边界','',
        'Stage 3 到此停止。所有 proposal 均未获批准、未执行写回；MAPPING_STRONG 仍需显式复核，UNRESOLVED 保持隔离。']
    (reports/'Q1_ALIGNMENT_REPAIR_STAGE3_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default='.'); ap.add_argument('--formal-run',default='runs/q1_full_20260924'); ap.add_argument('--stage2-run',default='runs/q1_alignment_repair_stage2'); ap.add_argument('--output-run',default='runs/q1_alignment_repair_stage3'); args=ap.parse_args()
    project=Path(args.project_root).resolve(); formal=(project/args.formal_run).resolve(); stage2=(project/args.stage2_run).resolve(); reports=(project/args.output_run).resolve()/'reports'; s2=stage2/'reports'; auto=formal/'reports/auto_resolution'
    manifest=read_jsonl(formal/'manifest.jsonl'); stage2_status=read_csv(s2/'alignment_repair_status.csv'); part={x['sample_id']:x for x in read_csv(auto/'problem_partition.csv')}
    visual_rows=read_csv(reports/'vision_summary_stage3.csv'); mapping_rows=read_csv(reports/'mapping_validation_stage3.csv'); cells=defaultdict(list)
    for x in read_csv(reports/'vision_segment_stage3.csv'): cells[x['sample_id']].append(x)
    final_rows,visual,resolution,reasons=final_status_rows(manifest,stage2_status,part,visual_rows,mapping_rows,reports)
    series,lookup=build_series(formal,manifest,s2,reports,part,cells)
    word=build_word_candidate(s2,reports,manifest,part,cells,series,lookup,resolution)
    s2by={x['sample_id']:x for x in stage2_status}; semantic={sid:x['semantic_pairing_status'] for sid,x in s2by.items()}; temporal={sid:x['temporal_alignment_status'] for sid,x in s2by.items()}
    compact,comparisons,pack=build_compact_candidate(formal,s2,reports,manifest,part,series,resolution,semantic,temporal,visual)
    final_rows=enrich_final_rows(reports,final_rows,pack)
    write_proposals(reports,final_rows,pack,{x['sample_id']:x for x in visual_rows},{x['sample_id']:x for x in mapping_rows},cells)
    mfa_ids=[x['sample_id'] for x in stage2_status if x['semantic_pairing_status']=='MATCH_STRONG' and x['temporal_alignment_status']=='PARTIAL' and float(x.get('midpoint_p90') or 0)>0.60]
    write_json(reports/'mfa_stage3_status.json',{'status':'not_run_optional','reason':'Only frozen-rule MATCH_STRONG plus PARTIAL cases with midpoint P90 > 0.60s are recommended; MFA remains an optional third-layer diagnostic','mfa_recommended_sample_ids':mfa_ids,'mfa_recommended_count':len(mfa_ids),'production_alignment_replaced':False})
    frozen=verify_frozen(project,reports); engineering=read_json(reports/'stage3_engineering_verification.json',{})
    acceptance={'ok':bool(word['ok'] and compact['ok'] and frozen['ok'] and engineering.get('ok') and len(final_rows)==100),'sample_count':len(final_rows),
        'resolution_status_counts':dict(Counter(x['resolution_status'] for x in final_rows)),'mapping_status_counts':dict(Counter(x['mapping_status'] for x in mapping_rows)),
        'visual_status_original37':dict(Counter(x['stage3_visual_status'] for x in visual_rows if int(x['original_identity_risk']))),
        'word_candidate_ok':word['ok'],'compact_candidate_ok':compact['ok'],'frozen_inputs_unchanged':frozen['ok'],
        'engineering_verification_ok':engineering.get('ok',False),'pytest':engineering.get('pytest',''),'pip_check':engineering.get('pip_check',''),
        'candidate_only':True,'production_write':False}
    write_json(reports/'stage3_acceptance.json',acceptance); write_report(reports,mapping_rows,visual_rows,final_rows,comparisons,word,compact,frozen)
    print(json.dumps(acceptance,ensure_ascii=False,indent=2)); return 0


if __name__=='__main__': raise SystemExit(main())

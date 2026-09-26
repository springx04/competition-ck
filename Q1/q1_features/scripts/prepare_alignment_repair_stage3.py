from __future__ import annotations
import argparse, hashlib, json, shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from q1_features.alignment_repair import best_assignment, sha256_file, text_pair_metrics
from q1_features.auto_resolution import read_csv
from q1_features.stage3_repair import (
    build_stage3_visual_timeline, classify_mapping_evidence,
    mapping_proposal_row, summarize_stage3_visual,
)
from q1_features.storage import read_json, read_jsonl, write_csv, write_json


def as_float(value: Any) -> float | None:
    try: out=float(value)
    except (TypeError,ValueError): return None
    return out if np.isfinite(out) else None


def freeze_inputs(project: Path, formal: Path, stage2: Path, reports: Path) -> dict[str,Any]:
    paths=[
        stage2/'reports/alignment_repair_status.csv', stage2/'reports/semantic_pairing_status.csv',
        stage2/'reports/official_text_forced_alignment.jsonl', stage2/'reports/vision_sample_summary.csv',
        stage2/'reports/vision_segment_candidates.csv', stage2/'reports/talknet_identity_timeline.csv',
        stage2/'reports/q1_word_aligned_repair_candidate.npz', stage2/'reports/q1_compact50_repair_candidate.npz',
        formal/'features/q1_compact50.npz', project/'configs/alignment_review.csv',
        project/'configs/target_face_segments.csv',
    ]
    rows=[]
    for path in paths:
        if not path.is_file(): raise FileNotFoundError(path)
        rows.append({'path':str(path.relative_to(project)),'bytes':path.stat().st_size,'sha256':sha256_file(path)})
    combined=hashlib.sha256('\n'.join(f"{x['sha256']}  {x['path']}" for x in rows).encode()).hexdigest()
    out={'file_count':len(rows),'combined_sha256':combined,'files':rows,'candidate_only':True,'production_write':False}
    write_json(reports/'stage3_baseline_hashes.json',out); return out


def input_state(stage2_reports: Path, auto: Path, reports: Path) -> tuple[dict,list[dict],dict[str,dict]]:
    status=read_csv(stage2_reports/'alignment_repair_status.csv')
    part={x['sample_id']:x for x in read_csv(auto/'problem_partition.csv')}
    mapping=read_csv(stage2_reports/'within_video_mapping_candidates.csv')
    identity={sid for sid,x in part.items() if int(x['identity_issue'])}
    out={
        'sample_count':len(status),
        'mapping_suspected_count':sum(int(x['mapping_suspected']) for x in mapping),
        'mapping_suspected_ids':sorted(x['sample_id'] for x in mapping if int(x['mapping_suspected'])),
        'semantic':dict(Counter(x['semantic_pairing_status'] for x in status)),
        'temporal':dict(Counter(x['temporal_alignment_status'] for x in status)),
        'original_37_visual_risk':dict(Counter(x['visual_candidate_status'] for x in status if x['sample_id'] in identity)),
        'no_face_count':sum(int(x['no_face']) for x in part.values()),
    }
    accept=read_json(stage2_reports/'stage2_acceptance.json'); prep=read_json(stage2_reports/'stage2_prepare_summary.json')
    trans=read_csv(stage2_reports/'visual_status_transition.csv')
    expected_visual=dict(Counter(x['stage2_talknet_segment_status'] for x in trans))
    errors=[]
    if not accept.get('ok') or out['sample_count']!=accept.get('sample_count'): errors.append('acceptance_or_sample_count')
    if out['semantic']!=accept.get('semantic_status_all100'): errors.append('semantic_counts')
    if out['temporal']!=accept.get('temporal_status_all100'): errors.append('temporal_counts')
    if out['mapping_suspected_count']!=prep.get('mapping_suspected'): errors.append('mapping_count')
    if out['original_37_visual_risk']!=expected_visual: errors.append('visual37_counts')
    out.update(stage2_consistency_ok=not errors,inconsistencies=errors)
    write_json(reports/'stage3_input_state_summary.json',out)
    if errors: raise RuntimeError('Stage3 input inconsistent: '+','.join(errors))
    return out,status,part


def metric_matrix(items:list[dict], tr:Mapping[str,dict], asr:str, metric:str)->np.ndarray:
    field='whisperx_transcript' if asr=='whisper' else 'ctc_diagnostic_transcript'
    m=np.zeros((len(items),len(items)),float)
    for i,a in enumerate(items):
        for j,b in enumerate(items):
            m[i,j]=text_pair_metrics(tr[a['sample_id']]['official_transcript'],tr[b['sample_id']][field])[metric]
    return m


def validate_mappings(manifest:list[dict], stage2_reports:Path, reports:Path, suspect:set[str]):
    trs=read_csv(stage2_reports/'asr_transcripts_all.csv'); tr={x['sample_id']:x for x in trs}
    by_video=defaultdict(list)
    for x in manifest: by_video[str(x['video_id'])].append(x)
    specs={'whisper_token':('whisper','token_f1'),'whisper_char':('whisper','character_similarity'),
           'ctc_token':('ctc','token_f1'),'ctc_char':('ctc','character_similarity')}
    all_ids=[x['sample_id'] for x in manifest]; rows=[]; proposals=[]; statuses={}
    for video,items in sorted(by_video.items()):
        relevant=[x for x in items if x['sample_id'] in suspect]
        if not relevant: continue
        items.sort(key=lambda x:int(x['sample_index'])); ids=[x['sample_id'] for x in items]
        matrices={name:metric_matrix(items,tr,*spec) for name,spec in specs.items()}
        assign={}
        for name,m in matrices.items():
            rr,cc=best_assignment(m); assign[name]={ids[int(i)]:ids[int(j)] for i,j in zip(rr,cc)}
        for item in relevant:
            sid=item['sample_id']; i=ids.index(sid); assigned={k:v[sid] for k,v in assign.items()}
            votes=Counter(v for v in assigned.values() if v!=sid)
            if votes:
                top=max(votes.values()); candidates=[x for x,n in votes.items() if n==top]
                def support(c):
                    j=ids.index(c); return sum(float(m[i,j]-m[i,i]) for m in matrices.values())
                candidate=max(sorted(candidates),key=lambda c:(support(c),c))
            else: candidate=sid
            j=ids.index(candidate); pairs={k:(float(m[i,i]),float(m[i,j])) for k,m in matrices.items()}
            scored=[]; official=tr[sid]['official_transcript']
            for aid in all_ids: scored.append((text_pair_metrics(official,tr[aid]['whisperx_transcript'])['score'],aid))
            scored.sort(key=lambda x:(-x[0],x[1])); ranks={aid:n for n,(_,aid) in enumerate(scored,1)}
            status,reason=classify_mapping_evidence(sample_id=sid,candidate_id=candidate,assignments=assigned,
                score_pairs=pairs,global_true_rank=ranks[sid],global_candidate_rank=ranks[candidate])
            cross=text_pair_metrics(tr[sid]['whisperx_transcript'],tr[sid]['ctc_diagnostic_transcript'])
            row={'sample_id':sid,'video_id':video,'official_clip_id':item['clip_id'],'candidate_sample_id':candidate,
                'candidate_clip_id':next(x['clip_id'] for x in items if x['sample_id']==candidate),
                'whisper_token_original_score':pairs['whisper_token'][0],'whisper_token_candidate_score':pairs['whisper_token'][1],
                'ctc_token_original_score':pairs['ctc_token'][0],'ctc_token_candidate_score':pairs['ctc_token'][1],
                'character_original_score':float(np.mean([pairs['whisper_char'][0],pairs['ctc_char'][0]])),
                'character_candidate_score':float(np.mean([pairs['whisper_char'][1],pairs['ctc_char'][1]])),
                'whisper_char_original_score':pairs['whisper_char'][0],'whisper_char_candidate_score':pairs['whisper_char'][1],
                'ctc_char_original_score':pairs['ctc_char'][0],'ctc_char_candidate_score':pairs['ctc_char'][1],
                'whisper_ctc_token_f1':cross['token_f1'],'hungarian_whisper_candidate':assigned['whisper_token'],
                'hungarian_ctc_candidate':assigned['ctc_token'],'hungarian_whisper_char_candidate':assigned['whisper_char'],
                'hungarian_ctc_char_candidate':assigned['ctc_char'],
                'cross_metric_agreement':sum(v==candidate for v in assigned.values())/4,
                'global_true_rank':ranks[sid],'global_candidate_rank':ranks[candidate],
                'mapping_status':status,'reason':reason}
            rows.append(row); proposals.append(mapping_proposal_row(sid,candidate,status,reason)); statuses[sid]=status
    write_csv(reports/'mapping_validation_stage3.csv',rows,list(rows[0]))
    write_csv(reports/'proposed_mapping_correction.csv',proposals,list(proposals[0]))
    return rows,proposals,statuses


def primary_clusters(auto:Path):
    out={}
    for row in read_csv(auto/'arcface_identity_clusters.csv'):
        if abs(float(row['threshold'])-.60)>1e-9: continue
        clusters=json.loads(row['clusters_json'])
        out[row['sample_id']]={'stable':bool(int(row['stable_partition_0_50_0_60_0_70'])),
            'episode_to_cluster':{ep:i for i,eps in enumerate(clusters) for ep in eps},
            'cluster_count':int(row['identity_cluster_count'])}
    return out


def visual_inputs(stage2_reports:Path,auto:Path,risk:set[str]):
    timeline=defaultdict(list); sources=defaultdict(list); av=defaultdict(dict)
    for x in read_csv(stage2_reports/'talknet_identity_timeline.csv'):
        if x['sample_id'] in risk: timeline[x['sample_id']].append(x)
    for x in read_jsonl(stage2_reports/'raw_vision_candidate_sources.jsonl'): sources[x['sample_id']].append(x)
    for x in read_csv(auto/'av_sync_scores.csv'):
        v=as_float(x.get('empirical_percentile'))
        if v is not None: av[x['sample_id']][int(x['cluster_id'])]=v
    return timeline,sources,av


def run_visual(manifest:list[dict],stage2_reports:Path,auto:Path,reports:Path,part:dict[str,dict]):
    identity={sid for sid,x in part.items() if int(x['identity_issue'])}; noface={sid for sid,x in part.items() if int(x['no_face'])}; risk=identity|noface
    timelines,sources,av=visual_inputs(stage2_reports,auto,risk); clusters=primary_clusters(auto)
    s2={x['sample_id']:x for x in read_csv(stage2_reports/'vision_sample_summary.csv')}; byid={x['sample_id']:x for x in manifest}
    cells=[]; anchors=[]; summaries=[]; cells_by_id={}
    for sid in sorted(risk,key=lambda x:int(byid[x]['sample_index'])):
        rows=timelines[sid]; duration=max(float(x['end']) for x in rows) if rows else 0.0
        stable=clusters.get(sid,{'stable':True})['stable'] if sid not in noface else True
        tl,an=build_stage3_visual_timeline(sample_id=sid,duration=duration,stage2_timeline=rows,
            source_rows=sources.get(sid,[]),partition_stable=stable,av_percentiles=av.get(sid,{}),max_identity_gap=1.0)
        cells.extend(tl); anchors.extend(an); cells_by_id[sid]=tl
        summaries.append({'sample_id':sid,'original_identity_risk':int(sid in identity),'original_no_face':int(sid in noface),
            'arcface_partition_stable':int(stable),**summarize_stage3_visual(tl,float(s2[sid]['usable_fraction']))})
    write_csv(reports/'vision_segment_stage3.csv',cells,list(cells[0])); write_csv(reports/'talknet_speaker_anchors_stage3.csv',anchors,list(anchors[0])); write_csv(reports/'vision_summary_stage3.csv',summaries,list(summaries[0]))
    sensitivity=[]
    for gap in (.5,1.0,1.5):
        selected=[]
        for sid in sorted(identity):
            rows=timelines[sid]; duration=max(float(x['end']) for x in rows)
            tl,_=build_stage3_visual_timeline(sample_id=sid,duration=duration,stage2_timeline=rows,
                source_rows=sources[sid],partition_stable=clusters[sid]['stable'],av_percentiles=av.get(sid,{}),max_identity_gap=gap)
            selected.append(summarize_stage3_visual(tl,float(s2[sid]['usable_fraction'])))
        sensitivity.append({'max_identity_gap_s':gap,'is_primary':int(gap==1.0),'sample_count':len(selected),
            'fully_verified':sum(x['stage3_visual_status']=='FULLY_VERIFIED' for x in selected),
            'partially_verified':sum(x['stage3_visual_status']=='PARTIALLY_VERIFIED' for x in selected),
            'ambiguous':sum(x['stage3_visual_status']=='AMBIGUOUS' for x in selected),
            'missing':sum(x['stage3_visual_status']=='MISSING' for x in selected),
            'unresolved':sum(x['stage3_visual_status']=='UNRESOLVED' for x in selected),
            'direct_fraction':float(np.mean([x['direct_talknet_verified_fraction'] for x in selected])),
            'continuity_fraction':float(np.mean([x['continuity_recovered_fraction'] for x in selected])),
            'usable_fraction':float(np.mean([x['total_usable_visual_fraction'] for x in selected]))})
    write_csv(reports/'visual_gap_sensitivity_stage3.csv',sensitivity,list(sensitivity[0]))
    return cells,anchors,summaries,cells_by_id,clusters


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default='.'); ap.add_argument('--formal-run',default='runs/q1_full_20260924'); ap.add_argument('--stage2-run',default='runs/q1_alignment_repair_stage2'); ap.add_argument('--output-run',default='runs/q1_alignment_repair_stage3'); args=ap.parse_args()
    project=Path(args.project_root).resolve(); formal=(project/args.formal_run).resolve(); stage2=(project/args.stage2_run).resolve(); out=(project/args.output_run).resolve(); reports=out/'reports'; reports.mkdir(parents=True,exist_ok=True)
    shutil.copy2(project/'configs/q1_alignment_repair_stage3.yaml',out/'config.snapshot.yaml')
    s2=stage2/'reports'; auto=formal/'reports/auto_resolution'; frozen=freeze_inputs(project,formal,stage2,reports)
    state,status,part=input_state(s2,auto,reports); manifest=read_jsonl(formal/'manifest.jsonl')
    mapping,proposals,mapping_status=validate_mappings(manifest,s2,reports,set(state['mapping_suspected_ids']))
    cells,anchors,visual,cells_by_id,clusters=run_visual(manifest,s2,auto,reports,part)
    summary={'input_state':state,'mapping_status':dict(Counter(x['mapping_status'] for x in mapping)),
        'visual_status_original37':dict(Counter(x['stage3_visual_status'] for x in visual if int(x['original_identity_risk']))),
        'stage3_baseline_combined_sha256':frozen['combined_sha256'],'candidate_only':True,'production_write':False}
    write_json(reports/'stage3_prepare_summary.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())

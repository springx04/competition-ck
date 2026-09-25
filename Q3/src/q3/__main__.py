import argparse
import json
from pathlib import Path
import numpy as np

from .config import load_config
from .data import load_samples
from .predictor import Predictor
from .interventions import apply_delete
from .attribution import view_result, shapley, modality_summary
from .results import write_json, append_jsonl, view_id
from .report import write_special_outputs
from .export import export_delivery
from .grouping import Unit, block3_units, text_word_units
from .selection import select_units, budget_count
from .statistics import summarize_faithfulness

def _run_dir(config, split):
    return Path(config["output"]["valid_run"] if split == "valid" else config["output"]["special_run"])

def prepare(config, split):
    samples = load_samples(config, split); run = _run_dir(config, split); run.mkdir(parents=True, exist_ok=True)
    manifest = run / "manifest.jsonl"
    if manifest.exists(): manifest.unlink()
    for sample in samples:
        append_jsonl(manifest, {"schema_version": 1, "split": split, "sample_index": sample.sample_index, "sample_id": sample.sample_id, "source_id": sample.source_id, "video_id": sample.video_id, "raw_text": sample.raw_text, "input_status": "failed" if sample.error else "ready", "reason": sample.error})
    write_json(run / "run.json", {"split": split, "expected_samples": len(samples), "status": "prepared"})
    print(f"prepared {split}: {len(samples)} samples -> {run}")

def attribute(config, split):
    samples = load_samples(config, split); run = _run_dir(config, split); run.mkdir(parents=True, exist_ok=True); bundle = config["project"]["bundle_dir"]; predictor = Predictor(bundle, config["runtime"].get("device", "cpu")); success = 0
    for sample in samples:
        sample_dir = run / "samples" / f"{sample.sample_index:04d}"; sample_dir.mkdir(parents=True, exist_ok=True)
        if sample.error:
            write_json(sample_dir / "status.json", {"status": "failed", "reason": sample.error}); continue
        try:
            raw = sample.raw; original = predictor.predict({k: v.unsqueeze(0) for k, v in raw.items()}); u0 = original.U[0]; c_star = int(np.argmax(original.probs[0])); cache = {}
            atoms = tuple(3 * t + m for t in range(50) for m in range(3) if u0[t, m])
            def evaluate(delete_atoms):
                key = tuple(sorted(delete_atoms))
                if key in cache: return cache[key]
                mask = np.zeros((50, 3), dtype=bool)
                for j in key: mask[j // 3, j % 3] = True
                altered = apply_delete(raw, mask, u0)
                result = view_result(key, original, predictor.predict({k: v.unsqueeze(0) for k, v in altered.items()}), c_star)
                cache[key] = result; return result
            views = {"v000000": evaluate(())}
            for j in atoms: views[view_id((j,))] = evaluate((j,))
            local_d = np.full((50, 3), np.nan); local_dc = np.full((50, 3), np.nan); local_ds = np.full((50, 3), np.nan); local_tv = np.full((50, 3), np.nan)
            for j in atoms:
                value = cache[(j,)]; local_d[j // 3, j % 3] = value.D; local_dc[j // 3, j % 3] = value.delta_class; local_ds[j // 3, j % 3] = value.delta_score_raw; local_tv[j // 3, j % 3] = value.TV
            joint_d = np.full(50, np.nan); joint_dc = np.full(50, np.nan); joint_ds = np.full(50, np.nan); joint_tv = np.full(50, np.nan); joint_observed = u0.any(axis=1)
            for t in np.where(joint_observed)[0]:
                key = tuple(3 * int(t) + m for m in range(3) if u0[t, m]); value = evaluate(key); joint_d[t] = value.D; joint_dc[t] = value.delta_class; joint_ds[t] = value.delta_score_raw; joint_tv[t] = value.TV
            modality = {}
            for m, name in enumerate("TAV"):
                key = tuple(3 * t + m for t in range(50) if u0[t, m]); modality[name] = evaluate(key) if key else None
            subset_probs = []; subset_scores = []
            for subset in range(8):
                keep = {m for m in range(3) if subset & (1 << m)}; delete = tuple(j for j in atoms if j % 3 not in keep); v = evaluate(delete); subset_probs.append(v.probs); subset_scores.append(v.score)
            phi, residual = shapley(np.asarray(subset_probs), np.asarray(subset_scores), c_star)
            write_json(sample_dir / "original.json", {"logits": original.logits[0].tolist(), "probs": original.probs[0].tolist(), "score": float(original.score[0]), "c_star": c_star, "U0": u0.tolist(), "J0": original.J[0].tolist(), "n_observed": u0.sum(0).tolist(), "prediction_status": "complete"})
            write_json(sample_dir / "modality.json", {"D": [v.D if v else 0.0 for v in modality.values()], "n_observed": u0.sum(0).tolist()})
            write_json(sample_dir / "shapley.json", {"phi_class": phi[:, 0].tolist(), "phi_score_scaled": phi[:, 1].tolist(), "phi_score_raw": (6 * phi[:, 1]).tolist(), "residual": residual.tolist(), "subsets": [{"bitmask": i, "probs": subset_probs[i].tolist(), "score": subset_scores[i]} for i in range(8)]})
            np.savez_compressed(sample_dir / "local.npz", observed=u0, D=local_d, delta_class=local_dc, delta_score_raw=local_ds, TV=local_tv, joint_observed=joint_observed, joint_D=joint_d, joint_delta_class=joint_dc, joint_delta_score_raw=joint_ds, joint_TV=joint_tv)
            for value in cache.values(): append_jsonl(sample_dir / "views.jsonl", {"view_id": view_id(value.delete_j), "delete_j": list(value.delete_j), "logits": value.logits.tolist(), "probs": value.probs.tolist(), "score": value.score, "predicted_class": value.predicted_class, "delta_class": value.delta_class, "delta_score_raw": value.delta_score_raw, "delta_score": value.delta_score, "D": value.D, "TV": value.TV, "class_changed": value.class_changed})
            write_json(sample_dir / "status.json", {"status": "complete", "unique_views": len(cache)}); success += 1
        except Exception as exc:
            write_json(sample_dir / "status.json", {"status": "failed", "reason": str(exc)})
    print(f"attribute {split}: {success}/{len(samples)} completed")

def _write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.unlink()
    for row in rows: append_jsonl(path, row)

def select_stage(config, split):
    run = _run_dir(config, split); completed = 0; samples=load_samples(config,split)
    try:
        from transformers import BertTokenizerFast
        tokenizer=BertTokenizerFast.from_pretrained(Path(config["project"]["bundle_dir"])/"assets"/"text_encoder",local_files_only=True)
    except Exception: tokenizer=None
    for sample_dir in sorted((run / "samples").glob("[0-9][0-9][0-9][0-9]")):
        local_path = sample_dir / "local.npz"
        if not local_path.exists(): continue
        with np.load(local_path) as local:
            observed = local["observed"]; d = local["D"]; delta_class=local["delta_class"]; joint_d = local["joint_D"]
        rows = []
        for modality, name in enumerate("TAV"):
            base = [Unit(f"{name}.t{t}", (3*t+modality,), t, t+1, 1, float(d[t,modality]), name) for t in range(50) if observed[t,modality]]
            for pct in (10,20,30):
                budget = budget_count(len(base), pct); key = select_units(base,budget,3,"max",True); low = select_units(base,budget,3,"min",True)
                for role, selection in (("key",key),("low",low)):
                    rows.append({"selection_id":f"{role}.{name}.p{pct}.point","experiment_kind":role,"path":name,"granularity":"point","budget_pct":pct,"target_budget":budget,"actual_cost":selection.actual_cost,"counts_by_modality":selection.counts_by_modality,"intervals":selection.intervals,"delete_j":list(selection.atoms),"proxy_value":selection.proxy_value,"status":selection.status,"reason":selection.reason})
            for role,sign in (("support",1.0),("suppress",-1.0)):
                direction=[Unit(unit.unit_id,unit.atoms,unit.lo,unit.hi,unit.cost,max(sign*float(delta_class[unit.lo,modality]),0.0),name) for unit in base]
                budget=budget_count(len(base),20);selected=select_units(direction,budget,3,"max",False)
                retained_proxy=selected.proxy_value if selected.proxy_value>0 else 0.0
                rows.append({"selection_id":f"{role}.{name}.p20.point","experiment_kind":role,"path":name,"granularity":"point","budget_pct":20,"target_budget":budget,"actual_cost":selected.actual_cost,"counts_by_modality":selected.counts_by_modality,"intervals":selected.intervals,"delete_j":list(selected.atoms),"proxy_value":retained_proxy,"status":selected.status if retained_proxy>0 else "not_applicable","reason":selected.reason if retained_proxy>0 else "no_directional_proxy"})
            groups = []
            for unit in block3_units(observed,name):
                proxy = sum(float(d[a//3,a%3]) for a in unit.atoms); groups.append(Unit(unit.unit_id,unit.atoms,unit.lo,unit.hi,unit.cost,proxy,name))
            budget = budget_count(len(base),20); selected = select_units(groups,budget,3,"max",False)
            rows.append({"selection_id":f"key.{name}.p20.block3","experiment_kind":"key","path":name,"granularity":"block3","budget_pct":20,"target_budget":budget,"actual_cost":selected.actual_cost,"counts_by_modality":selected.counts_by_modality,"intervals":selected.intervals,"delete_j":list(selected.atoms),"proxy_value":selected.proxy_value,"status":selected.status,"reason":selected.reason})
            if name=="T" and tokenizer is not None:
                sample=samples[int(sample_dir.name)]
                if sample.raw_text:
                    encoded=tokenizer(sample.raw_text,add_special_tokens=True,truncation=True,max_length=50,padding="max_length",return_offsets_mapping=True)
                    if list(encoded["input_ids"])==sample.raw["input_ids"].tolist():
                        word_units=[]
                        for unit in text_word_units(sample.raw_text,encoded["offset_mapping"],observed):
                            proxy=sum(float(d[a//3,a%3]) for a in unit.atoms);word_units.append(Unit(unit.unit_id,unit.atoms,unit.lo,unit.hi,unit.cost,proxy,"T"))
                        selected=select_units(word_units,budget,3,"max",False)
                        rows.append({"selection_id":"key.T.p20.word","experiment_kind":"key","path":"T","granularity":"word","budget_pct":20,"target_budget":budget,"actual_cost":selected.actual_cost,"counts_by_modality":selected.counts_by_modality,"intervals":selected.intervals,"delete_j":list(selected.atoms),"proxy_value":selected.proxy_value,"status":selected.status,"reason":selected.reason})
        joint_units=[]
        for t in range(50):
            atoms=tuple(3*t+m for m in range(3) if observed[t,m])
            if atoms: joint_units.append(Unit(f"J.t{t}",atoms,t,t+1,len(atoms),float(joint_d[t]),"J"))
        for pct in (10,20,30):
            budget=budget_count(int(observed.sum()),pct); selected=select_units(joint_units,budget,3,"max",False)
            rows.append({"selection_id":f"key.J.p{pct}.joint","experiment_kind":"key","path":"J","granularity":"joint","budget_pct":pct,"target_budget":budget,"actual_cost":selected.actual_cost,"counts_by_modality":selected.counts_by_modality,"intervals":selected.intervals,"delete_j":list(selected.atoms),"proxy_value":selected.proxy_value,"status":selected.status,"reason":selected.reason})
        _write_jsonl(sample_dir / "selections.jsonl",rows); completed += 1
    print(f"select {split}: {completed} samples")

def controls_stage(config, split):
    run=_run_dir(config,split); seed=int(config["runtime"]["seed"]); completed=0
    for sample_dir in sorted((run/"samples").glob("[0-9][0-9][0-9][0-9]")):
        path=sample_dir/"selections.jsonl"
        if not path.exists(): continue
        selections=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]; rows=[]
        original=json.loads((sample_dir/"original.json").read_text(encoding="utf-8")); observed=np.asarray(original["U0"],bool)
        for index,sel in enumerate(selections):
            if sel["experiment_kind"]!="key" or sel["actual_cost"]<=0: continue
            rng=np.random.default_rng(np.random.SeedSequence([seed,int(sample_dir.name),index])); target=tuple(sel["delete_j"]); seen=set(); lengths=sorted([b-a for a,b in sel["intervals"]]); path_name=sel["path"]
            for control_index in range(50):
                candidate=None
                for _ in range(5000):
                    intervals=sorted((start,start+length) for length in lengths for start in [int(rng.integers(0,51-length))])
                    if any(intervals[i][1]>intervals[i+1][0] for i in range(len(intervals)-1)): continue
                    modalities=range(3) if path_name=="J" else ("TAV".index(path_name),)
                    picked=tuple(sorted(3*t+m for a,b in intervals for t in range(a,b) for m in modalities if observed[t,m]))
                    counts=[sum(atom%3==m for atom in picked) for m in range(3)]
                    if counts!=sel["counts_by_modality"]: continue
                    if picked not in seen and picked!=target: candidate=picked;break
                if candidate is None: break
                seen.add(candidate);rows.append({"selection_id":sel["selection_id"],"control_index":control_index,"delete_j":list(candidate),"counts_by_modality":[sum(a%3==m for a in candidate) for m in range(3)],"interval_lengths":lengths,"status":"complete"})
        _write_jsonl(sample_dir/"controls.jsonl",rows);completed+=1
    print(f"controls {split}: {completed} samples")

def _batch_views(sample, original, predictor, sets, batch_size=128):
    u0=original.U[0]; c_star=int(np.argmax(original.probs[0])); unique=[];seen=set()
    for atoms in sets:
        key=tuple(sorted(atoms))
        if key not in seen:seen.add(key);unique.append(key)
    results={}
    for start in range(0,len(unique),batch_size):
        keys=unique[start:start+batch_size]; batches=[]
        for key in keys:
            mask=np.zeros((50,3),bool)
            for atom in key:mask[atom//3,atom%3]=True
            batches.append(apply_delete(sample.raw,mask,u0))
        raw={name:__import__('torch').stack([batch[name] for batch in batches]) for name in sample.raw}
        prediction=predictor.predict(raw)
        for i,key in enumerate(keys):
            one=type(original)(prediction.logits[i:i+1],prediction.probs[i:i+1],prediction.score[i:i+1],prediction.U[i:i+1],prediction.J[i:i+1]);results[key]=view_result(key,original,one,c_star)
    return results

def evaluate_stage(config,split):
    samples=load_samples(config,split);run=_run_dir(config,split);predictor=Predictor(config["project"]["bundle_dir"],config["runtime"].get("device","cpu"));completed=0
    for sample in samples:
        sample_dir=run/"samples"/f"{sample.sample_index:04d}";selection_path=sample_dir/"selections.jsonl";control_path=sample_dir/"controls.jsonl"
        if sample.error or not selection_path.exists():continue
        selections=[json.loads(line) for line in selection_path.read_text(encoding="utf-8").splitlines() if line];controls=[json.loads(line) for line in control_path.read_text(encoding="utf-8").splitlines() if line] if control_path.exists() else []
        original=predictor.predict({k:v.unsqueeze(0) for k,v in sample.raw.items()});all_observed=tuple(3*t+m for t in range(50) for m in range(3) if original.U[0,t,m]);sets=[]
        for row in selections:
            if row["status"]=="complete":
                key=tuple(row["delete_j"]);sets.append(key);sets.append(tuple(sorted(set(all_observed)-set(key))))
                modalities=range(3) if row["path"]=="J" else ("TAV".index(row["path"]),);domain={atom for atom in all_observed if atom%3 in modalities};sets.append(tuple(sorted(domain-set(key))))
        for row in controls:sets.append(tuple(row["delete_j"]))
        results=_batch_views(sample,original,predictor,sets,int(config["runtime"].get("view_batch_size",128)));rows=[]
        for sel in selections:
            if sel["status"]!="complete":continue
            key=tuple(sel["delete_j"]);v=results[key];keep_key=tuple(sorted(set(all_observed)-set(key)));kv=results[keep_key]
            rows.append({"experiment_id":sel["selection_id"]+".delete","selection_id":sel["selection_id"],"operation":"delete","control_role":sel["experiment_kind"],"delete_j":list(key),"D":v.D,"TV":v.TV,"delta_class":v.delta_class,"delta_score_raw":v.delta_score_raw,"cost":len(key),"status":"complete"})
            rows.append({"experiment_id":sel["selection_id"]+".global_keep","selection_id":sel["selection_id"],"operation":"global_keep","control_role":sel["experiment_kind"],"delete_j":list(keep_key),"S":1.0-kv.D,"D":kv.D,"TV":kv.TV,"cost":len(key),"status":"complete"})
            modalities=range(3) if sel["path"]=="J" else ("TAV".index(sel["path"]),);domain={atom for atom in all_observed if atom%3 in modalities};conditional_key=tuple(sorted(domain-set(key)));cv=results[conditional_key]
            rows.append({"experiment_id":sel["selection_id"]+".conditional_keep","selection_id":sel["selection_id"],"operation":"conditional_keep","control_role":sel["experiment_kind"],"delete_j":list(conditional_key),"S":1.0-cv.D,"D":cv.D,"TV":cv.TV,"cost":len(key),"status":"complete"})
        for control in controls:
            key=tuple(control["delete_j"]);v=results[key];rows.append({"experiment_id":control["selection_id"]+f".random.{control['control_index']:02d}","selection_id":control["selection_id"],"operation":"delete","control_role":"random","control_index":control["control_index"],"delete_j":list(key),"D":v.D,"TV":v.TV,"delta_class":v.delta_class,"delta_score_raw":v.delta_score_raw,"cost":len(key),"status":"complete"})
        _write_jsonl(sample_dir/"experiments.jsonl",rows);completed+=1
        if completed%25==0:print(f"evaluate {split}: {completed}/{len(samples)}",flush=True)
    print(f"evaluate {split}: {completed}/{len(samples)} completed")

def summarize_stage(config,split):
    run=_run_dir(config,split);grouped={};samples=load_samples(config,split);predicted=[];true=[];scores=[];score_true=[]
    for sample in samples:
        original=run/"samples"/f"{sample.sample_index:04d}"/"original.json"
        if original.exists():
            row=json.loads(original.read_text(encoding="utf-8"));predicted.append(int(row["c_star"]));scores.append(float(row["score"]));true.append(sample.label);score_true.append(sample.score_label)
    for path in (run/"samples").glob("[0-9][0-9][0-9][0-9]/experiments.jsonl"):
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line];targets={row["selection_id"]:row for row in rows if row["operation"]=="delete" and row["control_role"]=="key"};randoms={}
        for row in rows:
            if row["control_role"]=="random":randoms.setdefault(row["selection_id"],[]).append(row["D"])
        for key,target in targets.items():
            if randoms.get(key):grouped.setdefault(key,[]).append({"target":target["D"],"random":float(np.mean(randoms[key]))})
    summary={}
    for key,rows in grouped.items():
        item=summarize_faithfulness(rows);values=np.asarray([r["target"]-r["random"] for r in rows]);rng=np.random.default_rng(np.random.SeedSequence([int(config["runtime"]["seed"]),sum(map(ord,key))]));means=np.asarray([rng.choice(values,len(values),replace=True).mean() for _ in range(int(config["method"].get("bootstrap_repeats",1000)))]) if len(values) else np.asarray([])
        if len(means):item["ci_low"],item["ci_high"]=[float(x) for x in np.quantile(means,[.025,.975])]
        summary[key]=item
    write_json(run/"summaries"/"faithfulness.json",summary)
    if split=="valid" and all(value is not None for value in true):
        from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error
        write_json(run/"summaries"/"prediction_metrics.json",{"n":len(true),"accuracy":float(accuracy_score(true,predicted)),"macro_f1":float(f1_score(true,predicted,average="macro")),"mae":float(mean_absolute_error(score_true,scores))})
    print(f"summarize {split}: {len(summary)} experiment groups")

def validate(config, stage):
    report = Path(config["project"]["root"]) / "reports" / f"validation_{stage}.json"; report.parent.mkdir(parents=True, exist_ok=True); result = {"stage": stage, "status": "complete", "issues": []}
    if stage == "inputs":
        for split, expected in (("valid", 728), ("special", 20)):
            try: result[split] = len(load_samples(config, split));
            except Exception as exc: result["issues"].append(f"{split}: {exc}")
    if stage == "final":
        for split, expected in (("valid", 728), ("special", 20)):
            run = _run_dir(config, split); status_files = list((run / "samples").glob("[0-9][0-9][0-9][0-9]/status.json")); experiments = list((run / "samples").glob("[0-9][0-9][0-9][0-9]/experiments.jsonl"))
            complete = sum(json.loads(path.read_text(encoding="utf-8")).get("status") == "complete" for path in status_files)
            result[split] = {"expected": expected, "attribute_complete": complete, "experiment_files": len(experiments)}
            if complete != expected or len(experiments) != expected: result["issues"].append(f"{split}: incomplete artifacts")
        csv_path = Path(config["project"]["root"]) / "outputs" / "Q3_attachment4_predictions.csv"
        if not csv_path.exists() or len(csv_path.read_text(encoding="utf-8-sig").splitlines()) != 21: result["issues"].append("special CSV is not 20 rows")
        result["material_status"] = "partial"
        result["material_reason"] = "word-time review and official A/V provenance remain unresolved"
    result["status"] = "complete" if not result["issues"] else "failed"
    write_json(report, result); print(json.dumps(result, ensure_ascii=False)); return 0 if not result["issues"] else 1

def check_predictor(config):
    predictor=Predictor(config["project"]["bundle_dir"],config["runtime"].get("device","cpu"));probes=load_samples(config,"valid")[:8]+[s for s in load_samples(config,"special") if not s.error]
    max_prob=max_score=max_logits=0.0;class_consistent=True
    for sample in probes:
        single=predictor.predict({k:v.unsqueeze(0) for k,v in sample.raw.items()});batch=predictor.predict({k:__import__('torch').stack([v,v]) for k,v in sample.raw.items()})
        max_prob=max(max_prob,float(np.max(np.abs(single.probs[0]-batch.probs[0]))));max_score=max(max_score,float(abs(single.score[0]-batch.score[0])));max_logits=max(max_logits,float(np.max(np.abs(single.logits[0]-batch.logits[0]))));class_consistent &= int(np.argmax(single.probs[0]))==int(np.argmax(batch.probs[0]))==int(np.argmax(batch.probs[1]))
    result={"probe_count":len(probes),"max_logits_difference":max_logits,"max_probability_difference":max_prob,"max_score_difference":max_score,"class_consistent":bool(class_consistent),"eps_prob":max(1e-7,10*max_prob),"eps_score":max(1e-6,10*max_score)};result["eps_D"]=.5*result["eps_prob"]+.5*result["eps_score"]/6
    write_json(Path(config["project"]["root"])/"reports"/"numeric_tolerance.json",result);print(json.dumps(result));return 0 if class_consistent and max_prob<=1e-4 and max_score<=1e-3 else 1

def report(config, split):
    if split != "special":
        run=_run_dir(config,split);summary_path=run/"summaries"/"faithfulness.json";summary=json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {};report_path=Path(config["project"]["root"])/"reports"/"Q3_实验报告.md"
        lines=["# Q3 实验报告","","## 执行状态","",f"valid 样本数：{len(list((run/'samples').glob('[0-9][0-9][0-9][0-9]/experiments.jsonl')))}。预测器为 Q2 最终 late_attn_tune seed1111；归因固定跟踪原预测类别。","","## 忠实性汇总","","| 实验 | n | mean G | median G | 正增益比例 |","|---|---:|---:|---:|---:|"]
        for key,value in sorted(summary.items()):lines.append(f"| {key} | {value.get('n_eligible',0)} | {value.get('mean_G',float('nan')):.6f} | {value.get('median_G',float('nan')):.6f} | {value.get('positive_fraction',float('nan')):.3f} |")
        lines += ["","## 限制","","valid 曾参与 Q2 选模，因此这里只作为开发集方法评价。附件4无标签，不报告准确率。CTC词时人工审核与官方A/V行来源仍单独标记为 partial/unresolved。"]
        report_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
        print(f"report valid: {report_path}")
        return 0
    run = _run_dir(config, split); rows = []
    for index in range(20):
        sample_dir = run / "samples" / f"{index:04d}"; sample_id = f"{index + 1:02d}"
        original_path = sample_dir / "original.json"; modality_path = sample_dir / "modality.json"
        if not original_path.exists():
            rows.append({"sample_id": sample_id, "status": "failed", "mapping_status": "unresolved"}); continue
        original = json.loads(original_path.read_text(encoding="utf-8")); modality = json.loads(modality_path.read_text(encoding="utf-8"))
        probs = original["probs"]; d_sum = float(sum(modality["D"])); weights = None if d_sum <= 1e-6 else [d / d_sum for d in modality["D"]]
        selections=[json.loads(line) for line in (sample_dir/"selections.jsonl").read_text(encoding="utf-8").splitlines()] if (sample_dir/"selections.jsonl").exists() else [];experiments=[json.loads(line) for line in (sample_dir/"experiments.jsonl").read_text(encoding="utf-8").splitlines()] if (sample_dir/"experiments.jsonl").exists() else [];shapley=json.loads((sample_dir/"shapley.json").read_text(encoding="utf-8")) if (sample_dir/"shapley.json").exists() else {};ctc_path=run/"alignment"/sample_id/"words.jsonl";ctc=[json.loads(line) for line in ctc_path.read_text(encoding="utf-8").splitlines()] if ctc_path.exists() else []
        evidence={name:next((row for row in selections if row["selection_id"]==f"key.{name}.p20.point"),None) for name in "TAV"};primary=[] if not weights else ["TAV"[i] for i,value in enumerate(modality["D"]) if value>=max(modality["D"])-1e-6 and original["n_observed"][i]>0]
        rows.append({"sample_id": sample_id, "source_id": sample_id, "status": "complete", "predicted_class_id": original["c_star"], "predicted_class_name": config["method"]["class_names"][original["c_star"]], "sentiment_score": original["score"], "p_negative": probs[0], "p_neutral": probs[1], "p_positive": probs[2], "n_T": original["n_observed"][0], "n_A": original["n_observed"][1], "n_V": original["n_observed"][2], "D_T": modality["D"][0], "D_A": modality["D"][1], "D_V": modality["D"][2], "D_sum": d_sum, "w_T": weights[0] if weights else None, "w_A": weights[1] if weights else None, "w_V": weights[2] if weights else None, "primary_modalities": json.dumps(primary), "head_top_disagreement": None, "evidence_T":json.dumps(evidence["T"]["intervals"] if evidence["T"] else []),"evidence_A":json.dumps(evidence["A"]["intervals"] if evidence["A"] else []),"evidence_V":json.dumps(evidence["V"]["intervals"] if evidence["V"] else []), "mapping_status": "partial_ctc_pending_av_unresolved","shapley":shapley,"selections":selections,"experiments":experiments,"ctc_words":ctc})
    write_special_outputs(rows, Path(config["project"]["root"]) / "outputs")
    print(f"wrote special report rows: {len(rows)}")
    return 0

def main():
    parser = argparse.ArgumentParser(prog="q3"); parser.add_argument("--config", required=True); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "attribute", "select", "controls", "evaluate", "summarize", "map", "report"):
        p = sub.add_parser(name); p.add_argument("--split", choices=("valid", "special"), required=True); p.add_argument("--resume", action="store_true")
    sub.add_parser("check-predictor"); p = sub.add_parser("export"); p = sub.add_parser("validate"); p.add_argument("--stage", choices=("inputs", "final"), required=True)
    args = parser.parse_args(); config = load_config(args.config)
    if args.command == "prepare": return prepare(config, args.split)
    if args.command == "attribute": return attribute(config, args.split)
    if args.command == "select": return select_stage(config, args.split)
    if args.command == "controls": return controls_stage(config, args.split)
    if args.command == "evaluate": return evaluate_stage(config, args.split)
    if args.command == "summarize": return summarize_stage(config, args.split)
    if args.command == "validate": return validate(config, args.stage)
    if args.command == "report": return report(config, args.split)
    if args.command == "export": print(export_delivery(config)); return 0
    if args.command == "check-predictor": return check_predictor(config)
    print(f"{args.command}: stage is represented by the run records; no-op without completed attribute inputs")
    return 0

if __name__ == "__main__": raise SystemExit(main())

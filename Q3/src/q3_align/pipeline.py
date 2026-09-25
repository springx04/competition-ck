from pathlib import Path
import csv,json,pickle,subprocess,sys
import numpy as np
import yaml

def load_config(path):
    path=Path(path).resolve();cfg=yaml.safe_load(path.read_text(encoding="utf-8"));root=path.parent.parent.resolve()
    for section in ("project","alignment","output"):
        for key,value in list(cfg.get(section,{}).items()):
            if isinstance(value,str) and not Path(value).is_absolute() and (section=="output" or key.endswith(("_dir","_root","_file","_path")) or key in {"root","bundle_dir","valid_dir","review_file","av_provenance_file"}):cfg[section][key]=str((root/value).resolve())
    return cfg
def _special_dir(cfg):return Path(cfg["project"]["data_root"])/"附件4-可解释专项视频样本与特征文件"/"附件4-可解释专项视频样本与特征文件"/"对齐版本"
def _record(path):
    with path.open("rb") as handle:
        try:return pickle.load(handle)
        except ModuleNotFoundError as exc:
            if not (exc.name or "").startswith("numpy._core"):raise
            handle.seek(0)
            class Compat(pickle.Unpickler):
                def find_class(self,module,name):return super().find_class(module.replace("numpy._core","numpy.core",1),name)
            return Compat(handle).load()
def media(cfg,resume=False):
    source=_special_dir(cfg);run=Path(cfg["output"]["special_run"]);results=[]
    for number in range(1,21):
        sid=f"{number:02d}";dest=run/"media"/sid;dest.mkdir(parents=True,exist_ok=True);wav=dest/"audio_16k.wav";video=source/"videos"/f"{sid}.mp4"
        try:
            if not resume or not wav.exists():subprocess.run(["ffmpeg","-y","-v","error","-i",str(video),"-vn","-ac","1","-ar","16000",str(wav)],check=True,timeout=int(cfg["alignment"].get("external_command_timeout_s",120)))
            probe=subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","json",str(video)],text=True);meta={"sample_id":sid,"status":"complete","video_path":str(video),"wav_path":str(wav),"ffprobe":json.loads(probe)}
        except Exception as exc:meta={"sample_id":sid,"status":"failed","reason":str(exc),"video_path":str(video)}
        (dest/"media.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8");results.append(meta)
    print(f"media: {sum(r['status']=='complete' for r in results)}/20 complete")
def align(cfg,resume=False):
    project=Path(cfg["project"]["root"]);q1_root=project.parent/"Q1"/"q1_features";sys.path.insert(0,str(q1_root/"src"))
    from q1_features.alignment import find_ctc_files,load_ctc_model,ctc_forward,align_token_arrays_detailed
    from q1_features.text_map import build_words,prepare_ctc_rows
    import sentencepiece as spm
    import soundfile as sf
    ctc_dir=q1_root/"models"/"ctc";device=cfg["alignment"].get("device","cuda:0");model,_=load_ctc_model(ctc_dir,device);_,_,bpe=find_ctc_files(ctc_dir);sp=spm.SentencePieceProcessor(model_file=str(bpe));run=Path(cfg["output"]["special_run"]);source=_special_dir(cfg);completed=0
    for number in range(1,21):
        sid=f"{number:02d}";dest=run/"alignment"/sid;dest.mkdir(parents=True,exist_ok=True);out=dest/"words.jsonl"
        if resume and out.exists():completed+=1;continue
        try:
            record=_record(source/f"{sid}.pkl");raw_text=str(record.get("raw_text",record.get("text",record.get("id",""))));words=build_words(raw_text);prep=prepare_ctc_rows(words,sp,model.token_list);wav,sr=sf.read(run/"media"/sid/"audio_16k.wav",dtype="float32",always_2d=False)
            if sr!=16000:raise ValueError(f"unexpected sample rate {sr}")
            if wav.ndim==2:wav=wav.mean(axis=1)
            lpz,index_duration=ctc_forward(model,np.asarray(wav,np.float32),device);detail=align_token_arrays_detailed(lpz,[np.asarray(x,np.int64) for x in prep.token_arrays],prep.ctc_texts,model.token_list,index_duration=index_duration,min_window_size=int(cfg["alignment"]["min_window_size"]),max_window_size=int(cfg["alignment"]["max_window_size"]),score_min_mean_over_L=int(cfg["alignment"]["score_min_mean_over_L"]));segments=detail["segments"]
            with out.open("w",encoding="utf-8") as handle:
                for row_id,(start,end,score) in enumerate(segments):
                    word_id=prep.ctc_row_to_word_id[row_id];word=words[word_id];status="accepted" if np.isfinite([start,end,score]).all() and start<end and score>=float(cfg["alignment"]["min_log_score"]) else "rejected";handle.write(json.dumps({"sample_id":sid,"row_id":row_id,"word_id":word_id,"raw_word":word.raw_word,"start_common":start,"end_common":end,"log_score":score,"algorithm_status":status,"review_status":"pending"},ensure_ascii=False)+"\n")
            (dest/"diagnostics.json").write_text(json.dumps({"status":"complete","ctc_frames":int(lpz.shape[0]),"index_duration":index_duration,"rows":len(segments)},indent=2),encoding="utf-8");completed+=1
        except Exception as exc:(dest/"diagnostics.json").write_text(json.dumps({"status":"failed","reason":str(exc)},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"align: {completed}/20 complete")
def make_review(cfg):
    run=Path(cfg["output"]["special_run"]);review=Path(cfg["alignment"]["review_file"]);review.parent.mkdir(parents=True,exist_ok=True);existing={}
    if review.exists():
        with review.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):existing[(row["sample_id"],row["word_id"])]=row
    fields=["sample_id","word_id","raw_word","auto_start_common","auto_end_common","auto_log_score","algorithm_status","review_status","reviewer","reviewed_at","review_start_common","review_end_common","notes"]
    with review.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for path in sorted((run/"alignment").glob("*/words.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                item=json.loads(line);key=(item["sample_id"],str(item["word_id"]));row={"sample_id":item["sample_id"],"word_id":item["word_id"],"raw_word":item["raw_word"],"auto_start_common":item["start_common"],"auto_end_common":item["end_common"],"auto_log_score":item["log_score"],"algorithm_status":item["algorithm_status"],"review_status":"pending","reviewer":"","reviewed_at":"","review_start_common":"","review_end_common":"","notes":""};row.update(existing.get(key,{}));writer.writerow(row)
    print(f"review: {review}")

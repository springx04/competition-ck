"""Build candidate playback and inspect media clocks; never approve word timings."""
import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
import subprocess
import wave
from PIL import Image, ImageDraw
from q3.config import load_config
from q3.results import write_json
from q3.workflow import run_dir


def probe(path, frames=False):
    entries = 'frame=best_effort_timestamp_time,pkt_duration_time' if frames else 'stream=index,codec_type,start_time,duration,time_base:format=start_time,duration'
    args = ['ffprobe', '-v', 'error'] + (['-select_streams', 'v:0'] if frames else [])
    return json.loads(subprocess.check_output(args+['-show_entries', entries, '-of', 'json', str(path)], text=True))


def build(config):
    root = Path(config['project']['root']); run = run_dir(config, 'special')
    assets = root/'outputs_v2/assets'; assets.mkdir(parents=True, exist_ok=True)
    source = Path(config['project']['data_root'])/'附件4-可解释专项视频样本与特征文件'/'附件4-可解释专项视频样本与特征文件'/'对齐版本'/'videos'
    records = []; queue = []; total = Counter(); thumbnails = []
    for number in range(1, 21):
        sid = f'{number:02d}'; video = source/f'{sid}.mp4'
        mapping = json.loads((run/'samples'/f'{number-1:04d}'/'mapping.json').read_text(encoding='utf-8'))
        words = mapping['ctc']['word_times']; selected = set(mapping['evidence'].get('key.T.p20.point', {}).get('word_ids', []))
        metadata = probe(video); audio = next(s for s in metadata['streams'] if s['codec_type']=='audio')
        start = float(audio.get('start_time', 0)); duration = float(metadata['format']['duration'])
        frame_rows = probe(video, True)['frames']
        frames = [(i, float(r['best_effort_timestamp_time'])) for i,r in enumerate(frame_rows) if 'best_effort_timestamp_time' in r]
        monotonic = all(a[1] <= b[1] for a,b in zip(frames, frames[1:]))
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(video),'-vn','-ac','1','-ar','16000','-b:a','48k',str(assets/f'{sid}.mp3')], check=True)
        wav = root/'runs/special_main/media'/sid/'audio_16k.wav'; decoded_duration = None
        if wav.exists():
            with wave.open(str(wav), 'rb') as f: decoded_duration = f.getnframes()/f.getframerate()
        bounds = []; previous = 0.; rows = []
        for word in words:
            lo = word.get('candidate_start_common', word.get('start_common')); hi = word.get('candidate_end_common', word.get('end_common'))
            audio_duration = decoded_duration if decoded_duration is not None else float(audio['duration'])
            valid = lo is not None and hi is not None and 0 <= lo < hi <= audio_duration+.02 and lo >= previous-.02
            if hi is not None: previous = hi
            if not valid: bounds.append(word['word_id'])
            rejected = word.get('algorithm_status')=='rejected'; is_main = int(word['word_id']) in selected
            priority = 1 if rejected else 2 if is_main else 3
            queue.append(dict(sample_id=sid, word_id=word['word_id'], raw_word=word['raw_word'], start_candidate=lo, end_candidate=hi, algorithm_status=word['algorithm_status'], review_status=word.get('review_status','pending'), main_text_evidence=is_main, priority=priority, bounds_order_check='pass' if valid else 'fail', audio_offset_s=start, review_page=f'assets/{sid}_review.html'))
            total['words'] += 1; total['rejected_words'] += rejected; total['main_evidence_words'] += is_main; total['human_verified'] += word.get('verified',False)
            label = html.escape(word['raw_word']); status = html.escape(word.get('material_status','pending'))
            tag = '算法拒绝 · 优先核验' if rejected else '主文本证据 · 待听审' if is_main else '待听审'
            clip_lo = max(0,float(lo or 0)-.2); clip_hi = min(audio_duration,float(hi or 0)+.2)
            row = f'<tr><td>{word["word_id"]}</td><td>{label}</td><td>{float(lo):.3f}–{float(hi):.3f}s</td><td>{tag} / {status}</td><td><button onclick="play({clip_lo:.6f},{clip_hi:.6f})">播放候选±0.2s</button></td></tr>'
            rows.append((priority,int(word['word_id']),row))
        representatives = []
        for j,fraction in enumerate((.2,.5,.8)):
            requested_pts = frames[0][1] + (frames[-1][1]-frames[0][1])*fraction
            index,pts = min(frames,key=lambda f:abs(f[1]-requested_pts))
            name = f'{sid}_context{j+1}.jpg'; path = assets/name
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(video),'-vf',f'select=eq(n\,{index}),scale=320:-2','-frames:v','1',str(path)],check=True)
            representatives.append(dict(frame_index=index,pts_s=pts,asset=name,role='source_context_only_not_model_visual_evidence'))
            thumbnails.append((sid,pts,path))
        records.append(dict(sample_id=sid,source_video=str(video),ffprobe=metadata,video_pts_count=len(frames),video_pts_monotonic=monotonic,audio_offset_s=start,decoded_audio_duration=decoded_duration,candidate_clock_compatible=abs(start)<=.02,word_bounds_order_failures=bounds,representative_frames=representatives,auditory_review='not_performed',official_av_provenance='unresolved'))
        cards = ''.join(f'<figure><img src="{r["asset"]}"><figcaption>PTS {r["pts_s"]:.3f}s；素材上下文</figcaption></figure>' for r in representatives)
        page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{sid} 词时候选复核</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:30px auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}figure{{display:inline-block;margin:8px}}img{{max-width:300px}}button{{padding:7px}}.note{{background:#fff3cd;padding:16px}}</style>
<h1>{sid} 词时候选复核</h1><p><a href="../cases/{sid}.html">返回解释卡</a></p>
<p class="note">下列时间来自 CTC；结构检查与实际听审分别记录。本页不自动批准任何词时。参考帧按真实解码 PTS 选取，只展示视频上下文；不证明它们对应模型官方视觉特征行。</p>
<p>原文：{html.escape(mapping['raw_text'])}</p><audio id="audio" controls preload="metadata" src="{sid}.mp3"></audio>
<p>音频起点 {start:.6f}s；视频时长 {duration:.3f}s；边界/顺序异常 {len(bounds)}。复核后填写 data/review/word_times.csv 的审核列，再运行 map 和 report。</p>{cards}
<h2>按复核优先级排列</h2><table><tr><th>word_id</th><th>原词</th><th>候选区间</th><th>状态</th><th>播放</th></tr>{''.join(r[2] for r in sorted(rows))}</table>
<script>const audio=document.getElementById('audio');let end=null;function play(lo,hi){{end=hi;audio.currentTime=lo;audio.play();}}audio.addEventListener('timeupdate',()=>{{if(end!==null && audio.currentTime>=end){{audio.pause();end=null;}}}});</script></html>'''
        (assets/f'{sid}_review.html').write_text(page,encoding='utf-8')
    queue.sort(key=lambda r:(r['priority'],r['sample_id'],r['word_id']))
    destination = root/'reports/iteration_v2';destination.mkdir(parents=True,exist_ok=True)
    with (destination/'word_review_queue.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(queue[0]));writer.writeheader();writer.writerows(queue)
    write_json(destination/'media_audit.json',dict(counts=dict(total),samples=records,auditory_review='not_performed'))
    for part in range(2):
        canvas=Image.new('RGB',(990,2000),(248,248,248));draw=ImageDraw.Draw(canvas)
        for i,(sid,pts,path) in enumerate(thumbnails[part*30:(part+1)*30]):
            x=(i%3)*330;y=(i//3)*200
            with Image.open(path) as picture:
                picture.thumbnail((320,168));canvas.paste(picture,(x,y+24))
            draw.text((x+8,y+5),f'{sid}  PTS {pts:.3f}s',fill=(0,0,0))
        canvas.save(destination/f'context_contact_{part+1}.jpg',quality=88)
    print(json.dumps(dict(samples=len(records),counts=dict(total),bounds_failures=sum(len(r['word_bounds_order_failures']) for r in records),clock_compatible=sum(r['candidate_clock_compatible'] for r in records))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);build(load_config(p.parse_args().config))

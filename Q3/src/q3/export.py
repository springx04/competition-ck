"""Package the current iteration without duplicating data, models or old results."""
from pathlib import Path
import json
import zipfile
from .results import write_json


def export_delivery(config):
    root = Path(config['project']['root']); delivery = root/'delivery'
    delivery.mkdir(parents=True, exist_ok=True)
    validation = json.loads((root/'reports/validation_v2_final.json').read_text(encoding='utf-8'))
    if validation['status'] != 'complete':
        raise ValueError('Final computational validation has not passed')
    archive = delivery/'q3_submission_v2.zip'
    files = [root/name for name in ('README.md','THIRD_PARTY.md','pyproject.toml','requirements-main.txt','requirements-align.txt',
             'configs/q3_v2.yaml','reports/numeric_tolerance_v2.json','reports/validation_v2_final.json','data/review/word_times.csv')]
    files.extend((root/'configs').glob('q3_v2_*.yaml'))
    for folder, suffixes in [('src',{'.py'}),('scripts',{'.py','.sh'}),('tests',{'.py'}),('outputs_v2',None),('reports/iteration_v2',None)]:
        files.extend(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and (suffixes is None or p.suffix in suffixes))
    for split in ('valid','special'):
        files.extend(p for p in (Path(config['output'][f'{split}_run'])/'summaries').rglob('*') if p.is_file())
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
        for path in sorted(set(files)):
            if path.is_file(): output.write(path,path.relative_to(root))
    write_json(delivery/'size_report_v2.json',dict(q3_zip_bytes=archive.stat().st_size,archive=archive.name,
               full_competition_total_bytes=None,reason='Q1 final combined delivery not supplied; Q2 model is a shared external dependency',
               includes_current_cases=20,includes_source_videos=False,includes_ctc_weights=False,
               material_status=validation['material_status']))
    return archive

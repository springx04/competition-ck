import argparse,platform,subprocess,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    lines=[f'python={sys.executable}',platform.platform()]
    try: lines.append(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,stderr=subprocess.STDOUT))
    except Exception as exc: lines.append(f'pip_freeze_error={exc}')
    out.write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()

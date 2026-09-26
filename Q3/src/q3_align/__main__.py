import argparse
from .pipeline import load_config,media,align,make_review
def main():
    parser=argparse.ArgumentParser(prog="q3_align");parser.add_argument("--config",required=True);parser.add_argument("command",choices=("media","align","make-review"));parser.add_argument("--resume",action="store_true");args=parser.parse_args();cfg=load_config(args.config)
    if args.command=="media":return media(cfg,args.resume)
    if args.command=="align":return align(cfg,args.resume)
    return make_review(cfg)
if __name__=="__main__":raise SystemExit(main())

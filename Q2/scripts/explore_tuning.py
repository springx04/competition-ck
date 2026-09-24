"""Named validation-only experiments; each directory retains its full config."""
import argparse
import json
from pathlib import Path
from q2.config import load_config, validate_config
from q2.trainer import train_one
from q2.model.network import TUNED_VARIANTS

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--variant", choices=TUNED_VARIANTS, required=True)
parser.add_argument("--text-lr", type=float, default=3e-5)
parser.add_argument("--regression-weight", type=float, default=.25)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--seed", type=int, default=1111)
parser.add_argument("--student-lr", type=float, default=3e-4)
parser.add_argument("--class-weight-power", type=float, default=None)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--clip-z", type=float, default=None)
parser.add_argument("--unfrozen-layers", type=int, default=None)
parser.add_argument("--video-sampling-power", type=float, default=None)
parser.add_argument("--cls-context", action="store_true",
                    help="add the optional BERT CLS summary to observed token rows")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
config = load_config(root / args.config)
config["project"]["output_root"] = str(root / "experiments" / args.name)
config["text"].update(frozen=False, learning_rate=args.text_lr)
config["text"]["cls_context"] = args.cls_context
config["loss"]["regression_weight"] = args.regression_weight
if args.class_weight_power is not None:
    config["loss"]["class_weight_power"] = args.class_weight_power
if args.clip_z is not None:
    config["data"]["clip_z"] = args.clip_z
if args.unfrozen_layers is not None:
    config["text"]["unfrozen_layers"] = args.unfrozen_layers
if args.video_sampling_power is not None:
    config["data"]["video_sampling_power"] = args.video_sampling_power
config["train"].update(epochs=args.epochs, warmup_epochs=2,
                       seeds=[args.seed], eval_epochs=list(range(5, args.epochs + 1, 5)),
                       learning_rate=args.student_lr, min_learning_rate=args.student_lr / 10)
config["evaluation"]["deployed_seed"] = args.seed
config["data"]["num_workers"] = 0
validate_config(config)
print(json.dumps({"name": args.name, "variant": args.variant, "config": config}), flush=True)
print(json.dumps(train_one(root, config, args.variant, args.seed, resume=args.resume)), flush=True)

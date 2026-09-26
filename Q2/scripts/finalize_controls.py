"""Read out fixed historical controls for the final paper (no fitting on test)."""
import json
from pathlib import Path

import torch

from q2.__main__ import evaluate_best
from q2.evaluate import selection_key


def main():
    root = Path(__file__).resolve().parents[1]
    paths = [f"runs_baseline_20260924/{v}/seed_{s}/best.pt"
             for v in ("full", "late_clean", "late_aug") for s in (1111, 1112, 1113)]
    paths += [f"experiments/{name}/runs/{variant}/seed_1111/best.pt" for name, variant in (
        ("full_tune_v3", "full_tune"), ("full_tune_stress_v3", "full_tune_stress"),
        ("pool_clsctx_v3", "late_pool_tune"),
        ("attn_reg0.05_v1", "late_attn_tune"), ("attn_reg1.0_v1", "late_attn_tune"))]
    directory = root / "reports/final_controls"
    directory.mkdir(exist_ok=True)
    (directory / "planned_checkpoints.json").write_text(json.dumps(paths, indent=2))
    for relative in paths:
        path = root / relative
        variant, seed = path.parent.parent.name, int(path.parent.name.split("_")[-1])
        name = path.parts[-5] if relative.startswith("experiments/") else "baseline_" + variant
        out = directory / f"{name}_seed{seed}"
        if (out / "summary.json").exists():
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        selection = {"checkpoint": relative, "variant": variant, "seed": seed,
                     "epoch": ckpt["epoch"], "class_bias": None, "split": "test",
                     "independent_holdout": False,
                     "selection_basis": "Historical valid-selected checkpoint; final adaptive test readout"}
        out.mkdir(exist_ok=True)
        (out / "selection.json").write_text(json.dumps(selection, indent=2))
        rows, _ = evaluate_best(root, ckpt["config"], variant, seed, split="test",
                                stress=False, output_dir=out, selection=selection)
        result = dict(zip(("missing_macro_f1", "missing_mae", "clean_macro_f1", "clean_mae"),
                          selection_key(rows, ckpt["epoch"])[:4]))
        (out / "summary.json").write_text(json.dumps({**selection, **result}, indent=2))
        print(json.dumps({"name": name, "seed": seed, **result}), flush=True)


if __name__ == "__main__":
    main()

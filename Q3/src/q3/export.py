from pathlib import Path
import json
import zipfile

def export_delivery(config):
    root = Path(config["project"]["root"]); delivery = root / "delivery"; delivery.mkdir(parents=True, exist_ok=True); archive = delivery / "q3_submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in [root / "README.md", root / "THIRD_PARTY.md", root / "pyproject.toml", root / "requirements-main.txt", root / "requirements-align.txt", root / "configs" / "q3.yaml"]:
            if path.is_file(): output.write(path, path.relative_to(root))
        for path in (root / "src").rglob("*.py"):
            output.write(path, path.relative_to(root))
        for directory in (root / "outputs", root / "reports"):
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix not in {".log"}:
                    output.write(path, path.relative_to(root))
        review = root / "data" / "review" / "word_times.csv"
        if review.is_file(): output.write(review, review.relative_to(root))
    (delivery / "size_report.json").write_text(json.dumps({"q3_zip_bytes": archive.stat().st_size, "full_competition_total_bytes": None}, indent=2), encoding="utf-8")
    return archive

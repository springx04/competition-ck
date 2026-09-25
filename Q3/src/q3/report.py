from pathlib import Path
import csv
import json
import html
from .results import write_json

FIELDS = ["sample_id", "source_id", "status", "predicted_class_id", "predicted_class_name", "sentiment_score", "p_negative", "p_neutral", "p_positive", "n_T", "n_A", "n_V", "D_T", "D_A", "D_V", "D_sum", "w_T", "w_A", "w_V", "primary_modalities", "head_top_disagreement", "evidence_T", "evidence_A", "evidence_V", "mapping_status", "case_json", "case_html"]

def write_special_outputs(rows, output_dir):
    output = Path(output_dir); cases = output / "cases"; cases.mkdir(parents=True, exist_ok=True)
    with (output / "Q3_attachment4_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
        for row in rows:
            sample_id = str(row.get("sample_id")); case_json = cases / f"{sample_id}.json"; case_html = cases / f"{sample_id}.html"
            write_json(case_json, row); case_html.write_text("<html><body><pre>" + html.escape(json.dumps(row, ensure_ascii=False, indent=2)) + "</pre></body></html>", encoding="utf-8")
            values = {field: row.get(field) for field in FIELDS}; values["case_json"] = f"cases/{sample_id}.json"; values["case_html"] = f"cases/{sample_id}.html"; writer.writerow(values)

"""객체 검출 실험 결과의 집계와 CSV 저장 기능."""

import csv
from collections import defaultdict
from pathlib import Path


def summarize_records(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["method"], record["condition"])].append(record)

    summary = []
    for (method, condition), group in sorted(groups.items()):
        detected = [record for record in group if record["is_detection"]]
        total_count, detection_count = len(group), len(detected)
        summary.append(
            {
                "method": method,
                "condition": condition,
                "total_cases": total_count,
                "detection_count": detection_count,
                "success_rate": detection_count / total_count if total_count else 0.0,
                "mean_elapsed": sum(record["elapsed"] for record in group)
                / total_count,
                "mean_match_count": sum(record["match_count"] for record in detected)
                / detection_count
                if detection_count
                else 0.0,
                "mean_inlier_count": sum(record["inlier_count"] for record in detected)
                / detection_count
                if detection_count
                else 0.0,
                "mean_inlier_ratio": sum(record["inlier_ratio"] for record in detected)
                / detection_count
                if detection_count
                else 0.0,
            }
        )
    return summary


def save_experiment_csv(records, summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "records": output_dir / "experiment_records.csv",
        "summary": output_dir / "method_condition_summary.csv",
    }
    for name, rows in (("records", records), ("summary", summary)):
        fields = sorted({field for row in rows for field in row})
        with paths[name].open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return {name: str(path) for name, path in paths.items()}

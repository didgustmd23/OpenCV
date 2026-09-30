"""파노라마 실험의 세트 탐색, 합성 실행, 통계 수집 기능."""

import csv
import re
from pathlib import Path

import numpy as np

from src.common.project_paths import PANORAMA_RESULT_DIR, PANORAMA_SOURCE_DIR
from src.panorama.stitcher import stitch_panorama


# ==========================================
# 파일명 정렬용 보조 함수
# - 숫자 접미사는 문자열 순서가 아닌 숫자 순서로 정렬
# ==========================================
def _numeric_sort_key(value):
    return (0, int(value)) if value.isdigit() else (1, value.lower())


# ==========================================
# 파노라마 원본 세트 자동 탐색
# - data/set01, data/set02 ... 형식의 폴더를 번호순으로 수집
# - 반환: (set01, Path(...)) 형태의 목록
# ==========================================
def discover_panorama_source_sets(source_root=PANORAMA_SOURCE_DIR):
    source_root = Path(source_root)
    source_sets = []

    for path in source_root.glob("set*"):
        if not path.is_dir():
            continue

        match = re.fullmatch(
            r"set(\d+)",
            path.name,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(
                f"파노라마 세트 폴더는 set번호 형식이어야 합니다: {path.name}"
            )

        set_name = f"set{int(match.group(1)):02d}"
        source_sets.append((set_name, path))

    if not source_sets:
        raise FileNotFoundError(f"파노라마 세트 폴더를 찾을 수 없습니다: {source_root}")

    return sorted(
        source_sets,
        key=lambda item: _numeric_sort_key(item[0][3:]),
    )


# ==========================================
# 파노라마 합성 일괄 실행
# - 발견한 모든 세트에 지정한 특징점 방법을 적용
# - stitcher.py의 결과를 세트·방법별 폴더에 저장
# ==========================================
def run_panorama_experiments(
    source_root,
    methods,
    max_side,
    *,
    output_root=PANORAMA_RESULT_DIR,
):
    results = []

    for set_name, source_dir in discover_panorama_source_sets(source_root):
        for method in methods:
            result = stitch_panorama(
                input_dir=source_dir,
                method=method,
                max_side=max_side,
                output_root=output_root,
                set_name=set_name,
            )
            results.append(result)

    return results


# ==========================================
# 파노라마 합성 통계 수집
# - 세트·방법별 matching_stats.csv를 자동 탐색
# - 이전 세트 없는 결과 폴더는 집계에서 제외
# ==========================================
def collect_panorama_summary(panorama_result_dir, methods):
    panorama_result_dir = Path(panorama_result_dir)
    method_names = {str(method).upper() for method in methods}
    summary_rows = []

    for stats_path in sorted(panorama_result_dir.rglob("matching_stats.csv")):
        method_name = stats_path.parent.name.upper()

        if method_name not in method_names:
            continue

        set_dir = stats_path.parent.parent

        if set_dir == panorama_result_dir:
            continue

        with stats_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            pair_rows = list(csv.DictReader(file))

        if not pair_rows:
            continue

        panorama_path = stats_path.parent / "panorama.jpg"
        summary_rows.append(
            {
                "set": set_dir.name,
                "method": method_name,
                "pair_count": len(pair_rows),
                "mean_raw_matches": float(
                    np.mean([int(row["raw_match_count"]) for row in pair_rows])
                ),
                "mean_filtered_matches": float(
                    np.mean([int(row["filtered_match_count"]) for row in pair_rows])
                ),
                "mean_inliers": float(
                    np.mean([int(row["inlier_count"]) for row in pair_rows])
                ),
                "mean_inlier_ratio": float(
                    np.mean([float(row["inlier_ratio"]) for row in pair_rows])
                ),
                "panorama_path": (
                    str(panorama_path) if panorama_path.is_file() else None
                ),
            }
        )

    if not summary_rows:
        raise FileNotFoundError(
            "세트별 파노라마 matching_stats.csv를 찾을 수 없습니다."
        )

    return summary_rows

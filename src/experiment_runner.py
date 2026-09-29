import csv
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import src.object_finder as object_finder
from src.feature_match import detect_features
from src.object_finder import get_global_corners
from src.project_paths import (
    MATCHING_RESULT_DIR,
    OBJECT_DETECTION_RESULT_DIR,
    PANORAMA_RESULT_DIR,
    PANORAMA_SOURCE_DIR,
)
from src.settings import OBJECT_DETECTION_CONFIG
from src.stitcher import stitch_panorama
from src.tile_scanner import scan_scene


# ==========================================
# 객체 검출 실험 환경 설정
# - 노트북과 일괄 실험에서 동일한 타일·방법·결과 경로를 사용
# - 기본 탐색 값은 config.json의 object_detection 영역을 재사용
# ==========================================
@dataclass(frozen=True)
class ExperimentSettings:
    panorama_max_side: int
    methods: tuple
    tile_size: tuple
    coarse_stride: tuple
    fine_stride: tuple
    coarse_top_k: int
    early_stop_inliers: int
    result_root: Path
    matching_root: Path
    metrics_dir: Path
    conditions: tuple
    condition_labels: dict


# ==========================================
# 객체 검출 실험 환경 생성
# - 노트북에서 직접 선언하던 공통 환경값을 한 곳에서 관리
# ==========================================
def get_experiment_settings():
    return ExperimentSettings(
        panorama_max_side=2048,
        methods=tuple(OBJECT_DETECTION_CONFIG["methods"]),
        tile_size=tuple(OBJECT_DETECTION_CONFIG["tile_size"]),
        coarse_stride=tuple(OBJECT_DETECTION_CONFIG["coarse_stride"]),
        fine_stride=tuple(OBJECT_DETECTION_CONFIG["fine_stride"]),
        coarse_top_k=OBJECT_DETECTION_CONFIG["coarse_top_k"],
        early_stop_inliers=OBJECT_DETECTION_CONFIG["early_stop_inliers"],
        result_root=OBJECT_DETECTION_RESULT_DIR / "batch",
        matching_root=MATCHING_RESULT_DIR,
        metrics_dir=OBJECT_DETECTION_RESULT_DIR / "metrics",
        conditions=(
            {
                "name": "baseline",
                "rotation_deg": 0,
                "scale_factor": 1.0,
                "brightness_factor": 1.0,
                "save_visuals": True,
            },
            {
                "name": "rotation_minus30",
                "rotation_deg": -30,
                "scale_factor": 1.0,
                "brightness_factor": 1.0,
                "save_visuals": False,
            },
            {
                "name": "rotation_plus30",
                "rotation_deg": 30,
                "scale_factor": 1.0,
                "brightness_factor": 1.0,
                "save_visuals": False,
            },
            {
                "name": "scale_075",
                "rotation_deg": 0,
                "scale_factor": 0.75,
                "brightness_factor": 1.0,
                "save_visuals": False,
            },
            {
                "name": "scale_125",
                "rotation_deg": 0,
                "scale_factor": 1.25,
                "brightness_factor": 1.0,
                "save_visuals": False,
            },
            {
                "name": "brightness_070",
                "rotation_deg": 0,
                "scale_factor": 1.0,
                "brightness_factor": 0.70,
                "save_visuals": False,
            },
            {
                "name": "brightness_130",
                "rotation_deg": 0,
                "scale_factor": 1.0,
                "brightness_factor": 1.30,
                "save_visuals": False,
            },
        ),
        condition_labels={
            "baseline": "기본",
            "rotation_minus30": "회전 -30°",
            "rotation_plus30": "회전 +30°",
            "scale_075": "크기 75%",
            "scale_125": "크기 125%",
            "brightness_070": "조명 70%",
            "brightness_130": "조명 130%",
        },
    )


# ==========================================
# 파일명 정렬용 보조 함수
# - 숫자 접미사는 문자열 순서가 아닌 숫자 순서로 정렬
# - 예: 2, 10 순서를 10, 2로 잘못 정렬하지 않도록 처리
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
                "파노라마 세트 폴더는 set번호 형식이어야 합니다: "
                f"{path.name}"
            )

        set_name = f"set{int(match.group(1)):02d}"
        source_sets.append((set_name, path))

    if not source_sets:
        raise FileNotFoundError(
            "파노라마 세트 폴더를 찾을 수 없습니다: "
            f"{source_root}"
        )

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
# 접두사 기준 이미지 파일 수집
# - 지정한 폴더에서 *.jpg 파일 탐색
# - set01/target.jpg -> {"1": Path(...)} 형태로 반환
# - panorama1.jpg도 같은 방식으로 처리
# ==========================================
def _collect_images(directory, prefix, recursive=False):
    directory = Path(directory)
    images = {}

    paths = directory.rglob("*.jpg") if recursive else directory.glob("*.jpg")

    for path in paths:
        if path.stem.lower().startswith(prefix):
            suffix = path.stem[len(prefix):]
            # setN/target.jpg, panoramaN.jpg 형식의 숫자 번호만 테스트 케이스로 사용
            # panorama_outline.jpg, panorama.jpg 같은 보조 결과는 제외
            if not suffix.isdigit():
                continue

            if suffix in images:
                raise ValueError(
                    f"{prefix}{suffix}.jpg 파일이 여러 개 발견되었습니다: "
                    f"{images[suffix]}, {path}"
                )

            images[suffix] = path

    return images


# ==========================================
# 세트별 파노라마 결과 수집
# - results/panorama/set01/SIFT/panorama.jpg 형식을 사용
# - set01의 번호를 set01/target.jpg와 연결할 실험 번호로 사용
# ==========================================
# ==========================================
# 세트별 기준 물체 이미지 수집
# - data/objects/set01/target.jpg 형식을 사용
# - set01의 번호를 {"1": Path(...)} 형태로 반환
# ==========================================
def _collect_targets(directory):
    directory = Path(directory)
    targets = {}

    for path in directory.rglob("*"):
        if not path.is_file() or path.name.lower() != "target.jpg":
            continue

        match = re.fullmatch(
            r"set(\d+)",
            path.parent.name,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(
                "기준 물체는 set번호/target.jpg 형식으로 저장해야 합니다: "
                f"{path}"
            )

        case_id = str(int(match.group(1)))

        if case_id in targets:
            raise ValueError(
                f"기준 물체 세트 번호 {case_id}가 중복됩니다. "
                f"{targets[case_id]}, {path}"
            )

        targets[case_id] = path

    return targets


def _collect_panorama_results(panorama_result_dir, method):
    panorama_result_dir = Path(panorama_result_dir)
    method_name = str(method).upper()
    panoramas = {}

    for path in panorama_result_dir.rglob("panorama.jpg"):
        if path.parent.name.upper() != method_name:
            continue

        set_name = path.parent.parent.name

        # 이전 구조(results/panorama/SIFT/panorama.jpg)는 세트 번호가 없어
        # 새 세트별 실험 탐색에서 제외한다.
        if path.parent.parent == panorama_result_dir:
            continue

        match = re.search(r"(\d+)$", set_name)

        if match is None:
            raise ValueError(
                "파노라마 세트 폴더명 끝에 번호가 필요합니다: "
                f"{path.parent.parent}"
            )

        case_id = str(int(match.group()))

        if case_id in panoramas:
            raise ValueError(
                f"{method_name}의 세트 번호 {case_id}가 중복됩니다: "
                f"{panoramas[case_id]}, {path}"
            )

        panoramas[case_id] = path

    return panoramas


# ==========================================
# 자동 테스트 세트 생성
# - setN/target.jpg와 방법별 setN/panorama.jpg를 번호 기준으로 연결
# - 짝이 없는 파일은 조용히 제외하지 않고 오류로 안내
# - 반환: (case_name, reference_path, {method: panorama_path}) 목록
# ==========================================
def discover_test_cases(target_dir, panorama_result_dir, methods):
    targets = _collect_targets(target_dir)
    panorama_result_dir = Path(panorama_result_dir)
    panorama_paths = {}

    # ==========================================
    # 방법별 파노라마 결과 수집
    # - SIFT / ORB / ALIKED가 각각 합성한 세트별 panorama.jpg 사용
    # - 객체 검출도 같은 방법으로 실행해 전체 파이프라인 비교
    # ==========================================
    for method in methods:
        method_name = str(method).upper()
        panoramas = _collect_panorama_results(
            panorama_result_dir,
            method_name,
        )

        missing_panoramas = sorted(
            set(targets) - set(panoramas),
            key=_numeric_sort_key,
        )
        missing_targets = sorted(
            set(panoramas) - set(targets),
            key=_numeric_sort_key,
        )

        if missing_panoramas or missing_targets:
            raise ValueError(
                f"{method_name} 파노라마-기준 물체 짝이 맞지 않습니다. "
                f"파노라마 없음={missing_panoramas}, "
                f"기준 물체 없음={missing_targets}"
            )

        panorama_paths[method_name] = panoramas

    case_ids = sorted(targets, key=_numeric_sort_key)

    if not case_ids:
        raise FileNotFoundError(
            "setN/target.jpg 파일을 찾을 수 없습니다."
        )

    return [
        (
            f"set{int(case_id):02d}",
            str(targets[case_id]),
            {
                method: str(panorama_paths[method][case_id])
                for method in panorama_paths
            },
        )
        for case_id in case_ids
    ]


# ==========================================
# 파노라마 합성 통계 수집
# - 세트·방법별 matching_stats.csv를 자동 탐색
# - 이전 세트 없는 결과 폴더는 집계에서 제외
# ==========================================
def collect_panorama_summary(panorama_result_dir, methods):
    panorama_result_dir = Path(panorama_result_dir)
    method_names = {str(method).upper() for method in methods}
    summary_rows = []

    for stats_path in sorted(
        panorama_result_dir.rglob("matching_stats.csv")
    ):
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
                    np.mean(
                        [
                            int(row["raw_match_count"])
                            for row in pair_rows
                        ]
                    )
                ),
                "mean_filtered_matches": float(
                    np.mean(
                        [
                            int(row["filtered_match_count"])
                            for row in pair_rows
                        ]
                    )
                ),
                "mean_inliers": float(
                    np.mean(
                        [
                            int(row["inlier_count"])
                            for row in pair_rows
                        ]
                    )
                ),
                "mean_inlier_ratio": float(
                    np.mean(
                        [
                            float(row["inlier_ratio"])
                            for row in pair_rows
                        ]
                    )
                ),
                "panorama_path": (
                    str(panorama_path)
                    if panorama_path.is_file()
                    else None
                ),
            }
        )

    if not summary_rows:
        raise FileNotFoundError(
            "세트별 파노라마 matching_stats.csv를 찾을 수 없습니다."
        )

    return summary_rows


# ==========================================
# 파노라마 합성 통계표 생성
# - Matplotlib 객체를 받아 노트북에서 재사용
# - 표 생성만 담당하고 출력은 호출한 셀에서 처리
# ==========================================
def create_panorama_summary_figure(summary_rows, pyplot):
    table_rows = [
        [
            row["set"],
            row["method"],
            row["pair_count"],
            f"{row['mean_raw_matches']:.1f}",
            f"{row['mean_filtered_matches']:.1f}",
            f"{row['mean_inliers']:.1f}",
            f"{row['mean_inlier_ratio']:.3f}",
        ]
        for row in summary_rows
    ]

    figure, axis = pyplot.subplots(
        figsize=(12, max(3, 0.5 * len(table_rows) + 1.5))
    )
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=[
            "세트",
            "방법",
            "사진 쌍",
            "평균 원시 매칭",
            "평균 필터링 매칭",
            "평균 Inlier",
            "평균 Inlier 비율",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    axis.set_title(
        "방법별 파노라마 합성 매칭 통계",
        fontsize=16,
        pad=18,
    )

    return figure


# ==========================================
# 파노라마 결과 시각화용 세트 행 선택
# - set_name을 생략하면 첫 번째 세트를 사용
# - 지정한 세트의 방법별 통계 행을 정렬해 반환
# ==========================================
def _select_panorama_visual_rows(summary_rows, set_name=None):
    available_sets = sorted({row["set"] for row in summary_rows})

    if not available_sets:
        raise ValueError("시각화할 파노라마 통계가 없습니다.")

    selected_set = set_name or available_sets[0]
    rows = sorted(
        (
            row
            for row in summary_rows
            if row["set"] == selected_set
        ),
        key=lambda row: row["method"],
    )

    if not rows:
        raise ValueError(
            f"시각화할 파노라마 세트가 없습니다: {selected_set}"
        )

    return selected_set, rows


# ==========================================
# 방법별 필터링 매칭 이미지 비교 Figure 생성
# - 지정한 인접 사진 쌍의 filtered 매칭 이미지를 방법별로 표시
# ==========================================
def create_panorama_filtered_match_figure(
    summary_rows,
    panorama_result_dir,
    pyplot,
    *,
    set_name=None,
    pair_name="pair_01_02",
):
    selected_set, rows = _select_panorama_visual_rows(
        summary_rows,
        set_name,
    )
    panorama_result_dir = Path(panorama_result_dir)
    figure, axes = pyplot.subplots(
        1,
        len(rows),
        figsize=(7 * len(rows), 4),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for axis, row in zip(axes, rows):
        matching_path = (
            panorama_result_dir
            / selected_set
            / row["method"]
            / "matching"
            / f"{pair_name}_filtered.png"
        )

        if matching_path.is_file():
            axis.imshow(
                read_image_for_display(
                    matching_path,
                    max_width=900,
                )
            )
        else:
            axis.text(
                0.5,
                0.5,
                "필터링 매칭 이미지가 없습니다.",
                ha="center",
                va="center",
                wrap=True,
            )

        axis.set_title(
            f"{row['method']} / {pair_name} 필터링 매칭"
        )
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle(
        f"방법별 인접 사진 필터링 매칭: {selected_set}",
        fontsize=16,
    )

    return figure


# ==========================================
# 파노라마 매칭 단계별 이미지 Figure 생성
# - raw / filtered / ransac 매칭 이미지를 한 화면에 표시
# ==========================================
def create_panorama_matching_gallery_figure(
    panorama_result_dir,
    set_name,
    method,
    pyplot,
    *,
    column_count=3,
):
    matching_dir = (
        Path(panorama_result_dir)
        / str(set_name)
        / str(method).upper()
        / "matching"
    )
    matching_paths = sorted(matching_dir.glob("*.png"))

    if not matching_paths:
        raise FileNotFoundError(
            f"매칭 이미지를 찾을 수 없습니다: {matching_dir}"
        )
    if column_count <= 0:
        raise ValueError("column_count는 양수여야 합니다.")

    row_count = (
        len(matching_paths) + column_count - 1
    ) // column_count
    figure, axes = pyplot.subplots(
        row_count,
        column_count,
        figsize=(6 * column_count, 5 * row_count),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for axis, matching_path in zip(axes, matching_paths):
        axis.imshow(
            read_image_for_display(
                matching_path,
                max_width=800,
            )
        )
        axis.set_title(
            matching_path.stem.replace("_", " "),
            fontsize=11,
        )
        axis.set_xticks([])
        axis.set_yticks([])

    for axis in axes[len(matching_paths):]:
        axis.axis("off")

    figure.suptitle(
        f"{set_name} / {str(method).upper()} 파노라마 매칭 단계별 결과",
        fontsize=16,
    )

    return figure


# ==========================================
# 방법별 파노라마·윤곽선 Figure 생성
# - 첫 행에는 합성 결과, 둘째 행에는 원본 사진 배치 윤곽선 표시
# ==========================================
def create_panorama_result_comparison_figure(
    summary_rows,
    pyplot,
    *,
    set_name=None,
):
    selected_set, rows = _select_panorama_visual_rows(
        summary_rows,
        set_name,
    )
    figure, axes = pyplot.subplots(
        2,
        len(rows),
        figsize=(7 * len(rows), 9),
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(2, len(rows))

    for column, row in enumerate(rows):
        panorama_axis = axes[0, column]
        outline_axis = axes[1, column]
        panorama_path = row["panorama_path"]

        if panorama_path is not None and Path(panorama_path).is_file():
            panorama_path = Path(panorama_path)
            panorama_axis.imshow(
                read_image_for_display(
                    panorama_path,
                    max_width=900,
                )
            )
            outline_path = panorama_path.parent / "panorama_outline.jpg"

            if outline_path.is_file():
                outline_axis.imshow(
                    read_image_for_display(
                        outline_path,
                        max_width=900,
                    )
                )
            else:
                outline_axis.text(
                    0.5,
                    0.5,
                    "합성 윤곽선 이미지가 없습니다.",
                    ha="center",
                    va="center",
                    wrap=True,
                )
        else:
            panorama_axis.text(
                0.5,
                0.5,
                "파노라마 이미지가 없습니다.",
                ha="center",
                va="center",
                wrap=True,
            )
            outline_axis.text(
                0.5,
                0.5,
                "합성 윤곽선 이미지가 없습니다.",
                ha="center",
                va="center",
                wrap=True,
            )

        panorama_axis.set_title(f"{row['method']} 파노라마")
        outline_axis.set_title(f"{row['method']} 합성 윤곽선")

        for axis in (panorama_axis, outline_axis):
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        f"방법별 파노라마 합성 결과: {selected_set}",
        fontsize=18,
    )

    return figure


# ==========================================
# OpenCV 내장 Stitcher 결과 Figure 생성
# - stitcher_create.py가 생성한 세트별 panorama.jpg를 표시
# - 직접 구현한 방법별 결과와 시각적으로 비교할 때 사용
# ==========================================
def create_builtin_stitcher_result_figure(result_rows, pyplot):
    if not result_rows:
        raise ValueError("시각화할 OpenCV 내장 Stitcher 결과가 없습니다.")

    figure, axes = pyplot.subplots(
        1,
        len(result_rows),
        figsize=(7 * len(result_rows), 4),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for axis, result in zip(axes, result_rows):
        output_path = result.get("output_path")

        if output_path is not None and Path(output_path).is_file():
            axis.imshow(
                read_image_for_display(
                    output_path,
                    max_width=900,
                )
            )
        else:
            axis.text(
                0.5,
                0.5,
                "내장 Stitcher 파노라마가 생성되지 않았습니다.\n"
                + result.get("note", ""),
                ha="center",
                va="center",
                wrap=True,
            )

        axis.set_title(
            f"{result['set']} / OpenCV Stitcher\n"
            f"{result['elapsed']:.2f}초",
            fontsize=11,
        )
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle(
        "OpenCV 내장 Stitcher 파노라마 결과",
        fontsize=16,
    )

    return figure


# ==========================================
# 객체 검출 정량 결과 공통 데이터 준비
# - 조건·방법 조합별 요약 행을 빠르게 조회할 수 있도록 구성
# ==========================================
def _prepare_detection_summary(summary_rows, experiment):
    condition_names = [
        condition["name"]
        for condition in experiment.conditions
    ]
    summary_by_key = {
        (row["method"], row["condition"]): row
        for row in summary_rows
    }

    missing_keys = [
        (method, condition_name)
        for condition_name in condition_names
        for method in experiment.methods
        if (method, condition_name) not in summary_by_key
    ]

    if missing_keys:
        raise ValueError(
            "정량 비교에 필요한 실험 요약 행이 없습니다: "
            f"{missing_keys}"
        )

    return condition_names, summary_by_key


# ==========================================
# 객체 검출 정량 비교표 생성
# - 방법·변형 조건별 성공률, 시간, Inlier 비율을 표로 표시
# ==========================================
def create_detection_summary_table_figure(
    summary_rows,
    experiment,
    pyplot,
):
    condition_names, summary_by_key = _prepare_detection_summary(
        summary_rows,
        experiment,
    )
    table_rows = []

    for condition_name in condition_names:
        for method in experiment.methods:
            row = summary_by_key[(method, condition_name)]
            table_rows.append(
                [
                    experiment.condition_labels[condition_name],
                    method,
                    f"{row['detection_count']}/{row['total_cases']}",
                    f"{row['success_rate'] * 100:.1f}%",
                    f"{row['mean_elapsed']:.2f}초",
                    f"{row['mean_inlier_ratio']:.3f}",
                ]
            )

    figure, axis = pyplot.subplots(
        figsize=(12, max(4, 0.42 * len(table_rows) + 1.5))
    )
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=[
            "조건",
            "방법",
            "성공/전체",
            "검출 성공률",
            "평균 시간",
            "평균 Inlier 비율",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    axis.set_title(
        "SIFT / ORB / ALIKED 매칭 정확도·속도 비교표",
        fontsize=16,
        pad=18,
    )

    return figure


# ==========================================
# 객체 검출 최종 Tile 매칭 통계표 생성
# - 지정 조건에서 검출에 성공한 최종 Fine Tile의 매칭 품질을 방법별 비교
# - match_count는 Ratio Test 또는 LightGlue 이후, RANSAC 이전 매칭 수
# ==========================================
def create_detection_matching_summary_figure(
    records,
    experiment,
    pyplot,
    visual_condition="baseline",
):
    condition_records = [
        record
        for record in records
        if record["condition"] == visual_condition
    ]
    if not condition_records:
        raise ValueError(
            "매칭 통계를 만들 객체 검출 기록이 없습니다: "
            f"condition={visual_condition}"
        )

    table_rows = []
    for method in experiment.methods:
        method_records = [
            record
            for record in condition_records
            if record["method"] == method
        ]
        if not method_records:
            raise ValueError(
                "매칭 통계를 만들 방법별 객체 검출 기록이 없습니다: "
                f"method={method}, condition={visual_condition}"
            )

        detection_records = [
            record
            for record in method_records
            if record["is_detection"]
        ]
        detection_count = len(detection_records)
        mean_keypoints = sum(
            record["keypoints"]
            for record in method_records
        ) / len(method_records)

        if detection_count:
            mean_match_count = sum(
                record["match_count"]
                for record in detection_records
            ) / detection_count
            mean_inlier_count = sum(
                record["inlier_count"]
                for record in detection_records
            ) / detection_count
            mean_inlier_ratio = sum(
                record["inlier_ratio"]
                for record in detection_records
            ) / detection_count
            matching_values = [
                f"{mean_match_count:.1f}",
                f"{mean_inlier_count:.1f}",
                f"{mean_inlier_ratio:.3f}",
            ]
        else:
            matching_values = ["-", "-", "-"]

        table_rows.append(
            [
                method,
                str(len(method_records)),
                f"{detection_count}/{len(method_records)}",
                f"{mean_keypoints:.1f}",
                *matching_values,
            ]
        )

    condition_label = experiment.condition_labels.get(
        visual_condition,
        visual_condition,
    )
    figure, axis = pyplot.subplots(figsize=(14, 4.5))
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=[
            "방법",
            "전체 케이스",
            "검출 성공",
            "평균 기준 특징점",
            "평균 RANSAC 전 매칭",
            "평균 Inlier",
            "평균 Inlier 비율",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axis.set_title(
        f"{condition_label} 조건: 방법별 객체 검출 최종 Tile 매칭 통계",
        fontsize=16,
        pad=18,
    )

    return figure


# ==========================================
# 객체 검출 정량 비교 그래프 생성
# - 조건별 검출 성공률과 평균 실행 시간을 막대그래프로 비교
# ==========================================
def create_detection_summary_chart_figure(
    summary_rows,
    experiment,
    pyplot,
):
    condition_names, summary_by_key = _prepare_detection_summary(
        summary_rows,
        experiment,
    )
    x_positions = np.arange(len(condition_names))
    bar_width = 0.24
    figure, axes = pyplot.subplots(
        1,
        2,
        figsize=(18, 6),
        constrained_layout=True,
    )
    method_center = (len(experiment.methods) - 1) / 2

    for index, method in enumerate(experiment.methods):
        success_rates = [
            summary_by_key[(method, name)]["success_rate"] * 100
            for name in condition_names
        ]
        elapsed_times = [
            summary_by_key[(method, name)]["mean_elapsed"]
            for name in condition_names
        ]
        offset = (index - method_center) * bar_width
        axes[0].bar(
            x_positions + offset,
            success_rates,
            bar_width,
            label=method,
        )
        axes[1].bar(
            x_positions + offset,
            elapsed_times,
            bar_width,
            label=method,
        )

    labels = [
        experiment.condition_labels[name]
        for name in condition_names
    ]
    axes[0].set_title("조건별 검출 성공률")
    axes[0].set_ylabel("성공률 (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_title("조건별 평균 실행 시간")
    axes[1].set_ylabel("실행 시간 (초)")

    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)

    return figure


# ==========================================
# 기준 물체 이미지 변형
# - 회전: 이미지 잘림을 피하도록 확장된 캔버스 사용
# - 크기: scale_factor 비율로 Reference 크기 변경
# - 조명: brightness_factor를 밝기 배율로 적용
# ==========================================
def transform_reference(
    image,
    *,
    rotation_deg=0.0,
    scale_factor=1.0,
    brightness_factor=1.0,
):
    transformed = image.copy()

    if scale_factor <= 0:
        raise ValueError("scale_factor는 0보다 커야 합니다.")
    if brightness_factor <= 0:
        raise ValueError("brightness_factor는 0보다 커야 합니다.")

    # ==========================================
    # 크기 변형
    # ==========================================
    if scale_factor != 1.0:
        height, width = transformed.shape[:2]
        transformed = cv2.resize(
            transformed,
            (
                max(1, round(width * scale_factor)),
                max(1, round(height * scale_factor)),
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    # ==========================================
    # 회전 변형
    # - 확장된 출력 크기를 계산해 Reference가 잘리지 않도록 처리
    # ==========================================
    if rotation_deg != 0.0:
        height, width = transformed.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(
            center,
            rotation_deg,
            1.0,
        )

        cosine = abs(matrix[0, 0])
        sine = abs(matrix[0, 1])
        bound_width = int(height * sine + width * cosine)
        bound_height = int(height * cosine + width * sine)

        matrix[0, 2] += bound_width / 2 - center[0]
        matrix[1, 2] += bound_height / 2 - center[1]

        transformed = cv2.warpAffine(
            transformed,
            matrix,
            (bound_width, bound_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    # ==========================================
    # 조명 변형
    # ==========================================
    if brightness_factor != 1.0:
        transformed = cv2.convertScaleAbs(
            transformed,
            alpha=brightness_factor,
            beta=0,
        )

    return transformed


# ==========================================
# 단일 객체 검출 및 결과 기록
# - Reference Feature 추출
# - Coarse-to-Fine Tile Scan
# - 최종 Global Polygon 생성
# - 선택한 조건에서만 결과·매칭 이미지 저장
# ==========================================
def detect_and_save(
    case_name,
    ref_path,
    scene_path,
    method,
    *,
    condition,
    tile_size,
    coarse_stride,
    fine_stride,
    coarse_top_k,
    early_stop_inliers,
    result_root,
    matching_root,
):
    condition_name = condition["name"]
    save_visuals = condition.get(
        "save_visuals",
        condition_name == "baseline",
    )

    # ==========================================
    # Reference / Panorama 이미지 로드 및 Reference 변형
    # ==========================================
    ref_img = cv2.imread(ref_path)
    scene_img = cv2.imread(scene_path)

    if ref_img is None:
        raise FileNotFoundError(
            f"참조 이미지를 불러올 수 없습니다: {ref_path}"
        )
    if scene_img is None:
        raise FileNotFoundError(
            f"파노라마 이미지를 불러올 수 없습니다: {scene_path}"
        )

    ref_img = transform_reference(
        ref_img,
        rotation_deg=condition.get("rotation_deg", 0.0),
        scale_factor=condition.get("scale_factor", 1.0),
        brightness_factor=condition.get("brightness_factor", 1.0),
    )

    # ==========================================
    # Reference 특징점 사전 계산 및 Panorama 탐색
    # - 변형 실험은 중간 Tile 매칭 이미지를 저장하지 않아 I/O를 줄임
    # ==========================================
    start_time = time.perf_counter()
    kp_ref, des_ref = detect_features(ref_img, method=method)

    previous_save_matches = object_finder.SAVE_FEATURE_MATCHES
    object_finder.SAVE_FEATURE_MATCHES = save_visuals

    try:
        scan_result = scan_scene(
            ref_img=ref_img,
            sce_img=scene_img,
            ref_features=(kp_ref, des_ref),
            tile_size=tile_size,
            coarse_stride=coarse_stride,
            fine_stride=fine_stride,
            coarse_top_k=coarse_top_k,
            early_stop_inliers=early_stop_inliers,
            ref_name=ref_path,
            method=method,
        )
    finally:
        object_finder.SAVE_FEATURE_MATCHES = previous_save_matches

    # ==========================================
    # 기본 결과 레코드 준비
    # ==========================================
    result_image = scene_img.copy()
    best = scan_result["best"]
    case_dir = (
        Path(result_root)
        / case_name
        / condition_name
    )

    record = {
        "case": case_name,
        "method": method,
        "panorama_path": str(scene_path),
        "condition": condition_name,
        "rotation_deg": condition.get("rotation_deg", 0.0),
        "scale_factor": condition.get("scale_factor", 1.0),
        "brightness_factor": condition.get("brightness_factor", 1.0),
        "keypoints": len(kp_ref),
        "elapsed": time.perf_counter() - start_time,
        "is_detection": False,
        "result_path": None,
        "match_image_path": None,
    }

    # ==========================================
    # 최종 Fine Tile 매칭 이미지 저장
    # - 기본 조건에서만 케이스별로 보존
    # - 원본 매칭 파일은 다음 케이스 실행 시 덮어써질 수 있음
    # ==========================================
    if save_visuals and best is not None:
        case_dir.mkdir(parents=True, exist_ok=True)
        source_match_path = (
            Path(matching_root)
            / method
            / f"fine_{best['index']}_matches.png"
        )
        saved_match_path = (
            case_dir
            / f"{method}_keypoint_matches.png"
        )
        match_image = cv2.imread(str(source_match_path))

        if match_image is not None:
            if not cv2.imwrite(
                str(saved_match_path),
                match_image,
            ):
                raise OSError(
                    "매칭 이미지를 저장할 수 없습니다: "
                    f"{saved_match_path}"
                )
            record["match_image_path"] = str(saved_match_path)

    # ==========================================
    # Global Polygon 생성 및 검출 지표 기록
    # ==========================================
    if (
        best is not None
        and best["result"].get("is_detection", False)
        and best["result"].get("H") is not None
    ):
        best_result = best["result"]
        corner_result = get_global_corners(
            ref_img=ref_img,
            H_tile=best_result["H"],
            tile_origin=(best["x"], best["y"]),
        )
        corners_global = np.int32(
            np.round(corner_result["global"])
        )

        cv2.polylines(
            result_image,
            [corners_global],
            True,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

        record.update(
            {
                "is_detection": True,
                "match_count": best_result["match_count"],
                "inlier_count": best_result["inlier_count"],
                "inlier_ratio": best_result["inlier_ratio"],
                "tile": (best["x"], best["y"]),
            }
        )

    # ==========================================
    # 기본 조건의 최종 검출 결과 이미지 저장
    # ==========================================
    if save_visuals:
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / f"{method}_result.png"

        if not cv2.imwrite(
            str(result_path),
            result_image,
        ):
            raise OSError(
                f"검출 결과를 저장할 수 없습니다: {result_path}"
            )

        record["result_path"] = str(result_path)

    return record


# ==========================================
# 전체 테스트 세트 일괄 실행
# - 모든 (Reference, 방법별 Panorama) 케이스를 순차 처리
# - 각 방법이 직접 합성한 Panorama에서 같은 방법으로 객체 검출 실행
# - 결과 레코드 목록 반환
# ==========================================
def run_batch(
    test_cases,
    methods,
    conditions,
    *,
    tile_size,
    coarse_stride,
    fine_stride,
    coarse_top_k,
    early_stop_inliers,
    result_root,
    matching_root,
):
    records = []

    for condition in conditions:
        for case_name, ref_path, panorama_paths in test_cases:
            for method in methods:
                method_name = str(method).upper()
                record = detect_and_save(
                    case_name,
                    ref_path,
                    panorama_paths[method_name],
                    method_name,
                    condition=condition,
                    tile_size=tile_size,
                    coarse_stride=coarse_stride,
                    fine_stride=fine_stride,
                    coarse_top_k=coarse_top_k,
                    early_stop_inliers=early_stop_inliers,
                    result_root=result_root,
                    matching_root=matching_root,
                )
                records.append(record)

    return records


# ==========================================
# 방법·조건별 성능 요약
# - 정확도: 정답 쌍에서의 Detection Success Rate
# - 속도: 전체 실행 시간의 평균
# - 평균 Match / Inlier 수 및 Inlier Ratio 함께 집계
# ==========================================
def summarize_records(records):
    groups = defaultdict(list)

    for record in records:
        groups[
            (
                record["method"],
                record["condition"],
            )
        ].append(record)

    summary = []

    for (method, condition), group in sorted(groups.items()):
        detection_records = [
            record
            for record in group
            if record["is_detection"]
        ]
        total_count = len(group)
        detection_count = len(detection_records)

        summary.append(
            {
                "method": method,
                "condition": condition,
                "total_cases": total_count,
                "detection_count": detection_count,
                "success_rate": (
                    detection_count / total_count
                    if total_count > 0
                    else 0.0
                ),
                "mean_elapsed": sum(
                    record["elapsed"]
                    for record in group
                ) / total_count,
                "mean_match_count": (
                    sum(
                        record["match_count"]
                        for record in detection_records
                    ) / detection_count
                    if detection_count > 0
                    else 0.0
                ),
                "mean_inlier_count": (
                    sum(
                        record["inlier_count"]
                        for record in detection_records
                    ) / detection_count
                    if detection_count > 0
                    else 0.0
                ),
                "mean_inlier_ratio": (
                    sum(
                        record["inlier_ratio"]
                        for record in detection_records
                    ) / detection_count
                    if detection_count > 0
                    else 0.0
                ),
            }
        )

    return summary


# ==========================================
# 실험 원본 기록 및 요약 CSV 저장
# - records: 케이스별 원본 결과
# - summary: 방법·조건별 집계 결과
# - 반환: 저장된 CSV 경로
# ==========================================
def save_experiment_csv(records, summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record_path = output_dir / "experiment_records.csv"
    summary_path = output_dir / "method_condition_summary.csv"

    record_fields = sorted(
        {
            field
            for record in records
            for field in record
        }
    )
    summary_fields = sorted(
        {
            field
            for row in summary
            for field in row
        }
    )

    with record_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=record_fields,
        )
        writer.writeheader()
        writer.writerows(records)

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )
        writer.writeheader()
        writer.writerows(summary)

    return {
        "records": str(record_path),
        "summary": str(summary_path),
    }


# ==========================================
# Notebook 시각화용 이미지 변환
# - 큰 결과 이미지는 지정 폭 이하로 축소
# - OpenCV BGR 이미지를 Matplotlib RGB 형식으로 변환
# ==========================================
def read_image_for_display(path, max_width=500):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(
            f"결과 이미지를 불러올 수 없습니다: {path}"
        )

    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(
            image,
            (max_width, int(height * scale)),
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==========================================
# 객체 검출 정성적 결과 시각화
# - 위: Global Polygon이 표시된 최종 검출 결과
# - 아래: 최종 Fine Tile Keypoint Matches
# ==========================================
def create_detection_qualitative_figure(
    test_cases,
    experiment,
    pyplot,
    visual_condition="baseline",
):
    if not test_cases:
        raise ValueError("시각화할 객체 검출 테스트 세트가 없습니다.")

    methods = tuple(experiment.methods)
    if not methods:
        raise ValueError("시각화할 특징점 방법이 없습니다.")

    figure, axes = pyplot.subplots(
        2 * len(test_cases),
        len(methods),
        figsize=(20, 8 * len(test_cases)),
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(
        2 * len(test_cases),
        len(methods),
    )

    for row, (case_name, _, _) in enumerate(test_cases):
        case_dir = experiment.result_root / case_name / visual_condition

        for column, method in enumerate(methods):
            result_axis = axes[2 * row, column]
            result_path = case_dir / f"{method}_result.png"
            result_axis.imshow(read_image_for_display(result_path))
            result_axis.set_title(
                f"{case_name} / {method} 검출 결과",
                fontsize=11,
            )

            match_axis = axes[2 * row + 1, column]
            match_path = case_dir / f"{method}_keypoint_matches.png"
            if match_path.is_file():
                match_axis.imshow(
                    read_image_for_display(
                        match_path,
                        max_width=700,
                    )
                )
            else:
                match_axis.text(
                    0.5,
                    0.5,
                    "최종 Fine 후보 또는 매칭 이미지가 없습니다.",
                    ha="center",
                    va="center",
                    wrap=True,
                )
            match_axis.set_title(
                f"{case_name} / {method} 최종 Tile 매칭",
                fontsize=11,
            )

            for axis in (result_axis, match_axis):
                axis.set_xticks([])
                axis.set_yticks([])

    figure.suptitle(
        "기본 조건: 파노라마 객체 검출 결과와 최종 Tile Keypoint Matches",
        fontsize=18,
    )

    return figure

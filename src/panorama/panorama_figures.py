"""최종보고서 파노라마 실험 결과를 Matplotlib Figure로 만드는 기능."""

from pathlib import Path

import numpy as np

from src.reporting.report_images import read_image_for_display


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
    figure, axis = pyplot.subplots(figsize=(15, max(3, 0.5 * len(table_rows) + 1.5)))
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=[
            "세트",
            "방법",
            "사진 쌍",
            "평균 검사 전 매칭 수",
            "평균 검사 후 매칭 수",
            "평균 RANSAC Inlier 수",
            "평균 RANSAC Inlier 비율",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    axis.set_title("방법별 파노라마 합성 매칭 통계", fontsize=16, pad=18)
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
            axis.imshow(read_image_for_display(output_path, max_width=900))
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
            f"{result['set']} / OpenCV Stitcher\n{result['elapsed']:.2f}초",
            fontsize=11,
        )
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle("OpenCV 내장 Stitcher 파노라마 결과", fontsize=16)
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
        (row for row in summary_rows if row["set"] == selected_set),
        key=lambda row: row["method"],
    )
    if not rows:
        raise ValueError(f"시각화할 파노라마 세트가 없습니다: {selected_set}")
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
    selected_set, rows = _select_panorama_visual_rows(summary_rows, set_name)
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
            axis.imshow(read_image_for_display(matching_path, max_width=900))
        else:
            axis.text(
                0.5,
                0.5,
                "필터링 매칭 이미지가 없습니다.",
                ha="center",
                va="center",
                wrap=True,
            )
        axis.set_title(f"{row['method']} / {pair_name} 필터링 매칭")
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle(
        f"방법별 인접 사진 필터링 매칭: {selected_set}",
        fontsize=16,
    )
    return figure


def create_panorama_matching_gallery_figure(
    panorama_result_dir,
    set_name,
    method,
    pyplot,
    *,
    column_count=3,
):
    matching_dir = (
        Path(panorama_result_dir) / str(set_name) / str(method).upper() / "matching"
    )
    matching_paths = sorted(matching_dir.glob("*.png"))
    if not matching_paths:
        raise FileNotFoundError(f"매칭 이미지를 찾을 수 없습니다: {matching_dir}")
    if column_count <= 0:
        raise ValueError("column_count는 양수여야 합니다.")
    row_count = (len(matching_paths) + column_count - 1) // column_count
    figure, axes = pyplot.subplots(
        row_count,
        column_count,
        figsize=(6 * column_count, 5 * row_count),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for axis, matching_path in zip(axes, matching_paths):
        axis.imshow(read_image_for_display(matching_path, max_width=800))
        axis.set_title(matching_path.stem.replace("_", " "), fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes[len(matching_paths) :]:
        axis.axis("off")
    figure.suptitle(
        f"{set_name} / {str(method).upper()} 파노라마 매칭 단계별 결과", fontsize=16
    )
    return figure


def create_panorama_result_comparison_figure(summary_rows, pyplot, *, set_name=None):
    selected_set, rows = _select_panorama_visual_rows(summary_rows, set_name)
    figure, axes = pyplot.subplots(
        2, len(rows), figsize=(7 * len(rows), 9), constrained_layout=True
    )
    axes = np.asarray(axes, dtype=object).reshape(2, len(rows))
    for column, row in enumerate(rows):
        panorama_axis, outline_axis = axes[0, column], axes[1, column]
        panorama_path = row["panorama_path"]
        if panorama_path is not None and Path(panorama_path).is_file():
            panorama_path = Path(panorama_path)
            panorama_axis.imshow(read_image_for_display(panorama_path, max_width=900))
            outline_path = panorama_path.parent / "panorama_outline.jpg"
            if outline_path.is_file():
                outline_axis.imshow(read_image_for_display(outline_path, max_width=900))
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
    figure.suptitle(f"방법별 파노라마 합성 결과: {selected_set}", fontsize=18)
    return figure

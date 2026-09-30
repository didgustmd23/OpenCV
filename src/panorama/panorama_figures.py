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
def create_builtin_stitcher_result_figure(
    result_rows,
    pyplot,
    *,
    direct_summary_rows=None,
    direct_method=None,
):
    if not result_rows:
        raise ValueError("시각화할 OpenCV 내장 Stitcher 결과가 없습니다.")

    # 직접 구현 결과를 함께 받으면 세트별로 아래 행에 배치한다.
    direct_rows_by_set = {}
    if direct_summary_rows is not None:
        if direct_method is None:
            raise ValueError("직접 구현 결과를 표시하려면 direct_method가 필요합니다.")
        selected_method = str(direct_method).upper()
        direct_rows_by_set = {
            row["set"]: row
            for row in direct_summary_rows
            if str(row["method"]).upper() == selected_method
        }

    row_count = 2 if direct_rows_by_set else 1
    figure, axes = pyplot.subplots(
        row_count,
        len(result_rows),
        figsize=(7 * len(result_rows), 4 * row_count),
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(row_count, len(result_rows))

    for column, result in enumerate(result_rows):
        axis = axes[0, column]
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

        direct_row = direct_rows_by_set.get(result["set"])
        if direct_row is None:
            continue

        direct_axis = axes[1, column]
        direct_path = direct_row.get("panorama_path")
        if direct_path is not None and Path(direct_path).is_file():
            direct_axis.imshow(read_image_for_display(direct_path, max_width=900))
        else:
            direct_axis.text(
                0.5,
                0.5,
                "직접 구현 파노라마가 생성되지 않았습니다.",
                ha="center",
                va="center",
                wrap=True,
            )
        direct_axis.set_title(
            f"{result['set']} / 직접 구현 {selected_method}",
            fontsize=11,
        )
        direct_axis.set_xticks([])
        direct_axis.set_yticks([])

    if direct_rows_by_set:
        figure.suptitle(
            f"OpenCV 내장 Stitcher와 직접 구현 {selected_method} 결과 비교",
            fontsize=16,
        )
    else:
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
    stage_order = {"raw": 0, "filtered": 1, "ransac": 2}
    matching_paths = sorted(
        matching_dir.glob("*.png"),
        key=lambda path: (
            path.stem.rpartition("_")[0],
            stage_order.get(path.stem.rpartition("_")[2], len(stage_order)),
            path.name,
        ),
    )
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


# ==========================================
# 파노라마 합성 실패 사례 매칭 Figure 생성
# - failures 하위의 데이터셋·방법 폴더를 자동 탐색
# - 기본값은 기하 검증을 통과한 RANSAC 매칭 이미지만 표시
# - stage를 None으로 지정하면 raw·filtered·ransac 전체 이미지 표시
# ==========================================
def create_panorama_failure_match_figures(
    failure_dir,
    pyplot,
    *,
    stage="ransac",
    column_count=2,
):
    if stage not in {None, "raw", "filtered", "ransac"}:
        raise ValueError("stage는 raw, filtered, ransac 또는 None이어야 합니다.")
    if column_count <= 0:
        raise ValueError("column_count는 양수여야 합니다.")

    failure_dir = Path(failure_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        path
        for path in failure_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    if stage is not None:
        image_paths = [
            path for path in image_paths if path.stem.endswith(f"_{stage}")
        ]
    if not image_paths:
        return []

    grouped_paths = {}
    for image_path in image_paths:
        grouped_paths.setdefault(image_path.parent, []).append(image_path)

    figures = []
    stage_label = "전체 매칭 단계" if stage is None else stage.upper()
    for group_dir, group_paths in grouped_paths.items():
        group_name = group_dir.name
        for method_name in ("ALIKED", "SIFT", "ORB"):
            if group_name.endswith(method_name):
                dataset_name = group_name[: -len(method_name)]
                group_name = f"{dataset_name} / {method_name}"
                break
        row_count = (len(group_paths) + column_count - 1) // column_count
        figure, axes = pyplot.subplots(
            row_count,
            column_count,
            figsize=(7 * column_count, 4.5 * row_count),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()

        for axis, image_path in zip(axes, group_paths):
            axis.imshow(read_image_for_display(image_path, max_width=900))
            axis.set_title(image_path.stem.replace("_", " "), fontsize=11)
            axis.set_xticks([])
            axis.set_yticks([])
        for axis in axes[len(group_paths) :]:
            axis.axis("off")

        figure.suptitle(
            f"파노라마 합성 실패 사례: {group_name} / {stage_label}",
            fontsize=16,
        )
        figures.append(figure)

    return figures


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

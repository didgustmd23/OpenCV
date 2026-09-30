"""최종보고서 객체 검출 정량 결과를 Matplotlib Figure로 만드는 기능."""

import numpy as np

from src.reporting.report_images import read_image_for_display
from src.reporting.report_tables import prepare_detection_summary


# ==========================================
# 객체 검출 정량 비교표 생성
# - 방법·변형 조건별 성공률, 시간, Inlier 비율을 표로 표시
# ==========================================
def create_detection_summary_table_figure(
    summary_rows,
    experiment,
    pyplot,
):
    condition_names, summary_by_key = prepare_detection_summary(
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

    figure, axis = pyplot.subplots(figsize=(12, max(4, 0.42 * len(table_rows) + 1.5)))
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
        record for record in records if record["condition"] == visual_condition
    ]
    if not condition_records:
        raise ValueError(
            f"매칭 통계를 만들 객체 검출 기록이 없습니다: condition={visual_condition}"
        )

    table_rows = []
    for method in experiment.methods:
        method_records = [
            record for record in condition_records if record["method"] == method
        ]
        if not method_records:
            raise ValueError(
                "매칭 통계를 만들 방법별 객체 검출 기록이 없습니다: "
                f"method={method}, condition={visual_condition}"
            )

        detection_records = [
            record for record in method_records if record["is_detection"]
        ]
        detection_count = len(detection_records)
        mean_keypoints = sum(record["keypoints"] for record in method_records) / len(
            method_records
        )

        if detection_count:
            mean_match_count = (
                sum(record["match_count"] for record in detection_records)
                / detection_count
            )
            mean_inlier_count = (
                sum(record["inlier_count"] for record in detection_records)
                / detection_count
            )
            mean_inlier_ratio = (
                sum(record["inlier_ratio"] for record in detection_records)
                / detection_count
            )
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
    condition_names, summary_by_key = prepare_detection_summary(
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
            summary_by_key[(method, name)]["mean_elapsed"] for name in condition_names
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

    labels = [experiment.condition_labels[name] for name in condition_names]
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

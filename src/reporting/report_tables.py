"""최종보고서에서 사용하는 Pandas 표 생성 기능을 모은 모듈."""


# ==========================================
# 보고서용 pandas 모듈 불러오기
# - 일반 파이프라인 실행에는 pandas를 요구하지 않음
# - 최종보고서 표 생성 시에만 선택적으로 사용
# ==========================================
def _get_pandas():
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "최종보고서 표를 표시하려면 pandas가 필요합니다. "
            "'pip install pandas'를 실행한 뒤 다시 시도하세요."
        ) from error

    return pd


# ==========================================
# 객체 검출 정량 결과 공통 데이터 준비
# - 조건·방법 조합별 요약 행을 빠르게 조회할 수 있도록 구성
# ==========================================
def prepare_detection_summary(summary_rows, experiment):
    condition_names = [condition["name"] for condition in experiment.conditions]
    summary_by_key = {(row["method"], row["condition"]): row for row in summary_rows}
    missing_keys = [
        (method, condition_name)
        for condition_name in condition_names
        for method in experiment.methods
        if (method, condition_name) not in summary_by_key
    ]

    if missing_keys:
        raise ValueError(f"정량 비교에 필요한 실험 요약 행이 없습니다: {missing_keys}")

    return condition_names, summary_by_key


# ==========================================
# 파노라마 합성 매칭 통계 DataFrame 생성
# - 세트·방법별 평균 매칭 수와 RANSAC Inlier 통계를 표로 반환
# ==========================================
def build_panorama_matching_table(summary_rows):
    pd = _get_pandas()
    columns = [
        "set",
        "method",
        "pair_count",
        "mean_raw_matches",
        "mean_filtered_matches",
        "mean_inliers",
        "mean_inlier_ratio",
    ]
    column_names = {
        "set": "세트",
        "method": "방법",
        "pair_count": "사진 쌍",
        "mean_raw_matches": "평균 검사 전 매칭 수",
        "mean_filtered_matches": "평균 검사 후 매칭 수",
        "mean_inliers": "평균 RANSAC Inlier 수",
        "mean_inlier_ratio": "평균 RANSAC Inlier 비율",
    }

    return (
        pd.DataFrame(summary_rows)
        .loc[:, columns]
        .rename(columns=column_names)
        .sort_values(["세트", "방법"])
        .reset_index(drop=True)
    )


# ==========================================
# 객체 검출 정량 결과 DataFrame 생성
# - 방법·변형 조건별 성공률, 시간, Inlier 비율을 표로 반환
# ==========================================
def build_detection_summary_table(summary_rows, experiment):
    pd = _get_pandas()
    condition_names, summary_by_key = prepare_detection_summary(
        summary_rows,
        experiment,
    )
    table_rows = []

    for condition_name in condition_names:
        for method in experiment.methods:
            row = summary_by_key[(method, condition_name)]
            table_rows.append(
                {
                    "조건": experiment.condition_labels[condition_name],
                    "방법": method,
                    "검출 성공": (f"{row['detection_count']}/{row['total_cases']}"),
                    "검출 성공률": row["success_rate"],
                    "평균 시간(초)": row["mean_elapsed"],
                    "평균 Inlier": row["mean_inlier_count"],
                    "평균 Inlier 비율": row["mean_inlier_ratio"],
                }
            )

    return pd.DataFrame(table_rows)


# ==========================================
# 객체 검출 최종 Tile 매칭 통계 DataFrame 생성
# - 지정 조건에서 성공한 결과만 이용해 매칭 품질을 방법별 집계
# ==========================================
def build_detection_matching_table(
    records,
    experiment,
    visual_condition="baseline",
):
    pd = _get_pandas()
    record_table = pd.DataFrame(records)
    condition_table = record_table.loc[
        record_table["condition"] == visual_condition
    ].copy()

    if condition_table.empty:
        raise ValueError(
            f"매칭 통계를 만들 객체 검출 기록이 없습니다: condition={visual_condition}"
        )

    all_cases = condition_table.groupby("method").agg(
        전체_케이스=("case", "size"),
        검출_성공=("is_detection", "sum"),
        평균_기준_특징점=("keypoints", "mean"),
    )
    detected_cases = condition_table.loc[condition_table["is_detection"]]
    matching_quality = detected_cases.groupby("method").agg(
        평균_RANSAC_전_매칭=("match_count", "mean"),
        평균_Inlier=("inlier_count", "mean"),
        평균_Inlier_비율=("inlier_ratio", "mean"),
    )
    table = (
        all_cases.join(matching_quality)
        .reindex(experiment.methods)
        .reset_index(names="방법")
        .rename(
            columns={
                "전체_케이스": "전체 케이스",
                "검출_성공": "검출 성공",
                "평균_기준_특징점": "평균 기준 특징점",
                "평균_RANSAC_전_매칭": "평균 RANSAC 전 매칭",
                "평균_Inlier": "평균 Inlier",
                "평균_Inlier_비율": "평균 Inlier 비율",
            }
        )
    )
    table["검출 성공"] = (
        table["검출 성공"].astype(int).astype(str)
        + "/"
        + table["전체 케이스"].astype(int).astype(str)
    )

    return table.drop(columns="전체 케이스")


# ==========================================
# 최종보고서 DataFrame 표시 형식 적용
# - pandas Styler를 반환해 노트북에서 display로 출력
# ==========================================
def style_report_table(table, number_formats):
    return table.style.format(number_formats)

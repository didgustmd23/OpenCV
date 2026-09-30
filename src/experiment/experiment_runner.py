from dataclasses import dataclass
from pathlib import Path

from src.common.project_paths import (
    MATCHING_RESULT_DIR,
    OBJECT_DETECTION_RESULT_DIR,
)
from src.common.settings import OBJECT_DETECTION_CONFIG
from src.detection.detection_experiments import (
    run_batch_with_options,
)


# ==========================================
# 객체 검출 실험 환경 설정
# - 노트북과 일괄 실험에서 동일한 타일·방법·결과 경로를 사용
# - 기본 탐색 값은 config.json의 object_detection 영역을 재사용
# ==========================================
@dataclass(frozen=True)
class ExperimentSettings:
    panorama_max_side: int
    detection_max_side: int
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
# 객체 검출 1회 실행 옵션
# - run_batch()에서 모든 케이스에 공통으로 적용되는 값을 묶는다.
# - 내부 함수 사이에 긴 인자 목록을 반복 전달하지 않도록 한다.
# ==========================================
@dataclass(frozen=True)
class DetectionRunOptions:
    tile_size: tuple
    coarse_stride: tuple
    fine_stride: tuple
    coarse_top_k: int
    early_stop_inliers: int
    result_root: Path
    matching_root: Path
    detection_max_side: int


# ==========================================
# 객체 검출 실험 환경 생성
# - 노트북에서 직접 선언하던 공통 환경값을 한 곳에서 관리
# ==========================================
def get_experiment_settings():
    return ExperimentSettings(
        panorama_max_side=2048,
        detection_max_side=OBJECT_DETECTION_CONFIG["detection_max_side"],
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
# 파노라마 매칭 단계별 이미지 Figure 생성
# - raw / filtered / ransac 매칭 이미지를 한 화면에 표시
# ==========================================
# ==========================================
# 객체 검출 일괄 실행 호환 함수
# - 기존 노트북의 인자 형식을 유지한다.
# - 공통 실행 옵션은 DetectionRunOptions로 변환한 뒤 내부 함수에 전달한다.
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
    detection_max_side,
):
    options = DetectionRunOptions(
        tile_size=tuple(tile_size),
        coarse_stride=tuple(coarse_stride),
        fine_stride=tuple(fine_stride),
        coarse_top_k=coarse_top_k,
        early_stop_inliers=early_stop_inliers,
        result_root=Path(result_root),
        matching_root=Path(matching_root),
        detection_max_side=detection_max_side,
    )

    return run_batch_with_options(
        test_cases,
        methods,
        conditions,
        options,
    )

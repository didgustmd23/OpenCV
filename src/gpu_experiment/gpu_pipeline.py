"""GPU 객체 검출의 실제 실행 흐름.

벤치마크와 분리해 기준 물체·파노라마 입력부터 결과 이미지·JSON 저장까지
담당한다. 기존 OpenCV CPU 파이프라인은 호출하지 않는다.
"""

import json
from pathlib import Path
from time import perf_counter

import cv2
import torch

from src.gpu_experiment.gpu_detector import GPUObjectDetector
from src.gpu_experiment.gpu_visualization import save_final_visuals


# ==========================================
# 공용 입력 이미지 로드
# ==========================================
def load_detection_images(reference_path, panorama_path):
    """기준 물체와 파노라마 이미지를 OpenCV BGR 배열로 불러온다."""
    reference_path = Path(reference_path)
    panorama_path = Path(panorama_path)
    reference_image = cv2.imread(str(reference_path))
    panorama_image = cv2.imread(str(panorama_path))

    if reference_image is None:
        raise FileNotFoundError(
            f"기준 물체 이미지를 불러올 수 없습니다: {reference_path}"
        )
    if panorama_image is None:
        raise FileNotFoundError(
            f"파노라마 이미지를 불러올 수 없습니다: {panorama_path}"
        )
    return reference_image, panorama_image


# ==========================================
# 데이터 세트 자동 탐색
# - data/objects/setNN/target.jpg
# - results/panorama/setNN/ALIKED/panorama.jpg
# ==========================================
def discover_detection_cases(objects_dir, panorama_root):
    """번호가 같은 기준 물체와 ALIKED 파노라마 경로를 연결한다."""
    objects_dir = Path(objects_dir)
    panorama_root = Path(panorama_root)
    cases = []

    for target_dir in sorted(objects_dir.glob("set*")):
        if not target_dir.is_dir():
            continue

        case_name = target_dir.name.lower()
        reference_path = target_dir / "target.jpg"
        panorama_path = (
            panorama_root / case_name / "ALIKED" / "panorama.jpg"
        )
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"기준 물체 이미지를 찾을 수 없습니다: {reference_path}"
            )
        if not panorama_path.is_file():
            raise FileNotFoundError(
                f"ALIKED 파노라마를 찾을 수 없습니다: {panorama_path}"
            )
        cases.append((case_name, reference_path, panorama_path))

    if not cases:
        raise FileNotFoundError(
            f"객체 검출 세트를 찾을 수 없습니다: {objects_dir}"
        )
    return cases


# ==========================================
# 단일 GPU 객체 검출
# - detector를 전달하면 여러 실행에서 가중치를 재사용
# - 결과 시각화는 검출 시간 측정 뒤 별도로 저장
# ==========================================
def run_detection(
    reference_path,
    panorama_path,
    *,
    detector=None,
    detection_max_side=3072,
    visual_output_dir=None,
):
    """한 쌍의 입력에서 GPU 객체 검출을 실행하고 저장용 결과를 반환한다."""
    reference_path = Path(reference_path)
    panorama_path = Path(panorama_path)
    reference_image, panorama_image = load_detection_images(
        reference_path,
        panorama_path,
    )
    # 단일 실행일 때만 검출기를 생성한다. 일괄 실행은 하나를 전달해
    # 모델 가중치 로드와 CUDA 메모리 할당을 반복하지 않는다.
    detector = detector or GPUObjectDetector()
    detection_result = detector.detect(
        reference_image,
        panorama_image,
        detection_max_side=detection_max_side,
    )
    record = {
        "reference_path": str(reference_path),
        "panorama_path": str(panorama_path),
        **summarize_detection(detection_result),
    }

    if visual_output_dir is not None:
        record["visuals"] = save_final_visuals(
            detector,
            reference_image,
            panorama_image,
            detection_result,
            visual_output_dir,
        )
    return record


# ==========================================
# 여러 세트 GPU 객체 검출
# - 실제 세트마다 한 번씩만 검출
# - 워밍업용 중복 검출은 수행하지 않음
# ==========================================
def run_detection_batch(
    cases,
    *,
    tile_size=(512, 512),
    coarse_stride=(512, 512),
    fine_stride=(256, 256),
    coarse_top_k=3,
    early_stop_inliers=50,
    detection_max_side=3072,
    visual_root=None,
):
    """여러 세트의 실제 GPU 객체 검출 결과를 하나로 반환한다."""
    if not cases:
        raise ValueError("객체 검출에 사용할 세트가 없습니다.")

    initialization_start = perf_counter()
    detector = GPUObjectDetector(
        tile_size=tile_size,
        coarse_stride=coarse_stride,
        fine_stride=fine_stride,
        coarse_top_k=coarse_top_k,
        early_stop_inliers=early_stop_inliers,
    )
    initialization_elapsed = perf_counter() - initialization_start

    results = []
    for case_name, reference_path, panorama_path in cases:
        record = run_detection(
            reference_path,
            panorama_path,
            detector=detector,
            detection_max_side=detection_max_side,
            visual_output_dir=(
                Path(visual_root) / case_name / "baseline"
                if visual_root is not None
                else None
            ),
        )
        record["case"] = case_name
        results.append(record)

    return {
        "mode": "gpu_object_detection",
        "device": torch.cuda.get_device_name(0),
        "model_initialization_elapsed": initialization_elapsed,
        "settings": {
            "tile_size": list(tile_size),
            "coarse_stride": list(coarse_stride),
            "fine_stride": list(fine_stride),
            "coarse_top_k": coarse_top_k,
            "early_stop_inliers": early_stop_inliers,
            "detection_max_side": detection_max_side,
        },
        "cases": results,
    }


# ==========================================
# 검출 결과 요약과 JSON 저장
# ==========================================
def summarize_detection(detection_result):
    """GPU 검출 결과를 JSON 직렬화 가능한 기본 자료형으로 변환한다."""
    best = detection_result.best
    return {
        "is_detection": detection_result.is_detection,
        "elapsed": detection_result.stats["elapsed"],
        "match_count": best.match_count if best is not None else None,
        "inlier_count": best.inlier_count if best is not None else None,
        "inlier_ratio": best.inlier_ratio if best is not None else None,
        "tile": (
            [best.x, best.y, best.width, best.height]
            if best is not None
            else None
        ),
        "stats": detection_result.stats,
    }


def save_detection_result(result, output_path):
    """검출 결과를 UTF-8 JSON 파일로 저장하고 경로를 반환한다."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return output_path

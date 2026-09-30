"""OpenCV CPU ALIKED와 PyTorch GPU ALIKED·LightGlue 비교 도구."""

import json
from pathlib import Path
from time import perf_counter

import torch

from src.gpu_experiment.gpu_detector import GPUObjectDetector
from src.gpu_experiment.gpu_pipeline import load_detection_images
from src.gpu_experiment.gpu_visualization import save_final_visuals
from src.object_finder import (
    build_object_detection_result,
    run_object_detection_scan,
)


# ==========================================
# OpenCV CPU ALIKED 검출 실행
# - 특징점 매칭 이미지 저장을 끄고 순수 검출 시간만 비교
# ==========================================
def run_cpu_aliked(
    reference_image,
    panorama_image,
    reference_path,
    *,
    tile_size=(512, 512),
    coarse_stride=(512, 512),
    fine_stride=(256, 256),
    coarse_top_k=3,
    early_stop_inliers=50,
    detection_max_side=3072,
):
    """기존 OpenCV ALIKED 객체 검출을 실행해 비교용 결과를 반환한다."""
    # 기존 소스의 공통 탐색 함수를 그대로 사용한다.
    # save_feature_matches=False로 두어 이미지 저장 I/O가 시간 비교에 섞이지 않게 한다.
    scan_data = run_object_detection_scan(
        ref_img=reference_image,
        panorama_img=panorama_image,
        reference_path=reference_path,
        method="ALIKED",
        tile_size=tile_size,
        coarse_stride=coarse_stride,
        fine_stride=fine_stride,
        coarse_top_k=coarse_top_k,
        early_stop_inliers=early_stop_inliers,
        detection_max_side=detection_max_side,
        save_feature_matches=False,
    )
    # CPU의 Polygon 복원과 검출 성공 기준도 기존 코드와 동일하게 적용한다.
    _, record = build_object_detection_result(
        ref_img=reference_image,
        panorama_img=panorama_image,
        scan_data=scan_data,
    )
    return _cpu_summary(record)


# ==========================================
# GPU 검출 실행
# - 모델 생성, 첫 실행, 워밍업 후 실행 시간을 구분해 기록
# ==========================================
def run_gpu_aliked(
    reference_image,
    panorama_image,
    *,
    tile_size=(512, 512),
    coarse_stride=(512, 512),
    fine_stride=(256, 256),
    coarse_top_k=3,
    early_stop_inliers=50,
    detection_max_side=3072,
    visual_output_dir=None,
):
    """GPU ALIKED·LightGlue 검출의 초기화·첫 실행·워밍업 결과를 반환한다."""
    # 모델 가중치 로드와 GPU 메모리 할당 시간은 검출 시간과 분리해 기록한다.
    initialization_start = perf_counter()
    detector = GPUObjectDetector(
        tile_size=tile_size,
        coarse_stride=coarse_stride,
        fine_stride=fine_stride,
        coarse_top_k=coarse_top_k,
        early_stop_inliers=early_stop_inliers,
    )
    initialization_elapsed = perf_counter() - initialization_start

    # 첫 실행에는 CUDA 커널 초기화·메모리 allocator 준비 비용이 포함될 수 있다.
    first_result = detector.detect(
        reference_image,
        panorama_image,
        detection_max_side=detection_max_side,
    )
    # 동일 객체를 바로 다시 실행해 반복 실험에서 사용할 워밍업 시간을 측정한다.
    warm_result = detector.detect(
        reference_image,
        panorama_image,
        detection_max_side=detection_max_side,
    )

    result = {
        "device": torch.cuda.get_device_name(0),
        "model_initialization_elapsed": initialization_elapsed,
        "first_run": summarize_detection(first_result),
        "warm_run": summarize_detection(warm_result),
    }

    # 시각화는 워밍업 검출이 끝난 뒤 별도로 생성한다.
    # 이미지 변환·저장 시간은 성능 비교의 elapsed에 포함하지 않는다.
    if visual_output_dir is not None:
        result["visuals"] = save_final_visuals(
            detector,
            reference_image,
            panorama_image,
            warm_result,
            visual_output_dir,
        )
    return result


# ==========================================
# CPU·GPU 공통 조건 벤치마크
# - 기본값은 CPU와 GPU를 모두 실행
# - CPU 실행을 생략하면 GPU 동작만 빠르게 점검 가능
# ==========================================
def run_benchmark(
    reference_path,
    panorama_path,
    *,
    tile_size=(512, 512),
    coarse_stride=(512, 512),
    fine_stride=(256, 256),
    coarse_top_k=3,
    early_stop_inliers=50,
    detection_max_side=3072,
    include_cpu=True,
    visual_output_dir=None,
):
    """동일 입력에서 OpenCV CPU와 PyTorch GPU 검출 결과를 비교한다."""
    reference_image, panorama_image = load_detection_images(
        reference_path,
        panorama_path,
    )
    # 두 구현에 같은 탐색 설정을 전달해야 시간과 매칭 통계를 비교할 수 있다.
    # JSON 저장 호환성을 위해 tuple은 list로 바꾼다.
    settings = {
        "tile_size": list(tile_size),
        "coarse_stride": list(coarse_stride),
        "fine_stride": list(fine_stride),
        "coarse_top_k": coarse_top_k,
        "early_stop_inliers": early_stop_inliers,
        "detection_max_side": detection_max_side,
    }
    result = {
        "reference_path": str(reference_path),
        "panorama_path": str(panorama_path),
        "settings": settings,
        "gpu": run_gpu_aliked(
            reference_image,
            panorama_image,
            visual_output_dir=visual_output_dir,
            **settings,
        ),
        "cpu": None,
        "gpu_warm_speedup": None,
    }

    # CPU 비교는 오래 걸리므로 GPU 기능만 점검할 때는 생략할 수 있다.
    if include_cpu:
        result["cpu"] = run_cpu_aliked(
            reference_image,
            panorama_image,
            reference_path,
            **settings,
        )
        gpu_elapsed = result["gpu"]["warm_run"]["elapsed"]
        # speedup은 초기화 비용이 빠진 GPU 워밍업 시간을 기준으로 계산한다.
        if gpu_elapsed > 0:
            result["gpu_warm_speedup"] = result["cpu"]["elapsed"] / gpu_elapsed

    return result


# ==========================================
# 벤치마크 JSON 저장
# ==========================================
def save_benchmark(result, output_path):
    """벤치마크 결과를 GPU 실험 전용 JSON 파일로 저장한다."""
    output_path = Path(output_path)
    # 결과 폴더가 없으면 함께 생성한다. 기존 CPU results 폴더와 경로를 분리한다.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return output_path


def format_benchmark_summary(result):
    """노트북 또는 콘솔 출력용 핵심 비교 문장을 반환한다."""
    # JSON 전체를 열지 않아도 콘솔에서 핵심 시간만 확인할 수 있는 요약이다.
    gpu_warm = result["gpu"]["warm_run"]
    lines = [
        "GPU device: " + result["gpu"]["device"],
        "GPU first run: {:.3f}s".format(result["gpu"]["first_run"]["elapsed"]),
        "GPU warm run: {:.3f}s".format(gpu_warm["elapsed"]),
        "GPU warm detection: {}".format(gpu_warm["is_detection"]),
    ]
    if result["cpu"] is not None:
        lines.append("CPU run: {:.3f}s".format(result["cpu"]["elapsed"]))
        lines.append(
            "CPU / GPU warm speedup: {:.1f}x".format(
                result["gpu_warm_speedup"]
            )
        )
    return "\n".join(lines)


def _cpu_summary(record):
    # CPU/GPU JSON 구조를 맞춰 후속 표 생성 코드에서 같은 열을 사용할 수 있게 한다.
    return {
        "is_detection": record["is_detection"],
        "elapsed": record["elapsed"],
        "match_count": record.get("match_count"),
        "inlier_count": record.get("inlier_count"),
        "inlier_ratio": record.get("inlier_ratio"),
        "tile": record.get("tile"),
        "stats": {
            "coarse_checked": record["coarse_checked"],
            "fine_checked": record["fine_checked"],
            "total_skipped_black": record["total_skipped_black"],
        },
    }



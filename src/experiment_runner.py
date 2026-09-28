import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import src.object_finder as object_finder
from src.feature_match import detect_features
from src.object_finder import get_global_corners
from src.tile_scanner import scan_scene


# ==========================================
# 파일명 정렬용 보조 함수
# - 숫자 접미사는 문자열 순서가 아닌 숫자 순서로 정렬
# - 예: 2, 10 순서를 10, 2로 잘못 정렬하지 않도록 처리
# ==========================================
def _numeric_sort_key(value):
    return (0, int(value)) if value.isdigit() else (1, value.lower())


# ==========================================
# 접두사 기준 이미지 파일 수집
# - 지정한 폴더에서 *.jpg 파일 탐색
# - target1.jpg -> {"1": Path(...)} 형태로 반환
# - panorama1.jpg도 같은 방식으로 처리
# ==========================================
def _collect_images(directory, prefix, recursive=False):
    directory = Path(directory)
    images = {}

    paths = directory.rglob("*.jpg") if recursive else directory.glob("*.jpg")

    for path in paths:
        if path.stem.lower().startswith(prefix):
            suffix = path.stem[len(prefix):]
            # targetN.jpg, panoramaN.jpg 형식의 숫자 번호만 테스트 케이스로 사용
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
# 자동 테스트 세트 생성
# - targetN.jpg와 방법별 panoramaN.jpg를 번호 기준으로 연결
# - 짝이 없는 파일은 조용히 제외하지 않고 오류로 안내
# - 반환: (case_name, reference_path, {method: panorama_path}) 목록
# ==========================================
def discover_test_cases(target_dir, panorama_result_dir, methods):
    targets = _collect_images(target_dir, "target")
    panorama_result_dir = Path(panorama_result_dir)
    panorama_paths = {}

    # ==========================================
    # 방법별 파노라마 결과 수집
    # - SIFT / ORB / ALIKED가 각각 합성한 panoramaN.jpg 사용
    # - 객체 검출도 같은 방법으로 실행해 전체 파이프라인 비교
    # ==========================================
    for method in methods:
        method_name = str(method).upper()
        panoramas = _collect_images(
            panorama_result_dir / method_name,
            "panorama",
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
            "targetN.jpg 파일을 찾을 수 없습니다."
        )

    return [
        (
            f"target{case_id}_panorama{case_id}",
            str(targets[case_id]),
            {
                method: str(panorama_paths[method][case_id])
                for method in panorama_paths
            },
        )
        for case_id in case_ids
    ]


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

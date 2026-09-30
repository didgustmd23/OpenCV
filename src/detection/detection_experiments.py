"""객체 검출 실험의 단일 케이스 및 일괄 실행 기능."""

from pathlib import Path

import cv2

from src.detection.detection_transforms import transform_reference
from src.detection.object_finder import (
    build_object_detection_result,
    run_object_detection_scan,
)


def detect_and_save(case_name, ref_path, scene_path, method, *, condition, options):
    condition_name = condition["name"]
    save_visuals = condition.get("save_visuals", condition_name == "baseline")
    ref_img, scene_img = cv2.imread(str(ref_path)), cv2.imread(str(scene_path))
    if ref_img is None:
        raise FileNotFoundError(f"참조 이미지를 불러올 수 없습니다: {ref_path}")
    if scene_img is None:
        raise FileNotFoundError(f"파노라마 이미지를 불러올 수 없습니다: {scene_path}")
    ref_img = transform_reference(
        ref_img,
        rotation_deg=condition.get("rotation_deg", 0.0),
        scale_factor=condition.get("scale_factor", 1.0),
        brightness_factor=condition.get("brightness_factor", 1.0),
    )
    scan_data = run_object_detection_scan(
        ref_img=ref_img,
        panorama_img=scene_img,
        reference_path=ref_path,
        method=method,
        tile_size=options.tile_size,
        coarse_stride=options.coarse_stride,
        fine_stride=options.fine_stride,
        coarse_top_k=options.coarse_top_k,
        early_stop_inliers=options.early_stop_inliers,
        detection_max_side=options.detection_max_side,
        feature_match_dir=options.matching_root,
        save_feature_matches=save_visuals,
    )
    result_image, detection = build_object_detection_result(
        ref_img=ref_img, panorama_img=scene_img, scan_data=scan_data
    )
    method, best = scan_data["method"], scan_data["scan_result"]["best"]
    case_dir = Path(options.result_root) / case_name / condition_name
    record = {
        "case": case_name,
        "method": method,
        "panorama_path": str(scene_path),
        "condition": condition_name,
        "rotation_deg": condition.get("rotation_deg", 0.0),
        "scale_factor": condition.get("scale_factor", 1.0),
        "brightness_factor": condition.get("brightness_factor", 1.0),
        "keypoints": detection["reference_keypoints"],
        "detection_scale": detection["detection_scale"],
        "detection_size": detection["detection_size"],
        "coarse_checked": detection["coarse_checked"],
        "coarse_skipped_black": detection["coarse_skipped_black"],
        "fine_checked": detection["fine_checked"],
        "fine_skipped_black": detection["fine_skipped_black"],
        "total_skipped_black": detection["total_skipped_black"],
        "elapsed": detection["elapsed"],
        "is_detection": detection["is_detection"],
        "result_path": detection["result_path"],
        "match_image_path": None,
    }
    for key in (
        "tile",
        "match_count",
        "inlier_count",
        "inlier_ratio",
        "best_tile",
        "best_match_count",
        "best_inlier_count",
        "best_inlier_ratio",
        "minimum_inliers",
    ):
        if key in detection:
            record[key] = detection[key]
    if save_visuals:
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / f"{method}_result.png"
        if not cv2.imwrite(str(result_path), result_image):
            raise OSError(f"검출 결과를 저장할 수 없습니다: {result_path}")
        record["result_path"] = str(result_path)
        if best is not None:
            source = (
                Path(options.matching_root)
                / method
                / f"fine_{best['index']}_matches.png"
            )
            target = case_dir / f"{method}_keypoint_matches.png"
            match_image = cv2.imread(str(source))
            if match_image is not None:
                if not cv2.imwrite(str(target), match_image):
                    raise OSError(f"매칭 이미지를 저장할 수 없습니다: {target}")
                record["match_image_path"] = str(target)
    return record


def run_batch_with_options(test_cases, methods, conditions, options):
    records = []
    for condition in conditions:
        for case_name, ref_path, panorama_paths in test_cases:
            for method in methods:
                method_name = str(method).upper()
                records.append(
                    detect_and_save(
                        case_name,
                        ref_path,
                        panorama_paths[method_name],
                        method_name,
                        condition=condition,
                        options=options,
                    )
                )
    return records

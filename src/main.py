import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import src.object_finder as object_finder
from src.feature_match import detect_features
from src.image_io import read_image, save_image
from src.logger import logger
from src.project_paths import (
    MATCHING_RESULT_DIR,
    OBJECT_DETECTION_RESULT_DIR,
    project_path,
)
from src.settings import (
    LOGGING_CONFIG,
    OBJECT_DETECTION_CONFIG,
    PIPELINE_CONFIG,
)
from src.stitcher import SUPPORTED_METHODS, stitch_panorama
from src.tile_scanner import scan_scene


# ==========================================
# OpenCV 로그 수준 설정
# - config.json의 logging.opencv_level 값 적용
# ==========================================
cv2_log_level = getattr(
    cv2.utils.logging,
    f"LOG_LEVEL_{LOGGING_CONFIG['opencv_level']}",
)
cv2.utils.logging.setLogLevel(cv2_log_level)


# ==========================================
# 파이프라인 결과 폴더 생성
# - 파노라마 입력 세트와 검출 방법별로 결과 분리
# - 예: results/object_detection/set01/SIFT/
# ==========================================
def get_output_dir(input_dir, method):
    input_dir = project_path(input_dir)

    return (
        OBJECT_DETECTION_RESULT_DIR
        / input_dir.name
        / method.upper()
    )


# ==========================================
# 최종 선택 Tile 객체 검출
# - Coarse-to-Fine 탐색으로 가장 유력한 Tile 선택
# - Coarse / Fine 모든 Tile의 매칭 이미지를 저장
# - 최종 후보 Tile은 final_matches.png로 한 번 더 저장
# ==========================================
def detect_object_in_panorama(
    ref_img,
    panorama_img,
    reference_path,
    method,
    output_dir,
):
    tile_size = tuple(OBJECT_DETECTION_CONFIG["tile_size"])
    coarse_stride = tuple(OBJECT_DETECTION_CONFIG["coarse_stride"])
    fine_stride = tuple(OBJECT_DETECTION_CONFIG["fine_stride"])

    # ==========================================
    # Reference 특징점 사전 계산
    # - 모든 Tile에서 재사용해 반복 연산 방지
    # ==========================================
    keypoints_ref, descriptors_ref = detect_features(
        ref_img,
        method=method,
    )
    ref_features = (keypoints_ref, descriptors_ref)

    logger.info(
        "객체 검출 시작: method=%s, reference_keypoints=%d",
        method,
        len(keypoints_ref),
    )

    # ==========================================
    # Coarse-to-Fine Tile 탐색
    # - 모든 Tile의 매칭 이미지를 세트·방법별 폴더에 저장
    # - 예: results/matching/keypoint_matches/set01/SIFT/
    # ==========================================
    previous_match_dir = object_finder.FEATURE_MATCH_DIR
    matching_root = MATCHING_RESULT_DIR / output_dir.parent.name
    object_finder.FEATURE_MATCH_DIR = str(matching_root)

    try:
        scan_result = scan_scene(
            ref_img=ref_img,
            sce_img=panorama_img,
            ref_features=ref_features,
            tile_size=tile_size,
            coarse_stride=coarse_stride,
            fine_stride=fine_stride,
            coarse_top_k=OBJECT_DETECTION_CONFIG["coarse_top_k"],
            early_stop_inliers=(
                OBJECT_DETECTION_CONFIG["early_stop_inliers"]
            ),
            ref_name=str(reference_path),
            method=method,
        )
    finally:
        object_finder.FEATURE_MATCH_DIR = previous_match_dir

    best = scan_result["best"]
    result_image = panorama_img.copy()

    record = {
        "method": method,
        "is_detection": False,
        "reference_keypoints": len(keypoints_ref),
        "coarse_checked": scan_result["stats"]["coarse_checked"],
        "fine_checked": scan_result["stats"]["fine_checked"],
        "elapsed": scan_result["stats"]["elapsed"],
        "result_path": None,
        "matching_path": None,
    }

    # ==========================================
    # 최종 후보 Tile 매칭 이미지 저장
    # - 검출 성공 여부와 관계없이 최고 후보의 매칭 이미지를 저장
    # - 결과 위치: results/matching/keypoint_matches/세트명/방법/
    # ==========================================
    if best is None:
        logger.warning("객체 검출 실패: 유효한 Tile 후보가 없습니다.")
        return result_image, record

    best_result = best["result"]
    tile_x = best["x"]
    tile_y = best["y"]
    scene_height, scene_width = panorama_img.shape[:2]
    tile_width, tile_height = tile_size
    tile_x2 = min(tile_x + tile_width, scene_width)
    tile_y2 = min(tile_y + tile_height, scene_height)
    best_tile = panorama_img[tile_y:tile_y2, tile_x:tile_x2]

    previous_save_matches = object_finder.SAVE_FEATURE_MATCHES
    previous_match_dir = object_finder.FEATURE_MATCH_DIR
    object_finder.SAVE_FEATURE_MATCHES = object_finder.SAVE_FINAL_MATCHES
    object_finder.FEATURE_MATCH_DIR = str(matching_root)

    try:
        final_result = object_finder.find_object(
            ref_img,
            best_tile,
            method=method,
            ref_name=str(reference_path),
            sce_name="final",
            ref_features=ref_features,
            coarse_mode=False,
        )
    finally:
        object_finder.SAVE_FEATURE_MATCHES = previous_save_matches
        object_finder.FEATURE_MATCH_DIR = previous_match_dir

    matching_path = (
        matching_root
        / method
        / "final_matches.png"
    )
    record["matching_path"] = (
        str(matching_path)
        if object_finder.SAVE_FINAL_MATCHES and matching_path.is_file()
        else None
    )

    # ==========================================
    # 검출 실패 결과 기록
    # - 최종 후보의 매칭 이미지는 이미 저장되어 육안 분석 가능
    # ==========================================
    if (
        not best_result.get("is_detection", False)
        or best_result.get("H") is None
    ):
        record.update(
            {
                "best_tile": [tile_x, tile_y],
                "best_match_count": best_result["match_count"],
                "best_inlier_count": best_result["inlier_count"],
                "best_inlier_ratio": best_result["inlier_ratio"],
                "minimum_inliers": object_finder.MIN_INLIERS,
            }
        )
        logger.warning(
            "객체 검출 실패: best_tile=(%d, %d), "
            "matches=%d, inliers=%d, min_required=%d",
            tile_x,
            tile_y,
            best_result["match_count"],
            best_result["inlier_count"],
            object_finder.MIN_INLIERS,
        )
        return result_image, record

    # 재검출이 실패하면 탐색 단계의 결과로 Polygon은 계속 표시
    detection_result = final_result or best_result

    corner_result = object_finder.get_global_corners(
        ref_img=ref_img,
        H_tile=detection_result["H"],
        tile_origin=(tile_x, tile_y),
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
            "tile": [tile_x, tile_y, tile_x2, tile_y2],
            "match_count": detection_result["match_count"],
            "inlier_count": detection_result["inlier_count"],
            "inlier_ratio": detection_result["inlier_ratio"],
        }
    )

    logger.info(
        "객체 검출 완료: tile=%s, matches=%d, inliers=%d, ratio=%.3f",
        record["tile"],
        record["match_count"],
        record["inlier_count"],
        record["inlier_ratio"],
    )

    return result_image, record


# ==========================================
# 파노라마 합성 → 객체 검출 전체 실행
# - 입력 연속 사진으로 파노라마를 생성
# - 메모리에 있는 완성 파노라마에서 기준 물체를 바로 검출
# - 파노라마·검출 결과·매칭 결과·요약 JSON 저장
# ==========================================
def run_pipeline(
    input_dir,
    reference_path,
    method,
    *,
    max_side=None,
):
    method = method.upper()

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"지원하지 않는 방법입니다: {method}")

    reference_path = project_path(reference_path)
    ref_img = read_image(reference_path)
    output_dir = get_output_dir(input_dir, method)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1단계: 파노라마 합성
    # - stitcher.py의 공통 합성 함수를 사용
    # ==========================================
    stitch_options = {}

    if max_side is not None:
        stitch_options["max_side"] = max_side

    panorama_result = stitch_panorama(
        input_dir=input_dir,
        method=method,
        **stitch_options,
    )
    panorama_img = panorama_result["panorama"]

    # ==========================================
    # 2단계: 완성 파노라마 객체 검출
    # ==========================================
    detection_image, detection_record = detect_object_in_panorama(
        ref_img=ref_img,
        panorama_img=panorama_img,
        reference_path=reference_path,
        method=method,
        output_dir=output_dir,
    )

    result_path = save_image(
        output_dir / "detection_result.png",
        detection_image,
    )
    detection_record["result_path"] = str(result_path)

    # ==========================================
    # 파이프라인 요약 저장
    # - 보고서 작성과 실행 결과 추적에 사용
    # ==========================================
    summary = {
        "input_dir": str(project_path(input_dir)),
        "reference_path": str(reference_path),
        "method": method,
        "panorama_path": str(panorama_result["panorama_path"]),
        "panorama_outline_path": str(panorama_result["outline_path"]),
        "panorama_matching_stats": str(panorama_result["stats_path"]),
        "detection": detection_record,
    }
    summary_path = output_dir / "pipeline_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    summary["summary_path"] = str(summary_path)
    return summary


# ==========================================
# 명령줄 인자 설정
# - --input: 연속 사진 폴더
# - --reference: 파노라마에서 찾을 기준 물체 이미지
# - --method: 합성과 검출에 공통 적용할 특징점 방법
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="파노라마 합성과 객체 검출 전체 실행",
    )
    parser.add_argument(
        "--input",
        default=PIPELINE_CONFIG["default_input_dir"],
        help="연속 사진이 들어 있는 폴더",
    )
    parser.add_argument(
        "--reference",
        default=PIPELINE_CONFIG["default_reference_path"],
        help="찾을 기준 물체 이미지 경로",
    )
    parser.add_argument(
        "--method",
        type=str.lower,
        choices=[method.lower() for method in SUPPORTED_METHODS],
        default=OBJECT_DETECTION_CONFIG["default_method"].lower(),
        help="파노라마 합성과 객체 검출에 사용할 방법",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="파노라마 입력 이미지 긴 변의 최대 크기",
    )

    return parser.parse_args()


# ==========================================
# 명령줄 진입점
# ==========================================
def main():
    args = parse_arguments()
    summary = run_pipeline(
        input_dir=args.input,
        reference_path=args.reference,
        method=args.method,
        max_side=args.max_side,
    )

    print("\n파이프라인 실행 완료")
    print(f"파노라마: {summary['panorama_path']}")
    print(f"검출 결과: {summary['detection']['result_path']}")
    print(f"요약: {summary['summary_path']}")


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        FileNotFoundError,
        OSError,
        cv2.error,
        np.linalg.LinAlgError,
    ) as error:
        raise SystemExit(f"오류: {error}")

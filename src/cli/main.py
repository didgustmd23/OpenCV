import argparse

import cv2
import numpy as np

from src.common.image_io import read_image, save_image
from src.common.project_paths import RESULTS_DIR, project_path
from src.common.settings import (
    LOGGING_CONFIG,
    OBJECT_DETECTION_CONFIG,
    PIPELINE_CONFIG,
)
from src.detection.object_finder import detect_object_in_panorama
from src.panorama.stitcher import SUPPORTED_METHODS, stitch_panorama

# ==========================================
# OpenCV 로그 출력 설정
# - config.json의 logging.opencv_level 값을 적용
# ==========================================
cv2_log_level = getattr(
    cv2.utils.logging,
    f"LOG_LEVEL_{LOGGING_CONFIG['opencv_level']}",
)
cv2.utils.logging.setLogLevel(cv2_log_level)


# ==========================================
# 파노라마 합성 단위 테스트
# - 전체 파이프라인이 아닌 stitcher.py만 개별 확인
# - 결과는 results/debug/panorama 아래에 저장
# ==========================================
def run_stitch_debug(input_dir, method, max_side=None):
    return stitch_panorama(
        input_dir=input_dir,
        method=str(method).upper(),
        max_side=max_side,
        output_root=RESULTS_DIR / "debug" / "panorama",
    )


# ==========================================
# 객체 검출 단위 테스트
# - 이미 생성된 파노라마와 기준 물체로 검출 기능만 확인
# - 결과는 results/debug 아래에 저장
# ==========================================
def run_detection_debug(
    scene_path,
    reference_path,
    method,
    output_path=None,
):
    scene_path = project_path(scene_path)
    reference_path = project_path(reference_path)
    method = str(method).upper()

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"지원하지 않는 방법입니다: {method}")

    if output_path is None:
        output_path = RESULTS_DIR / "debug" / f"detection_{method}.png"
    else:
        output_path = project_path(output_path)

    reference_image = read_image(reference_path)
    scene_image = read_image(scene_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    result_image, record = detect_object_in_panorama(
        ref_img=reference_image,
        panorama_img=scene_image,
        reference_path=reference_path,
        method=method,
        output_dir=output_dir,
    )
    record["result_path"] = str(save_image(output_path, result_image))

    return record


# ==========================================
# 디버그 명령줄 인자 설정
# - stitch: 연속 사진 합성 기능만 테스트
# - detect: 완성 파노라마 객체 검출 기능만 테스트
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="파노라마·객체 검출 개별 기능 테스트 및 디버그",
    )
    parser.add_argument(
        "--mode",
        choices=("stitch", "detect"),
        required=True,
        help="테스트할 개별 기능",
    )
    parser.add_argument(
        "--input",
        default=PIPELINE_CONFIG["default_input_dir"],
        help="stitch 모드에서 사용할 연속 사진 폴더",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="detect 모드에서 사용할 완성 파노라마 이미지",
    )
    parser.add_argument(
        "--reference",
        default=PIPELINE_CONFIG["default_reference_path"],
        help="detect 모드에서 찾을 기준 물체 이미지",
    )
    parser.add_argument(
        "--method",
        type=str.lower,
        choices=[method.lower() for method in SUPPORTED_METHODS],
        default=OBJECT_DETECTION_CONFIG["default_method"].lower(),
        help="테스트할 특징점 방법",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="stitch 모드의 입력 이미지 최대 변 길이",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="detect 모드의 검출 결과 저장 경로",
    )

    return parser.parse_args()


# ==========================================
# 디버그 명령줄 진입점
# ==========================================
def main():
    args = parse_arguments()

    if args.mode == "stitch":
        result = run_stitch_debug(
            input_dir=args.input,
            method=args.method,
            max_side=args.max_side,
        )
        print("\n파노라마 합성 디버그 완료")
        print(f"파노라마: {result['panorama_path']}")
        print(f"통계: {result['stats_path']}")
        return

    if args.scene is None:
        raise ValueError("detect 모드에는 --scene 파노라마 이미지 경로가 필요합니다.")

    record = run_detection_debug(
        scene_path=args.scene,
        reference_path=args.reference,
        method=args.method,
        output_path=args.output,
    )
    print("\n객체 검출 디버그 완료")
    print(f"검출 성공: {record['is_detection']}")
    print(f"결과: {record['result_path']}")


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        cv2.error,
        np.linalg.LinAlgError,
    ) as error:
        raise SystemExit(f"오류: {error}")

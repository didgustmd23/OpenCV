import argparse
import json

import cv2
import numpy as np

from src.image_io import read_image, save_image
from src.object_finder import detect_object_in_panorama
from src.project_paths import get_object_detection_output_dir, project_path
from src.settings import (
    LOGGING_CONFIG,
    OBJECT_DETECTION_CONFIG,
    PIPELINE_CONFIG,
)
from src.stitcher import SUPPORTED_METHODS, stitch_panorama


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
# 파노라마 합성 · 객체 검출 전체 실행
# - stitcher.py에서 연속 사진을 파노라마로 합성
# - object_finder.py에서 합성 결과를 대상으로 기준 물체 검출
# - main.py는 모듈을 연결하고 결과를 저장하는 역할만 담당
# ==========================================
def run_pipeline(
    input_dir,
    reference_path,
    method,
    *,
    max_side=None,
    export_path=None,
):
    method = str(method).upper()

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"지원하지 않는 방법입니다: {method}")

    reference_path = project_path(reference_path)
    ref_img = read_image(reference_path)
    output_dir = get_object_detection_output_dir(input_dir, method)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1단계: 파노라마 합성
    # - 필요할 때만 입력 이미지 최대 변 길이와 내보내기 경로를 전달
    # ==========================================
    stitch_options = {}

    if max_side is not None:
        stitch_options["max_side"] = max_side

    if export_path is not None:
        stitch_options["export_path"] = export_path

    panorama_result = stitch_panorama(
        input_dir=input_dir,
        method=method,
        **stitch_options,
    )

    # ==========================================
    # 2단계: 완성 파노라마 객체 검출
    # - 세부 Tile 탐색과 Polygon 계산은 object_finder.py에서 처리
    # ==========================================
    detection_image, detection_record = detect_object_in_panorama(
        ref_img=ref_img,
        panorama_img=panorama_result["panorama"],
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
    # 전체 실행 요약 저장
    # - 노트북 보고서와 결과 추적에 사용할 경로·통계를 JSON으로 기록
    # ==========================================
    summary = {
        "input_dir": str(project_path(input_dir)),
        "reference_path": str(reference_path),
        "method": method,
        "panorama_path": str(panorama_result["panorama_path"]),
        "panorama_outline_path": str(panorama_result["outline_path"]),
        "panorama_matching_stats": str(panorama_result["stats_path"]),
        "panorama_export_path": (
            str(panorama_result["export_path"])
            if panorama_result["export_path"] is not None
            else None
        ),
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
        help="파노라마 입력 이미지 변의 최대 길이",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="완성된 파노라마를 별도로 저장할 경로",
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
        export_path=args.export,
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

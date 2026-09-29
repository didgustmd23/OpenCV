import argparse
import time
from pathlib import Path

import cv2

from src.image_io import list_image_files, read_image, save_image
from src.project_paths import PANORAMA_RESULT_DIR, PANORAMA_SOURCE_DIR
from src.settings import PANORAMA_CONFIG


# ==========================================
# OpenCV 내장 Stitcher 결과 저장 경로
# - 직접 구현한 stitcher.py 결과와 분리해 비교
# - 예: results/panorama/create/set01/panorama.jpg
# ==========================================
CREATE_RESULT_DIR = PANORAMA_RESULT_DIR / "create"


# ==========================================
# OpenCV 내장 Stitcher로 파노라마 생성
# - 현재 프로젝트의 이미지 입출력과 최소 사진 수 설정을 재사용
# - 직접 구현한 stitcher.py와 비교하기 위한 기준 결과 생성
# ==========================================
def create_panorama_from_folder(
    input_dir,
    output_path,
    *,
    max_side=PANORAMA_CONFIG["max_side"],
):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    image_paths = list_image_files(input_dir)
    minimum_images = PANORAMA_CONFIG["min_images"]

    if len(image_paths) < minimum_images:
        return (
            False,
            0.0,
            "이미지 부족 "
            f"(최소 {minimum_images}장 필요, 현재 {len(image_paths)}장)",
        )

    images = [
        read_image(path, max_side=max_side)
        for path in image_paths
    ]

    if not hasattr(cv2, "Stitcher_create"):
        raise RuntimeError(
            "현재 OpenCV 환경에서 Stitcher_create를 찾을 수 없습니다."
        )

    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    start_time = time.perf_counter()
    status, panorama = stitcher.stitch(images)
    elapsed_time = time.perf_counter() - start_time

    if status == cv2.Stitcher_OK:
        saved_path = save_image(output_path, panorama)
        height, width = panorama.shape[:2]
        return (
            True,
            elapsed_time,
            f"성공 ({width}x{height} px, path={saved_path})",
        )

    status_messages = {
        cv2.Stitcher_ERR_NEED_MORE_IMGS:
            "ERR_NEED_MORE_IMGS (특징점 부족 또는 겹침 없음)",
        cv2.Stitcher_ERR_HOMOGRAPHY_ESTIMATION_FAIL:
            "ERR_HOMOGRAPHY_ESTIMATION_FAIL (호모그래피 계산 실패)",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            "ERR_CAMERA_PARAMS_ADJUST_FAIL (카메라 파라미터 조정 실패)",
    }
    return (
        False,
        elapsed_time,
        status_messages.get(status, f"Error Code: {status}"),
    )


# ==========================================
# OpenCV 내장 Stitcher 일괄 실행
# - data/set01, data/set02 ... 형식의 현재 프로젝트 세트 사용
# - 결과는 results/panorama/create/세트명/에 저장
# ==========================================
def run_experiments(
    target_set=None,
    *,
    max_side=PANORAMA_CONFIG["max_side"],
):
    if target_set is not None:
        target_path = PANORAMA_SOURCE_DIR / str(target_set)

        if not target_path.is_dir():
            raise FileNotFoundError(
                "지정한 파노라마 세트 폴더를 찾을 수 없습니다: "
                f"{target_path}"
            )

        set_dirs = [target_path]
    else:
        set_dirs = sorted(
            (
                path
                for path in PANORAMA_SOURCE_DIR.glob("set*")
                if path.is_dir()
            ),
            key=lambda path: path.name,
        )

    if not set_dirs:
        raise FileNotFoundError(
            "실행할 파노라마 세트 폴더를 찾을 수 없습니다: "
            f"{PANORAMA_SOURCE_DIR}"
        )

    summary_results = []

    for input_dir in set_dirs:
        output_path = (
            CREATE_RESULT_DIR
            / input_dir.name
            / "panorama.jpg"
        )
        success, elapsed_time, note = create_panorama_from_folder(
            input_dir=input_dir,
            output_path=output_path,
            max_side=max_side,
        )
        summary_results.append(
            {
                "set": input_dir.name,
                "status": "SUCCESS" if success else "FAILED",
                "elapsed": elapsed_time,
                "output_path": str(output_path) if success else None,
                "note": note,
            }
        )

    return summary_results


# ==========================================
# 내장 Stitcher 명령줄 실행
# - --set을 생략하면 data 아래의 모든 setNN 세트를 실행
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="OpenCV 내장 Stitcher 파노라마 비교 실행",
    )
    parser.add_argument(
        "--set",
        default=None,
        help="실행할 세트 폴더명 (예: set01)",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=PANORAMA_CONFIG["max_side"],
        help="입력 이미지의 최대 변 길이",
    )

    return parser.parse_args()


# ==========================================
# 명령줄 진입점
# ==========================================
def main():
    args = parse_arguments()
    summary_results = run_experiments(
        target_set=args.set,
        max_side=args.max_side,
    )

    print("OpenCV 내장 Stitcher 실행 결과")
    for result in summary_results:
        print(
            f"{result['set']}: {result['status']} | "
            f"{result['elapsed']:.2f}초 | {result['note']}"
        )


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        ValueError,
        cv2.error,
    ) as error:
        raise SystemExit(f"오류: {error}")

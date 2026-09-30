"""GPU ALIKED·LightGlue 객체 검출 실행 진입점."""

import argparse
from pathlib import Path
import sys


# ==========================================
# 파일 직접 실행 지원
# - python src/gpu_experiment/main.py 형태에서는 프로젝트 루트가
#   모듈 탐색 경로에 없으므로 실행 전에 추가
# ==========================================
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gpu_experiment.gpu_pipeline import (
    discover_detection_cases,
    run_detection,
    run_detection_batch,
    save_detection_result,
)


# ==========================================
# 명령줄 인자 구성
# - 기본값은 set01의 실제 GPU 객체 검출
# - 전체 세트는 번호가 같은 target·panorama를 자동 연결
# ==========================================
def parse_arguments():
    project_root = Path(__file__).resolve().parents[2]
    # argparse는 명령줄 옵션을 받아 GPU 객체 검출 함수에 전달한다.
    parser = argparse.ArgumentParser(
        description="GPU ALIKED·LightGlue 객체 검출",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=project_root / "data/objects/set01/target.jpg",
        help="기준 물체 이미지 경로",
    )
    parser.add_argument(
        "--panorama",
        type=Path,
        default=project_root / "results/panorama/set01/ALIKED/panorama.jpg",
        help="검출 대상 파노라마 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="검출 요약 JSON 저장 경로(생략 시 기본 경로 사용)",
    )
    parser.add_argument(
        "--all-sets",
        action="store_true",
        help="set01부터 set05까지 GPU 객체 검출 일괄 실행",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=3072,
        help="검출 전 파노라마 최대 변 길이",
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="최종 Polygon·키포인트 매칭 이미지를 GPU 결과 폴더에 저장",
    )
    parser.add_argument(
        "--visual-output",
        type=Path,
        default=None,
        help="단일 실행의 최종 결과 이미지 저장 폴더(생략 시 세트명 사용)",
    )
    return parser.parse_args()


# ==========================================
# GPU 객체 검출 실행
# ==========================================
def main():
    # 1) 인자 읽기  2) 단일/일괄 GPU 검출  3) JSON 저장  4) 콘솔 요약 출력
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[2]

    if arguments.max_side <= 0:
        raise ValueError("--max-side는 1 이상의 정수여야 합니다.")

    if arguments.all_sets:
        cases = discover_detection_cases(
            project_root / "data/objects",
            project_root / "results/panorama",
        )
        result = run_detection_batch(
            cases,
            detection_max_side=arguments.max_side,
            visual_root=(
                project_root / "results/gpu_experiment/object_detection"
                if arguments.save_visuals
                else None
            ),
        )
        output_path = save_detection_result(
            result,
            arguments.output
            or project_root
            / "results/gpu_experiment/object_detection/batch_summary.json",
        )
        print(f"GPU 일괄 검출 완료: {len(result['cases'])}세트")
        print(f"저장 완료: {output_path}")
        return

    # 직접 지정한 입력도 target.jpg의 상위 폴더명(setNN)을 이용해
    # 결과를 서로 덮어쓰지 않는 세트별 폴더에 저장한다.
    case_name = arguments.reference.parent.name.lower()
    default_output_dir = (
        project_root / "results/gpu_experiment/object_detection" / case_name
    )
    result = run_detection(
        arguments.reference,
        arguments.panorama,
        detection_max_side=arguments.max_side,
        visual_output_dir=(
            (arguments.visual_output or default_output_dir / "baseline")
            if arguments.save_visuals
            else None
        ),
    )
    output_path = save_detection_result(
        result,
        arguments.output
        or default_output_dir / "detection_summary.json",
    )
    print(f"GPU 검출 완료: detection={result['is_detection']}")
    print(
        "매칭/Inlier: "
        f"{result['match_count']} / {result['inlier_count']}, "
        f"시간: {result['elapsed']:.3f}s"
    )
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    main()

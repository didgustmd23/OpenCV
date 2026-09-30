"""기준 물체와 방법별 파노라마 결과로 실험 케이스를 구성한다."""

import re
from pathlib import Path


def _numeric_sort_key(value):
    return (0, int(value)) if value.isdigit() else (1, value.lower())


def _collect_targets(target_dir):
    targets = {}
    for path in Path(target_dir).rglob("target.jpg"):
        match = re.fullmatch(r"set(\d+)", path.parent.name, re.IGNORECASE)
        if match is None:
            raise ValueError(
                f"기준 물체는 set번호/target.jpg 형식이어야 합니다: {path}"
            )
        case_id = str(int(match.group(1)))
        if case_id in targets:
            raise ValueError(f"기준 물체 세트 번호 {case_id}가 중복됩니다.")
        targets[case_id] = path
    if not targets:
        raise FileNotFoundError("setN/target.jpg 파일을 찾을 수 없습니다.")
    return targets


def _collect_panoramas(panorama_root, method):
    panoramas, method_name = {}, str(method).upper()
    panorama_root = Path(panorama_root)
    for path in panorama_root.rglob("panorama.jpg"):
        if (
            path.parent.name.upper() != method_name
            or path.parent.parent == panorama_root
        ):
            continue
        match = re.search(r"(\d+)$", path.parent.parent.name)
        if match is None:
            raise ValueError(
                f"파노라마 세트 폴더명 끝에 번호가 필요합니다: {path.parent.parent}"
            )
        case_id = str(int(match.group()))
        if case_id in panoramas:
            raise ValueError(f"{method_name}의 세트 번호 {case_id}가 중복됩니다.")
        panoramas[case_id] = path
    return panoramas


def discover_test_cases(target_dir, panorama_result_dir, methods):
    targets, paths_by_method = _collect_targets(target_dir), {}
    for method in methods:
        method_name = str(method).upper()
        panoramas = _collect_panoramas(panorama_result_dir, method_name)
        missing_panoramas = sorted(set(targets) - set(panoramas), key=_numeric_sort_key)
        missing_targets = sorted(set(panoramas) - set(targets), key=_numeric_sort_key)
        if missing_panoramas or missing_targets:
            raise ValueError(
                f"{method_name} 파노라마-기준 물체 짝이 맞지 않습니다. 파노라마 없음={missing_panoramas}, 기준 물체 없음={missing_targets}"
            )
        paths_by_method[method_name] = panoramas
    return [
        (
            f"set{int(case_id):02d}",
            str(targets[case_id]),
            {method: str(paths[case_id]) for method, paths in paths_by_method.items()},
        )
        for case_id in sorted(targets, key=_numeric_sort_key)
    ]

from src.project_paths import RESULTS_DIR


# ==========================================
# 결과 파일 정리
# - results/ 내부에서 생성된 파일만 삭제
# - .gitkeep, opencv.log와 폴더 구조는 유지
# - dry_run=True이면 삭제 대상만 미리 확인
# ==========================================
def clear_results(dry_run=True):
    result_root = RESULTS_DIR.resolve()
    protected_names = {".gitkeep", "opencv.log"}
    target_paths = []

    if not result_root.is_dir():
        return target_paths

    for path in result_root.rglob("*"):
        if path.is_file() and path.name not in protected_names:
            target_paths.append(path)

    if not dry_run:
        for path in target_paths:
            path.unlink()

    return target_paths

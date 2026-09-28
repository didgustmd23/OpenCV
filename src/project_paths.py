from pathlib import Path

from src.settings import PATH_CONFIG


# ==========================================
# 프로젝트 공통 경로
# - 실행 위치와 관계없이 프로젝트 루트를 기준으로 경로를 계산
# - 데이터, 결과물, 모델 경로를 한 곳에서 관리
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / PATH_CONFIG["data_dir"]
OBJECT_DIR = PROJECT_ROOT / PATH_CONFIG["objects_dir"]

# 파노라마를 만들기 전의 연속 사진 세트
# 예: data/panorama_sets/set01/01.jpg, 02.jpg, 03.jpg
PANORAMA_SOURCE_DIR = PROJECT_ROOT / PATH_CONFIG["panorama_source_dir"]

# 객체 검출 실험에 사용하는 완성 파노라마
# 예: data/panorama/panorama1.jpg
PANORAMA_IMAGE_DIR = PROJECT_ROOT / PATH_CONFIG["panorama_image_dir"]

MODELS_DIR = PROJECT_ROOT / PATH_CONFIG["models_dir"]
RESULTS_DIR = PROJECT_ROOT / PATH_CONFIG["results_dir"]
LOG_DIR = PROJECT_ROOT / PATH_CONFIG["log_dir"]
MATCHING_RESULT_DIR = PROJECT_ROOT / PATH_CONFIG["matching_result_dir"]
OBJECT_DETECTION_RESULT_DIR = (
    PROJECT_ROOT / PATH_CONFIG["object_detection_result_dir"]
)
PANORAMA_RESULT_DIR = PROJECT_ROOT / PATH_CONFIG["panorama_result_dir"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ==========================================
# 프로젝트 기준 경로 변환
# - 절대 경로는 그대로 사용
# - 상대 경로는 프로젝트 루트 기준으로 변환
# ==========================================
def project_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path

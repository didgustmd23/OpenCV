import logging
from src.project_paths import LOG_DIR
from src.settings import LOGGING_CONFIG

# ==========================================
# 로그 저장 경로 준비
# - 프로젝트 공통 경로를 사용해 실행 위치가 달라도 동일한 위치에 저장
# ==========================================
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("opencv")
logger.setLevel(
    getattr(
        logging,
        LOGGING_CONFIG["python_level"].upper(),
        logging.INFO,
    )
)

# 중복 설정 방지
if not logger.handlers:

    # 파일 저장
    file_handler = logging.FileHandler(
        LOG_DIR / "opencv.log",
        encoding="utf-8"
    )

    # 터미널 출력
    console_handler = logging.StreamHandler()

    # 로그 형식
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

import logging
import os

os.makedirs("results/logs", exist_ok=True)

logger = logging.getLogger("opencv")
logger.setLevel(logging.INFO)

# 중복 설정 방지
if not logger.handlers:

    # 파일 저장
    file_handler = logging.FileHandler(
        "results/logs/opencv.log",
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
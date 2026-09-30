import re
from pathlib import Path

import cv2
import numpy as np

from src.common.project_paths import IMAGE_EXTENSIONS, project_path


# ==========================================
# 숫자가 포함된 파일명 정렬 키
# - 2.jpg, 10.jpg가 10.jpg, 2.jpg 순서로 섞이지 않도록 처리
# - 촬영 순서대로 파노라마를 연결하기 위한 보조 함수
# ==========================================
def natural_sort_key(path):
    return [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", Path(path).name)
    ]


# ==========================================
# 이미지 파일 목록 수집
# - 지원하는 이미지 확장자만 수집
# - 파일명 숫자를 고려한 자연 정렬 적용
# ==========================================
def list_image_files(directory):
    directory = project_path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"이미지 폴더를 찾을 수 없습니다: {directory}")

    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )


# ==========================================
# 이미지 크기 축소
# - max_side가 지정되면 긴 변 기준으로만 축소
# - 축소 이미지와 원본 사이의 좌표 변환용 배율을 함께 반환
# ==========================================
def resize_image(image, max_side=None):
    if image is None or image.size == 0:
        raise ValueError("축소할 이미지가 비어 있습니다.")

    if max_side is None:
        return image, 1.0

    if max_side <= 0:
        raise ValueError("max_side는 0보다 커야 합니다.")

    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))

    if scale < 1.0:
        image = cv2.resize(
            image,
            (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    return image, scale


# ==========================================
# 이미지 읽기
# - np.fromfile + cv2.imdecode를 사용해 한글 경로도 지원
# - max_side가 지정되면 긴 변 기준으로만 축소
# ==========================================
def read_image(path, max_side=None):
    path = project_path(path)
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {path}")

    image, _ = resize_image(image, max_side=max_side)
    return image


# ==========================================
# 이미지 저장
# - 저장 폴더가 없으면 자동 생성
# - cv2.imencode + tofile을 사용해 한글 경로도 지원
# ==========================================
def save_image(path, image):
    path = project_path(path)

    if not path.suffix:
        raise ValueError(f"이미지 확장자가 필요합니다: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)

    if not ok:
        raise ValueError(f"이미지 인코딩에 실패했습니다: {path}")

    encoded.tofile(str(path))
    return path

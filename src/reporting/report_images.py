"""최종보고서 Figure에서 공통으로 사용하는 이미지 로더."""

import cv2


# ==========================================
# Notebook 시각화용 이미지 변환
# - 큰 결과 이미지는 지정 폭 이하로 축소
# - OpenCV BGR 이미지를 Matplotlib RGB 형식으로 변환
# ==========================================
def read_image_for_display(path, max_width=500):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"결과 이미지를 불러올 수 없습니다: {path}")

    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(
            image,
            (max_width, int(height * scale)),
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

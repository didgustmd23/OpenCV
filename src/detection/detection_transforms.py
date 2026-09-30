"""객체 검출 실험에서 기준 물체에 적용하는 영상 변형."""

import cv2


def transform_reference(
    image, *, rotation_deg=0.0, scale_factor=1.0, brightness_factor=1.0
):
    if scale_factor <= 0 or brightness_factor <= 0:
        raise ValueError("scale_factor와 brightness_factor는 0보다 커야 합니다.")
    transformed = image.copy()
    if scale_factor != 1.0:
        height, width = transformed.shape[:2]
        transformed = cv2.resize(
            transformed,
            (max(1, round(width * scale_factor)), max(1, round(height * scale_factor))),
            interpolation=cv2.INTER_LINEAR,
        )
    if rotation_deg != 0.0:
        height, width = transformed.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        bound_width = int(height * abs(matrix[0, 1]) + width * abs(matrix[0, 0]))
        bound_height = int(height * abs(matrix[0, 0]) + width * abs(matrix[0, 1]))
        matrix[0, 2] += bound_width / 2 - center[0]
        matrix[1, 2] += bound_height / 2 - center[1]
        transformed = cv2.warpAffine(
            transformed,
            matrix,
            (bound_width, bound_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
    if brightness_factor != 1.0:
        transformed = cv2.convertScaleAbs(transformed, alpha=brightness_factor, beta=0)
    return transformed

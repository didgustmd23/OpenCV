import csv
from pathlib import Path

import cv2
import numpy as np

from src.common.feature_match import (
    detect_features,
    draw_matches,
    get_matched_points,
    match_features,
    ratio_test,
)
from src.common.image_io import list_image_files, read_image, save_image
from src.common.logger import logger
from src.common.project_paths import (
    PANORAMA_RESULT_DIR,
    project_path,
)
from src.common.settings import PANORAMA_CONFIG

SUPPORTED_METHODS = tuple(PANORAMA_CONFIG["methods"])


# ==========================================
# 인접 사진 특징점 추출
# - 객체 검출 모듈과 같은 detect_features() 사용
# - 각 사진의 Keypoint와 Descriptor를 한 번만 계산하여 재사용
# ==========================================
def extract_features(images, paths, method):
    features = []

    for image, path in zip(images, paths):
        keypoints, descriptors = detect_features(
            image,
            method=method,
        )

        if descriptors is None or len(keypoints) < 4:
            raise ValueError(f"특징점이 부족합니다: {path.name}")

        logger.info(
            "특징점 추출: image=%s, method=%s, keypoints=%d",
            path.name,
            method,
            len(keypoints),
        )
        features.append((keypoints, descriptors))

    return features


# ==========================================
# 방법별 매칭 결과 정리
# - SIFT / ORB: KNN 매칭 후 Lowe Ratio Test 적용
# - ALIKED: LightGlue 매칭 결과를 그대로 사용
# - raw_matches는 비율 검사 전 매칭 수 비교용으로 사용
# ==========================================
def get_pair_matches(kp1, desc1, image1, kp2, desc2, image2, method, ratio):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    matches = match_features(
        desc1,
        desc2,
        kp1,
        kp2,
        (width1, height1),
        (width2, height2),
        method=method,
    )

    if method == "ALIKED":
        # LightGlue는 자체 매칭 점수 기반으로 후보를 선택하므로
        # Lowe Ratio Test를 별도로 적용하지 않습니다.
        raw_matches = list(matches)
        good_matches = raw_matches
    else:
        knn_matches = [pair for pair in matches if len(pair) >= 2]
        raw_matches = [pair[0] for pair in knn_matches]
        good_matches = ratio_test(
            knn_matches,
            ratio=ratio,
        )

    return raw_matches, good_matches


# ==========================================
# 인접 사진 호모그래피 계산 및 매칭 결과 저장
# - 왼쪽 사진(i)의 좌표계로 오른쪽 사진(i+1)을 변환
# - 매칭 전, 필터 후, RANSAC Inlier 이미지를 모두 저장
# - 보고서용 매칭 수와 Inlier 수를 함께 반환
# ==========================================
def estimate_pair_transforms(
    images,
    paths,
    features,
    method,
    ratio,
    ransac_threshold,
    matching_dir,
):
    transforms = []
    pair_records = []

    for index in range(len(images) - 1):
        image1 = images[index]
        image2 = images[index + 1]
        kp1, desc1 = features[index]
        kp2, desc2 = features[index + 1]

        raw_matches, good_matches = get_pair_matches(
            kp1,
            desc1,
            image1,
            kp2,
            desc2,
            image2,
            method,
            ratio,
        )

        pair_name = f"pair_{index + 1:02d}_{index + 2:02d}"
        label = f"{paths[index].name} <-> {paths[index + 1].name}"

        # ==========================================
        # 매칭 단계별 시각화 저장
        # - ALIKED는 raw와 필터 후 결과가 동일할 수 있음
        # ==========================================
        save_image(
            matching_dir / f"{pair_name}_raw.png",
            draw_matches(image1, kp1, image2, kp2, raw_matches),
        )
        save_image(
            matching_dir / f"{pair_name}_filtered.png",
            draw_matches(image1, kp1, image2, kp2, good_matches),
        )

        logger.info(
            "인접 매칭: pair=%s, raw=%d, filtered=%d",
            label,
            len(raw_matches),
            len(good_matches),
        )

        if len(good_matches) < 4:
            raise ValueError(f"{label}: 필터 후 매칭이 4개 미만입니다.")

        points1, points2 = get_matched_points(
            kp1,
            kp2,
            good_matches,
        )

        # points2 -> points1 변환
        # 다음 사진을 현재 사진의 좌표계로 옮기기 위한 호모그래피
        matrix, mask = cv2.findHomography(
            np.float32(points2).reshape(-1, 1, 2),
            np.float32(points1).reshape(-1, 1, 2),
            cv2.RANSAC,
            ransac_threshold,
        )

        if matrix is None or mask is None or not np.isfinite(matrix).all():
            raise ValueError(f"{label}: 호모그래피 계산에 실패했습니다.")

        inlier_matches = [
            match for match, keep in zip(good_matches, mask.ravel()) if keep
        ]

        if len(inlier_matches) < 4:
            raise ValueError(f"{label}: RANSAC Inlier 매칭이 부족합니다.")

        save_image(
            matching_dir / f"{pair_name}_ransac.png",
            draw_matches(image1, kp1, image2, kp2, inlier_matches),
        )

        inlier_ratio = len(inlier_matches) / len(good_matches)
        logger.info(
            "RANSAC 결과: pair=%s, inliers=%d, ratio=%.3f",
            label,
            len(inlier_matches),
            inlier_ratio,
        )

        transforms.append(matrix)
        pair_records.append(
            {
                "pair": pair_name,
                "left_image": paths[index].name,
                "right_image": paths[index + 1].name,
                "method": method,
                "raw_match_count": len(raw_matches),
                "filtered_match_count": len(good_matches),
                "inlier_count": len(inlier_matches),
                "inlier_ratio": f"{inlier_ratio:.6f}",
            }
        )

    return transforms, pair_records


# ==========================================
# 중앙 사진 기준 정렬 행렬 생성
# - 중앙 사진은 단위 행렬로 유지
# - 오른쪽 사진은 이전 변환을 누적
# - 왼쪽 사진은 역행렬을 사용해 중앙 좌표계로 변환
# ==========================================
def align_to_center(pair_transforms, image_count):
    center_index = image_count // 2
    transforms = [None] * image_count
    transforms[center_index] = np.eye(3, dtype=np.float64)

    for index in range(center_index + 1, image_count):
        transforms[index] = transforms[index - 1] @ pair_transforms[index - 1]

    for index in range(center_index - 1, -1, -1):
        transforms[index] = transforms[index + 1] @ np.linalg.inv(
            pair_transforms[index]
        )

    return transforms


# ==========================================
# 이미지 모서리 좌표 생성
# - 좌상단부터 시계 방향으로 4개 모서리 반환
# - 변환 후 파노라마 캔버스 범위를 계산할 때 사용
# ==========================================
def image_corners(image):
    height, width = image.shape[:2]

    return np.float32(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ]
    ).reshape(-1, 1, 2)


# ==========================================
# 변환된 이미지 모서리 검증
# - 투영 분모 부호가 섞이면 무한대로 발산할 가능성이 있어 중단
# - 유한한 좌표만 반환하여 비정상 캔버스 생성을 방지
# ==========================================
def transformed_corners(image, matrix):
    corners = image_corners(image)
    xy = corners.reshape(-1, 2)
    denominator = xy @ matrix[2, :2] + matrix[2, 2]

    same_sign = np.all(denominator > 1e-8) or np.all(denominator < -1e-8)

    if not same_sign:
        raise ValueError("불안정한 호모그래피입니다. 매칭 결과를 확인하세요.")

    projected = cv2.perspectiveTransform(corners, matrix)

    if not np.isfinite(projected).all():
        raise ValueError("변환된 모서리 좌표가 유효하지 않습니다.")

    return projected


# ==========================================
# 페더 블렌딩 가중치 생성
# - 이미지 중앙은 크게, 가장자리는 작게 가중치를 설정
# - 겹치는 영역의 경계가 부드럽게 합성되도록 사용
# ==========================================
def feather_weight(image):
    height, width = image.shape[:2]

    x_weight = np.minimum(
        np.arange(width) + 1,
        width - np.arange(width),
    )
    y_weight = np.minimum(
        np.arange(height) + 1,
        height - np.arange(height),
    )

    weight = np.minimum(
        y_weight[:, None],
        x_weight[None, :],
    ).astype(np.float32)

    return weight / weight.max()


# ==========================================
# 파노라마 합성
# - 모든 사진이 들어가는 최소 캔버스를 계산
# - 워핑한 이미지에 페더 블렌딩을 적용
# - 유효 픽셀 영역만 잘라 검은 여백 제거
# - 원본 사진 배치 영역을 표시한 outline 이미지도 반환
# ==========================================
def build_panorama(images, transforms):
    polygons = [
        transformed_corners(image, matrix) for image, matrix in zip(images, transforms)
    ]

    all_points = np.concatenate(polygons, axis=0).reshape(-1, 2)
    lower = np.floor(all_points.min(axis=0))
    upper = np.ceil(all_points.max(axis=0))

    canvas_width = int(upper[0] - lower[0])
    canvas_height = int(upper[1] - lower[1])

    if (
        canvas_width <= 0
        or canvas_height <= 0
        or max(canvas_width, canvas_height) > PANORAMA_CONFIG["max_canvas_side"]
        or canvas_width * canvas_height > PANORAMA_CONFIG["max_canvas_pixels"]
    ):
        raise ValueError(
            f"파노라마 캔버스 크기가 비정상적입니다: {canvas_width}x{canvas_height}"
        )

    translation = np.array(
        [
            [1, 0, -float(lower[0])],
            [0, 1, -float(lower[1])],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    color_sum = np.zeros(
        (canvas_height, canvas_width, 3),
        dtype=np.float32,
    )
    weight_sum = np.zeros(
        (canvas_height, canvas_width),
        dtype=np.float32,
    )

    for image, matrix in zip(images, transforms):
        warp_matrix = translation @ matrix
        weight = feather_weight(image)
        weighted_image = image.astype(np.float32) * weight[:, :, None]

        color_sum += cv2.warpPerspective(
            weighted_image,
            warp_matrix,
            (canvas_width, canvas_height),
        )
        weight_sum += cv2.warpPerspective(
            weight,
            warp_matrix,
            (canvas_width, canvas_height),
        )

    panorama = np.clip(
        color_sum / np.maximum(weight_sum, 1e-6)[:, :, None],
        0,
        255,
    ).astype(np.uint8)

    outline = panorama.copy()

    for image, matrix in zip(images, transforms):
        polygon = cv2.perspectiveTransform(
            image_corners(image),
            translation @ matrix,
        )
        cv2.polylines(
            outline,
            [np.round(polygon).astype(np.int32)],
            True,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # 파노라마 실제 영역만 남겨 이후 객체 검출 시 불필요한 검은 타일을 제거
    valid_mask = np.uint8(weight_sum > 1e-6)
    valid_points = cv2.findNonZero(valid_mask)

    if valid_points is None:
        raise ValueError("합성된 파노라마에 유효한 픽셀이 없습니다.")

    crop_x, crop_y, crop_width, crop_height = cv2.boundingRect(valid_points)
    crop_slice = np.s_[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]

    return panorama[crop_slice], outline[crop_slice]


# ==========================================
# 매칭 통계 CSV 저장
# - 인접 사진별 매칭 전후 개수와 RANSAC Inlier 수 기록
# - 보고서의 매칭 품질 표에 바로 사용할 수 있도록 저장
# ==========================================
def save_pair_records(path, pair_records):
    fieldnames = [
        "pair",
        "left_image",
        "right_image",
        "method",
        "raw_match_count",
        "filtered_match_count",
        "inlier_count",
        "inlier_ratio",
    ]

    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_records)

    return path


# ==========================================
# 파노라마 생성 실행 함수
# - 연속 사진 폴더를 입력받아 한 장의 파노라마 생성
# - 결과는 results/panorama/세트명/방법/에 저장
# - export_path 지정 시 객체 검출용 data/panorama/에도 별도 저장
# ==========================================
def stitch_panorama(
    input_dir,
    method=PANORAMA_CONFIG["default_method"],
    *,
    ratio=PANORAMA_CONFIG["lowe_ratio"],
    ransac_threshold=PANORAMA_CONFIG["ransac_reproj_threshold"],
    max_side=PANORAMA_CONFIG["max_side"],
    output_root=PANORAMA_RESULT_DIR,
    set_name=None,
    export_path=None,
):
    method = method.upper()

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"지원하지 않는 방법입니다: {method}")
    if not 0 < ratio < 1:
        raise ValueError("ratio는 0보다 크고 1보다 작아야 합니다.")
    if ransac_threshold <= 0:
        raise ValueError("ransac_threshold는 양수여야 합니다.")
    if max_side <= 0:
        raise ValueError("max_side는 양수여야 합니다.")

    input_dir = project_path(input_dir)
    paths = list_image_files(input_dir)

    if len(paths) < PANORAMA_CONFIG["min_images"]:
        raise ValueError(
            "파노라마 생성에는 촬영 순서의 이미지가 "
            f"{PANORAMA_CONFIG['min_images']}장 이상 필요합니다."
        )

    output_root = project_path(output_root)
    set_name = str(set_name or input_dir.name)

    if not set_name or Path(set_name).name != set_name:
        raise ValueError("set_name은 폴더명만 지정해야 합니다.")

    # ==========================================
    # 세트·방법별 결과 폴더 생성
    # - 예: results/panorama/set01/SIFT/
    # - 여러 파노라마 세트의 합성 이미지와 매칭 통계를 분리
    # ==========================================
    output_dir = output_root / set_name / method
    matching_dir = output_dir / "matching"
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "파노라마 생성 시작: set=%s, method=%s, images=%d",
        input_dir.name,
        method,
        len(paths),
    )

    images = [read_image(path, max_side=max_side) for path in paths]
    features = extract_features(images, paths, method)
    pair_transforms, pair_records = estimate_pair_transforms(
        images,
        paths,
        features,
        method,
        ratio,
        ransac_threshold,
        matching_dir,
    )
    transforms = align_to_center(
        pair_transforms,
        len(images),
    )
    panorama, outline = build_panorama(images, transforms)

    panorama_path = save_image(
        output_dir / "panorama.jpg",
        panorama,
    )
    outline_path = save_image(
        output_dir / "panorama_outline.jpg",
        outline,
    )
    stats_path = save_pair_records(
        output_dir / "matching_stats.csv",
        pair_records,
    )

    export_result = None

    if export_path is not None:
        export_result = save_image(export_path, panorama)

    logger.info(
        "파노라마 생성 완료: size=%dx%d, path=%s",
        panorama.shape[1],
        panorama.shape[0],
        panorama_path,
    )

    return {
        "input_paths": paths,
        "method": method,
        "panorama": panorama,
        "pair_records": pair_records,
        "panorama_path": panorama_path,
        "outline_path": outline_path,
        "matching_dir": matching_dir,
        "stats_path": stats_path,
        "export_path": export_result,
    }

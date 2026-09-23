# 라이브러리 추가
import os
import cv2
import numpy as np

from logger import logger
from feature_match import (
    detect_features,
    match_features,
    ratio_test,
    get_matched_points,
    find_homography,
    draw_matches,
)


# 객체 검출로 인정하기 위한 최소 RANSAC Inlier 개수
MIN_INLIERS = 8

# Homography 계산 전 특징점 매칭 결과 저장 설정
SAVE_FEATURE_MATCHES = True
FEATURE_MATCH_DIR = "results/matching/keypoint_matches"


# ==========================================
# Reference 객체를 Scene 이미지에서 검출
# - Reference / Scene 이미지의 특징점과 Descriptor 추출
# - 특징점 매칭 및 Ratio Test 수행
# - RANSAC 기반 Homography 계산
# - Inlier 개수와 비율을 이용해 매칭 신뢰도 분석
# - Homography로 Reference의 네 모서리를 Scene 좌표로 변환
# - 검출 영역의 면적, 형태, Inlier 분포를 분석
# - 객체 검출 결과와 Homography 정보를 Dictionary로 반환
#
# ref_features:
# - 미리 계산한 Reference 특징점을 전달하면 재사용
# - Tile마다 Reference 특징을 반복 계산하지 않기 위한 최적화
#
# coarse_mode:
# - True이면 MIN_INLIERS 미만의 약한 결과도 후보 점수로 반환
# - False이면 MIN_INLIERS 미만의 결과를 검출 실패(None)로 처리
# ==========================================
def find_object(
    ref_img,
    sce_img,
    method="SIFT",
    ref_name="",
    sce_name="",
    ref_features=None,
    coarse_mode=False,
):

    logger.debug("Object Detection - find_object() - Start")

    # ==========================================
    # Reference 특징점 및 Descriptor 준비
    # - 사전 계산된 ref_features가 있으면 그대로 재사용
    # - 없으면 현재 Reference 이미지에서 새로 특징 추출
    # - Tile Scan 시 Reference 특징을 재사용하여 반복 계산 방지
    # ==========================================
    if ref_features is None:
        logger.debug("Reference 특징점 새로 계산")

        kp1, des1 = detect_features(
            ref_img,
            method
        )
    else:
        logger.debug("Reference 특징점 재사용")
        kp1, des1 = ref_features

    # ==========================================
    # Scene 특징점 및 Descriptor 추출
    # - 현재 검사할 Scene 또는 Tile에서 특징 추출
    # - Scene은 Tile마다 달라지므로 매번 새로 계산
    # ==========================================
    logger.debug("Scene 특징점 새로 계산")

    kp2, des2 = detect_features(
        sce_img,
        method
    )

    logger.debug("method %s: Reference Image: %s Scene Image: %s", method, ref_name, sce_name)
    logger.debug(f"reference keypoint: {len(kp1)}")
    logger.debug(f"scene keypoint: {len(kp2)}")

    # ==========================================
    # Reference와 Scene 특징점 매칭
    # - 이미지 크기를 Matcher에 전달
    # - SIFT / ORB / ALIKED에 맞는 매칭 방식 사용
    # - ALIKED는 LightGlue에서 이미지 크기 정보를 사용
    # ==========================================
    h1, w1 = ref_img.shape[:2]
    h2, w2 = sce_img.shape[:2]

    ref_size = (w1, h1)
    sce_size = (w2, h2)

    matches = match_features(
        des1,
        des2,
        kp1,
        kp2,
        ref_size,
        sce_size,
        method,
    )

    logger.debug(f"전체 매칭: {len(matches)}")

    # ==========================================
    # 좋은 매칭점 선택
    # - SIFT / ORB는 Lowe Ratio Test 적용
    # - ALIKED + LightGlue는 Matcher 결과를 그대로 사용
    # ==========================================
    if method == "ALIKED":
        good_matches = matches
    else:
        good_matches = ratio_test(matches, ratio=0.75)

    logger.debug(f"좋은 매칭: {len(good_matches)}")

    # ==========================================
    # 특징점 매칭 결과 저장
    # - Reference와 현재 Scene/Tile을 좌우로 배치
    # - 실제로 선택된 good_matches만 선으로 연결
    # - Homography 계산에 실패하더라도 매칭 상태를 확인할 수 있도록
    #   좌표 변환보다 먼저 저장
    # ==========================================
    if SAVE_FEATURE_MATCHES:
        method_name = str(method).upper()
        method_match_dir = os.path.join(
            FEATURE_MATCH_DIR,
            method_name,
        )
        os.makedirs(method_match_dir, exist_ok=True)

        match_image = draw_matches(
            ref_img,
            kp1,
            sce_img,
            kp2,
            good_matches,
        )

        safe_scene_name = os.path.basename(
            sce_name or "scene"
        )
        output_path = os.path.join(
            method_match_dir,
            f"{safe_scene_name}_matches.png",
        )

        if cv2.imwrite(output_path, match_image):
            logger.debug(
                "특징점 매칭 이미지 저장: scene=%s, matches=%d, path=%s",
                sce_name,
                len(good_matches),
                output_path,
            )
        else:
            logger.warning(
                "특징점 매칭 이미지 저장 실패: %s",
                output_path,
            )

    # ==========================================
    # 매칭 결과를 실제 이미지 좌표로 변환
    # - pts1: Reference 이미지의 대응점 좌표
    # - pts2: Scene 이미지의 대응점 좌표
    # - Homography 계산에 사용할 좌표 집합 생성
    # ==========================================
    pts1, pts2 = get_matched_points(
        kp1,
        kp2,
        good_matches,
    )

    logger.debug(f"pts1 개수: {len(pts1)}")
    logger.debug(f"pts2 개수: {len(pts2)}")

    # ==========================================
    # Homography 계산 가능 여부 확인
    # - Homography 계산에는 최소 4쌍의 대응점 필요
    # - 4개 미만이면 객체 위치 변환이 불가능하므로 종료
    # ==========================================
    if len(pts1) < 4 or len(pts2) < 4:
        logger.debug(
            "대응 점이 4개 미만라 계산이 불가합니다."
        )
        return None

    # ==========================================
    # RANSAC 기반 Homography 계산
    # - Reference와 Scene 대응점으로 Homography 계산
    # - 공간적으로 일관되지 않은 잘못된 매칭을 Outlier로 분리
    # - mask를 이용해 RANSAC을 통과한 Inlier 확인
    # ==========================================
    H, mask = find_homography(
        pts1,
        pts2,
    )

    match_count = len(good_matches)

    if H is None:
        logger.debug("Homography 계산 실패")
        return None

    # ==========================================
    # RANSAC Inlier 통계 계산
    # - match_count: Ratio Test 이후 전체 매칭 수
    # - inlier_count: RANSAC을 통과한 매칭 수
    # - inlier_ratio: 전체 매칭 중 Inlier가 차지하는 비율
    # ==========================================
    inlier_count = int(mask.sum())

    inlier_ratio = (
        inlier_count / match_count
        if match_count > 0
        else 0.0
    )

    logger.debug(
        "RANSAC 결과: matches=%d, inliers=%d, ratio=%.3f",
        match_count,
        inlier_count,
        inlier_ratio,
    )

    # ==========================================
    # 최소 Inlier 조건 검사
    # - MIN_INLIERS 이상이면 정상 객체 검출 후보로 계속 진행
    # - 미만이면 일반 검출에서는 실패(None) 처리
    # - coarse_mode에서는 약한 결과도 Coarse 후보 순위 계산에
    #   사용할 수 있도록 통계 정보만 반환
    # ==========================================
    if inlier_count < MIN_INLIERS:

        logger.debug(
            "RANSAC inlier 부족: inliers=%d, min_required=%d",
            inlier_count,
            MIN_INLIERS,
        )

        if coarse_mode:
            return {
                "match_count": match_count,
                "inlier_count": inlier_count,
                "inlier_ratio": inlier_ratio,
                "is_detection": False,
            }

        return None

    # ==========================================
    # Reference 객체의 네 모서리 좌표 생성
    # - Reference 전체 영역을 사각형으로 표현
    # - 좌상 → 우상 → 우하 → 좌하 순서
    # ==========================================
    h, w = ref_img.shape[:2]

    corners = np.float32([
        [0, 0],     # 왼쪽 위
        [w, 0],     # 오른쪽 위
        [w, h],     # 오른쪽 아래
        [0, h],     # 왼쪽 아래
    ]).reshape(-1, 1, 2)

    # ==========================================
    # Reference 모서리를 Scene 좌표로 변환
    # - RANSAC으로 계산한 Homography 사용
    # - 변환된 사각형이 Scene에서 검출된 객체 영역
    # ==========================================
    transformed_corners = cv2.perspectiveTransform(
        corners,
        H,
    )

    # ==========================================
    # Inlier Spatial Coverage 분석
    # - RANSAC Inlier가 Reference의 어느 범위에 분포하는지 확인
    # - 특정 좁은 영역에만 매칭이 몰리는지 분석하기 위한 지표
    # - coverage_x / coverage_y가 클수록 Reference의 넓은 영역에
    #   Inlier가 분포하고 있음을 의미
    # ==========================================
    inlier_mask = mask.ravel().astype(bool)

    # pts1이 list일 수도 있으므로 NumPy 배열로 변환
    pts1_np = np.asarray(
        pts1,
        dtype=np.float32,
    )

    # Reference 쪽에서 RANSAC을 통과한 Inlier 좌표만 추출
    ref_inlier_pts = pts1_np[inlier_mask]

    min_x = np.min(ref_inlier_pts[:, 0])
    max_x = np.max(ref_inlier_pts[:, 0])

    min_y = np.min(ref_inlier_pts[:, 1])
    max_y = np.max(ref_inlier_pts[:, 1])

    coverage_x = (max_x - min_x) / w
    coverage_y = (max_y - min_y) / h

    logger.debug(
        "Reference inlier points: %d",
        len(ref_inlier_pts),
    )

    logger.debug(
        "Inlier bbox: x=[%.2f, %.2f], y=[%.2f, %.2f]",
        min_x,
        max_x,
        min_y,
        max_y,
    )

    logger.debug(
        "Inlier coverage: x=%.3f, y=%.3f",
        coverage_x,
        coverage_y,
    )

    # ==========================================
    # Homography 검출 영역의 면적 분석
    # - Reference 원본 면적과 변환된 사각형 면적 비교
    # - area_ratio로 객체의 크기 변화 정도 확인
    # - 1.0에 가까우면 Reference와 비슷한 면적
    # ==========================================
    ref_area = float(w * h)

    detected_area = abs(
        cv2.contourArea(
            transformed_corners.astype(np.float32)
        )
    )

    area_ratio = (
        detected_area / ref_area
        if ref_area > 0
        else 0.0
    )

    # ==========================================
    # 검출 사각형의 기하학적 형태 분석
    # - Homography로 변환된 네 점이 Convex인지 확인
    # - 상/하 변 길이와 좌/우 변 길이를 비교
    # - 비정상적으로 뒤틀린 Homography를 분석하기 위한 지표
    # ==========================================
    quad = transformed_corners.reshape(4, 2)

    is_convex = cv2.isContourConvex(
        quad.astype(np.float32)
    )

    logger.debug(
        "Convex quadrilateral: %s",
        is_convex,
    )

    top = np.linalg.norm(
        quad[1] - quad[0]
    )
    right = np.linalg.norm(
        quad[2] - quad[1]
    )
    bottom = np.linalg.norm(
        quad[2] - quad[3]
    )
    left = np.linalg.norm(
        quad[3] - quad[0]
    )

    horizontal_ratio = (
        min(top, bottom) / max(top, bottom)
        if max(top, bottom) > 0
        else 0.0
    )

    vertical_ratio = (
        min(left, right) / max(left, right)
        if max(left, right) > 0
        else 0.0
    )

    logger.debug(
        "Reference area: %.2f",
        ref_area,
    )

    logger.debug(
        "Detected area: %.2f",
        detected_area,
    )

    logger.debug(
        "Area ratio: %.3f",
        area_ratio,
    )

    logger.debug(
        "Quad sides: top=%.2f, right=%.2f, bottom=%.2f, left=%.2f",
        top,
        right,
        bottom,
        left,
    )

    logger.debug(
        "Opposite side ratio: horizontal=%.3f, vertical=%.3f",
        horizontal_ratio,
        vertical_ratio,
    )

    # ==========================================
    # Convex 여부를 이용한 비정상 Homography 제거
    # - 정상적인 객체 사각형은 일반적으로 Convex 형태
    # - 네 모서리가 뒤집히거나 교차한 경우 검출 실패 처리
    # ==========================================
    if not is_convex:
        logger.debug(
            "Homography polygon이 convex하지 않아 객체 검출 실패"
        )
        return None

    # ==========================================
    # 검출 결과 시각화
    # - Homography를 이용해 Reference 영역을 Scene 좌표로 변환
    # - Scene 복사본 위에 검출된 객체의 외곽 사각형 표시
    # - 원본 Scene 이미지는 변경하지 않음
    # ==========================================
    sce_corners = cv2.perspectiveTransform(
        corners,
        H,
    )

    result = sce_img.copy()

    cv2.polylines(
        result,
        [np.int32(sce_corners)],
        True,
        (0, 255, 0),
        3,
    )

    logger.debug(
        "Object Detection - find_object() - END"
    )

    # ==========================================
    # 객체 검출 결과 반환
    # - H: Reference → 현재 Scene/Tile Homography
    # - matches: 최종 특징점 매칭 정보
    # - Inlier 관련 값: RANSAC 매칭 신뢰도 분석
    # - corners: 현재 Scene/Tile 좌표의 객체 영역
    # - Geometry 값: 면적/형태/분포 분석에 사용
    # ==========================================
    return {
        "image": result,
        "H": H,
        "matches": good_matches,
        "match_count": match_count,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "is_detection": True,
        "corners": transformed_corners,
        "detected_area": detected_area,
        "area_ratio": area_ratio,
        "horizontal_ratio": horizontal_ratio,
        "vertical_ratio": vertical_ratio,
        "is_convex": is_convex,
        "coverage_x": coverage_x,
        "coverage_y": coverage_y,
    }


# ==========================================
# 검출 Polygon을 전체 Scene 좌표로 변환
# - Reference 이미지의 네 모서리를 생성
# - Homography를 이용해 BEST Tile 좌표로 변환
# - Tile 시작 위치를 더해 전체 Scene 좌표로 변환
# - Reference / Tile / Global corner 좌표 반환
#
# H_tile:
# - Reference → BEST Tile 좌표계의 Homography
#
# tile_origin:
# - 전체 Scene에서 BEST Tile이 시작하는 (x, y) 좌표
# ==========================================
def get_global_corners(
    ref_img,
    H_tile,
    tile_origin,
):
    # ==========================================
    # Reference 이미지의 네 모서리 생성
    # ==========================================
    ref_h, ref_w = ref_img.shape[:2]

    ref_corners = np.float32([
        [0, 0],
        [ref_w, 0],
        [ref_w, ref_h],
        [0, ref_h],
    ]).reshape(-1, 1, 2)

    # ==========================================
    # Reference → BEST Tile 좌표 변환
    # - find_object()에서 계산한 Homography 사용
    # ==========================================
    corners_tile = cv2.perspectiveTransform(
        ref_corners,
        H_tile,
    )

    # ==========================================
    # BEST Tile → 전체 Scene 좌표 변환
    # - Tile 내부 좌표에 Tile 시작 위치를 더함
    #
    # Global X = Tile X + Tile 시작 X
    # Global Y = Tile Y + Tile 시작 Y
    # ==========================================
    tile_x, tile_y = tile_origin

    corners_global = corners_tile.copy()

    corners_global[:, :, 0] += tile_x
    corners_global[:, :, 1] += tile_y

    logger.debug(
        "Reference corners:\n%s",
        ref_corners.reshape(-1, 2),
    )

    logger.debug(
        "Tile corners:\n%s",
        corners_tile.reshape(-1, 2),
    )

    logger.debug(
        "Global corners:\n%s",
        corners_global.reshape(-1, 2),
    )

    # ==========================================
    # 좌표계별 Corner 반환
    # - reference: Reference 원본 좌표
    # - tile: BEST Tile 내부 좌표
    # - global: 전체 Scene / Panorama 좌표
    # ==========================================
    return {
        "reference": ref_corners,
        "tile": corners_tile,
        "global": corners_global,
    }

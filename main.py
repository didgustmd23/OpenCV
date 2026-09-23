import os
import cv2
import time
import numpy as np


from logger import logger

from object_finder import get_global_corners
from tile_scanner import scan_scene
from feature_match import detect_features

# ==========================================
# 탐색 설정
# ==========================================

TILE_W = 512
TILE_H = 512

COARSE_STRIDE_X = 512
COARSE_STRIDE_Y = 512

FINE_STRIDE_X = 256
FINE_STRIDE_Y = 256

COARSE_TOP_K = 3
EARLY_STOP_INLIERS = 50

METHOD = "ALIKED"

# ==========================================
# 로그 수준 설정 (오류만 출력)
# 설정 가능 옵션: LOG_LEVEL_SILENT, LOG_LEVEL_FATAL, LOG_LEVEL_ERROR, LOG_LEVEL_WARNING, LOG_LEVEL_INFO, LOG_LEVEL_VERBOSE
# ==========================================

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

# ==========================================
# 이미지 로드
# ==========================================

ref_path = "data/objects/Drone2.jpg"
sce_path = "data/objects/Drone.JPEG"

ref_name = os.path.basename(ref_path)
sce_name = os.path.basename(sce_path)

ref_img = cv2.imread(ref_path)  # 찾을 객체
sce_img = cv2.imread(sce_path)  # 원본 이미지

if ref_img is None:
    raise FileNotFoundError(
        f"Reference 이미지를 불러올 수 없습니다: {ref_path}"
    )

if sce_img is None:
    raise FileNotFoundError(
        f"Scene 이미지를 불러올 수 없습니다: {sce_path}"
    )

# ==========================================
# Reference 특징점 사전 계산
# - Reference는 모든 Tile에서 동일하게 사용
# - ALIKED 특징점을 한 번만 계산하여 재사용
# ==========================================

logger.info(
    "Reference %s 특징 추출 시작",
    METHOD,
)

kp_ref, des_ref = detect_features(
    ref_img,
    method=METHOD,
)

ref_features = (
    kp_ref,
    des_ref,
)

logger.info(
    "Reference %s 특징 추출 완료: "
    "keypoints=%d, descriptors=%s",
    METHOD,
    len(kp_ref),
    des_ref.shape,
)

# ==========================================
# Scene Coarse-to-Fine 탐색
# ==========================================

scan_result = scan_scene(
    ref_img=ref_img,
    sce_img=sce_img,
    ref_features=ref_features,
    tile_size=(
        TILE_W,
        TILE_H,
    ),
    coarse_stride=(
        COARSE_STRIDE_X,
        COARSE_STRIDE_Y,
    ),
    fine_stride=(
        FINE_STRIDE_X,
        FINE_STRIDE_Y,
    ),
    coarse_top_k=COARSE_TOP_K,
    early_stop_inliers=EARLY_STOP_INLIERS,
    ref_name=ref_path,
    method=METHOD,
)

best = scan_result["best"]

if best is None:
    logger.warning(
        "최종적으로 객체를 검출하지 못했습니다."
    )

else:
    best_result = best["result"]

    best_tile_index = best["index"]

    best_x = best["x"]
    best_y = best["y"]

    scene_h, scene_w = sce_img.shape[:2]

    best_x2 = min(
        best_x + TILE_W,
        scene_w,
    )

    best_y2 = min(
        best_y + TILE_H,
        scene_h,
    )

    best_tile_info = (
        best_x,
        best_y,
        best_x2,
        best_y2,
    )

    corner_result = get_global_corners(
        ref_img=ref_img,
        H_tile=best_result["H"],
        tile_origin=(
            best_x,
            best_y,
        ),
    )

    corners_global = corner_result["global"]

    logger.info(
        "최종 검출 결과: "
        "tile_index=%d, tile=%s, "
        "matches=%d, inliers=%d, ratio=%.3f, "
        "detected_area=%.2f, area_ratio=%.3f",
        best_tile_index,
        best_tile_info,
        best_result["match_count"],
        best_result["inlier_count"],
        best_result["inlier_ratio"],
        best_result["detected_area"],
        best_result["area_ratio"],
    )

    # ==========================================
    # 최종 검출 결과 시각화
    # - Scene 복사본 생성
    # - Global Polygon을 정수 좌표로 변환
    # - 검출된 객체 영역 표시
    # ==========================================

    scene_result = sce_img.copy()

    corners_global_int = np.int32(
        np.round(corners_global)
    )

    cv2.polylines(
        scene_result,
        [corners_global_int],
        True,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(f"results/matching/{METHOD}_result.png", scene_result)
    
    cv2.imshow("Scene result", scene_result)
    cv2.waitKey()
    cv2.destroyAllWindows()
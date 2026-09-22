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
)

def find_object(ref_img, sce_img, method="SIFT", ref_name="", sce_name=""):
    
    logger.info("Object Detection - find_object() - Start")

    # 특징점 검출
    kp1, des1 = detect_features(ref_img, method)
    kp2, des2 = detect_features(sce_img, method)
    
    logger.info(f"사용 방법: {method}")
    logger.info(f"Reference 이미지: {os.path.basename(ref_name)}")
    logger.info(f"Scene 이미지: {os.path.basename(sce_name)}")
    logger.info(f"reference 특징점: {len(kp1)}")
    logger.info(f"scene 특징점: {len(kp2)}")

    # 특징점 매칭
    matches = match_features(des1, des2, method)
    
    logger.info(f"전체 매칭: {len(matches)}")

    # ratio test 수행
    if method == "ORB":
        good_matches = ratio_test(matches, ratio=0.75)
    else:
        good_matches = ratio_test(matches, ratio=0.75)
    
    logger.info(f"좋은 매칭: {len(good_matches)}")

    # 매칭 좌표 가져오기
    pts1, pts2 = get_matched_points(kp1, kp2, good_matches)
    
    logger.info(f"pts1 개수: {len(pts1)}")
    logger.info(f"pts2 개수: {len(pts2)}")
    
    # 대응점 개수 확인 4개보다 작으면 계산 불가
    if len(pts1) < 4 or len(pts2) < 4:
        logger.info("대응 점이 4개 미만라 계산이 불가합니다.") 
        return None

    # 공간적으로 틀린 매칭 제거
    H, mask = find_homography(pts1, pts2)
    
    if H is None:
        logger.info("Homography 계산 실패")
        return None
    
    inlier_count = int(mask.sum())

    if inlier_count < 8:
        logger.info("RANSAC inlier가 8개 미만이라 객체 검출 실패")
        return None
    
    # RANSAC inlier 개수
    inliers = int(mask.sum())
    logger.info(f"RANSAC inlier: {inliers}")
    
    # 이미지 크기
    h, w = ref_img.shape[:2]
    
    # 객체의 네 모서리를 구함
    corners = np.float32([
        [0, 0],     # 왼쪽 위
        [w, 0],     # 오른쪽 위
        [w, h],     # 오른쪽 아래
        [0, h]      # 왼쪽 아래
    ]).reshape(-1, 1, 2)
    
    # Homography를 이용해서 Scene 위치로 변환
    sce_corners = cv2.perspectiveTransform(corners, H)
    
    # 이미지 복사
    result = sce_img.copy()
    
    # 객체 위치에 네모 그리기
    cv2.polylines(
        result,
        [np.int32(sce_corners)],
        True,
        (0, 255, 0),
        3
    )
    
    logger.info("Object Detection - find_object() - END")
    
    return result
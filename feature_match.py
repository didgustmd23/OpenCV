# 라이브러리
import cv2
import numpy as np
from logger import logger

ALIKED_PATH = "models/aliked-n16rot-top1k-1280.onnx"
LIGHTGLUE_ONNX_PATH = "models/lightglue_for_aliked.onnx"
DISK_PATH = "models/"

# ==========================================
# 특징점 검출 및 Descriptor 추출
# - 지정한 방법에 따라 Feature Detector 생성
# - 이미지에서 Keypoint와 Descriptor 추출
# - SIFT, ORB, ALIKED, DISK 지원
# ==========================================
def detect_features(image, method="SIFT"):
    
    # 특징점 검출 방법 선택 SIFT, ORB, AKAZE
    if method == "SIFT":
        detector = cv2.SIFT.create()
    elif method == "ORB":                       # ORB는 그레이스케일 이미지를 사용해야함
        detector = cv2.ORB.create(
            nfeatures=40000,                    # 최대 특징점 수
            scaleFactor=1.2,                    # 스케일 변화율
            nlevels=8,                          # 스케일의 레벨 수
            edgeThreshold=31,                   # 엣지 임곗값
            firstLevel=0,                       # 시작 피라미드 레벨
            WTA_K=2,                            # 비교점
            scoreType=cv2.ORB_HARRIS_SCORE,     # 점수 방식
            patchSize=31,                       # 패치 크기
            fastThreshold=20,                   # FAST 임곗값
        )
    # OpenCV 5.0에서 AKAZE는 현재 사용불가 해당 AKAZE_create가 없음
    # OpenCV 5.0에서 ALIKED, DISK 추가됨
    elif method == "ALIKED":
        
        params = cv2.ALIKED.Params()
        params.inputSize = (1280, 1280)
        params.normalizeDescriptors = True

        cv2.ocl.setUseOpenCL(False)
        
        detector = cv2.ALIKED.create(
            ALIKED_PATH,
            params
        )
        
            
    elif method == "DISK":
        detector = cv2.DISK.create(DISK_PATH)
    else:
        # 에러 발생시 즉시 중단하고 에러 출력
        raise ValueError(f"지원하지 않는 메소드 : {method}")
    
    # 선택된 detector로 특징점 검출 (None : Mask 사용하지 않고 전체에서 특징점 검출)
    keypoints, descriptors = detector.detectAndCompute(image, None)
    
    return keypoints, descriptors

# ==========================================
# 특징점 매칭
# - SIFT / ORB는 BFMatcher 사용
# - ALIKED는 LightGlueMatcher 사용
# - ALIKED 좌표는 원본 이미지 좌표 그대로 전달
# - setPairInfo()에 Reference / Scene 크기 전달
# ==========================================
def match_features(des1, des2, kp1, kp2, ref_size, sce_size, method="SIFT"):
    
    # =====================================
    # SIFT / ORB
    # =====================================
    if method == "SIFT":
        norm = cv2.NORM_L2

    elif method == "ORB":
        norm = cv2.NORM_HAMMING

    # =====================================
    # ALIKED + LightGlue
    # =====================================
    elif method == "ALIKED":
        matcher = cv2.LightGlueMatcher.create(
            LIGHTGLUE_ONNX_PATH
        )

        pts1 = np.ascontiguousarray(
            [kp.pt for kp in kp1],
            dtype=np.float32
        )

        pts2 = np.ascontiguousarray(
            [kp.pt for kp in kp2],
            dtype=np.float32
        )

        des1 = np.ascontiguousarray(
            des1,
            dtype=np.float32
        )

        des2 = np.ascontiguousarray(
            des2,
            dtype=np.float32
        )

        matcher.setPairInfo(
            queryKpts=pts1,
            trainKpts=pts2,
            queryImageSize=ref_size,
            trainImageSize=sce_size
        )

        matches = matcher.match(
            des1,
            des2
        )

        logger.debug(
            "LightGlue matches: %d",
            len(matches),
        )

        return matches

    else:
        raise ValueError(
            f"지원하지 않는 메소드: {method}"
        )
    
    bf = cv2.BFMatcher(norm)

    # =====================================
    # SIFT / ORB 매칭
    # =====================================
    matches = bf.knnMatch(
        des1,
        des2,
        k=2
    )

    return matches

# ==========================================
# Lowe Ratio Test
# - KNN 매칭 결과에서 모호한 매칭 제거
# - 가장 가까운 매칭이 두 번째 매칭보다
#   충분히 가까운 경우에만 유지
# ==========================================
def ratio_test(matches, ratio=0.75):
    good_matches = []

    for m, n in matches:
        # m의 거리기 n*ratio 보다 작은지 비교 작다면 good_matches에 m추가
        # m의 거리가 n의 거리보다 최소 25% 작아야 한다
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches

# ==========================================
# 매칭 결과를 좌표쌍으로 변환
# - DMatch의 queryIdx로 Reference 좌표 추출
# - DMatch의 trainIdx로 Scene 좌표 추출
# - Homography 계산에 사용할 좌표쌍 반환
# ==========================================
def get_matched_points(kp1, kp2, matches):
    pts1 = []
    pts2 = []

    # 생성된 매칭을 좌표쌍으로 변환
    for match in matches:
        pts1.append(kp1[match.queryIdx].pt)
        pts2.append(kp2[match.trainIdx].pt)

    return pts1, pts2

# ==========================================
# Homography 계산
# - 매칭된 좌표를 float32 NumPy 배열로 변환
# - 최소 4개의 대응점이 있는지 확인
# - RANSAC으로 이상 매칭을 제거하며 Homography 계산
# ==========================================
def find_homography(pts1, pts2):
    # Numpy 배열로 변환, 타입은 float32(실수형)
    pts1 = np.array(pts1, dtype=np.float32).reshape(-1, 2)
    pts2 = np.array(pts2, dtype=np.float32).reshape(-1, 2)
    
    # 대응점이 4개가 없으면 계산 불가
    if len(pts1) < 4 or len(pts2) < 4:
        return None, None

    # RANSAC를 사용하여 매칭 검사 pts1과 pts2의 점의 차이가 5픽셀 이내로 설정
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    return H, mask

# ==========================================
# 특징점 매칭 결과 시각화
# - 두 이미지 사이의 매칭된 특징점을 선으로 연결
# - 매칭되지 않은 특징점은 표시하지 않음
# ==========================================
def draw_matches(img1, kp1, img2, kp2, matches):
    # drawMatches함수를 이용하여 특징점끼리 선으로 연결
    # None : 새로운 이미지 생성하여 반환
    # DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS : 매칭되지 않은 점은 표시하지 않음    
    result = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return result
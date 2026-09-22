# 라이브러리
import cv2
import numpy as np
# import onnxruntime as ort

ALIKED_PATH = "models/aliked-n16rot-top2k-1280.onnx"
DISK_PATH = "models/"

# 특징점 찾는 함수
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
        
        # ========== 테스트 코드 ========
        # print("input size :", params.inputSize)
        # print("backend:", params.backend)
        # print("engine:", params.engine)
        # print("normalizeDescriptors:", params.normalizeDescriptors)
        # print("target:", params.target)
        # session = ort.InferenceSession(
        #     ALIKED_PATH,
        #     providers=["CPUExecutionProvider"]
        # )

        # print("=== INPUT ===")

        # for x in session.get_inputs():
        #     print(x.name)
        #     print(x.shape)
        #     print(x.type)

        # print("=== OUTPUT ===")

        # for x in session.get_outputs():
        #     print(x.name)
        #     print(x.shape)
        #     print(x.type)
        # # ===============================

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

# 특징점끼리 비교하는 함수
def match_features(des1, des2, method="SIFT"):
    # SIFT와 ORB는 비교 방식이 다르기 때문에 분기 필요
    # SIFT는 실수형태의 값을 사용
    if method == "SIFT" or method == "ALIKED" :
        norm = cv2.NORM_L2          # 유클리드 거리
    # ORB는 바이너리 형태의 값을 사용
    elif method == "ORB":
        norm = cv2.NORM_HAMMING     # 해밍 거리
    else:
        # 에러 발생시 즉시 중단하고 에러 출력
        raise ValueError(f"지원하지 않는 메소드 : {method}")
    
    # 이미지 1의 특징점을 이미지 2의 모든 특징점과 하나씩 비교
    bf = cv2.BFMatcher(norm)    
    
    # 가장 가까운 2개의 매칭 후보를 가져온다
    matches = bf.knnMatch(des1, des2, k=2)
    
    return matches

# 애매한 매칭 제거 함수
def ratio_test(matches, ratio=0.75):
    good_matches = []

    for m, n in matches:
        # m의 거리기 n*ratio 보다 작은지 비교 작다면 good_matches에 m추가
        # m보다 n이 최소 25% 이상 우수할 경우에 선택하는 것임
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches

# 좌표쌍 반환 함수
def get_matched_points(kp1, kp2, matches):
    pts1 = []
    pts2 = []

    # 생성된 매칭을 좌표쌍으로 변환
    for match in matches:
        pts1.append(kp1[match.queryIdx].pt)
        pts2.append(kp2[match.trainIdx].pt)

    return pts1, pts2

# 공간적으로 틀린 매칭 제거하고 homography반환하는 함수
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

# 매칭 결과 이미지 출력 함수
def draw_matches(img1, kp1, img2, kp2, matches):
    # drawMatches함수를 이용하여 특징점끼리 선으로 연결
    # None : 새로운 이미지 생성하여 반환
    # DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS : 매칭되지 않은 점은 표시하지 않음
    result = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return result
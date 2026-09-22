import cv2
from logger import logger
from feature_match import (
    detect_features,
    match_features,
    ratio_test, 
    get_matched_points,
    find_homography,
    draw_matches,
)


from object_finder import find_object


# ============ 테스트 코드 ============
# print(cv2.__version__)
# print(hasattr(cv2, "ALIKED"))
# print(hasattr(cv2.ALIKED, "create"))
# # print("opencv version : ",cv2.__version__)
# print(cv2.__version__)
# print(cv2.ALIKED)
# print(cv2.ALIKED.Params)
# ====================================

ref_path = "data/objects/notebook.png"
sce_path = "data/objects/desk.jpg"

ref_img = cv2.imread(ref_path)  # 찾을 객체
sce_img = cv2.imread(sce_path)  # 원본 이미지

# ORB는 그레이스케일 이미지 사용
# ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)  # 찾을 객체
# sce_img = cv2.imread(sce_path, cv2.COLOR_BGR2GRAY)  # 원본 이미지

if ref_img is None:
    logger.info("Reference 이미지를 불러오지 못했습니다.")
    exit()

if sce_img is None:
    logger.info("Scene 이미지를 불러오지 못했습니다.")
    exit()

# 객체 찾기 (SIFT, ORB)
# result = find_object(ref_img, sce_img, method="SIFT", ref_name=ref_path, sce_name=sce_path)
# result = find_object(ref_img, sce_img, method="ORB", ref_name=ref_path, sce_name=sce_path)
# OpenCV 5.0에서 추가된 특징 검출기
result = find_object(ref_img, sce_img, method="ALIKED", ref_name=ref_path, sce_name=sce_path)
# result = find_object(ref_img, sce_img, method="DISK", ref_name=ref_path, sce_name=sce_path)

if result is None:
    logger.info("객체를 찾지 못했습니다.")
    exit()

# 파일 생성
cv2.imwrite(
    "results/object_detection/object_result.jpg",
    result
)

# 이미지 출력
cv2.imshow("Object Detection", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
from feature_match import (
    detect_features,
    match_features,
    ratio_test,
    get_matched_points,
    find_homography,
    draw_matches,
)


from object_finder import find_object


ref_img = cv2.imread("data/objects/Lantern.png")  # 찾을 객체
sce_img = cv2.imread("data/objects/desk.jpg")  # 원본 이미지

if ref_img is None:
    print("Reference 이미지를 불러오지 못했습니다.")
    exit()

if sce_img is None:
    print("Scene 이미지를 불러오지 못했습니다.")
    exit()
    
result = find_object(
    ref_img,
    sce_img,
    method="SIFT"
)

if result is None:
    print("객체를 찾지 못했습니다.")
    exit()

cv2.imwrite(
    "results/object_detection/object_result.jpg",
    result
)

cv2.imshow("Object Detection", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
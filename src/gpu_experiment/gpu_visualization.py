"""GPU 객체 검출 결과 시각화 및 이미지 저장 기능."""

from pathlib import Path

import cv2
import numpy as np


# ==========================================
# 최종 객체 검출 Polygon 그리기
# - detector가 복원한 원본 파노라마 좌표를 그대로 사용
# ==========================================
def draw_detection_result(panorama_image, detection_result):
    """원본 파노라마에 최종 객체 Polygon을 그린 이미지를 반환한다."""
    result_image = panorama_image.copy()

    # 검출 실패 시에는 원본을 그대로 반환한다.
    # 실패 사례에서도 어떤 입력을 검사했는지 결과 이미지를 남길 수 있다.
    if detection_result.polygon is None:
        return result_image

    cv2.polylines(
        result_image,
        [detection_result.polygon],
        True,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )
    return result_image


# ==========================================
# 최종 후보 Tile 매칭 이미지 생성
# - 검출 결과의 최종 Tile만 한 번 다시 GPU 매칭
# - 탐색 과정 전체의 모든 Tile 이미지를 저장하지 않아 I/O를 줄임
# ==========================================
def draw_final_matches(
    detector,
    reference_image,
    panorama_image,
    detection_result,
):
    """기준 물체와 최종 후보 Tile의 LightGlue 매칭 이미지를 반환한다."""
    best = detection_result.best
    if best is None:
        return None

    # 검출 당시와 동일한 크기의 축소 파노라마를 재생성한다.
    # best.x, best.y는 이 축소 파노라마 좌표계 기준이다.
    detection_image = _resize_to_detection_size(
        panorama_image,
        detection_result.detection_size,
    )
    tile = detection_image[
        best.y:best.y + best.height,
        best.x:best.x + best.width,
    ]
    if tile.size == 0:
        return None

    # 저장용 매칭 이미지는 탐색 시간에 포함하지 않는다.
    # 탐색 시 사용한 추출기·매처를 재사용해 가중치 재로딩을 피한다.
    reference_features = detector.extractor.extract(reference_image)
    tile_features = detector.extractor.extract(tile)
    match_result = detector.matcher.match(reference_features, tile_features)

    # OpenCV drawMatches는 cv2.KeyPoint와 cv2.DMatch 형식을 요구한다.
    # GPU Tensor 좌표·인덱스를 이 형식으로만 변환해 화면에 그린다.
    reference_points = _to_keypoints(reference_features.features["keypoints"])
    tile_points = _to_keypoints(tile_features.features["keypoints"])
    matches = _to_dmatches(match_result.matches, match_result.scores)

    return cv2.drawMatches(
        reference_image,
        reference_points,
        tile,
        tile_points,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


# ==========================================
# GPU 최종 결과 이미지 저장
# ==========================================
def save_final_visuals(
    detector,
    reference_image,
    panorama_image,
    detection_result,
    output_dir,
    *,
    method_name="ALIKED_GPU",
):
    """최종 Polygon·키포인트 매칭 이미지를 저장하고 경로를 반환한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "result_path": None,
        "match_image_path": None,
    }

    # 검출 실패 여부와 관계없이 결과 파노라마는 저장한다.
    result_path = output_dir / f"{method_name}_result.png"
    result_image = draw_detection_result(panorama_image, detection_result)
    if not cv2.imwrite(str(result_path), result_image):
        raise OSError(f"GPU 검출 결과를 저장할 수 없습니다: {result_path}")
    paths["result_path"] = str(result_path)

    # 최종 후보가 없으면 매칭 이미지 자체를 만들 수 없다.
    match_image = draw_final_matches(
        detector,
        reference_image,
        panorama_image,
        detection_result,
    )
    if match_image is not None:
        match_path = output_dir / f"{method_name}_keypoint_matches.png"
        if not cv2.imwrite(str(match_path), match_image):
            raise OSError(f"GPU 매칭 이미지를 저장할 수 없습니다: {match_path}")
        paths["match_image_path"] = str(match_path)

    return paths


def _resize_to_detection_size(panorama_image, detection_size):
    """검출 당시 기록한 크기로 파노라마를 다시 축소한다."""
    width, height = detection_size
    if panorama_image.shape[1] == width and panorama_image.shape[0] == height:
        return panorama_image
    return cv2.resize(
        panorama_image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def _to_keypoints(keypoints_tensor):
    """[1, N, 2] GPU Tensor를 OpenCV KeyPoint 목록으로 변환한다."""
    points = keypoints_tensor[0].detach().cpu().numpy()
    return [
        cv2.KeyPoint(float(point[0]), float(point[1]), 1.0)
        for point in points
    ]


def _to_dmatches(matches_tensor, scores_tensor):
    """LightGlue 매칭 인덱스를 OpenCV DMatch 목록으로 변환한다."""
    matches = matches_tensor.detach().cpu().numpy()
    scores = scores_tensor.detach().cpu().numpy()
    return [
        # DMatch distance는 작을수록 좋은 값이다.
        # LightGlue score(클수록 좋음)를 1-score로 뒤집어 표현한다.
        cv2.DMatch(
            int(match[0]),
            int(match[1]),
            float(1.0 - score),
        )
        for match, score in zip(matches, scores)
    ]

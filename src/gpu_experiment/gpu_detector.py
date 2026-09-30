"""GPU ALIKED·LightGlue 기반 독립 객체 검출기.

기존 CPU 파이프라인을 호출하지 않고 GPU 특징점 추출과 매칭만 사용한다.
호모그래피 계산과 Polygon 좌표 복원에는 OpenCV의 CPU 기하 연산을 사용한다.
"""

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from src.gpu_experiment.gpu_aliked import GPUALIKEDExtractor
from src.gpu_experiment.gpu_lightglue import GPULightGlueMatcher


@dataclass(frozen=True)
class TileCandidate:
    """Tile 하나의 ALIKED·LightGlue·RANSAC 결과.

    homography는 기준 물체 좌표를 해당 Tile 내부 좌표로 변환한다.
    x, y를 더하면 축소 파노라마의 전역 좌표가 된다.
    """

    index: int
    x: int
    y: int
    width: int
    height: int
    match_count: int
    inlier_count: int
    inlier_ratio: float
    is_detection: bool
    homography: np.ndarray
    feature_elapsed: float
    matching_elapsed: float


@dataclass(frozen=True)
class GPUDetectionResult:
    """Coarse-to-Fine GPU 객체 검출 결과.

    polygon은 원본 파노라마 좌표계의 [4, 1, 2] 정수 배열이다.
    따라서 cv2.polylines()에 바로 전달할 수 있다.
    """

    is_detection: bool
    polygon: np.ndarray | None
    best: TileCandidate | None
    detection_scale: float
    detection_size: tuple
    stats: dict


class GPUObjectDetector:
    """GPU 특징점 매칭과 CPU RANSAC을 결합한 Coarse-to-Fine 검출기."""

    def __init__(
        self,
        *,
        tile_size=(512, 512),
        coarse_stride=(512, 512),
        fine_stride=(256, 256),
        coarse_top_k=3,
        early_stop_inliers=50,
        min_inliers=8,
        ransac_reproj_threshold=5.0,
        minimum_tile_size=(300, 250),
        black_pixel_threshold=8,
        max_black_tile_ratio=0.7,
        max_keypoints=1000,
        feature_resize=1280,
        extractor=None,
        matcher=None,
    ):
        # 아래 기본값은 기존 config.json의 객체 검출 기준과 맞춘 값이다.
        # GPU와 CPU를 비교할 때 탐색 조건이 달라지지 않도록 생성 시점에 고정한다.
        self.tile_size = tuple(tile_size)
        self.coarse_stride = tuple(coarse_stride)
        self.fine_stride = tuple(fine_stride)
        self.coarse_top_k = coarse_top_k
        self.early_stop_inliers = early_stop_inliers
        self.min_inliers = min_inliers
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.minimum_tile_size = tuple(minimum_tile_size)
        self.black_pixel_threshold = black_pixel_threshold
        self.max_black_tile_ratio = max_black_tile_ratio
        # 외부에서 extractor/matcher를 주입하면 테스트·실험 설정을 바꿀 수 있다.
        # 주입하지 않으면 GPU 기본 구현을 생성한다.
        self.extractor = extractor or GPUALIKEDExtractor(
            max_keypoints=max_keypoints,
            resize=feature_resize,
        )
        self.matcher = matcher or GPULightGlueMatcher()

    # ==========================================
    # 전체 파노라마 객체 검출
    # - 기준 물체 특징점은 한 번만 계산
    # - Coarse 탐색 뒤 상위 후보 주변에서 Fine 탐색
    # - 최종 Polygon은 원본 파노라마 좌표로 복원
    # ==========================================
    def detect(self, reference_image, panorama_image, *, detection_max_side=3072):
        if reference_image is None or panorama_image is None:
            raise ValueError("기준 물체와 파노라마 이미지를 모두 입력해야 합니다.")

        # 큰 파노라마는 먼저 축소해 Tile 개수와 GPU 추론 횟수를 줄인다.
        # detection_scale은 마지막 Polygon을 원본 크기로 되돌릴 때 사용한다.
        detection_image, detection_scale = self._resize_for_detection(
            panorama_image,
            detection_max_side,
        )
        start_time = perf_counter()
        # 기준 물체 특징점은 모든 Tile에서 같으므로 한 번만 계산해 재사용한다.
        reference_features = self.extractor.extract(reference_image)

        coarse_positions = self._generate_positions(
            detection_image.shape[:2],
            self.coarse_stride,
        )
        coarse_candidates, coarse_stats = self._scan_positions(
            reference_features,
            detection_image,
            coarse_positions,
        )
        # Inlier 수 -> Inlier 비율 -> 매칭 수 순으로 강한 후보를 우선한다.
        coarse_candidates.sort(key=self._candidate_sort_key, reverse=True)
        top_coarse = coarse_candidates[: self.coarse_top_k]

        # Coarse 전체를 다시 검사하지 않고 상위 후보 주변 3x3 위치만 세밀하게 검사한다.
        fine_positions = self._generate_fine_positions(
            top_coarse,
            set(coarse_positions),
            detection_image.shape[:2],
        )
        fine_candidates, fine_stats = self._scan_positions(
            reference_features,
            detection_image,
            fine_positions,
            early_stop=True,
        )
        fine_candidates.sort(key=self._candidate_sort_key, reverse=True)

        # Fine 후보가 없으면 Coarse 최상위 후보를 사용해 실패 원인을 남긴다.
        best = fine_candidates[0] if fine_candidates else (
            top_coarse[0] if top_coarse else None
        )
        # 최종 Homography와 Tile 시작 좌표를 합쳐 원본 파노라마 Polygon을 만든다.
        polygon = self._build_polygon(
            reference_image,
            best,
            detection_scale,
        )
        elapsed = perf_counter() - start_time

        stats = {
            # 특징점 수/시간은 기준 물체 한 장에 대한 값이다.
            "reference_keypoints": reference_features.keypoint_count,
            "reference_feature_elapsed": reference_features.elapsed,
            "coarse_checked": coarse_stats["checked"],
            "coarse_skipped_black": coarse_stats["skipped_black"],
            "fine_checked": fine_stats["checked"],
            "fine_skipped_black": fine_stats["skipped_black"],
            "total_checked": (
                coarse_stats["checked"] + fine_stats["checked"]
            ),
            "total_skipped_black": (
                coarse_stats["skipped_black"] + fine_stats["skipped_black"]
            ),
            "early_stopped": fine_stats["early_stopped"],
            "elapsed": elapsed,
        }
        return GPUDetectionResult(
            is_detection=best is not None and best.is_detection,
            polygon=polygon,
            best=best,
            detection_scale=detection_scale,
            detection_size=(
                detection_image.shape[1],
                detection_image.shape[0],
            ),
            stats=stats,
        )

    # ==========================================
    # Tile 탐색
    # - 검은 캔버스 비율이 높은 Tile은 GPU 추론 전에 제외
    # - Homography가 계산된 후보만 반환
    # ==========================================
    def _scan_positions(
        self,
        reference_features,
        scene_image,
        positions,
        *,
        early_stop=False,
    ):
        candidates = []
        checked = 0
        skipped_black = 0
        early_stopped = False
        scene_height, scene_width = scene_image.shape[:2]
        tile_width, tile_height = self.tile_size

        for index, (x, y) in enumerate(positions, start=1):
            x2 = min(x + tile_width, scene_width)
            y2 = min(y + tile_height, scene_height)
            tile = scene_image[y:y2, x:x2]

            # 직접 합성 파노라마의 검은 여백은 특징점이 거의 없으므로
            # GPU ALIKED 실행 전에 제외해 불필요한 추론을 방지한다.
            if self._black_ratio(tile) >= self.max_black_tile_ratio:
                skipped_black += 1
                continue

            checked += 1
            # Tile마다 Scene 특징점은 달라지므로 새로 계산해야 한다.
            candidate = self._detect_in_tile(
                reference_features,
                tile,
                index=index,
                x=x,
                y=y,
            )
            if candidate is None:
                continue

            candidates.append(candidate)
            # 충분한 Inlier가 나온 Fine 후보는 이미 신뢰도가 높으므로
            # 이후 후보를 생략해 전체 검출 시간을 줄인다.
            if early_stop and candidate.inlier_count >= self.early_stop_inliers:
                early_stopped = True
                break

        return candidates, {
            "checked": checked,
            "skipped_black": skipped_black,
            "early_stopped": early_stopped,
        }

    # ==========================================
    # Tile 하나의 특징점 매칭과 RANSAC
    # ==========================================
    def _detect_in_tile(self, reference_features, tile, *, index, x, y):
        # 1) Tile ALIKED 특징점 추출(GPU)
        scene_features = self.extractor.extract(tile)
        # 2) 기준 물체와 Tile의 LightGlue 매칭(GPU)
        match_result = self.matcher.match(reference_features, scene_features)
        # 3) 최종 대응점 좌표만 CPU로 가져와 RANSAC에 사용
        reference_points, scene_points = self.matcher.matched_points(
            reference_features,
            scene_features,
            match_result,
        )

        # Homography는 최소 네 개의 대응점이 필요하다.
        if len(reference_points) < 4:
            return None

        # RANSAC은 기하적으로 맞지 않는 오매칭(Outlier)을 제거한다.
        homography, mask = cv2.findHomography(
            reference_points,
            scene_points,
            cv2.RANSAC,
            self.ransac_reproj_threshold,
        )
        if homography is None or mask is None:
            return None

        match_count = len(reference_points)
        # mask=1인 대응점만 RANSAC 검증을 통과한 Inlier다.
        inlier_count = int(mask.sum())
        inlier_ratio = inlier_count / match_count
        return TileCandidate(
            index=index,
            x=x,
            y=y,
            width=tile.shape[1],
            height=tile.shape[0],
            match_count=match_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            is_detection=inlier_count >= self.min_inliers,
            homography=homography,
            feature_elapsed=scene_features.elapsed,
            matching_elapsed=match_result.elapsed,
        )

    # ==========================================
    # Coarse 및 Fine Tile 좌표 생성
    # ==========================================
    def _generate_positions(self, scene_shape, stride):
        scene_height, scene_width = scene_shape
        stride_x, stride_y = stride
        minimum_width, minimum_height = self.minimum_tile_size
        positions = []

        # 위에서 아래(y), 왼쪽에서 오른쪽(x) 순으로 Tile 시작 좌표를 생성한다.
        # 마지막 경계 Tile은 minimum_tile_size보다 작으면 정보량이 부족해 제외한다.
        for y in range(0, scene_height, stride_y):
            for x in range(0, scene_width, stride_x):
                if min(x + self.tile_size[0], scene_width) - x < minimum_width:
                    continue
                if min(y + self.tile_size[1], scene_height) - y < minimum_height:
                    continue
                positions.append((x, y))
        return positions

    def _generate_fine_positions(
        self,
        coarse_candidates,
        coarse_positions,
        scene_shape,
    ):
        scene_height, scene_width = scene_shape
        fine_stride_x, fine_stride_y = self.fine_stride
        minimum_width, minimum_height = self.minimum_tile_size
        positions = set()

        for candidate in coarse_candidates:
            # 각 Coarse 후보 주변의 3 x 3 위치를 Fine 후보로 만든다.
            for offset_y in (-fine_stride_y, 0, fine_stride_y):
                for offset_x in (-fine_stride_x, 0, fine_stride_x):
                    x = candidate.x + offset_x
                    y = candidate.y + offset_y
                    if x < 0 or y < 0 or x >= scene_width or y >= scene_height:
                        continue
                    if (x, y) in coarse_positions:
                        continue
                    if min(x + self.tile_size[0], scene_width) - x < minimum_width:
                        continue
                    if min(y + self.tile_size[1], scene_height) - y < minimum_height:
                        continue
                    positions.add((x, y))

        # Coarse 후보와 가까운 위치부터 검사해야 조기 종료가 더 빨라진다.
        return sorted(
            positions,
            key=lambda point: (
                min(
                    abs(point[0] - candidate.x)
                    + abs(point[1] - candidate.y)
                    for candidate in coarse_candidates
                ),
                point[1],
                point[0],
            ),
        ) if coarse_candidates else []

    # ==========================================
    # 원본 파노라마 좌표의 검출 Polygon 생성
    # ==========================================
    @staticmethod
    def _build_polygon(reference_image, candidate, detection_scale):
        if candidate is None or not candidate.is_detection:
            return None

        reference_height, reference_width = reference_image.shape[:2]
        # 기준 물체 이미지의 네 모서리를 Homography로 Tile 내부 좌표로 보낸다.
        corners = np.float32(
            [
                [0, 0],
                [reference_width, 0],
                [reference_width, reference_height],
                [0, reference_height],
            ]
        ).reshape(-1, 1, 2)
        polygon = cv2.perspectiveTransform(corners, candidate.homography)
        # Tile 내부 좌표 + Tile 시작 위치 = 축소 파노라마 전역 좌표
        polygon += np.float32([[[candidate.x, candidate.y]]])
        # 축소 탐색 좌표를 원본 파노라마 좌표로 복원한다.
        return np.int32(np.round(polygon / detection_scale))

    @staticmethod
    def _candidate_sort_key(candidate):
        return (
            candidate.inlier_count,
            candidate.inlier_ratio,
            candidate.match_count,
        )

    def _resize_for_detection(self, image, max_side):
        image_height, image_width = image.shape[:2]
        longest_side = max(image_height, image_width)
        if longest_side <= max_side:
            return image, 1.0

        # 가로세로 비율을 유지한 채 긴 변만 max_side에 맞춘다.
        scale = max_side / longest_side
        resized = cv2.resize(
            image,
            (
                round(image_width * scale),
                round(image_height * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _black_ratio(self, tile):
        if tile.size == 0:
            return 1.0
        # B, G, R 모두 임계값 이하인 픽셀만 검은 여백으로 간주한다.
        black_pixels = np.all(tile <= self.black_pixel_threshold, axis=2)
        return float(np.mean(black_pixels))

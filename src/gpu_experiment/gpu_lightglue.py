"""PyTorch CUDA 기반 LightGlue 특징점 매칭기."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from lightglue import LightGlue

from src.gpu_experiment.gpu_aliked import GPUFeatureResult


@dataclass(frozen=True)
class GPUMatchResult:
    """LightGlue 매칭 결과와 실행 시간.

    matches는 [N, 2] Tensor이며 각 행은
    [기준 물체 keypoint 번호, Tile keypoint 번호]를 뜻한다.
    """

    matches: torch.Tensor
    scores: torch.Tensor
    elapsed: float

    @property
    def match_count(self):
        """필터를 통과한 최종 매칭 개수를 반환한다."""
        return int(self.matches.shape[0])


class GPULightGlueMatcher:
    """CUDA에서 ALIKED 기술자를 LightGlue로 매칭한다."""

    def __init__(
        self,
        *,
        filter_threshold=0.1,
        device="cuda",
        model_cache_dir=None,
    ):
        # 이 모듈은 CPU fallback을 제공하지 않는다.
        # GPU 비교 중 CPU로 실행되는 실수를 방지하기 위해 명시적으로 확인한다.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA를 사용할 수 없습니다. GPU CUDA 환경에서 실행하세요."
            )

        self.device = torch.device(device)

        # ALIKED 추출기와 동일한 가중치 캐시를 사용한다.
        # 첫 실행에만 aliked_lightglue 가중치가 이 폴더에 다운로드된다.
        self.model_cache_dir = (
            Path(model_cache_dir)
            if model_cache_dir is not None
            else Path(__file__).resolve().parents[2]
            / "models"
            / "gpu_experiment"
        )
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(self.model_cache_dir))

        # features="aliked": 128차원 ALIKED descriptor용 학습 가중치 선택
        # filter_threshold: 매칭 신뢰도 임계값
        # mp=True: RTX GPU에서 float16 혼합 정밀도를 사용해 매칭을 가속
        self.matcher = LightGlue(
            features="aliked",
            filter_threshold=filter_threshold,
            mp=True,
        ).eval().to(self.device)

    def match(self, reference_features, scene_features):
        """두 ALIKED 결과를 입력받아 GPU에서 최종 매칭을 계산한다."""
        # GPUFeatureResult 또는 features dictionary를 모두 받을 수 있게 처리한다.
        reference = self._get_features(reference_features)
        scene = self._get_features(scene_features)
        self._validate_features(reference, "reference")
        self._validate_features(scene, "scene")

        # 특징점과 기술자는 GPU에 남긴 채 LightGlue에 바로 전달한다.
        # CPU로 복사하면 PCIe 전송 비용이 생겨 GPU 매칭 이점이 줄어든다.
        torch.cuda.synchronize(self.device)
        start_time = perf_counter()
        with torch.inference_mode():
            output = self.matcher(
                {
                    "image0": reference,
                    "image1": scene,
                }
            )
        torch.cuda.synchronize(self.device)

        return GPUMatchResult(
            matches=output["matches"][0],
            scores=output["scores"][0],
            elapsed=perf_counter() - start_time,
        )

    @staticmethod
    def matched_points(reference_features, scene_features, match_result):
        """호모그래피 계산용 대응점 좌표를 NumPy 배열로 변환한다."""
        reference = GPULightGlueMatcher._get_features(reference_features)
        scene = GPULightGlueMatcher._get_features(scene_features)
        matches = match_result.matches

        if match_result.match_count == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty

        # 이 지점에서만 RANSAC(OpenCV CPU)을 위해 GPU -> CPU 복사를 수행한다.
        # 복사 대상은 이미지 전체가 아니라 최종 대응점 좌표 N개뿐이다.
        reference_points = reference["keypoints"][0, matches[:, 0]]
        scene_points = scene["keypoints"][0, matches[:, 1]]
        return (
            reference_points.detach().cpu().numpy().astype(np.float32),
            scene_points.detach().cpu().numpy().astype(np.float32),
        )

    @staticmethod
    def _get_features(feature_result):
        # 추출 시간까지 담긴 dataclass를 받으면 LightGlue 입력 dictionary만 꺼낸다.
        if isinstance(feature_result, GPUFeatureResult):
            return feature_result.features
        return feature_result

    def _validate_features(self, features, label):
        # LightGlue는 좌표, descriptor, 원본 이미지 크기로 좌표를 정규화한다.
        required_keys = ("keypoints", "descriptors", "image_size")
        missing = [key for key in required_keys if key not in features]
        if missing:
            raise ValueError(f"{label} 특징점 결과에 필요한 값이 없습니다: {missing}")
        # torch.device("cuda")와 torch.device("cuda:0")는 표현만 다를 수 있다.
        # 따라서 type과 명시된 GPU 번호를 나누어 비교한다.
        descriptor_device = features["descriptors"].device
        expected_index = self.device.index
        if (
            descriptor_device.type != self.device.type
            or (
                expected_index is not None
                and descriptor_device.index != expected_index
            )
        ):
            raise ValueError(
                f"{label} 특징점이 {self.device}에 있지 않습니다. "
                "GPUALIKEDExtractor 결과를 그대로 사용하세요."
            )

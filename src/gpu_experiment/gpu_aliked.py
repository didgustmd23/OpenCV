"""PyTorch CUDA 기반 ALIKED 특징점 추출기.

이 모듈은 기존 OpenCV ALIKED 구현과 분리되어 동작한다.
입력 이미지는 OpenCV 형식의 BGR ``numpy.ndarray``이며, 반환 특징점은
다음 단계의 GPU LightGlue 매칭에서 GPU Tensor 상태로 그대로 사용한다.
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from lightglue import ALIKED


@dataclass(frozen=True)
class GPUFeatureResult:
    """ALIKED 특징점 추출 결과와 실행 시간.

    features에는 LightGlue가 요구하는 keypoints, descriptors,
    keypoint_scores, image_size Tensor가 GPU 상태로 들어 있다.
    """

    features: dict
    elapsed: float

    @property
    def keypoint_count(self):
        """배치 하나에 추출된 특징점 개수를 반환한다."""
        return int(self.features["keypoints"].shape[1])


class GPUALIKEDExtractor:
    """CUDA에서 ALIKED 특징점과 기술자를 추출한다."""

    def __init__(
        self,
        *,
        max_keypoints=1000,
        resize=1280,
        model_name="aliked-n16rot",
        device="cuda",
        model_cache_dir=None,
    ):
        # CUDA를 사용할 수 없는 환경에서 CPU로 조용히 전환하면
        # GPU 실험 결과가 왜곡될 수 있으므로 즉시 중단한다.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA를 사용할 수 없습니다. GPU CUDA 환경에서 실행하세요."
            )

        # ``cuda``는 기본 GPU(cuda:0)를 의미한다.
        # 이후 입력 Tensor와 모델을 같은 장치에 올려야 GPU 연산이 가능하다.
        self.device = torch.device(device)
        self.resize = resize

        # LightGlue 패키지는 가중치가 없을 때 torch.hub로 자동 다운로드한다.
        # 기본 사용자 캐시 대신 프로젝트 전용 폴더를 사용해 기존 환경과 분리한다.
        self.model_cache_dir = (
            Path(model_cache_dir)
            if model_cache_dir is not None
            else Path(__file__).resolve().parents[2]
            / "models"
            / "gpu_experiment"
        )
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(self.model_cache_dir))

        # detection_threshold=0.0이면 점수 임계값 대신 max_num_keypoints를 적용한다.
        # 기존 OpenCV ALIKED top1k 모델과 맞추기 위해 기본값을 1,000개로 둔다.
        # eval(): BatchNorm/Dropout을 추론 모드로 고정
        # to(device): 모델 가중치를 GPU 메모리로 이동
        self.model = ALIKED(
            model_name=model_name,
            max_num_keypoints=max_keypoints,
            detection_threshold=0.0,
        ).eval().to(self.device)

    def extract(self, image):
        """BGR 이미지를 입력받아 GPU 특징점·기술자를 반환한다."""
        # cv2.imread() 결과는 BGR 3채널 uint8 배열이다.
        # 이 형식이 아니면 RGB 변환과 Tensor 변환 결과를 보장할 수 없다.
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("BGR 3채널 이미지가 필요합니다.")

        # CPU NumPy 배열을 RGB float32 GPU Tensor [1, 3, H, W]로 변환한다.
        image_tensor = self._to_tensor(image)

        # CUDA 함수 호출은 기본적으로 비동기다.
        # 동기화 없이 시간을 재면 GPU 작업 완료 전 시간이 측정되어 지나치게 작아진다.
        torch.cuda.synchronize(self.device)
        start_time = perf_counter()

        # 추론에서는 gradient가 필요 없으므로 메모리 사용량과 연산을 줄인다.
        # resize는 긴 변을 기준으로 축소/확대해 모델 입력 크기를 제한한다.
        with torch.inference_mode():
            features = self.model.extract(
                image_tensor,
                resize=self.resize,
            )
        torch.cuda.synchronize(self.device)

        return GPUFeatureResult(
            features=features,
            elapsed=perf_counter() - start_time,
        )

    def _to_tensor(self, image):
        """OpenCV BGR uint8 이미지를 RGB float32 GPU Tensor로 변환한다."""
        # BGR -> RGB: PyTorch ALIKED 학습 입력은 RGB 순서를 사용한다.
        # ascontiguousarray: channel reverse 결과의 음수 stride를 제거해
        # torch.from_numpy()가 안전하게 공유할 수 있는 메모리로 만든다.
        rgb_image = np.ascontiguousarray(image[:, :, ::-1])
        return (
            # [H, W, C] -> [C, H, W], 배치 차원 추가 -> [1, C, H, W]
            torch.from_numpy(rgb_image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            # uint8 0~255 -> float32 0.0~1.0, GPU 메모리로 전송
            .to(self.device, dtype=torch.float32)
            / 255.0
        )

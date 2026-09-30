# GPU ALIKED · LightGlue 객체 검출

이 폴더는 기존 OpenCV 기반 CPU 객체 검출 코드와 **분리하여** ALIKED 특징점 추출과 LightGlue 매칭을 NVIDIA GPU에서 실행하는 객체 검출 기능입니다. 기존 `src/object_finder.py`, `src/tile_scanner.py` 및 최종 보고서용 CPU 실험 결과에는 영향을 주지 않습니다.

## 1. 처리 흐름

```text
기준 물체 이미지 + 파노라마 이미지
        ↓
GPU ALIKED-n16rot: 특징점·descriptor 추출
        ↓
GPU LightGlue: 특징점 매칭
        ↓
OpenCV CPU RANSAC: 호모그래피·Inlier 계산
        ↓
Coarse-to-Fine Tile 탐색 및 객체 Polygon 표시
```

ALIKED와 LightGlue의 신경망 추론 및 매칭은 GPU에서 수행합니다. 호모그래피 계산, 이미지 입출력, 결과 이미지 그리기는 OpenCV CPU 기능을 사용합니다.

## 2. 구성

```text
src/gpu_experiment/
├── main.py                 # 직접 실행 가능한 GPU 객체 검출 진입점
├── gpu_pipeline.py         # 입력·세트 탐색·실제 검출·결과 JSON 저장
├── benchmark.py            # CPU·GPU 성능 비교 실험용 보조 모듈
├── gpu_aliked.py           # PyTorch CUDA ALIKED 특징점 추출기
├── gpu_lightglue.py        # PyTorch CUDA LightGlue 매칭기
├── gpu_detector.py         # Coarse-to-Fine GPU 객체 검출기
├── gpu_visualization.py    # Polygon·최종 키포인트 매칭 이미지 저장
└── requirements-gpu.txt    # GPU 전용 가상환경 의존성

models/gpu_experiment/checkpoints/
├── aliked-n16rot.pth
└── aliked_lightglue_v0-1_arxiv.pth

results/gpu_experiment/
└── object_detection/       # 검출 요약 JSON, 세트별 Polygon·최종 매칭 이미지
```

모델 가중치는 첫 실행 시 `models/gpu_experiment/checkpoints/`에 자동으로 내려받습니다. 모델과 결과 파일은 `.gitignore` 대상입니다.

## 3. 설치 및 GPU 확인

GPU 실험은 기존 `cv2` 환경과 별도의 CUDA 환경에서 실행합니다. PowerShell에서 프로젝트 루트로 이동한 뒤 GPU 환경을 활성화합니다.

```powershell
conda activate gpu_cv
python -m pip install -r src/gpu_experiment/requirements-gpu.txt
```

CUDA와 PyTorch가 GPU를 인식하는지 확인합니다.

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`True`와 GPU 이름이 출력되어야 합니다. `ModuleNotFoundError: No module named 'torch'`가 발생하면 GPU 환경이 아닌 다른 인터프리터로 실행한 경우이므로 `conda activate gpu_cv`를 다시 실행합니다.

## 4. 입력 데이터

단일 세트 기본 경로는 다음과 같습니다.

```text
data/objects/set01/target.jpg
results/panorama/set01/ALIKED/panorama.jpg
```

전체 세트 실행은 아래 규칙으로 파일을 자동 연결합니다.

```text
data/objects/setNN/target.jpg
results/panorama/setNN/ALIKED/panorama.jpg
```

예를 들어 `set03`을 실행하려면 기준 물체는 `data/objects/set03/target.jpg`, 파노라마는 `results/panorama/set03/ALIKED/panorama.jpg`가 필요합니다.

## 5. 실행 방법

모든 명령은 프로젝트 루트에서 실행합니다. `main.py`는 모듈 실행 없이 직접 실행하도록 구성되어 있습니다.

### 5.1 set01 GPU 객체 검출

```powershell
python src/gpu_experiment/main.py
```

기본 결과는 `results/gpu_experiment/object_detection/set01/detection_summary.json`에 저장됩니다.

### 5.2 단일 세트 결과 이미지 저장

```powershell
python src/gpu_experiment/main.py --save-visuals
```

아래 두 이미지가 생성됩니다.

```text
results/gpu_experiment/object_detection/set01/baseline/
├── ALIKED_GPU_result.png             # 검출 Polygon 표시 결과
└── ALIKED_GPU_keypoint_matches.png   # 최종 Tile 특징점 매칭
```

### 5.3 기준 물체·파노라마를 직접 지정

```powershell
python src/gpu_experiment/main.py `
  --reference data/objects/set03/target.jpg `
  --panorama results/panorama/set03/ALIKED/panorama.jpg `
  --output results/gpu_experiment/object_detection/set03/detection_summary.json `
  --save-visuals
```

### 5.4 전체 세트 GPU 일괄 실행

```powershell
python src/gpu_experiment/main.py --all-sets --save-visuals
```

결과는 `results/gpu_experiment/object_detection/batch_summary.json` 및 각 `results/gpu_experiment/object_detection/setNN/baseline/`에 저장됩니다. 일괄 검출은 모델을 한 번만 불러오고, 각 세트는 한 번씩만 처리합니다.

### 5.5 원본 해상도 그대로 실행

기본값은 긴 변을 3072px로 제한합니다. 원본 중 가장 긴 변보다 큰 값을 지정하면 축소하지 않습니다. 현재 데이터의 최장 변이 10,350px이므로 아래 명령은 원본 해상도로 실행합니다.

```powershell
python src/gpu_experiment/main.py --all-sets --save-visuals --max-side 20000
```

원본 해상도는 작은 특징점을 보존해 검출 품질을 높일 수 있지만, 타일 수와 실행 시간이 늘어납니다.

## 6. 주요 설정값

| 항목 | 기본값 | 설명 |
| --- | ---: | --- |
| ALIKED 최대 특징점 수 | 1000 | 이미지 또는 타일별 추출 상한 |
| ALIKED 입력 크기 | 1280px | 모델 입력 시 긴 변 제한 |
| Coarse Tile | 512 × 512 | 파노라마 전체의 1차 탐색 영역 |
| Fine Tile 간격 | 256px | 상위 Coarse 후보 주변의 정밀 탐색 간격 |
| Coarse 상위 후보 수 | 3 | Fine 탐색 대상으로 넘길 후보 수 |
| 조기 종료 Inlier | 50 | 충분한 검출로 판단하는 기준 |
| 파노라마 최대 변 | 3072px | `--max-side`로 변경 가능 |

`--max-side`는 검출용 파노라마 크기만 조절합니다. 기준 물체와 최종 Polygon 좌표는 원본 파노라마 기준으로 유지됩니다.

## 7. 결과 JSON 읽기

단일 실행 JSON에는 검출 여부, 매칭 수, Inlier 수, Polygon Tile 정보가 기록됩니다. 전체 실행 JSON에는 세트별 결과가 `cases` 목록으로 저장됩니다.

주요 항목은 다음과 같습니다.

| 항목 | 의미 |
| --- | --- |
| `elapsed` | 해당 세트의 검출 시간(초) |
| `match_count` | 최종 후보 Tile에서 LightGlue가 선택한 매칭 수 |
| `inlier_count` | RANSAC 호모그래피에 일관되게 포함된 매칭 수 |
| `inlier_ratio` | `inlier_count / match_count` |
| `tile` | 검출된 Tile의 원본 파노라마 좌표 `[x, y, width, height]` |
| `stats.total_checked` | Coarse와 Fine 단계에서 실제 처리한 Tile 수 |
| `stats.total_skipped_black` | 검은 여백 비율이 높아 건너뛴 Tile 수 |

## 8. 주의 사항

- GPU 전용 구현은 기존 OpenCV DNN 기반 ALIKED가 아니라 PyTorch ALIKED·LightGlue 가중치를 사용합니다. 따라서 CPU와 GPU의 매칭 수가 완전히 같을 필요는 없습니다.
- GPU 실행 결과는 보고서에서 `ALIKED_GPU`로 구분해 기록하는 것이 좋습니다.
- `--all-sets --save-visuals`를 다시 실행하면 같은 세트의 GPU 결과 이미지와 `batch_summary.json`은 갱신됩니다.
- CUDA를 사용할 수 없는 환경에서는 CPU로 자동 전환하지 않고 오류를 발생시킵니다. GPU 성능 실험에서 CPU 결과가 섞이지 않도록 하기 위한 동작입니다.

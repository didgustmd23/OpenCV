# OpenCV 파노라마 · 객체 검출 프로젝트

## 1. 주제

특징점 매칭과 호모그래피를 이용해 연속 사진을 파노라마로 합성하고, 완성된 파노라마에서 기준 물체의 위치를 검출하는 프로젝트입니다. SIFT·ORB·ALIKED 방법의 매칭 정확도와 실행 시간을 비교하고, 회전·크기·조명 변화에 대한 검출 성능도 실험합니다. 과제 안내의 AKAZE는 현재 OpenCV 5.0 환경에서 사용할 수 없어 ALIKED로 대체했습니다.

## 2. 역할 분담

| 구분 | 담당자 | 담당 업무 |
| --- | --- | --- |
| 파노라마 | 담당 | 연속 사진 데이터 준비, 특징점 매칭·RANSAC 기반 파노라마 합성, 블렌딩 결과 확인 |
| 객체 검출 | 담당 | 기준 물체-파노라마 매칭, Coarse-to-Fine 타일 탐색, 객체 외곽 Polygon 표시 |
| 공동 작업 | 담당 | SIFT·ORB·ALIKED 성능 비교, 회전·크기·조명 변화 실험, 실패 사례 분석, 보고서·발표 자료 작성 |

## 3. 폴더 구성

```text
src/                           # 프로젝트 실행 코드
├── pipeline.py                # 파노라마 합성 · 객체 검출 전체 실행
├── main.py                    # 개별 기능 테스트 · 디버그 실행
├── stitcher.py                # 파노라마 합성 핵심 기능
├── stitcher_create.py         # OpenCV 내장 Stitcher 비교 실행
├── experiment_runner.py       # 일괄 실험·통계·노트북 시각화 도우미
├── feature_match.py           # 특징점 추출·매칭 공통 기능
├── object_finder.py           # 호모그래피 기반 객체 검출
├── tile_scanner.py            # Coarse-to-Fine 타일 탐색
├── image_io.py                # 이미지 파일 입출력
├── project_paths.py           # 공통 경로
├── settings.py                # config.json 로더
├── logger.py                  # 콘솔·opencv.log 기록 설정
├── result_cleanup.py          # 생성 결과 정리 기능
└── __init__.py                # src 패키지 표시 파일

data/
├── objects/                    # 기준 물체: set01/target.jpg ...
└── set01/                      # 파노라마 원본 연속 사진: 01.jpg, 02.jpg, 03.jpg ...

results/
├── panorama/                   # 파노라마 합성 결과
│   └── set01/SIFT/
│       ├── matching/           # raw / 필터 후 / RANSAC 매칭 이미지
│       ├── panorama.jpg
│       ├── panorama_outline.jpg
│       └── matching_stats.csv
│   └── create/set01/           # OpenCV 내장 Stitcher 비교 결과
│       └── panorama.jpg
├── object_detection/           # 객체 검출 결과와 실험 통계
├── matching/keypoint_matches/  # Coarse·Fine 타일 및 최종 후보 매칭 이미지
├── debug/                      # main.py 디버그 결과
└── logs/opencv.log             # 실행 로그
```

## 4. 실행 방법

### 4.1 파노라마 합성만 테스트

연속 사진의 합성 기능만 확인할 때는 `src.main`의 stitch 디버그 모드를 사용합니다. 실행 전 [config.json](config.json)에서 파노라마 매칭 기준, 모델 경로, 결과 폴더를 확인합니다.

```powershell
python -m src.main --mode stitch --input data/set01 --method sift --max-side 1600
```

결과는 `results/debug/panorama/set01/SIFT/`에 저장됩니다. 이 경로는 테스트 전용이므로, 정식 결과인 `results/panorama/`를 덮어쓰지 않습니다.

`stitcher.py`는 명령줄 진입점 없이 `stitch_panorama()`를 제공하는 내부 합성 모듈입니다. SIFT·ORB에는 BFMatcher와 Lowe Ratio Test를, ALIKED에는 LightGlue 매칭을 사용합니다.

### 4.2 파노라마 합성 · 객체 검출 전체 실행

연속 사진을 합성한 뒤, 생성된 파노라마 메모리 이미지에서 기준 물체를 바로 검출합니다.

```powershell
python -m src.pipeline --input data/set01 --reference data/objects/set01/target.jpg --method sift --max-side 1024
```

합성한 파노라마를 다른 경로에도 복사하려면 `--export`를 추가합니다. 기본 결과 경로는 그대로 유지됩니다.

```powershell
python -m src.pipeline --input data/set01 --reference data/objects/set01/target.jpg --method sift --export data/exported_panorama.jpg
```

결과는 아래 위치에 저장됩니다.

```text
results/panorama/{입력_폴더명}/SIFT/                   # 파노라마·인접 사진 매칭 결과
results/object_detection/{입력_폴더명}/SIFT/          # 검출 결과와 요약 JSON
results/matching/keypoint_matches/{입력_폴더명}/SIFT/ # Coarse·Fine Tile과 final_matches.png
```

`config.json`의 `object_detection.save_feature_matches`가 참이면 Coarse·Fine 탐색 타일의 매칭 이미지를 저장하고, `save_final_matches`가 참이면 최종 후보의 `final_matches.png`를 추가로 저장합니다.

### 4.3 노트북 일괄 파노라마 실험

최종보고서용 `FinalReport.ipynb`는 `data/` 아래의 `set01`, `set02` 형식 폴더를 자동 탐색합니다. 발견한 각 세트에 SIFT, ORB, ALIKED를 적용해 파노라마를 만들고, 결과를 `results/panorama/{세트명}/{방법}/`에 분리 저장합니다.

객체 검출 실험까지 이어서 수행하려면 세트 번호를 기준 물체 폴더와 맞춥니다. 예를 들어 `data/set01/`에는 연속 사진을, `data/objects/set01/target.jpg`에는 해당 세트의 기준 물체를 둡니다.

### 4.4 개별 기능 테스트·디버그

`src.main`은 전체 파이프라인을 실행하지 않습니다. 파노라마 합성 또는 완성된 파노라마의 객체 검출을 각각 확인할 때 사용합니다.

```powershell
# 완성 파노라마의 객체 검출 기능만 테스트
python -m src.main --mode detect --scene results/panorama/set01/SIFT/panorama.jpg --reference data/objects/set01/target.jpg --method sift
```

디버그 결과는 `results/debug/` 아래에 저장됩니다.

## 5. 실행 요약 파일: `pipeline_summary.json`

`src.pipeline` 실행이 끝나면 아래 경로에 파노라마 합성과 객체 검출의 핵심 결과를 JSON으로 저장합니다.

```text
results/object_detection/{입력 폴더명}/{METHOD}/pipeline_summary.json
```

예를 들어 `data/set01` 폴더를 SIFT로 실행하면 `results/object_detection/set01/SIFT/pipeline_summary.json`이 생성됩니다.

| 항목 | 설명 |
| --- | --- |
| `input_dir` | 파노라마 합성에 사용한 연속 사진 폴더 |
| `reference_path` | 장면에서 찾을 기준 물체 이미지 |
| `method` | 실행한 특징점 방법(SIFT, ORB, ALIKED) |
| `panorama_path` | 생성된 파노라마 이미지 경로 |
| `panorama_outline_path` | 파노라마 외곽선 확인 이미지 경로 |
| `panorama_matching_stats` | 인접 사진끼리의 원시·필터링·RANSAC 매칭 수 CSV 경로 |
| `panorama_export_path` | `--export`를 지정했을 때 추가 저장한 파노라마 경로. 미지정 시 `null` |
| `detection` | Coarse-to-Fine 객체 탐색 결과 |

`detection`에는 다음 값을 기록합니다.

| 항목 | 설명 |
| --- | --- |
| `is_detection` | 최종 객체 검출 성공 여부 |
| `reference_keypoints` | 기준 물체에서 검출된 특징점 수 |
| `coarse_checked`, `fine_checked` | 넓은 탐색과 세밀 탐색에서 검사한 타일 수 |
| `elapsed` | 객체 탐색에 걸린 시간(초) |
| `result_path` | 초록색 객체 Polygon이 표시된 검출 결과 이미지 경로 |
| `matching_path` | 최종 후보 타일의 `final_matches.png` 경로. `save_final_matches`가 꺼져 있으면 `null` |
| `tile` | 최종 후보 타일 범위 `[x1, y1, x2, y2]` |
| `match_count` | Ratio Test 또는 LightGlue 후 남은 좋은 매칭 수 |
| `inlier_count`, `inlier_ratio` | RANSAC을 통과한 매칭 수와 그 비율 |

`inlier_count`가 `config.json`의 `matching.min_inliers` 이상이면 검출 성공으로 판단합니다. 메서드별 `elapsed`, `match_count`, `inlier_count`, `inlier_ratio`를 비교하면 속도와 매칭 품질을 보고서 표로 정리할 수 있습니다.

## 6. 실험 결과 CSV

`experiment_runner.py`로 기준 물체의 회전, 크기, 밝기를 바꿔 실험하면 아래 두 CSV가 생성됩니다.

```text
results/object_detection/metrics/experiment_records.csv
results/object_detection/metrics/method_condition_summary.csv
```

### 6.1 `experiment_records.csv`

각 **테스트 케이스 × 조건 × 메서드**의 원본 결과를 한 행씩 기록합니다. 실패 사례를 확인하거나 특정 이미지의 검출 결과를 추적할 때 사용합니다. 기본 조건은 검출 결과와 최종 Tile 매칭 이미지를 저장하며, 나머지 변형 조건은 실행 시간을 줄이기 위해 기본적으로 수치만 기록합니다.

| 주요 열 | 설명 |
| --- | --- |
| `case` | 기준 물체와 파노라마를 조합한 테스트 케이스 이름 |
| `condition` | `baseline`, `rotation_plus30`, `scale_075`, `brightness_070` 등 변형 조건 |
| `method` | SIFT, ORB, ALIKED |
| `rotation_deg`, `scale_factor`, `brightness_factor` | 기준 물체 이미지에 적용한 회전각, 크기, 밝기 배율 |
| `is_detection` | 해당 조건에서 객체 검출 성공 여부 |
| `keypoints`, `match_count`, `inlier_count`, `inlier_ratio` | 특징점 수와 매칭 품질 지표 |
| `elapsed` | 해당 케이스의 탐색 시간(초) |
| `tile` | 최종 객체가 선택된 파노라마 타일 좌표 |
| `result_path`, `match_image_path` | 검출 Polygon 이미지와 특징점 매칭 이미지 경로 |

### 6.2 `method_condition_summary.csv`

동일한 **메서드 × 조건**의 `experiment_records.csv` 행을 집계한 비교표입니다. 발표·보고서에는 이 파일을 우선 사용합니다.

| 주요 열 | 설명 |
| --- | --- |
| `method`, `condition` | 비교 대상 특징점 방법과 실험 조건 |
| `total_cases` | 해당 조합으로 실행한 전체 케이스 수 |
| `detection_count` | 객체 검출에 성공한 케이스 수 |
| `success_rate` | 성공률 (`detection_count / total_cases`) |
| `mean_elapsed` | 평균 탐색 시간(초) |
| `mean_match_count` | 평균 좋은 매칭 수 |
| `mean_inlier_count`, `mean_inlier_ratio` | 평균 RANSAC Inlier 수와 비율 |

예를 들어 `rotation_plus30` 행의 `success_rate`는 30도 회전에도 해당 방법이 얼마나 안정적으로 물체를 찾는지 나타냅니다. 속도는 `mean_elapsed`, 매칭 신뢰도는 `mean_inlier_ratio`, 검출 안정성은 `success_rate`를 중심으로 비교합니다.

## 7. 공통 설정

프로젝트 공통 값은 `config.json`에서 관리합니다.

- `paths`: 데이터·모델·결과물 폴더
- `models`: ALIKED·LightGlue 모델 파일 경로
- `matching`: Lowe Ratio Test, RANSAC, 최소 Inlier 수
- `object_detection`: 타일 크기·간격·조기 종료 기준
- `panorama`: 파노라마 비율 검사·RANSAC·입력 축소·캔버스 제한
- `pipeline`: pipeline.py의 기본 연속 사진·기준 물체 입력 경로
- `logging`: Python·OpenCV 로그 수준

입력 폴더, 기준 물체/파노라마 조합, 실행 방법처럼 실행마다 달라지는 값은 명령줄 인자 또는 노트북 셀에서 지정합니다.

## 8. 결과 파일 정리

`result_cleanup.py`의 `clear_results()`는 `results/` 아래에서 생성된 파일만 정리합니다. `.gitkeep`, `opencv.log`, 폴더 구조는 유지합니다. 기본값은 삭제하지 않고 대상만 반환하는 미리 보기 모드입니다.

```python
from src.result_cleanup import clear_results

targets = clear_results()             # 삭제 대상 확인
clear_results(dry_run=False)          # 확인 후 실제 삭제
```

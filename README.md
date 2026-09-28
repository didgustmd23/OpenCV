# OpenCV 파노라마 · 객체 검출 프로젝트

## 주제

특징점 매칭과 호모그래피를 이용해 연속 사진을 파노라마로 합성하고, 완성된 파노라마에서 기준 물체의 위치를 검출하는 프로젝트입니다. SIFT·ORB·AKAZE 방법의 매칭 정확도와 실행 시간을 비교하고, 회전·크기·조명 변화에 대한 검출 성능도 실험합니다.

## 역할 분담

| 구분 | 담당 업무 |
| --- | --- |
| 파노라마 담당 | 연속 사진 데이터 준비, 특징점 매칭·RANSAC 기반 파노라마 합성, 블렌딩 결과 확인 |
| 객체 검출 담당 | 기준 물체-파노라마 매칭, Coarse-to-Fine 타일 탐색, 객체 외곽 Polygon 표시 |
| 공동 작업 | SIFT·ORB·ALIKED 성능 비교, 회전·크기·조명 변화 실험, 실패 사례 분석, 보고서·발표 자료 작성 |

## 폴더 구성

```text
src/                           # 프로젝트 실행 코드
├── main.py                    # 단일 객체 검출 실행
├── stitcher.py                # 파노라마 합성 실행
├── experiment_runner.py       # 객체 검출 일괄 실험
├── feature_match.py           # 특징점 추출·매칭 공통 기능
├── object_finder.py           # 호모그래피 기반 객체 검출
├── tile_scanner.py            # Coarse-to-Fine 타일 탐색
├── image_io.py                # 이미지 파일 입출력
├── project_paths.py           # 공통 경로
└── settings.py                # config.json 로더

data/
├── objects/                    # 기준 물체 이미지: target1.jpg ...
├── panorama_sets/              # 파노라마 원본 연속 사진
│   └── set01/                  # 01.jpg, 02.jpg, 03.jpg ...
└── panorama/                   # 객체 검출에 사용할 완성 파노라마
    └── panorama1.jpg

results/
├── panorama/                   # 파노라마 합성 결과
│   └── set01/SIFT/
│       ├── matching/           # raw / 필터 후 / RANSAC 매칭 이미지
│       ├── panorama.jpg
│       ├── panorama_outline.jpg
│       └── matching_stats.csv
├── object_detection/           # 객체 검출 결과와 실험 통계
├── matching/keypoint_matches/  # 객체 검출 최종 타일 매칭 이미지
└── logs/opencv.log
```

## 파노라마 생성

실행 전 [config.json](config.json)에서 공통 설정을 확인합니다. 기본값으로 바로 실행할 수 있으며, 타일 크기·매칭 기준·모델 경로·결과 폴더를 변경할 때만 수정하면 됩니다.

```powershell
python -m src.stitcher --input data/panorama_sets/set01 --method sift
```

객체 검출용 파노라마까지 함께 저장하려면 `--export` 경로를 지정합니다.

```powershell
python -m src.stitcher --input data/panorama_sets/set01 --method sift --export data/panorama/panorama1.jpg
```

`stitcher.py`는 SIFT·ORB는 BFMatcher와 Lowe Ratio Test를, ALIKED는 LightGlue 매칭을 공통 모듈에서 재사용합니다.

## 파노라마 합성 · 객체 검출 전체 실행

연속 사진을 합성한 뒤, 생성된 파노라마 메모리 이미지에서 기준 물체를 바로 검출합니다.

```powershell
python -m src.main --input data/panorama --reference data/objects/target.jpg --method sift --max-side 1024
```

결과는 아래 위치에 저장됩니다.

```text
results/panorama/SIFT/                         # 파노라마와 인접 사진 매칭 결과
results/object_detection/세트명/SIFT/          # 검출 결과와 요약 JSON
results/matching/keypoint_matches/세트명/SIFT/ # Coarse·Fine 모든 Tile과 최종 후보 매칭 이미지
```

## 실행 요약 파일: `pipeline_summary.json`

`src.main` 실행이 끝나면 아래 경로에 파노라마 합성과 객체 검출의 핵심 결과를 JSON으로 저장합니다.

```text
results/object_detection/{입력 폴더명}/{METHOD}/pipeline_summary.json
```

예를 들어 `data/panorama` 폴더를 SIFT로 실행하면 `results/object_detection/panorama/SIFT/pipeline_summary.json`이 생성됩니다.

| 항목 | 설명 |
| --- | --- |
| `input_dir` | 파노라마 합성에 사용한 연속 사진 폴더 |
| `reference_path` | 장면에서 찾을 기준 물체 이미지 |
| `method` | 실행한 특징점 방법(SIFT, ORB, ALIKED) |
| `panorama_path` | 생성된 파노라마 이미지 경로 |
| `panorama_outline_path` | 파노라마 외곽선 확인 이미지 경로 |
| `panorama_matching_stats` | 인접 사진끼리의 원시·필터링·RANSAC 매칭 수 CSV 경로 |
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

## 실험 결과 CSV

`experiment_runner.py`로 기준 물체의 회전, 크기, 밝기를 바꿔 실험하면 아래 두 CSV가 생성됩니다.

```text
results/object_detection/metrics/experiment_records.csv
results/object_detection/metrics/method_condition_summary.csv
```

### `experiment_records.csv`

각 **테스트 케이스 × 조건 × 메서드**의 원본 결과를 한 행씩 기록합니다. 실패 사례를 확인하거나 특정 이미지의 검출 결과를 추적할 때 사용합니다.

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

### `method_condition_summary.csv`

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

## 공통 설정

프로젝트 공통 값은 `config.json`에서 관리합니다.

- `paths`: 데이터·모델·결과물 폴더
- `models`: ALIKED·LightGlue 모델 파일 경로
- `matching`: Lowe Ratio Test, RANSAC, 최소 Inlier 수
- `object_detection`: 타일 크기·간격·조기 종료 기준
- `panorama`: 파노라마 비율 검사·RANSAC·입력 축소·캔버스 제한
- `pipeline`: main.py의 기본 연속 사진·기준 물체 입력 경로
- `logging`: Python·OpenCV 로그 수준

입력 폴더, 기준 물체/파노라마 조합, 실행 방법처럼 실행마다 달라지는 값은 명령줄 인자 또는 노트북 셀에서 지정합니다.

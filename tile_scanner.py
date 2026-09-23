import time
import cv2

from logger import logger
from object_finder import find_object

# ==========================================
# Fine 후보 우선순위 계산
# - Fine Tile과 Coarse 상위 후보 사이의 거리를 계산
# - Manhattan Distance(|dx| + |dy|) 사용
# - 값이 작을수록 Coarse 후보와 가까우므로 우선 탐색
# ==========================================
def get_fine_priority(
    position,
    coarse_candidates,
):
    x, y = position

    best_distance = float("inf")

    for candidate in coarse_candidates:
        coarse_x = candidate["x"]
        coarse_y = candidate["y"]

        dx = abs(x - coarse_x)
        dy = abs(y - coarse_y)

        distance = dx + dy

        if distance < best_distance:
            best_distance = distance

    return best_distance


# ==========================================
# Fine 탐색 후보 좌표 생성
# - Coarse 상위 후보 주변에 Fine Tile 좌표 생성
# - Scene 범위를 벗어나는 위치 제외
# - 너무 작은 경계 Tile 제외
# - 이미 Coarse에서 검사한 위치 제외
# - 중복 좌표 제거
# - Coarse 후보와 가까운 순서로 정렬
# ==========================================
def generate_fine_positions(
    coarse_candidates,
    coarse_positions,
    scene_size,
    tile_size,
    fine_stride,
):
    scene_w, scene_h = scene_size
    tile_w, tile_h = tile_size
    stride_x, stride_y = fine_stride

    fine_positions = set()

    for candidate in coarse_candidates:
        center_x = candidate["x"]
        center_y = candidate["y"]

        # Coarse Tile 중심 위치를 기준으로
        # 주변 3 x 3 위치에 Fine 후보 생성
        for dy in (-stride_y, 0, stride_y):
            for dx in (-stride_x, 0, stride_x):

                fine_x = center_x + dx
                fine_y = center_y + dy

                # Scene 바깥에서 시작하는 Tile 제외
                if fine_x < 0 or fine_y < 0:
                    continue

                if fine_x >= scene_w or fine_y >= scene_h:
                    continue

                # Scene 경계에 걸린 경우 실제 Tile 크기 계산
                fine_x2 = min(
                    fine_x + tile_w,
                    scene_w,
                )

                fine_y2 = min(
                    fine_y + tile_h,
                    scene_h,
                )

                # 너무 작은 경계 Tile 제외
                if (fine_x2 - fine_x) < 300:
                    continue

                if (fine_y2 - fine_y) < 250:
                    continue

                # Coarse 단계에서 이미 검사한 위치는
                # 다시 검사하지 않음
                if (fine_x, fine_y) in coarse_positions:
                    continue

                # set을 사용하여 중복 좌표 자동 제거
                fine_positions.add(
                    (fine_x, fine_y)
                )

    # Fine 후보 탐색 순서 결정
    #
    # 1순위: 가장 가까운 Coarse 후보와의 거리
    # 2순위: y 좌표
    # 3순위: x 좌표
    #
    # 동일한 입력에서는 항상 동일한 탐색 순서가
    # 나오도록 y, x를 추가 정렬 기준으로 사용
    fine_positions = sorted(
        fine_positions,
        key=lambda p: (
            get_fine_priority(
                p,
                coarse_candidates,
            ),
            p[1],
            p[0],
        )
    )

    return fine_positions


# ==========================================
# Fine Tile 탐색
# - 생성된 Fine 후보 좌표를 순서대로 검사
# - 각 Tile에서 객체 검출 수행
# - 유효한 검출 결과를 저장
# - 충분히 강한 검출 결과가 나오면 Early Stop
# - 탐색 결과와 탐색 통계를 반환
# ==========================================
def scan_fine(
    ref_img,
    sce_img,
    fine_positions,
    tile_size,
    ref_features,
    early_stop_inliers,
    ref_name,
    method="ALIKED",
):
    tile_w, tile_h = tile_size

    scene_h, scene_w = sce_img.shape[:2]

    fine_results = []

    fine_checked_count = 0
    early_stopped = False

    fine_start = time.time()

    logger.info(
        "Fine 탐색 시작: candidates=%d",
        len(fine_positions),
    )

    for fine_index, (x, y) in enumerate(
        fine_positions,
        start=1,
    ):
        x2 = min(x + tile_w, scene_w)
        y2 = min(y + tile_h, scene_h)

        tile = sce_img[y:y2, x:x2]

        fine_checked_count += 1

        logger.debug(
            "Fine Tile 시작: tile=%d, xy=(%d, %d), size=(%d, %d)",
            fine_index,
            x,
            y,
            x2 - x,
            y2 - y,
        )

        try:
            result = find_object(
                ref_img,
                tile,
                method=method,
                ref_name=ref_name,
                sce_name=f"fine_{fine_index}",
                ref_features=ref_features,
                coarse_mode=True,
            )

        except cv2.error as e:
            logger.warning(
                "Fine Tile OpenCV 오류: "
                "tile=%d, xy=(%d, %d), error=%s",
                fine_index,
                x,
                y,
                e,
            )
            continue

        except Exception:
            logger.exception(
                "Fine Tile 처리 중 예외 발생: "
                "tile=%d, xy=(%d, %d)",
                fine_index,
                x,
                y,
            )
            continue

        if result is None:
            logger.debug(
                "Fine 결과 없음: tile=%d, xy=(%d, %d)",
                fine_index,
                x,
                y,
            )
            continue

        logger.debug(
            "Fine 결과: tile=%d, xy=(%d, %d), "
            "matches=%d, inliers=%d, "
            "ratio=%.3f, detection=%s",
            fine_index,
            x,
            y,
            result["match_count"],
            result["inlier_count"],
            result["inlier_ratio"],
            result["is_detection"],
        )

        fine_results.append(
            {
                "index": fine_index,
                "x": x,
                "y": y,
                "result": result,
            }
        )

        # 충분히 강한 검출 결과가 나오면
        # 남은 Fine Tile을 검사하지 않고 종료
        if (
            result["is_detection"]
            and result["inlier_count"] >= early_stop_inliers
        ):
            early_stopped = True

            logger.info(
                "Fine Early Stop: xy=(%d, %d), "
                "matches=%d, inliers=%d, ratio=%.3f",
                x,
                y,
                result["match_count"],
                result["inlier_count"],
                result["inlier_ratio"],
            )

            break

    fine_elapsed = time.time() - fine_start

    logger.info(
        "Fine 탐색 완료: checked=%d, valid=%d, "
        "elapsed=%.2fs, early_stop=%s",
        fine_checked_count,
        len(fine_results),
        fine_elapsed,
        early_stopped,
    )

    # ==========================================
    # Fine BEST 결과 선택
    # - Fine 탐색에서 얻은 유효 결과 중 가장 강한 결과 선택
    # - Inlier 수가 많은 결과를 우선
    # - Inlier 수가 같으면 Match 수가 많은 결과를 우선
    # - 유효한 Fine 결과가 없으면 None 반환
    # ==========================================

    best_fine = None

    if fine_results:
        best_fine = max(
            fine_results,
            key=lambda item: (
                item["result"]["inlier_count"],
                item["result"]["match_count"],
            ),
        )

        best_result = best_fine["result"]

        logger.info(
            "Fine BEST: "
            "tile=%d, xy=(%d, %d), "
            "matches=%d, inliers=%d, ratio=%.3f",
            best_fine["index"],
            best_fine["x"],
            best_fine["y"],
            best_result["match_count"],
            best_result["inlier_count"],
            best_result["inlier_ratio"],
        )

    return {
        "results": fine_results,
        "best": best_fine,
        "checked_count": fine_checked_count,
        "early_stopped": early_stopped,
        "elapsed": fine_elapsed,
    }
    
# ==========================================
# Coarse Tile 탐색
# - Scene 전체를 Coarse 간격으로 탐색
# - 너무 작은 경계 Tile은 탐색 대상에서 제외
# - 각 Tile에서 객체 검출 수행
# - 검사한 Coarse 위치와 유효한 후보 결과를 저장
# - Fine 탐색에 필요한 Coarse 탐색 결과를 반환
# ==========================================
def scan_coarse(
    ref_img,
    sce_img,
    tile_size,
    coarse_stride,
    ref_features,
    ref_name,
    top_k,
    method="ALIKED",
):
    tile_w, tile_h = tile_size
    stride_x, stride_y = coarse_stride

    scene_h, scene_w = sce_img.shape[:2]

    coarse_candidates = []
    coarse_positions = set()

    coarse_index = 0

    coarse_start = time.perf_counter()

    logger.info("Coarse 탐색 시작")

    for y in range(0, scene_h, stride_y):
        for x in range(0, scene_w, stride_x):

            x2 = min(x + tile_w, scene_w)
            y2 = min(y + tile_h, scene_h)

            # 너무 작은 마지막 경계 Tile은 제외
            if (x2 - x) < 300 or (y2 - y) < 250:
                continue

            # 실제로 검사하는 Coarse 위치 저장
            # 이후 Fine 후보 생성 시 중복 검사를 막는 데 사용
            coarse_positions.add((x, y))

            tile = sce_img[y:y2, x:x2]

            coarse_index += 1

            logger.debug(
                "Coarse Tile 시작: "
                "tile=%d, xy=(%d, %d), size=(%d, %d)",
                coarse_index,
                x,
                y,
                x2 - x,
                y2 - y,
            )

            try:
                result = find_object(
                    ref_img,
                    tile,
                    method=method,
                    ref_name=ref_name,
                    sce_name=f"coarse_{coarse_index}",
                    ref_features=ref_features,
                    coarse_mode=True,
                )

            except cv2.error as e:
                logger.warning(
                    "Coarse Tile OpenCV 오류: "
                    "tile=%d, xy=(%d, %d), error=%s",
                    coarse_index,
                    x,
                    y,
                    e,
                )
                continue

            except Exception:
                logger.exception(
                    "Coarse Tile 처리 중 예외 발생: "
                    "tile=%d, xy=(%d, %d)",
                    coarse_index,
                    x,
                    y,
                )
                continue

            # 특징점 부족, Homography 실패 등으로
            # 결과 자체가 생성되지 않은 경우
            if result is None:
                logger.debug(
                    "Coarse 결과 없음: "
                    "tile=%d, xy=(%d, %d)",
                    coarse_index,
                    x,
                    y,
                )
                continue

            current_inliers = result["inlier_count"]
            current_matches = result["match_count"]

            logger.debug(
                "Coarse 후보: "
                "tile=%d, xy=(%d, %d), "
                "matches=%d, inliers=%d, "
                "ratio=%.3f, detection=%s",
                coarse_index,
                x,
                y,
                current_matches,
                current_inliers,
                result["inlier_ratio"],
                result["is_detection"],
            )

            coarse_candidates.append(
                {
                    "index": coarse_index,
                    "x": x,
                    "y": y,
                    "x2": x2,
                    "y2": y2,
                    "match_count": current_matches,
                    "inlier_count": current_inliers,
                    "inlier_ratio": result["inlier_ratio"],
                    "is_detection": result["is_detection"],
                    "result": result,
                }
            )

    coarse_elapsed = time.perf_counter() - coarse_start

    logger.info(
        "Coarse 탐색 완료: checked=%d, valid=%d, elapsed=%.2fs",
        coarse_index,
        len(coarse_candidates),
        coarse_elapsed,
    )

    # ==========================================
    # Coarse 후보 우선순위 정렬
    # - Inlier 수가 많은 후보를 우선
    # - Inlier 수가 같으면 Match 수가 많은 후보를 우선
    # ==========================================

    coarse_candidates.sort(
        key=lambda candidate: (
            candidate["inlier_count"],
            candidate["match_count"],
        ),
        reverse=True,
    )

    top_coarse_candidates = coarse_candidates[:top_k]

    for rank, candidate in enumerate(
        top_coarse_candidates,
        start=1,
    ):
        logger.info(
            "Coarse Top %d: "
            "tile=%d, xy=(%d, %d), "
            "matches=%d, inliers=%d, "
            "ratio=%.3f, detection=%s",
            rank,
            candidate["index"],
            candidate["x"],
            candidate["y"],
            candidate["match_count"],
            candidate["inlier_count"],
            candidate["inlier_ratio"],
            candidate["is_detection"],
        )

    return {
        "candidates": coarse_candidates,
        "top_candidates": top_coarse_candidates,
        "positions": coarse_positions,
        "checked_count": coarse_index,
        "elapsed": coarse_elapsed,
    }

# ==========================================
# Scene 전체 Coarse-to-Fine 탐색
# - Coarse 탐색으로 Scene 전체를 빠르게 검사
# - Coarse 상위 후보 주변에 Fine 후보 좌표 생성
# - Fine 탐색으로 객체 위치를 정밀하게 검사
# - Fine BEST 결과와 전체 탐색 통계를 반환
# - 좌표 변환 및 시각화는 수행하지 않음
# ==========================================
def scan_scene(
    ref_img,
    sce_img,
    ref_features,
    tile_size,
    coarse_stride,
    fine_stride,
    coarse_top_k,
    early_stop_inliers,
    ref_name,
    method="ALIKED",
):
    scan_start = time.perf_counter()

    scene_h, scene_w = sce_img.shape[:2]

    logger.info(
        "Scene 탐색 시작: "
        "scene_size=(%d, %d), tile_size=%s",
        scene_w,
        scene_h,
        tile_size,
    )

    # ==========================================
    # 1단계: Coarse 탐색
    # ==========================================

    coarse_scan = scan_coarse(
        ref_img=ref_img,
        sce_img=sce_img,
        tile_size=tile_size,
        coarse_stride=coarse_stride,
        ref_features=ref_features,
        ref_name=ref_name,
        top_k=coarse_top_k,
        method=method,
    )

    coarse_candidates = coarse_scan["candidates"]
    top_coarse_candidates = coarse_scan["top_candidates"]
    coarse_positions = coarse_scan["positions"]

    # Coarse 후보가 하나도 없으면
    # Fine 탐색을 진행할 수 없으므로 여기서 반환
    if not coarse_candidates:
        scan_elapsed = time.perf_counter() - scan_start

        logger.warning(
            "Coarse 후보를 찾지 못했습니다."
        )

        return {
            "best": None,
            "coarse": coarse_scan,
            "fine": None,
            "fine_positions": [],
            "stats": {
                "coarse_checked": coarse_scan["checked_count"],
                "fine_checked": 0,
                "total_checked": coarse_scan["checked_count"],
                "early_stopped": False,
                "elapsed": scan_elapsed,
            },
        }

    # ==========================================
    # 2단계: Fine 후보 좌표 생성
    # ==========================================

    fine_positions = generate_fine_positions(
        coarse_candidates=top_coarse_candidates,
        coarse_positions=coarse_positions,
        scene_size=(scene_w, scene_h),
        tile_size=tile_size,
        fine_stride=fine_stride,
    )

    logger.info(
        "Fine 후보 생성 완료: candidates=%d",
        len(fine_positions),
    )

    # ==========================================
    # 3단계: Fine 탐색
    # ==========================================

    fine_scan = scan_fine(
        ref_img=ref_img,
        sce_img=sce_img,
        fine_positions=fine_positions,
        tile_size=tile_size,
        ref_features=ref_features,
        early_stop_inliers=early_stop_inliers,
        ref_name=ref_name,
        method=method,
    )

    # ==========================================
    # 전체 탐색 통계 계산
    # ==========================================

    coarse_checked = coarse_scan["checked_count"]
    fine_checked = fine_scan["checked_count"]

    total_checked = coarse_checked + fine_checked

    scan_elapsed = time.perf_counter() - scan_start

    logger.info(
        "전체 탐색 완료: "
        "coarse=%d, fine=%d, total=%d, "
        "early_stop=%s, elapsed=%.2fs",
        coarse_checked,
        fine_checked,
        total_checked,
        fine_scan["early_stopped"],
        scan_elapsed,
    )

    return {
        "best": fine_scan["best"],
        "coarse": coarse_scan,
        "fine": fine_scan,
        "fine_positions": fine_positions,
        "stats": {
            "coarse_checked": coarse_checked,
            "fine_checked": fine_checked,
            "total_checked": total_checked,
            "early_stopped": fine_scan["early_stopped"],
            "elapsed": scan_elapsed,
        },
    }
import json
from pathlib import Path

# ==========================================
# JSON 설정 파일 로드
# - 모든 실행 모듈이 동일한 config.json 값을 사용
# - 설정 파일이 없거나 JSON 형식이 잘못되면 즉시 원인을 안내
# ==========================================
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.json"


def load_config(path=CONFIG_PATH):
    try:
        with Path(path).open(encoding="utf-8") as file:
            config = json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            "config.json 형식이 올바르지 않습니다: "
            f"line={error.lineno}, column={error.colno}"
        ) from error

    if not isinstance(config, dict):
        raise ValueError("config.json의 최상위 값은 객체여야 합니다.")

    return config


# ==========================================
# 설정 영역 조회
# - 필요한 영역이 누락되면 KeyError 대신 이해하기 쉬운 오류 제공
# ==========================================
def get_section(config, name):
    try:
        section = config[name]
    except KeyError as error:
        raise KeyError(f"config.json에 '{name}' 설정 영역이 없습니다.") from error

    if not isinstance(section, dict):
        raise ValueError(f"config.json의 '{name}' 설정은 객체여야 합니다.")

    return section


CONFIG = load_config()
PATH_CONFIG = get_section(CONFIG, "paths")
MODEL_CONFIG = get_section(CONFIG, "models")
MATCHING_CONFIG = get_section(CONFIG, "matching")
OBJECT_DETECTION_CONFIG = get_section(CONFIG, "object_detection")
PANORAMA_CONFIG = get_section(CONFIG, "panorama")
PIPELINE_CONFIG = get_section(CONFIG, "pipeline")
LOGGING_CONFIG = get_section(CONFIG, "logging")

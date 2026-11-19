import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import REPO_ROOT

from rpsai.config import AppConfig, Download, GestureConfig, load_config
from rpsai.downloads import fetch, sha256_of
from rpsai.export import (
    EXPORT_FORMAT,
    GESTURE_FIXTURE_FILE,
    GESTURE_MODEL_FILE,
    PREDICTOR_FILE,
    PREDICTOR_FIXTURE_FILE,
    gesture_fixture,
    predictor_fixture,
    predictor_payload,
    write_json,
)
from rpsai.gestures import GestureNetwork
from rpsai.predictors import EXPERT_NAMES, PredictorSettings


def test_repository_config_loads(repo_config: AppConfig) -> None:
    class_of = repo_config.gestures.class_of()
    assert class_of['fist'] == 0
    assert class_of['no_gesture'] == 3
    assert 'three' not in class_of


@pytest.mark.parametrize(
    'change',
    [
        {'classes': {'paper': ['palm'], 'rock': ['fist'], 'scissors': ['peace'], 'other': ['ok']}},
        {'classes': {'rock': ['fist'], 'paper': ['fist'], 'scissors': ['peace'], 'other': ['ok']}},
        {'hands_per_class': {'train': 1, 'test': 1}},
        {'stretch_range': [1.2, 2.0]},
    ],
)
def test_gesture_config_rejects_inconsistent_settings(
    repo_config: AppConfig, change: dict[str, Any]
) -> None:
    raw = {**repo_config.gestures.model_dump(), **change}
    with pytest.raises(ValueError):
        GestureConfig.model_validate(raw)


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    text = (REPO_ROOT / 'config' / 'config.yaml').read_text(encoding='utf-8')
    path = tmp_path / 'config.yaml'
    path.write_text(text + 'surprise: true\n', encoding='utf-8')
    with pytest.raises(ValueError):
        load_config(path)


def download_of(content: bytes) -> Download:
    return Download(
        url='https://example.org/file.zip',
        file='file.zip',
        sha256=hashlib.sha256(content).hexdigest(),
    )


class FakeResponse(io.BytesIO):
    def __enter__(self) -> 'FakeResponse':
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_fetch_downloads_and_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b'landmarks'
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: FakeResponse(content))
    path = fetch(download_of(content), tmp_path)
    assert path.read_bytes() == content
    assert sha256_of(path) == hashlib.sha256(content).hexdigest()


def test_fetch_reuses_an_intact_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b'already here'
    (tmp_path / 'file.zip').write_bytes(content)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError('no download expected')

    monkeypatch.setattr('urllib.request.urlopen', refuse)
    assert fetch(download_of(content), tmp_path).read_bytes() == content


def test_fetch_rejects_a_corrupt_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: FakeResponse(b'tampered'))
    with pytest.raises(ValueError, match='checksum'):
        fetch(download_of(b'original'), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_write_json_is_compact_by_default(tmp_path: Path) -> None:
    path = write_json({'a': [1, 2]}, tmp_path / 'out' / 'file.json')
    assert path.read_bytes() == b'{"a":[1,2]}\n'


def test_predictor_payload_lists_the_experts(settings: PredictorSettings) -> None:
    payload = predictor_payload(settings)
    assert payload['format'] == EXPORT_FORMAT
    assert payload['experts'] == list(EXPERT_NAMES)
    assert PredictorSettings(**payload['settings']) == settings


def assert_close(actual: Any, expected: Any, path: str = '') -> None:
    '''Compare nested JSON, allowing for the last bits of floating point maths across platforms.'''
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_close(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_close(left, right, f'{path}[{index}]')
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), path
    else:
        assert actual == expected, path


def test_committed_game_files_match_the_code() -> None:
    '''The JavaScript tests read these fixtures, so they must describe the current Python code.'''
    models = REPO_ROOT / 'web' / 'models'
    fixtures = REPO_ROOT / 'web' / 'test' / 'fixtures'
    network = GestureNetwork.from_json(json.loads((models / GESTURE_MODEL_FILE).read_text()))
    settings = PredictorSettings(**json.loads((models / PREDICTOR_FILE).read_text())['settings'])
    assert_close(
        gesture_fixture(network), json.loads((fixtures / GESTURE_FIXTURE_FILE).read_text())
    )
    assert_close(
        predictor_fixture(settings), json.loads((fixtures / PREDICTOR_FIXTURE_FILE).read_text())
    )

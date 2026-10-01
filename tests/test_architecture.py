"""
test_architecture.py — 계층 규칙 검증

  src/        : 솔루션 본체 (Docker 이미지에 들어가는 코드)
  validation/ : 시뮬레이터/평가/실험 도구 (솔루션 밖)

의존 방향: validation -> src 는 허용, src -> validation 은 금지.
(과거 DataCollector 가 시뮬레이터의 정답지 CSV를 직접 읽어 평가 정답이 학습에 새어 들어간 사례를 방지)

실행 방법:
  python -m pytest tests/test_architecture.py -v
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FORBIDDEN = ("validation", "tools", "tests", "scripts")


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_src_never_imports_validation_or_dev_folders():
    violations = []
    for py in (ROOT / "src").rglob("*.py"):
        for mod in _imports(py):
            if mod.split(".")[0] in FORBIDDEN:
                violations.append(f"{py.relative_to(ROOT)}: import {mod}")
    assert not violations, "src 는 검증/개발 도구를 import 할 수 없음:\n" + "\n".join(violations)


def test_src_does_not_reference_dev_folder_paths():
    """import 가 아니라 경로 문자열로 몰래 읽는 경우(eval_dataset.csv 사례)도 차단"""
    needles = ("validation/", "tools/simulator", "eval_dataset", "eval_runs")
    violations = []
    for py in (ROOT / "src").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for n in needles:
            if n in text:
                violations.append(f"{py.relative_to(ROOT)}: '{n}'")
    assert not violations, "src 에서 검증 도구 경로를 참조함:\n" + "\n".join(violations)


def test_validation_has_no_unexpected_top_level_dev_dirs():
    """검증 도구는 validation/ 한 곳에만 둔다 (tools/, scripts/ 가 다시 생기는 것을 방지)"""
    for name in ("tools", "scripts"):
        assert not (ROOT / name).exists(), f"{name}/ 폴더가 다시 생겼습니다. validation/ 으로 옮기세요."

# CLAUDE.md

이 프로젝트는 원래 Gemini CLI로 진행되었으며, 핵심 규범과 진행 상태는 `GEMINI.md`에 있습니다.
Claude Code도 아래 문서를 **동일한 규범**으로 따릅니다. (중복 관리를 피하기 위해 내용을 복사하지 않고 import 합니다.)

## 핵심 규범 및 현재 상태
@GEMINI.md

## 구현 계획 (Single Source of Truth)
@plan.md

## 학습 레슨 (작업 시작 전 반드시 확인)
@lessons.md

## Claude Code 운용 메모
- **상태 동기화**: 큰 작업 완료 시 `GEMINI.md`의 `Status` 섹션을 업데이트합니다. 진행 상태의 원본은 `GEMINI.md` 하나로 유지하며, 이 파일(`CLAUDE.md`)에는 상태를 따로 기록하지 않습니다.
- **테스트 실행**: `pytest -v` (모듈 단위: `pytest tests/models/ -v`). `tests/` 는 `src/` 구조를 따르는 모듈별 배치.
- **폴더 규칙**: `src/`·`ui/` 는 솔루션, `validation/` 은 시뮬레이터·평가·실험 도구(`validation/README.md`). `src` 는 `validation` 을 import 하지 않음(`tests/test_architecture.py`). 모델 평가는 `python validation/cli/evaluate_model.py`.
- **통합 실행**: `docker-compose up -d --build` (인프라만 실행: `docker-compose up -d kafka redis zookeeper`)
- **로컬 API 서버**: `python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`
- **환경**: Windows 11. `src/config.py`는 `src/config.py.example`을 기반으로 하며 DB 접속 정보 등을 포함합니다(`.gitignore` 처리됨).

# PTN 이상탐지 프로젝트 핵심 규범 (GEMINI.md)

이 파일은 PTN EMS 이상탐지 솔루션 개발을 위한 기본 지침을 담고 있습니다. 이 지침은 일반적인 워크플로우보다 우선합니다.

## 1. 하네스 코딩(Harness Coding) 원칙
모든 개발은 참고 문서에 정의된 **하네스 코딩** 방법론을 엄격히 따릅니다.
- **에이전트 = 모델 + 하네스**: 단순히 코드를 제안하는 것이 아니라, [목표 -> 조사 -> 설계 -> 구현 -> 테스트]의 제어 루프를 스스로 실행해야 합니다.
- **제어 루프 (Control Loop)**:
    1. **방향 제시(Target)**: 시작 전 목표와 제약 조건을 명확히 합니다.
    2. **가이드 제공(Guide)**: `plan.md` 및 기존 패턴을 준수합니다.
    3. **결과 분류(Result)**: 모든 동작은 검증(PASS/FAIL/SKIP)되어야 하며, 실패 시 원인을 분석하고 재시도합니다.
- **자가 학습 루프 (Lesson System)**:
    - 반복되는 실패(3-Strike 룰)는 반드시 `lessons.md`에 기록합니다.
    - 매 작업 시작 전 `lessons.md`를 읽어 동일한 실수를 방지합니다.

## 2. 기술 스택 및 표준
- **핵심**: Python, PyTorch (LSTM-Autoencoder), MySQL.
- **주기**: 15분 단위 데이터 수집 및 추론.
- **구조**: 데이터 수집, 전처리, 모델, 추론 레이어를 철저히 분리합니다.
- **검증**: `tests/` 디렉토리에 대응하는 테스트 케이스가 없는 코드는 완료된 것으로 간주하지 않습니다.

## 3. 현재 작업 상태 (Status)
- **현재 단계**: Phase 13 XAI 및 능동 학습 대기 중
- **완료 항목 (Phase 12까지)**:
    - RCA 엔진 코어 구현 및 룰 관리 프론트엔드 연동 완료. 
    - Feature Contribution 산출기, 구조화된 도메인 룰 JSON(Min/Max/기여도), 조치 방법(Action) 연동 완료.
    - Rule Management 동적 폼 및 대시보드 툴팁 UI 구축 완벽 적용.
    - **16종의 PTN 도메인 RCA 룰셋 주입 (Ratio, Trend Slope 적용 완료)**
    - **3-Sigma 기반 동적 임계치(Dynamic Threshold) 파이프라인 통합 완료**
    - **자동 재학습**: Data Drift 감지 및 백그라운드 모델 재학습 파이프라인 (완료)
    - 24시간 주기 MSE 평가, 저장된 기존 학습 파라미터(Epochs 등) 재사용 로직, UI 연동 완료.
    - **오프라인 모델 검증 및 오탐(FP) 최소화 완료**: 
        - TTF-Aware 평가 스크립트 기반 F1-Score 0.86 달성 및 정밀도 96.2% 방어 성공.
        - 알람 피로도 억제 및 장애 사후 방치 기간을 고려한 현실적인 채점(Ground Truth) 로직 보완 완료.
    - **MLOps 및 모델 배포 고도화 완료**:
        - 알람 피로도 억제를 위한 심각도별 차등 쿨다운(Alert Dampening) 적용.
        - 파생 변수(MA, Var, Lag)를 생성하는 모델 고도화 적용 (input_dim=15).
        - 버전 관리를 지원하는 Lightweight Model Registry 도입 (버전 저장/Active 지정. 자동 롤백은 미구현 → plan.md Backlog).
    - **대용량 분산 처리 및 고가용성 아키텍처 완료**:
        - Apache Kafka 기반 스트림 프로세싱 도입 (Producer/Consumer 완벽 분리).
        - Redis를 활용한 분산 환경에서의 시계열 Rolling Window 상태 관리 구축.
        - API, UI, Producer, Consumer 각 서비스별 컨테이너화(Dockerization) 완료 (K8s/HPA는 미구현 → plan.md Backlog).
    - **정합성 점검 및 버그 수정 (2026-10-01)**: 컨테이너 DB 호스트/모델·룰 공유 볼륨, DB silent failure 제거, 포트 필터(.any) 버그, `PATHS` 전역 오염, 레지스트리 경로, 드리프트 baseline, 결측 0-채움 제거. 낡은 테스트 정비 및 회귀 테스트 추가 (전체 36 PASS).

    - **2차 점검 수정 (2026-10-01)**: 포트 단위 알람 발생/해제(`alarm_tracker`), Consumer의 CRITICAL 복구 CLEAR 전달, 학습 코드의 정답지 암묵 의존 제거(`exclude_path`/`TRAIN_EXCLUDE_CSV`로 명시 주입), SQL 파라미터 바인딩, 학습 중복 실행 409 가드, `.dockerignore`·Producer 전용 requirements, 수동 실행 스크립트를 `tests/` 에서 분리 (전체 50 PASS).
    - **평가 재설계 (2026-10-01)**: 구 F1 0.86은 관대한 채점(정답 구간 93.9%)으로 모델 성능의 근거가 될 수 없음. 시드 고정 시나리오 생성기 + 이벤트 지표 + 베이스라인 병기 평가로 교체 (`validation/cli/evaluate_model.py`). 현재 활성 모델(v1) 결과: 노이즈 포함 이벤트 F1 0.325 / 제거 시 0.612 (고정 임계 0.77, 롤링 규칙 0.92보다 낮음).
    - **격리 재학습 실험 (2026-10-01)**: 학습 코드의 스케일러가 검증 데이터로 재 fit되던 버그를 수정하고(모든 기존 모델에 영향), 장애 구간을 제거한 정상 데이터로 재학습. 조기 탐지율 44%→79~83%, 광 열화 조기 탐지 16%→93~98%, AUPRC 0.29→0.60. **정상적인 변동(버스트/산발 에러)까지 포함해 학습한 모델**이 노이즈 환경에서 오탐 0.51→0.11건/포트·일, 이벤트 F1 0.32→0.63. 그러나 롤링 규칙(F1 0.89~0.92)에는 여전히 미달(주로 오탐: 이벤트 정밀도 약 50% vs 95%). 활성 모델은 아직 교체하지 않음.

    - **폴더 구조 정리 (2026-10-01)**: 솔루션(`src/`, `ui/`)과 검증 도구를 분리. `tools/`+`scripts/` → `validation/`(simulator, evaluation, cli, runs). `src` → `validation` 방향 import 금지(`tests/test_architecture.py`). 구형 코드 정리: `collect_data.py`, 구형 이력 생성기 3종(`generate_history`, `generate_rca_history`, `realtime_injector`) 삭제 → `validation/cli/seed_db.py`로 대체, 레거시 정답지는 `validation/runs/legacy/` 로 보관. `tests/`는 Phase 이름 기반 파일을 해체해 `src/` 구조를 따르는 모듈별 배치로 재정리 (72 PASS).

- **대기 중 (Phase 13)**:
    - **XAI (설명 가능한 AI)**: SHAP/LIME을 통한 폭포수 차트 구현.
    - **능동 학습 (Active Learning)**: 관리자 피드백 기반 실시간 라벨링 및 RLHF 파이프라인.

- **최종 업데이트**: 2026-10-01
- **참고**: 2026-10-01부터 Claude Code도 함께 사용합니다. `CLAUDE.md`가 이 파일을 import하므로 이 파일이 규범·상태의 유일한 원본입니다.

## 4. 워크플로우
- `plan.md`를 구현 단계의 유일한 진실 공급원(Single Source of Truth)으로 참조합니다.
- 모든 프로젝트 상태는 임시 폴더가 아닌 현재 루트 디렉토리(`D:\PTN_Anomaly_Detection`)에서 관리합니다.
- **큰 단위의 작업이 완료되면 반드시 다음 세 파일을 업데이트하여 상태를 동기화합니다:**
    1. `GEMINI.md`: `Status` 섹션에 현재 단계 및 완료 항목 기록.
    2. `plan.md`: 해당 Phase의 태그를 `(Complete)`로 변경하고 다음 단계를 `(Go)`로 설정.
    3. `lessons.md`: 작업 중 겪은 시행착오나 얻은 교훈 기록 (필요 시).

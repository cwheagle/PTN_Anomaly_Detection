# PTN EMS 이상탐지 프로젝트 실행 계획 (plan.md)

## 1. Background & Motivation (배경 및 목적)
- **배경:** PTN(Packet Transport Network) EMS(Element Management System) 장비의 15분 단위 성능 데이터(In Packet, Error, BPS, PPS)와 광 모듈 상태 데이터(Tx/Rx)가 다중 MySQL DB 서버에 분산 저장되어 있음.
- **목표:** 해당 시계열 데이터를 기반으로 딥러닝 기술을 적용하여 장애를 사전에 탐지하고 예지 정비가 가능한 통합 엔진 개발 및 시각화 서비스 구축.
- **개발 원칙:** AI 에이전트 기반 개발(Harness Coding) 방법론을 적용하여, 목표-조사-설계-구현-테스트-학습의 루프를 통한 코드 품질과 유지보수성 확보.

## 2. Scope & Architecture (범위 및 아키텍처)
- **개발 언어 및 프레임워크:** Python, PyTorch (Deep Learning), Pandas/NumPy (Data Processing), SQLAlchemy (DB Connection), APScheduler (Task Scheduling).
- **데이터베이스:** MySQL (원본 성능 데이터 및 분석 결과 저장).
- **핵심 컴포넌트:**
  1. `DataFetcher`: MySQL DB에서 주기적(15분)으로 최신 데이터를 조회.
  2. `Preprocessor`: 결측치, 이상치(예: 큰 Packet) 처리 및 스케일링(RobustScaler).
  3. `AnomalyDetector (Model)`: LSTM-Autoencoder 모델 아키텍처 설계(Train) 및 추론(Inference) 기능.
  4. `InferenceEngine`: 스케줄러를 통한 주기적(Real-time) 이상 탐지 수행.

## 3. Implementation Steps (구현 단계별 세부 단계)

### Phase 1: 기반 설정 및 데이터 연동 (Complete)
- **목표:** MySQL 연결 및 데이터 추출, 전처리 로직 구현.
- **세부 내용:**
  - `config.py` 구축 (DB 접속 정보, 모델 하이퍼파라미터 등 설정 관리).
  - `db_connector.py` 구현 (데이터베이스 연결 및 쿼리 시스템).
  - `data_processor.py` 구현 (데이터 정제, 스케일링, 시퀀스(Window) 생성).
- **검증:** 단위 데이터 전처리 결과 확인 및 DB 연결 성공 여부 유닛 테스트 (`test_data.py`).

### Phase 2: 모델 개발 및 학습 로직 (Complete)
- **목표:** LSTM-AE 모델 구조 설계 및 학습 코드 개발.
- **세부 내용:**
  - `model.py` 구현 (PyTorch `nn.Module` 기반 LSTM-Autoencoder 클래스 구현).
  - `trainer.py` 구현 (Loss function(MSE), Optimizer 설정, 모델 학습 루프 및 모델 가중치 저장).
- **검증:** 더미 데이터를 이용한 학습 프로세스 정상 동작 여부 및 모델 아키텍처 차원 확인 테스트 (`test_model.py`).

### Phase 3: 추론 엔진 및 스케줄러 통합 (Complete)
- **목표:** 15분 주기로 동작하는 실시간 추론 및 결과 저장 시스템 구축.
- **세부 내용:**
  - `inference.py` 구현 (학습 모델 로드, 최신 데이터 예측, Reconstruction Error 기반 점수 및 임계치 산출).
  - `scheduler.py` 구현 (APScheduler 적용, 15분마다 `inference.py` 핵심 함수 호출).
  - 추론 결과를 DB에 저장하는 로직 추가.
- **검증:** 전체 파이프라인(조회-전처리-추론-저장) 통합 테스트.

### Phase 4: 세부 아키텍처 데이터 표준화 (Complete)
- **목표:** Traffic과 Optical 트랙을 분리하여 데이터 수집 및 분석을 다원화하는 기반 마련.
- **세부 내용:**
  - 데이터 스키마 표준화 (`ip_addr`, `cid`, `lid` 기반 식별 체계 확립).
  - Traffic/Optical 분리 분석 구조 구축 및 세부 트랙 로직 구현.
  - 탐지 시 단순 여부가 아닌 구체적 사유(`anomaly_reason`) 생성 로직 추가.
- **검증:** 과거 실제 데이터 기반의 트랙별 추론 결과 검토.

### Phase 5: 모델 고도화 및 통합 검증 (Complete)
- **목표:** 실제 현장 데이터의 편차와 노이즈를 반영한 모델 성능 및 안정성 강화.
- **세부 내용:**
  - **데이터 전처리 고도화**: Traffic 데이터 `log1p` 변환 및 `RobustScaler` 적용을 통한 이상치 왜곡 최소화.
  - **모델 성능 개선**: LSTM-Autoencoder 구조 최적화 및 최적의 임계치 percentile 설정.
  - **통합 검증기**: `test_inference.py`를 강화하여 [추론 + CSV 저장 + 탐지 사유 리포팅] 기능을 통합.
- **검증:** 
    - 과거 장애 시점 데이터를 활용한 실제 탐지 성능(Hit Rate) 평가 완료.
    - **장기 가동 테스트**: 15분 주기의 실시간 운영 시 데이터 수집 및 추론 엔진 무결성 검증 완료. (2026-05-11)

### Phase 6: 예지 정비 및 지능형 분석 고도화 (Complete)
- **목표:** 단순 이상 탐지를 넘어, 추세 분석을 통한 미래 시점 예측 및 지능형 알람 구현.
- **세부 내용:**
  - **심각도 스코어링 (Severity Scoring)**: 원본 MSE 점수를 0~100 사이의 상대적 심각도 점수로 변환하는 로직 구현.
  - **다단계 경보 체계 (Alerting)**: 심각도에 따른 주의(Minor), 경고(Major), 심각(Critical) 등급 부여 로직 완료.
  - **추세 분석 및 기울기(Slope) 분석**: 연속된 추론 결과의 변화율을 통해 초기 장애 포착 및 급격한 악화 시점 감지 로직 구현.
  - **잔여 수명 예측 (RUL Prediction)**: 현재 추세 기반 미래 시점의 임계치 도달 예상 시간을 계산, 15분 단위의 올림(Ceil) 처리 로직 구현 완료. (2026-05-11)
- **검증:** 실제 장비 장애 패턴과의 비교를 통한 예측 시점 및 RUL 정확도 검토 완료.

### Phase 7: 지능형 엔진 코어 완성 (Complete)
- **목표:** 동적 임계치 및 MLOps 기반의 자동 재학습 체계를 구축하여 사람의 개입 없이 진화하는 AI 엔진 코어 완성.
- **세부 내용:**
  - **Backend API & 인프라**: FastAPI 서버, 실시간 SSE, 무중단 모델 갱신(Hot-Reload), Blackwell(GB10) 최적화 및 DB 동기화 로직 구축 완료.
- **검증:** 
  - FastAPI 서버 통합 테스트 완료.
  - 실시간 SSE 이벤트 스트리밍 및 Hot-Reload 무중단 배포 확인 완료.

### Phase 8: RCA 엔진 코어 구현 및 관리자 UI 구축 (Complete)
- **목표:** '이상이 탐지됐다'를 넘어, '어떤 장애가 의심된다'는 구체적 진단명과 조치 방법을 출력하는 RCA(Root Cause Analysis) 엔진 코어와 이를 관리할 프론트엔드 UI를 구축.
- **배경:** 화웨이 iMaster NCE, 노키아 WaveSuite 등 선도 솔루션의 핵심 차별화 포인트가 단순 이상 감지가 아닌 **장애 원인 진단 + Feature 기여도 분석**임을 확인. 운용자가 실질적으로 필요한 정보는 '수치가 이상하다'가 아닌 '이 장비의 이 포트에서 이런 장애가 의심된다'는 진단 및 조치 방법임.
- **세부 내용:**
  - **Feature Contribution 분석기**: 이상 시점에서 각 Feature(tx_packet, rx_packet, error_packet, tx_avg_power, rx_avg_power)가 MSE 상승에 기여한 비율(%)을 계산하여, 주요 원인 Feature를 식별.
  - **도메인 룰 인터페이스 설계 (Fully Structured JSON)**: 도메인 전문가(네트워크 엔지니어)가 룰을 정의하여 주입할 수 있는 안전한 인터페이스 구축.
    - 예: `{"contributions": {"error_packet": 60}, "diagnosis": "CRC/비트 오류 의심", "action": "광 커넥터 청소"}`
  - **장애명 진단 출력**: 룰이 없어도 Feature Contribution 기반의 일반 진단이 동작하도록 기본(Default) 룰 세트를 함께 제공. (예: `RX 전력 기여도 1위 → 광 수신 열화 의심`).
  - **DB 스키마 확장 및 파이프라인 연동**: `rca_diagnosis`, `feature_contribution`, `rca_action` 컬럼을 동적으로 추가하고 파이프라인(DB-API-UI) 전체를 연동.
  - **UI/UX 개선 및 룰 관리 프론트엔드**:
    - 대시보드 UI의 Recent Anomalies 표에 툴팁을 활용한 추천 조치(Action) 렌더링 추가.
    - Rule Management UI 전면 개편: 구조화된 JSON 기반의 동적 입력 폼(Min/Max, 기여도) 프론트엔드 연동 완성.
- **검증:**
  - 모의 이상 데이터 주입 시 Feature Contribution 비율 정확성 검토.
  - 기본 룰 세트 적용 후 도출된 진단명의 직관적 타당성 검토.

### Phase 9: 지능형 MLOps 및 시스템 안정화 (Complete)
- **목표:** 도메인 전문가 룰을 주입하여 RCA 정확도를 고도화하고, 동적 임계치·자동 재학습을 완성하며, 상용 수준의 대시보드를 구현.
- **세부 내용:**
  - **도메인 룰 주입 (Rule Injection)**: 네트워크 엔지니어가 정의한 룰을 Phase 8의 인터페이스에 등록 및 검증 (디테일한 룰셋 구축 작업).
  - **동적 임계치 (Dynamic Thresholding)**: 시간대/요일별 계절성(Seasonality)을 반영한 가변 임계치로 오탐(False Positive) 최소화.
  - **자동 재학습 파이프라인 (Auto-Retraining MLOps)**: 성능 저하(Drift) 감지 시 백그라운드 학습 트리거 → Hot-Reload 무중단 배포.
- **검증:**
  - 3-Sigma 기반 동적 임계치(Dynamic Threshold) 파이프라인 검증.
  - 백그라운드 재학습 → 무중단 배포 파이프라인 자동화 무결성 테스트.

### Phase 10: 오프라인 모델 검증 및 성능 평가 (Complete)
- **목표:** 예지 정비(Predictive Maintenance)의 특수성을 고려하여, 잔여 수명(TTF)을 활용한 동적 평가 지표(TTF-Aware F1-Score) 평가 프레임워크 완성.
- **주요 작업:**
  - 시뮬레이터가 생성한 **정답지(Ground Truth) 데이터셋**(`eval_dataset.csv`) 연동 및 검증 완료.
  - **데이터 오염 방지(Data Sanitization)**: 훈련셋 구성 시 장애 구간 데이터를 철저히 도려내어 모델 오염 방지 기능 적용 완료.
  - TTF-Aware 다이나믹 타임 윈도우 기반 Precision, Recall, F1-Score 산출 로직 구현 완료.
  - **오탐(False Positive) 억제 튜닝 완료**: 
    - 훈련 데이터 순수화로 인한 'Over-Sensitivity' 부작용 해결 (최소 임계치 1.0x 강제 보장).
    - Minor(50) 이상 시점에만 TTF를 산출하도록 로직 보완하여 자연 노이즈 무시.
    - 결과적으로 오탐률을 0.03% 수준(FP 110개)으로 감소시키고, F1-Score 0.84 달성.

### Phase 11: MLOps 및 모델 배포 고도화 (Final Polish) (Complete)
- **목표:** Phase 10의 평가 프레임워크를 바탕으로, 엔터프라이즈 상용망 수준의 예측 정확도 및 모델 생명주기 관리(MLOps) 안정성을 확보.
- **세부 내용 (핵심 과제):**
  1. **알람 피로도 억제 및 심각도별 차등 쿨다운 (Alert Dampening)**:
     - 단순 임계치 초과 시 즉각 알람을 쏘는 구조를 탈피하여 추론 엔진 단에 방파제 추가.
     - CRITICAL(즉시), MAJOR(2회/30분), MINOR(3회/45분) 등 심각도에 따른 차등 지속성 검증 로직 구현.
  2. **시계열 파생 변수(Feature Engineering) 고도화**:
     - 원본 15분 단위 메트릭에 이동 평균(1h/4h), 이동 변동성, 시차 데이터(Lag) 자동 생성.
  3. **모델 레지스트리 및 자동 롤백 (Advanced MLOps)**:
     - MLflow 등 모델 버저닝 및 성능 하락 감지 시 자동 롤백(Rollback) 체계 도입.
- **검증:**
  - Phase 10의 `evaluate_model.py`를 활용하여 **심각도별 쿨다운 도입 후 Precision 90% 이상 향상** 검증.

### Phase 12: 대용량 분산 처리 및 폐쇄망 배포 아키텍처 (Scalability & HA) (Go)
- **목표:** 전국망 단위(10만 대 이상)의 노드를 지연 없이 실시간으로 분석할 수 있는 상용 엔터프라이즈 인프라 구축 및 폐쇄망 배포 환경 구성.
- **세부 내용:**
  1. **마이크로서비스 컨테이너화 (Dockerization)**: Kubernetes에 배포하기 위한 필수 전제 조건. UI, API, Producer, 추론 엔진(Consumer)을 각각 독립된 Docker 이미지로 빌드하여 폐쇄망 오프라인 배포 파이프라인 구축.
  2. **스트림 프로세싱(Stream Processing) 도입**: 기존 DB 스케줄러 폴링(Polling) 방식을 탈피하여, Apache Kafka 기반의 실시간 데이터 파이프라인으로 마이그레이션.
  3. **클라우드 네이티브 고가용성 (Kubernetes)**: 빌드된 도커 컨테이너들을 K8s 클러스터에 배포. 트래픽 부하에 따라 AI 추론 파드(Pod)를 HPA(오토스케일링)로 동적 확장하고, 장애 시 즉각 페일오버(Failover)하는 완벽한 HA 체계 완성.

### Phase 13: XAI(설명 가능한 AI) 및 능동 학습 (Active Learning)
- **목표:** 현업 엔지니어의 신뢰도를 높이고, 인간의 피드백을 통해 AI가 스스로 진화하는 플라이휠(Flywheel) 완성.
- **세부 내용:**
  1. **XAI 폭포수 차트 (SHAP / LIME 적용)**: RCA 결과 도출 시, 각 메트릭이 최종 Severity(95점)에 기여한 비중을 시각적인 폭포수 차트로 대시보드에 제공.
  2. **Human-in-the-loop (피드백 기반 RLHF) 및 실시간 정답지 자동화 연동**: 
     - NMS UI 수동 입력(관리자의 [👍실제 장애 / 👎오탐임] 피드백) 및 장비의 Link Down/Up 원시 로그를 추적하여 실제 정답지(Ground Truth) 자동 구축.
     - 수집된 라벨링 데이터를 즉시 반영하여 다음 재학습 시 모델의 가중치를 교정하는 능동 학습(Active Learning) 파이프라인 완성.

---

## 4. Harness Checklists & Rules (실행 단계별 체크리스트 및 수칙)
- **(방향 제시) Target-Plan 매칭:** 모든 구현 코드는 `plan.md`의 단계와 연결되어야 함. 목적 없는 코드 생성 금지.
- **(기술 표준) 레이어 분리:** 데이터 연동(DB), 비즈니스 로직(추론), 모델 로직(PyTorch)의 책임을 명확히 분리하여 이식성 최적화.
- **(결과 분류 및 검증 루프) 유닛 테스트:** 모든 핵심 함수는 대응하는 테스트 코드를 생성 및 실행(PASS/FAIL 판정).
- **(학습 루프) 3-Strike 룰:** 동일한 에러(예: Tensor 차원 불일치) 3회 반복 시 작업을 중단하고 `lessons.md`에 기록 및 해결책 수립 후 재개.

## 5. Verification & Testing (검증 계획)
- **Unit Test:** `pytest` 기반의 모듈별 독립 검증.
- **Integration Test:** 데이터 수집부터 DB 저장까지의 전체 파이프라인 검증.
- **내결함성 테스트:** DB 연결 끊김, 데이터 결측 등 예외 상황에서의 시스템 안정성 및 자동 복구 로직(Try-Except) 검증.

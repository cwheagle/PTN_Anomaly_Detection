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

### Phase 7: 통합 서비스 아키텍처 및 Web UI 구현 및 고도화 (Go)
- **목표:** 개발된 엔진을 서비스 형태로 패키징하고, 사용자 대시보드를 상용 수준으로 구현 및 고도화하여 운영 가시성 확보.
- **세부 내용:**
  - **Backend API 서버 (FastAPI)**: 스케줄러 엔진 통합 API 서버 구축 및 REST 규격 준수 완료.
  - **실시간 이벤트 알림 (SSE)**: Critical 경보 발생 시 프론트엔드 즉시 푸시(SSE) 구현 완료.
  - **통합 모니터링 대시보드 (Vue 3)**: 실시간 경보 리스트, 모델 상태, 추세 차트 기본 구축 완료.
  - **모델 관리 시스템**: 훈련/추론 설정 분리 및 실시간 임계치 업데이트, 학습 상태 추적 및 중지 기능 API 구현 완료.
  - **신뢰성 고도화**: API 버전 체계(v1.0.0) 정립 및 APScheduler 비동기 최적화, MySQL 데이터 무결성 처리(NaN/inf) 완료.
  - **데이터 신뢰성 및 동기화**: DB Upsert 로직 도입으로 분석 결과 일관성 확보, 클라이언트 간 알람 상태 동기화(Pull+Push) 구현 완료.
  - **Blackwell 최적화**: GB10 GPU 전용 `torch.compile` 및 FP8 연산(Transformer Engine) 반영, 하드웨어 이식성(Fallback) 확보 완료.
  - **UI 고도화 (UI Advancement)**: 디자인 폴리싱(글래스모피즘, 글로우 효과), 애니메이션 강화, 전문적인 차트 스타일링 적용. (진행 중)
- **검증:** 
  - 프론트엔드-백엔드 실시간 데이터 스트리밍 및 모델 제어 무결성 검증 완료.
  - UI 고도화 후 시각적 전문성 및 사용자 경험(UX) 개선 여부 검증.

### Phase 8: 동적 임계치 및 자동 재학습 체계 (Pending)
- **목표:** 장기 데이터 분석 기반 동적 임계치 도입 및 모델 성능 하락 시 자동 재학습 파이프라인 구축.
- **세부 내용:**
  - **계절성 분석 (Seasonality)**: 시간대별/요일별 데이터 패턴 분석을 통한 가변 임계치 엔진 개발.
  - **성능 모니터링**: 추론 결과의 Drift(편차) 감지 및 성능 저하 판단 로직 구현.
  - **자동 재학습 (Auto-Retraining)**: 성능 저하 감지 시 백그라운드에서 최신 데이터 수집 및 모델 자동 갱신.
  - **알림 채널 확장**: Slack/Email 등 외부 알림 연동 및 경보 전파 체계 강화.
- **검증:** 
  - 동적 임계치 적용 전후의 오탐/미탐 비율 비교 검증.
  - 자동 재학습 파이프라인의 엔드투엔드 안정성 테스트.

---

## 4. Harness Checklists & Rules (실행 단계별 체크리스트 및 수칙)
- **(방향 제시) Target-Plan 매칭:** 모든 구현 코드는 `plan.md`의 단계와 연결되어야 함. 목적 없는 코드 생성 금지.
- **(기술 표준) 레이어 분리:** 데이터 연동(DB), 비즈니스 로직(추론), 모델 로직(PyTorch)의 책임을 명확히 분리하여 이식성 최적화.
- **(결과 분류 및 검증 루프) 유닛 테스트:** 모든 핵심 함수는 대응하는 테스트 코드를 생성 및 실행(PASS/FAIL 판정).
- **(학습 루프) 3-Strike 룰:** 동일한 에러(예: Tensor 차원 불일치) 3회 반복 시 작업을 중단하고 `_lessons.md`에 기록 및 해결책 수립 후 재개.

## 5. Verification & Testing (검증 계획)
- **Unit Test:** `pytest` 기반의 모듈별 독립 검증.
- **Integration Test:** 데이터 수집부터 DB 저장까지의 전체 파이프라인 검증.
- **내결함성 테스트:** DB 연결 끊김, 데이터 결측 등 예외 상황에서의 시스템 안정성 및 자동 복구 로직(Try-Except) 검증.

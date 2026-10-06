# P1-1 설계: 재학습 파이프라인 정상화

> 작성: 2026-10-01 · 근거: plan.md Backlog **결정 사항 2)**, P1-1 행, lessons #19·#22·#24·#25·#28·#30·#31
> 선행: P0-1 (`src/models/registry.py` — 후보 저장/승격/롤백) 완료. 이 설계는 그 위에 얹는다.
> 상태: **확정(2026-10-06)** — 10장 결정 사항 확정, 5장 기본값 그대로. 구현은 dev.

## 0. 요약
| 문제 (결정 사항 2) | 이 설계의 해법 |
|---|---|
| ① 학습 구간 `[-37d, -7d]` 이 트리거 시점과 어긋남 | 학습 구간을 **최근까지** 포함 `[T-28d, T]`. 검증은 시간이 아니라 **포트 단위 홀드아웃(10%)** 으로 분리 → 최근 패턴을 학습하면서도 처음 보는 데이터로 검증 |
| ② 드리프트 신호가 전체 평균 score 1개 | **포트별 평균 score 의 중앙값 비율** + **드리프트 포트 비율**. 소수 포트의 급등(장애)과 다수 포트의 이동(정상 패턴 변화)을 구분 |
| ③ v1 은 baseline 지표가 달라 사실상 미트리거 | 기준값을 **같은 지표(포트별 평균의 분포)로 학습 시 저장**. 구버전 모델은 `baseline_legacy` 로 표시하고 **감지만** (재학습 트리거 안 함) |
| ④ 장애 구간을 정상으로 학습할 위험 | **장애 의심 구간 제외**: 자기 알람 이력 + 모델과 무관한 규칙 + 명시 CSV 의 합집합. 의심 비율이 높으면 학습 중단 |
| ⑤ 검증 없이 자동 배포 | **승격 게이트** (아티팩트 / 임계치·검증 손실 / 홀드아웃 포트의 알람 비율 / 카나리 AUPRC / 임계치 상승 추세) 결과를 후보에 기록, 승격 시 사용 |
| ⑥ 재학습 쿨다운 없음 | **쿨다운 72h + 지속성(3회 연속 감지) + 대기 중 후보 있으면 생략**, 상태는 파일에 영속화 |
| (신규) 모드 | `DRIFT_RETRAIN_MODE = off / candidate(기본) / auto` |

조사 중 확인한 사실 (설계에 반영)
- **F1. 드리프트 평균에 0 이 섞임**: `inference.detect()` 는 없는 트랙의 score 를 `0.0` 으로 채워 저장(`src/pipeline/inference.py:331`)하고, `DriftMonitor` 는 이를 그대로 `AVG` 함(`src/pipeline/drift_monitor.py:79-86`). 광 데이터가 없는 포트가 많으면 평균이 낮아짐. E2E 비율 0.31/0.13 의 원인 중 하나일 **가능성**(미확인).
- **F2. 모델 교체 직후 섞임**: 최근 24h 의 score 에는 이전 모델의 score 가 섞인다(행에 모델 버전 컬럼 없음). 승격 직후 드리프트 판정이 틀어질 수 있음.
- **F3. 알람 이력 보존 30일**: `RETENTION_DAYS = 30` 으로 `anomaly_detection` 이 삭제됨. 알람 이력 기반 제외는 최근 30일까지만 가능 → 학습 구간 기본값 28일.
- **F4. 중복 코드**: 드리프트 재학습 트리거가 `run_daily_maintenance`(`src/api/main.py:85-121`)와 `/api/drift/check`(`:540-592`)에 복제되어 있고, 학습 날짜 기본값도 `/api/model/train`(`:360-364`)까지 3곳에 하드코딩.
- **F5. 비활성 버전 로드 불가**: `AnomalyDetector.reload_model` 은 활성 버전만 로드 → 게이트가 후보를 추론하려면 버전 지정 로드가 필요.
- **F6. `filter_excluded` 는 구간 수만큼 전체 행을 순회**(`src/data/data_collector.py:28-42`) → 수천 구간 × 수백만 행에서 느림.

---

## 1. 전체 흐름

```
[03:00 일일 유지보수 / POST /api/drift/check]
        │
        ▼
DriftMonitor.check_drift()            ── 포트별 통계 → 트랙별 {median_ratio, drifted_port_fraction, kind}
        │
        ▼
retrain_policy.decide(result, state, now, policy)    ── 순수 함수
        │   none      : 정상 / 국소(장애 의심) / legacy 기준값 / 활성화 직후
        │   notify    : 드리프트이나 mode=off, 지속성 미달, 쿨다운, 대기 후보 있음
        │   train     : mode ∈ {candidate, auto} & 광역 드리프트 & 지속 N회 & 쿨다운 경과
        ▼
run_training_pipeline(ft, cfg, plan, trigger="drift")
        │  1) train_window.plan_window(now)           → 기간
        │  2) DataCollector.collect_and_save(..., val_port_fraction=0.1, suspect_policy=policy)
        │        내부에서 suspect = 알람 이력 ∪ 규칙(원본 기반) ∪ 명시 CSV (병합)
        │        의심 비율 > 20% → CSV 저장 없이 {'skipped': ...} 반환 → 학습 중단(상태 기록)
        │  3) Trainer.train()   → 후보 vN (+ 포트 분포 기준값, 학습 구간, 제외 통계)
        │  4) promotion_gate.run_gate(ft, vN)  → 결과를 레지스트리 vN.gate 에 기록
        │  5) mode=auto & trigger=drift & gate=PASS → registry.promote(reason="auto-gate") + reload
        │     그 외 → 후보로 대기, 알림
        ▼
사람: 모델 관리 UI 에서 gate 결과 확인 → /api/model/promote (FAIL 이면 force 필요)
```

UI 의 Train 버튼(수동 학습)도 같은 1)~4)를 거친다(트리거 `manual`, 쿨다운·지속성은 적용 안 함, 자동 승격 없음).

---

## 2. 컴포넌트 설계

### 2.1 학습 구간과 데이터 분할 — `src/data/train_window.py` (신규)

**구간**: 기준시각 T(트리거 시각) 에 대해 `[T - train_days, T]`, 기본 `train_days = 28` (F3).

**포트 단위 홀드아웃**: 포트 키 `(ip_addr, cid, lid)` 의 안정적 해시로 `val_port_fraction`(기본 10%)를 검증 포트로 고정.
- 학습 CSV = 학습 포트 × 전체 구간 − 의심 구간
- 검증 CSV = 검증 포트 × 전체 구간 − 의심 구간 (early stopping, `final_val_loss`, 드리프트 기준값 산출)
- 게이트 데이터 = 검증 포트 × 최근 `gate_days`(3일) **원본(제외 없음)**
- 시간 분할 대신 포트 분할을 택한 이유: 드리프트는 "최근 며칠"에 생기므로 시간 분할을 하면 최근 패턴이 검증/게이트 쪽에만 들어가 학습되지 않는다(문제 ①의 재발). 포트 분할은 최근 패턴을 학습하면서 **처음 보는 포트**로 일반화를 검증한다.
- 해시는 `hashlib.md5(f"{salt}|{ip}|{cid}|{lid}")` (파이썬 `hash()` 는 프로세스마다 달라 금지). 재학습마다 같은 포트가 검증 포트로 남아 버전 간 비교가 가능.
- 한계: 망 전체에 같이 생기는 패턴(계절성)은 학습/검증 포트에 동시에 나타나므로 검증 포트도 "완전히 처음 보는 시기"는 아님 — 게이트 G3 은 "새 포트에서 알람을 과하게 내지 않는가"를 본다.

**장애 의심 구간**: 아래 세 소스의 합집합을 포트별로 병합(겹치거나 맞닿은 구간 합침).
| 소스 | 내용 | 기본 |
|---|---|---|
| A. 자기 알람 이력 | `anomaly_detection` 에서 `alarm_level >= 1` 인 행 → 포트별 연속 구간(간격 ≤ 4스텝 묶음), `[첫 알람 − 16스텝, 마지막 알람 + 4스텝]` 확장. 앞쪽 16스텝(4h)은 장애 직전 램프 구간(lessons #24 의 "영향권 16스텝") | 사용 |
| B. 모델과 무관한 규칙 | 포트별 직전 24h 중앙값 대비: 에러 ≥ 20, 광 수신 ≥ 3dB 하락, 트래픽 < 15%. **조건이 4스텝(1h) 이상 연속**일 때만 의심 구간(산발 에러·버스트 같은 정상 변동은 남겨야 함, lessons #24). 구간 앞뒤 확장은 A와 동일 | 사용 |
| C. 명시 CSV | 기존 `exclude_path` / `TRAIN_EXCLUDE_CSV` (이후 EMS 알람 이력, Phase 13 피드백도 이 형식으로 주입) | 지정 시 |

- B 는 `validation/evaluation/baselines.rolling_rule` 과 같은 아이디어지만 `src → validation` import 가 금지이므로 **src 에 별도 구현**한다. 목적이 다름(평가 베이스라인 vs 학습 데이터 정제)이므로 파라미터를 공유하지 않는다. 문서/주석에 관계를 명시.
- **중단 조건**: 학습 구간 행 중 의심 비율 > `max_excluded_fraction`(20%) → 학습하지 않고 `skipped: suspect_fraction=…` 상태 기록. 대규모 장애 중에 재학습하지 않기 위함(lessons #28).
- **포트 제외**: 한 포트의 의심 비율이 `port_drop_fraction`(50%) 초과 → 그 포트 전체를 학습에서 뺌(만성 불량 포트를 "정상"으로 배우지 않도록).
- 제외 구간 행은 삭제되어 시간축에 공백이 생긴다. `DataProcessor.preprocess` 는 포트별로 reindex 한 뒤 **보간(`limit=1, limit_direction='both'`)으로 공백 양끝을 1행씩 메우고** 나머지를 NaN 으로 남겨, NaN 이 포함된 윈도우를 버린다(`src/data/data_processor.py` reindex·보간·`create_sequences`).
  - 따라서 **3스텝 이상 공백은 가로지르는 시퀀스가 0건**이고, **2스텝 이하 공백은 보간으로 메워져 시퀀스가 가로지른다**(한계).
  - 자동 의심 구간(A·B)은 앞 16·뒤 4스텝 확장으로 항상 20스텝 이상이라 영향이 없다. 해당되는 것은 **명시 CSV(C)의 2스텝 이하 짧은 구간**뿐이며, 이 경우 제외 구간의 원본 행은 학습에 쓰이지 않지만 보간 값으로 채워진 윈도우는 학습에 들어간다.
  - 두 동작 모두 테스트로 고정(7장 T-W5).

인터페이스
```python
@dataclass(frozen=True)
class TrainWindow:
    start: datetime; end: datetime            # 학습·검증 공통 기간
    gate_start: datetime; gate_end: datetime  # 게이트 데이터 기간 (= [end - gate_days, end])

def plan_window(now: datetime, policy: RetrainPolicy) -> TrainWindow
def is_val_port(ip_addr, cid, lid, fraction: float, salt: str) -> bool
def split_ports(df: pd.DataFrame, fraction, salt) -> tuple[pd.DataFrame, pd.DataFrame]   # (train, val)
def intervals_from_alarms(alarm_rows: pd.DataFrame, pre_steps, post_steps, gap_steps=4) -> pd.DataFrame
def intervals_from_rules(raw: pd.DataFrame, policy) -> pd.DataFrame
def merge_intervals(*frames) -> pd.DataFrame          # 컬럼: ip_addr, cid, lid, start_time, end_time, source
def suspect_stats(df, intervals) -> dict              # {rows, excluded_rows, fraction, dropped_ports, by_source}
```
구간 컬럼은 `start_time, end_time` 으로 통일하고, 기존 CSV 형식(`failure_time`)은 `load_exclusions` 에서 `end_time` 으로 별칭 처리(하위 호환).

### 2.2 `DataCollector` 변경 — `src/data/data_collector.py`
- `collect_and_save(train_start, train_end, test_start=None, test_end=None, ..., exclusions=None, val_port_fraction=None, split_salt="ptn", suspect_policy=None)` (구현 기준).
  - `val_port_fraction` 이 주어지면 **포트 분할 모드**: `train/test` 날짜 인자 대신 같은 기간을 포트로 나눠 `<ft>_train.csv`/`<ft>_test.csv` 저장. 주어지지 않으면 기존 날짜 분할(하위 호환 — `validation/` 스크립트와 기존 테스트 유지).
  - **의심 구간 산출은 DataCollector 내부에서 일괄 수행**: `suspect_policy`(RetrainPolicy)가 주어지면 `exclude_from_alarms` 이면 알람 이력 구간(A), `exclude_from_rules` 이면 **트랙별로 조회한 원본 데이터**에서 규칙 구간(B)을 만든다(규칙 B 는 원본 값이 필요하므로 수집 단계에서 계산). 여기에 `exclusions`(호출자 주입 DataFrame)와 기존 `exclude_path`/`TRAIN_EXCLUDE_CSV`(C)를 합집합으로 병합.
  - 반환값: 정상 시 `{ft: {'train': n, 'test': n, 'suspect_stats': {...}}}`. 의심 비율이 `max_excluded_fraction` 을 넘으면 **CSV 를 저장하지 않고** `{'skipped': 'suspect_fraction=…', 'suspect_stats': {...}}` 반환 → API 는 이를 `training_status[ft].last_error` 와 `retrain_state.last_outcome` 에 기록하고 학습하지 않음.
  - 따라서 API 는 의심 구간을 따로 만들지 않고 `suspect_policy` 만 넘긴다 (1장 흐름 2), 2.8).
- `filter_excluded` 를 구간 조인 방식(포트별 `merge_asof` 또는 정렬 후 `searchsorted`)으로 교체 (F6). 결과는 기존 구현과 동일해야 함(테스트 T-C2).
- 알람 이력 조회는 `DBConnector` 에 `fetch_alarm_rows(start, end, min_level)` 추가 (파라미터 바인딩, SELECT `occur_date, ip_addr, cid, lid, alarm_level`).

### 2.3 `Trainer` 변경 — `src/models/trainer.py`
- `__init__(..., trigger: str = "manual", window_info: dict | None = None)` — 메타데이터에 기록만 함.
- 학습 종료 후 **검증 포트 기준 분포**를 계산해 메타데이터/레지스트리에 저장 (드리프트 기준, lessons #19 "같은 지표"):
  - 검증 CSV → `create_sequences(is_train=False)` → 포트별 마지막 시점 MSE 평균 → 그 분포의 `median`, `p90`
  - 메타 키: `baseline_port_median`, `baseline_port_p90`, `baseline_ports`(포트 수)
  - 검증 데이터가 없으면 학습 데이터로 계산하고 `baseline_source: "train"` 표시
- 메타/레지스트리 엔트리 추가 키: `trigger`, `training_window {start, end, val_port_fraction, salt}`, `suspect_stats`, `baseline_port_median`, `baseline_port_p90`.
- 기존 `baseline_mse` 는 유지(하위 호환).

### 2.4 드리프트 신호 — `src/pipeline/drift_monitor.py`
쿼리 (트랙별, 파라미터 바인딩):
```sql
SELECT ip_addr, cid, lid, AVG({ft}_score) AS mean_score, COUNT(*) AS n
FROM anomaly_detection
WHERE occur_date >= %s AND {ft}_score > 0          -- F1: 0.0 은 '트랙 없음'으로 간주
GROUP BY ip_addr, cid, lid
```
- 조회 시작 = `max(now − drift_window_hours, 활성 버전의 activated_at)` (F2). 활성화 후 경과 < `min_hours_since_activation`(24h) 이면 `status: "warming_up"` 으로 판정하지 않음.
- `n < drift_min_port_rows`(48 = 12h) 인 포트는 제외.
- 트랙별 출력 (기존 키 `status, mean_mse, baseline_mse, drift_ratio` 유지 + 추가):
  ```
  median_ratio            = median(mean_score_p) / baseline_port_median
  drifted_port_fraction   = mean(mean_score_p > baseline_port_p90)       # 정상이면 ≈ 0.10
  ports                   = 판정에 쓴 포트 수
  top_ports               = mean_score 상위 10개 (조치 참고용)
  kind                    = "none" | "localized" | "widespread"
  baseline_legacy         = 기준값이 포트 분포가 아닌 구버전(baseline_mse/val_loss)인지
  ```
- 판정:
  - `widespread`: `median_ratio > drift_median_factor(1.5)` **또는** `drifted_port_fraction > drift_port_fraction(0.30)` (정상 기대치 0.10 의 3배)
  - `localized`: widespread 가 아니고 `top_ports` 중 `mean_score > baseline_port_p90 × 3` 인 포트가 있음 → **장애 의심, 재학습 대상 아님**
  - 기준값이 legacy 면 위 계산은 `baseline_mse` 로 근사하되 `baseline_legacy: true` 로 표시 → `decide()` 가 재학습하지 않음
- 기존 응답 구조(`drift_detected`, `drifted_tracks`)는 UI 호환을 위해 유지하되 `drift_detected` 는 `widespread` 일 때만 true.
- DB/판정 로직 분리: `check_drift()` 는 조회만, 판정은 순수 함수 `evaluate_track(port_stats: DataFrame, baseline: dict, policy) -> dict` 로 분리해 DB 없이 테스트.

### 2.5 재학습 결정 — `src/pipeline/retrain_policy.py` (신규)
```python
@dataclass(frozen=True)
class RetrainPolicy:   # 5장 기본값. load_retrain_policy() 가 config.RETRAIN_POLICY(선택) + 환경변수로 덮어씀
    mode: str = "candidate"   # off | candidate | auto
    ...

@dataclass(frozen=True)
class Decision:
    action: str               # none | notify | train
    reason: str               # 사람이 읽는 사유 (UI/로그)
    track: str

def load_retrain_policy() -> RetrainPolicy
def decide(track_result: dict, state: dict, now: datetime, policy: RetrainPolicy,
           pending_candidate: bool) -> Decision          # 순수 함수
def record_check(state: dict, track_result: dict, now) -> dict    # 지속성 카운트 갱신 (같은 날 중복 집계 안 함)
def record_training(state: dict, now, version, outcome) -> dict
def load_state(model_dir, ft) -> dict; def save_state(model_dir, ft, state)   # 원자적 쓰기 (registry.save 와 같은 방식)
```
상태 파일 `<model_dir>/<ft>_retrain_state.json` (공유 볼륨, API 재시작에도 유지):
`{consecutive_widespread: [날짜...], last_trigger_at, last_outcome, last_version, last_skipped_at}`

**결과별 상태 기록 (2026-10-06 확정, 6.3 E2E 발견 반영)**
| 결과 | `last_trigger_at` (쿨다운 시작) | 지속성(`consecutive_widespread`) | `last_skipped_at` |
|---|---|---|---|
| 수집 단계 중단: `skipped: suspect_fraction=…` | 기록 안 함 (쿨다운 미적용) | **유지** | 기록 → 24h 백오프 |
| 수집 오류 (DB 실패 등) | 기록 안 함 (쿨다운 미적용) | **유지** | 기록 → 24h 백오프 |
| 수집 통과 후 Trainer 시작 (이후 후보 생성·학습 예외·게이트 ERROR 모두) | **기록** (쿨다운 72h) | 초기화 | — |

- 쿨다운은 **후보가 연달아 생기는 것**을 막는 장치이고, 의심 비율 중단의 목적은 "장애 중에 재학습하지 않음"이지 "장애 뒤 72h 금지"가 아니다. 따라서 후보를 만들지 않은 중단·수집 오류는 쿨다운을 걸지 않는다.
- 학습 시작 후 실패에 쿨다운을 유지하는 이유: 같은 학습 실패가 매일 반복되는 비용을 막기 위함.
- `record_training(state, now, version, outcome)` 은 Trainer 시작 시점에만 `last_trigger_at` 을 기록한다. 중단·수집 오류는 `record_skip(state, now, outcome)`(`last_outcome`, `last_skipped_at` 만 갱신)으로 분리한다.
- 결과: 장애 구간이 28일 학습 창에서 빠져 의심 비율이 20% 아래로 내려가면, 그다음 일일 점검에서 바로 재학습된다.

`decide()` 규칙 (위에서부터 첫 번째 해당):
| 조건 | action | reason 예 |
|---|---|---|
| 판정 불가(error/warming_up/데이터 부족) | none | `warming_up: 활성화 후 6h` |
| kind == none | none | |
| kind == localized | notify | `국소 이상 3포트 — 장애 의심, 재학습 안 함` |
| baseline_legacy | notify | `구버전 기준값 — 감지만 (재학습하려면 수동 학습)` |
| mode == off | notify | |
| 연속 widespread 일수 < `drift_persist_checks`(3) | notify | `광역 드리프트 1/3일` |
| 마지막 드리프트 트리거 후 < `cooldown_hours`(72) | notify | `쿨다운 41h 남음` |
| 마지막 중단·수집 오류(`last_skipped_at`) 후 < 24h (일일 점검 주기와 같은 고정값) | notify | `재학습 보류: 의심 비율 32.4% > 20% — 장애 의심 (15h 후 재시도)` |
| 드리프트 트리거로 만든 미승격 후보가 활성 버전보다 새로 있음 | notify | `대기 중 후보 v5 검토 필요` |
| 학습 중 | notify | |
| 그 외 | **train** | |

- 수동 `/api/drift/check` 를 같은 날 여러 번 호출해도 지속성 일수는 하루 1회만 증가 (`record_check` 가 날짜 집합으로 관리).
- `notify` 는 지금은 로그 + `last_drift_result.retrain` 필드 + UI 표시. SSE/외부 알림은 범위 밖.

### 2.6 승격 게이트 — `src/models/promotion_gate.py` (신규)
학습 직후 1회 실행해 결과를 레지스트리 엔트리 `gate` 에 저장. 승격 API 는 저장된 결과를 사용(승격 시 재계산하지 않음 → 결과 재현성, 응답 지연 없음). `POST /api/model/gate?ft&version` 으로 재실행 가능.

| ID | 검사 | 판정 | 비고 |
|---|---|---|---|
| G1 | 아티팩트: 파일 존재, threshold·val_loss 가 유한하고 > 0, input_dim 이 현재 DataProcessor 와 일치 | FAIL | force 로도 승격 불가 (깨진 모델) |
| G2 | 임계치/검증 손실 비율 (기존 `registry.compare_with_active`, 3배) | FAIL | P0-1 동작 유지. **활성 모델 기준 상대 비교라서 활성 모델이 비정상이면 정상 후보도 FAIL 한다(의도된 동작)** → `force` 로 승격하거나, 롤백으로 정상 버전을 활성화한 뒤 승격. G3 도 활성 모델 기준 상대 비교라 같은 영향을 받음 |
| G3 | **홀드아웃 포트 알람 비율**: 게이트 데이터(검증 포트 × 최근 3일, 최대 2,000포트 샘플)에서 후보와 활성 모델의 해당 트랙 알람을 **운영과 같은 정책으로** 계산 → 인시던트/포트·일 (1시간 묶음) | `cand ≤ max(active × 1.5, 0.01)` 그리고 `cand ≤ 0.05` 이면 PASS, 아니면 FAIL. **단, 게이트 데이터의 포트·일(검증 포트 수 × 일수) < `gate_min_port_days`(150) 이면 SKIP** (메시지 `포트·일 n<150, 판정 불가`) | 보고 패키지의 합격 기준 C1(오탐 ≤ 0.02)과 맞춰 절대 상한 0.05(장애 포함 전체 알람). **최소 포트·일 가드 (2026-10-06 확정)**: 포트·일이 작으면 인시던트 1건의 값이 하한·상한보다 커서(예: 4포트 × 3일 = 12포트·일 → 1건 = 0.083) 학습 난수만으로 PASS/FAIL 이 갈림(6.3 E2E 발견). 150 이면 1건 = 0.0067 로 하한 0.01 보다 작고, 섀도 최소 규모 500포트(검증 50포트 × 3일)에서도 판정됨. value 에는 비율과 함께 **인시던트 수·포트·일(후보/활성)** 을 기록. 활성 모델과 겹치는 인시던트 비율은 **정보로만** 기록(판정에 쓰지 않음: 활성 모델의 알람이 오탐이었을 수 있음) |
| G4 | **카나리 구분력**: `<model_dir>/<ft>_canary.csv`(라벨된 고정 장애/정상 데이터)가 있으면 후보와 활성의 AUPRC 비교 | `cand ≥ active − 0.05` 이면 PASS, 아니면 FAIL. 파일 없으면 SKIP | 순환 위험 완화용(8장). 카나리는 `validation/` 도구가 시뮬레이터로 생성해 모델 볼륨에 둔다 (src 는 파일만 읽음 → import 규칙 준수) |
| G5 | **임계치 상승 추세**: 최근 활성화된 3개 버전 + 후보의 threshold 가 단조 증가하고 누적 2배 이상 | WARN | 점진적 둔감화(장애를 정상으로 계속 흡수) 경보 |

- 종합: 하나라도 FAIL → `FAIL`, FAIL 없이 WARN → `WARN`, 나머지 → `PASS`. 게이트 실행 자체가 실패하면 `ERROR`.
  - **G3 가 SKIP 이면 (FAIL 이 없을 때) 종합은 `WARN`** (2026-10-06 확정). 알람 부하를 판정하지 못한 후보가 auto 모드에서 자동 승격되지 않게 하기 위함(auto 는 `PASS` 에서만 승격, 1장 5). 수동 승격은 WARN 과 같이 승격 + warnings 반환(2.7). 다른 검사의 SKIP(예: 카나리 파일 없는 G4)은 종합에 영향 없음.
- **배율 표기 (2026-10-06 확정)**: G2·G3 메시지는 같은 포맷 함수 하나로 배율을 표기한다. 비율 ≥ 1 이면 `×37.5`, < 1 이면 역수로 `1/37.5` 로 쓰고 원래 값과 기준을 함께 쓴다(예: `임계치 16.48 → 0.4399 (1/37.5, 기준 1/3 ~ ×3)`). G2 기준이 3배 초과·1/3 미만으로 대칭이므로, 감소도 같은 척도로 보여야 한다(이전 표기 `0.0배` 는 정보가 없었음).
- 활성 버전이 없는 최초 학습은 기존대로 자동 활성화(게이트는 기록만).

알람 계산은 운영 코드와 같아야 한다(lessons #31). 이를 위해:
- `validation/evaluation/tuning.py` 의 `_track_frame`, `simulate_alarms` 를 **`src/pipeline/alerting.py` 로 이동** (`alarms_from_track_scores(track_scores, policy)`), `validation` 쪽은 이를 import 해서 기존 이름으로 재노출. 기존 `tests/pipeline/test_tuning_equivalence.py` 가 그대로 통과해야 함.
- 인시던트 묶음 `count_incidents(alarms, gap_steps=4)` 를 `src/pipeline/alerting.py` 에 추가. `validation/evaluation/metrics.make_incidents` 와 같은 결과임을 테스트로 고정(T-A2).
- `AnomalyDetector` 에 버전 지정 로드 추가 (F5): `reload_model(ft, version=None)` → 경로 해석을 `_resolve_paths(ft, version)` 로 분리. 게이트는 별도 인스턴스 2개(활성/후보)를 만들어 `track_scores()` 만 사용(RCA·댐프닝 상태 불필요). 운영 Consumer 의 인스턴스에는 영향 없음.

인터페이스
```python
@dataclass
class GateCheck: id: str; status: str; value: Any; limit: Any; message: str   # status: PASS|FAIL|WARN|SKIP
@dataclass
class GateResult: status: str; checks: list[GateCheck]; evaluated_at: str; data: dict  # data: 기간, 포트 수, 정책

def evaluate(cand_meta, active_meta, cand_stats, active_stats, canary, history, limits) -> GateResult   # 순수
def run_gate(model_dir, ft, version, window: TrainWindow, policy: RetrainPolicy,
             fetch_raw: Callable, alert_policy: AlertPolicy) -> GateResult                          # 추론 포함
```

### 2.7 레지스트리 변경 — `src/models/registry.py`
- 엔트리에 `gate`, `trigger`, `training_window`, `suspect_stats`, `baseline_port_*` 저장 (스키마는 자유 dict 라 마이그레이션 불필요. `list_versions` 출력에 `gate.status`, `trigger` 추가).
- `set_gate(model_dir, ft, version, gate_result)` 추가 (transaction 안에서 갱신).
- `promote(..., force)` 판정 순서:
  1. `_check_files` (기존)
  2. `gate` 가 있으면: G1 FAIL → `RegistryError`(force 불가). 그 외 FAIL/ERROR → `PromotionWarning(gate 메시지)` (force 로 진행). WARN → 승격하되 `warnings` 로 반환.
  3. `gate` 가 없으면(P0-1 이전 후보, 게이트 실패로 미기록): 기존 `compare_with_active` 경고 그대로.
- 활성화 이력 `reason` 에 `auto-gate` 추가.

### 2.8 API 변경 — `src/api/main.py`
- 중복 제거(F4): `handle_drift(result, trigger_source)` 하나로 `run_daily_maintenance` 와 `/api/drift/check` 가 공유. 학습 날짜 기본값은 `train_window.plan_window()` 로 일원화.
- `run_training_pipeline(ft, cfg, date_params=None, trigger="manual")`:
  - `date_params` 가 없으면 포트 분할 모드(2.1), 있으면 사용자가 지정한 날짜 분할(기존 UI 호환).
  - 수집(`suspect_policy` 전달 → DataCollector 가 의심 구간 제외, `skipped` 면 중단) → 학습 → 게이트 → (auto & drift & PASS) 승격 + reload 브로드캐스트.
  - 결과를 `retrain_state` 에 기록(드리프트 트리거일 때).
  - `training_status[ft]` 에 `gate_status`, `suspect_fraction` 추가.
- `/api/model/train`: `exclude_suspect: bool = True` 쿼리 추가. 날짜 미지정 시 새 기본 구간.
- `/api/drift/check` 응답: 기존 키 + 트랙별 `kind/median_ratio/drifted_port_fraction/top_ports` + `retrain: {track: {action, reason}}`. `auto_retrain_triggered` 는 `action == "train"` 인 트랙이 있으면 true (하위 호환).
- `/api/drift/status`: `last_result` + 트랙별 `retrain_state`.
- `POST /api/model/gate?ft&version`: 게이트 재실행(학습 중이면 409).
- `/api/model/promote`: 409 응답 `detail` 에 `gate.checks` 포함.

### 2.9 UI — `ui/` (dev 가 해당 화면 위치 확인)
- 모델 관리: 버전 목록에 `trigger`, `gate` 상태 배지(PASS/WARN/FAIL/ERROR/—), 클릭 시 검사별 값/한계/메시지.
- 드리프트 상태: 트랙별 kind, median_ratio, drifted_port_fraction, 결정(action/reason), 지속 일수, 쿨다운 남은 시간.

### 2.10 카나리 생성 도구 — `validation/cli/build_canary.py` (신규, 도입 확정)
- 시뮬레이터로 시드 고정 데이터 생성(학습·평가 시드와 분리, 예: 시드 900) → 트랙별 원본 + `label`(state>0) CSV 를 `models/<ft>_canary.csv` 로 저장.
- 크기 상한(예: 40포트 × 7일) — 게이트 시간 1분 이내 목표.
- 카나리가 시뮬레이터 장애 유형만 대표한다는 한계는 8장.

---

## 3. 수정/신규 파일
| 파일 | 구분 | 내용 |
|---|---|---|
| `src/data/train_window.py` | 신규 | 구간·포트 분할·의심 구간 (2.1) |
| `src/pipeline/retrain_policy.py` | 신규 | RetrainPolicy, decide, 상태 영속화 (2.5) |
| `src/models/promotion_gate.py` | 신규 | 게이트 (2.6) |
| `src/data/data_collector.py` | 수정 | 포트 분할 모드, exclusions 인자, 구간 조인 (2.2) |
| `src/data/db_connector.py` | 수정 | `fetch_alarm_rows`, 포트별 드리프트 통계 쿼리 |
| `src/models/trainer.py` | 수정 | 포트 분포 기준값, trigger/window 메타 (2.3) |
| `src/models/registry.py` | 수정 | gate 저장·승격 판정 (2.7) |
| `src/pipeline/drift_monitor.py` | 수정 | 포트별 신호, 판정 순수 함수 분리 (2.4) |
| `src/pipeline/alerting.py` | 수정 | `alarms_from_track_scores`, `count_incidents` 이동/추가 (2.6) |
| `src/pipeline/inference.py` | 수정 | 버전 지정 로드 (F5) |
| `src/api/main.py` | 수정 | handle_drift 통합, 파이프라인·엔드포인트 (2.8) |
| `src/config.py.example` | 수정 | `RETRAIN_POLICY` 주석 블록 (5장). **`src/config.py` 는 건드리지 않음** — 모든 키는 코드 기본값을 가지고 `getattr(config, "RETRAIN_POLICY", {})` 로 덮어씀 (`alerting.load_policy` 패턴) |
| `validation/evaluation/tuning.py` | 수정 | 이동한 함수 재노출 |
| `validation/cli/build_canary.py` | 신규(도입 확정) | 카나리 생성 (2.10) |
| `validation/cli/check_retrain_policy.py` | 신규 | 6장 검증 실험 |
| `ui/` 모델 관리·드리프트 화면 | 수정 | 2.9 |
| `docker-compose.yml` | 수정 | api 에 `DRIFT_RETRAIN_MODE=${DRIFT_RETRAIN_MODE:-candidate}` |
| `tests/…` | 신규/수정 | 7장 |

---

## 4. 구현 순서 (dev 작업 단위, 각 단계 끝에 전체 pytest)
1. **A — 데이터**: `train_window.py`, DataCollector 포트 분할·구간 조인, `fetch_alarm_rows`, Trainer 메타 확장. (API 동작 불변, 기존 날짜 분할 경로 유지)
2. **B — 드리프트/결정**: DriftMonitor 포트별 신호, `retrain_policy.py`, API `handle_drift` 통합, 모드 환경변수.
3. **C — 게이트**: alerting 이동(+동등성 테스트), 버전 지정 로드, `promotion_gate.py`, registry 연동, API/UI.
4. **D — 검증**: 6장 실험(`check_retrain_policy.py`) + 컨테이너 E2E 시나리오(6.3). 결과를 `docs/performance_report.md` 에 추가.

---

## 5. 설정 키 (`RETRAIN_POLICY`, 코드 기본값 = 아래 제안값)
| 키 | 기본 | 근거 |
|---|---|---|
| `mode` | `candidate` | 결정 사항 2). 환경변수 `DRIFT_RETRAIN_MODE` 가 우선 |
| `train_days` | 28 | 알람 이력 보존 30일(F3) |
| `val_port_fraction` / `split_salt` | 0.10 / `"ptn"` | |
| `gate_days` / `gate_max_ports` | 3 / 2000 | 게이트 시간 제한 |
| `exclude_from_alarms` / `alarm_min_level` | True / 1 (MINOR) | |
| `suspect_pre_steps` / `suspect_post_steps` | 16 / 4 | lessons #24 (오탐 96.6% 가 직전 16스텝 영향권) |
| `exclude_from_rules` | True | 순환 완화 (8장) |
| `rule_error_ge` / `rule_rx_drop_db` / `rule_traffic_ratio` / `rule_min_consecutive` | **10 / 2.0 / 0.3 / 6** (초안 20 / 3.0 / 0.15 / 4) | 6.1 실험 결과: 선택 규칙 통과 후보 0/81 → 차선 채택. 검증 시드 장애 제외율 평균 90.6%(3시드 중 2 PASS, 경계선). `performance_report.md` 7.1 |
| `max_excluded_fraction` / `port_drop_fraction` | 0.20 / 0.50 | |
| `drift_window_hours` / `drift_min_port_rows` | 24 / 48 | |
| `drift_median_factor` / `drift_port_fraction` | 1.5 / 0.30 | 기존 `drift_factor=1.5` 유지, 포트 비율은 정상 기대치(0.10)의 3배 |
| `localized_factor` | 3.0 | 국소 장애 표시 |
| `drift_persist_checks` | 3 (일) | 일시적 장애로 재학습하지 않도록 |
| `min_hours_since_activation` | 24 | F2 |
| `cooldown_hours` | 72 | |
| `gate_max_alarm_ratio` / `gate_alarm_floor` / `gate_max_incidents_per_port_day` | 1.5 / 0.01 / 0.05 | 보고 패키지 C1 과 정합 |
| `gate_min_port_days` | 150 | 2026-10-06 확정. 미달 시 G3 SKIP + 종합 WARN (2.6). 1건 = 0.0067 < 하한 0.01, 섀도 최소 500포트에서 판정 가능 |
| `gate_canary_auprc_drop` | 0.05 | |
| `gate_threshold_trend_versions` / `gate_threshold_trend_factor` | 3 / 2.0 | |

---

## 6. 검증 계획 (테스트 외)
### 6.1 의심 구간 품질 (시뮬레이터, `check_retrain_policy.py`)
시뮬레이터 데이터(정답 있음)에 의심 구간 산출을 적용해 측정. **선택 규칙을 데이터 보기 전에 정의**(lessons #30): 개발 시드 7·11 로 규칙 파라미터 선택, 검증 시드 23·31·47 로 확인.
- 장애 스텝(state>0) 중 제외된 비율 ≥ 90% (A 단독, B 단독, A∪B 각각 보고)
- 정상 교란(nuisance) 스텝 중 제외된 비율 ≤ 10% (정상 변동을 남겨야 함, lessons #24)
- 전체 제외 비율 (유병률 4~5% 데이터에서 20% 미만이어야 중단 조건에 안 걸림)

**결과 (2026-10-06, 상세 `docs/performance_report.md` 7.1)**: 개발 시드에서 기준 통과 후보 0/81 → 차선(10 / 2.0 / 0.3 / 6) 채택, 기준은 완화하지 않음.
검증 시드 B 단독 장애/nuisance/전체 평균 90.6 / 1.3 / 3.3% — 시드 31 이 장애 89.6% 로 FAIL(3시드 중 2 PASS). 원인은 traffic_drop 제외율 31~54%(직전 24h 중앙값이 점진 감소를 따라감, P1-4 와 같은 취약점).
C(명시 구간)는 전 시드 PASS. A 가 B 에 더하는 효과 약 +0.7%p, 제외 없이 학습한 모델의 A 는 0.17~0.24(순환 위험 실증, 8장).

### 6.2 자동 정제로 학습한 모델의 품질
같은 학습 데이터로 (a) 정답 기반 제외(현재 `train_isolated.py` 방식) vs (b) 자동 의심 구간 제외 vs (c) 제외 없음 학습 → 검증 시드에서 AUPRC, 이벤트 F1(선택 정책). **합격: (b) 가 (a) 대비 AUPRC −0.05, F1 −0.05 이내이고 (c) 보다 나음.** 미달 시 기본값 `exclude_from_rules` 등을 재검토하고 lead 에게 보고.

**결과 (2026-10-06, `validation/cli/compare_train_exclusion.py`, 상세 `docs/performance_report.md` 7.2)**: 합격 기준 4건 모두 PASS.
선택 정책 AUPRC auto 0.641/0.632 vs truth 0.621/0.603 vs none 0.149/0.159, 이벤트 F1 0.812/0.798 vs 0.783/0.754 vs 0.013/0.020.
단, 반복 2회라 ±0.03 이내 차이는 우열 근거 아님(“auto 가 truth 에 뒤지지 않음”까지만 주장). none 모델은 조기 탐지 0~0.3%로 장애 학습 위험 실증. 시뮬레이션 기준.

### 6.3 컨테이너 E2E (`docs/e2e_guide.md` 에 시나리오 추가)
1. mode=candidate, 시뮬레이션 데이터로 광역 드리프트 3일 재현 → 3일째 후보 생성, gate 기록, 활성 불변
2. 쿨다운 중 재감지 → notify 만
3. 국소 장애 주입 → localized, 재학습 없음
4. mode=auto + PASS 후보 → 자동 승격 + Consumer reload, 이력 `auto-gate`
5. 비정상 후보(2에폭) → G2/G3 FAIL → 승격 409, force 시 진행
6. 활성화 직후 드리프트 체크 → warming_up

---

## 7. 테스트 항목 (`tests/` 는 `src/` 구조를 따름)
| 파일 | ID | 검증 내용 |
|---|---|---|
| `tests/data/test_train_window.py` (신규) | T-W1 | `plan_window` 경계(T 포함, gate 구간 = 마지막 3일) |
| | T-W2 | `is_val_port` 가 프로세스와 무관하게 결정적이고 비율 ≈ 10% (1만 포트 ±1%p) |
| | T-W3 | 알람 행 → 구간: 4스텝 이하 간격 병합, 앞 16·뒤 4스텝 확장 |
| | T-W4 | 규칙 구간: 1~3스텝 산발 에러는 제외 안 됨, 4스텝 연속은 제외 |
| | T-W5 | 제외로 생긴 3스텝 이상 공백을 가로지르는 시퀀스가 생성되지 않음 + 2스텝 이하 공백은 보간되어 가로지름(한계 고정) |
| | T-W6 | 의심 비율 > 20% 판정, 포트 50% 초과 시 포트 제외 |
| `tests/data/test_data_collector.py` (추가) | T-C1 | 포트 분할 모드: train/test 포트가 겹치지 않고 기간은 동일 |
| | T-C2 | 새 `filter_excluded` 결과가 기존 구현과 동일 (무작위 구간 비교) |
| | T-C3 | 날짜 분할(기존 경로) 동작 불변, `failure_time` 컬럼 CSV 호환 |
| `tests/models/test_trainer.py` (추가) | T-T1 | 메타에 `baseline_port_median/p90`, `trigger`, `training_window` 기록, 레지스트리 엔트리에도 반영 |
| `tests/pipeline/test_drift_monitor.py` (추가) | T-D1 | score 0 행은 집계에서 제외 (F1) |
| | T-D2 | 소수 포트 급등 → localized, `drift_detected` false |
| | T-D3 | 다수 포트 이동 → widespread |
| | T-D4 | 활성화 후 24h 미만 → warming_up; 조회 시작이 activated_at 이후 |
| | T-D5 | legacy 기준값 → `baseline_legacy` true |
| `tests/pipeline/test_retrain_policy.py` (신규) | T-P1 | `decide()` 표의 각 행 (mode off/candidate/auto, 지속성, 쿨다운, 중단 후 24h 백오프, 대기 후보, 학습 중) + 2.5 결과별 상태 기록: ① `skipped: suspect_fraction` 뒤 쿨다운 미적용·지속성 유지, 24h 이내 재점검은 notify, 24h 경과 후 train ② 수집 오류도 ①과 같음 ③ Trainer 시작 후 실패(학습 예외·게이트 ERROR)는 쿨다운 적용 |
| | T-P2 | 같은 날 중복 체크는 지속성 1회 |
| | T-P3 | 상태 파일 원자적 저장, 손상 파일은 빈 상태로 로드 |
| | T-P4 | 환경변수 `DRIFT_RETRAIN_MODE` 가 config 보다 우선, 잘못된 값은 `candidate` 로 + 경고 |
| `tests/models/test_promotion_gate.py` (신규) | T-G1 | `evaluate()` 각 검사 PASS/FAIL/WARN/SKIP 경계값 |
| | T-G2 | 종합 상태 규칙 (FAIL > WARN > PASS, ERROR) |
| | T-G1b | 배율 포맷 함수: 비율 ≥ 1 은 `×N`, < 1 은 `1/N`, 원래 값과 기준을 함께 표기. G2·G3 메시지가 같은 함수를 사용 |
| | T-G3 | 임계치 추세 WARN |
| | T-G5 | G3 최소 포트·일: 149포트·일 → G3 SKIP(메시지 `포트·일 n<150`) + 다른 검사 PASS 여도 종합 WARN / 150포트·일 → 기존 규칙대로 PASS·FAIL 판정 / value 에 인시던트 수·포트·일(후보/활성) 기록 |
| | T-G4 | `run_gate` 가 작은 더미 모델 2개로 끝까지 동작 (DB 대신 `fetch_raw` 주입) |
| `tests/models/test_registry.py` (추가) | T-R1 | gate FAIL → PromotionWarning, force 로 승격 / G1 FAIL 은 force 로도 거부 |
| | T-R2 | gate WARN → 승격 + warnings 반환 |
| | T-R3 | gate 없는 후보 → 기존 compare_with_active 동작 |
| `tests/pipeline/test_alerting.py` (추가) | T-A1 | `alarms_from_track_scores` 가 `detect()` 와 일치 (기존 test_tuning_equivalence 유지) |
| | T-A2 | `count_incidents` == `validation.evaluation.metrics.make_incidents` 개수 |
| `tests/pipeline/test_inference_golden.py` | T-I1 | 버전 지정 로드 추가 후 골든 출력 불변 |
| `tests/api/test_model_endpoints.py` (추가) | T-E1 | drift check 응답 하위 호환 키 + retrain 필드 |
| | T-E2 | mode=auto & PASS → 승격·reload publish 호출 / candidate → 활성 불변 / mode=auto & G3 SKIP(종합 WARN) → 승격하지 않고 활성 불변 |
| | T-E3 | `/api/model/gate` 학습 중 409 |
| `tests/test_architecture.py` | (기존) | 신규 src 모듈이 validation 을 import 하지 않음 |

---

## 8. 순환 위험과 한계 (명시)
**순환 위험**: 학습 데이터를 "현재 모델이 알람을 낸 구간"으로 거르면, **현재 모델이 못 잡는 장애는 제외되지 않고 정상으로 학습**된다. 재학습할수록 그 장애에 더 둔감해지는 자기 강화가 생길 수 있다(결정 사항 2 의 한계).

이 설계의 완화책 (완전한 해결은 아님)
1. **모델과 무관한 규칙(B)** 을 합집합으로 사용 → 모델이 놓쳐도 에러·광 하락·트래픽 급감이 지속된 구간은 제외. 단, 규칙도 못 보는 장애(약한 장애, 점진적 트래픽 감소 — `performance_report.md` 4장의 약점)는 여전히 남는다.
2. **카나리 게이트(G4)** → 고정된 라벨 데이터에서 구분력이 떨어지면 승격 차단. 단, 카나리는 시뮬레이터 장애 유형만 대표하므로 **실데이터 장애 유형의 둔감화는 잡지 못한다**.
3. **임계치 상승 추세 경고(G5)** → 버전마다 임계치가 오르는 점진적 둔감화를 사람에게 알림.
4. **지속성 3일 + 쿨다운 + 의심 비율 20% 중단** → 대규모 장애 기간에 재학습하지 않음.
5. **기본 모드 candidate** → 사람이 게이트 결과를 보고 승격. auto 는 게이트 신뢰가 쌓인 뒤(아래) 옵션으로만.
6. **명시 제외 CSV(C)** → EMS 알람 이력(P1-2 의 약한 정답 포맷)과 Phase 13 피드백을 같은 형식으로 넣을 수 있게 해 둠. **근본 해결은 사람 피드백/외부 정답**이다.

그 밖의 한계
- G3 의 "알람 비율"은 정답 없는 대리 지표다. 후보의 알람이 적은 것이 오탐 감소인지 둔감화인지 G3 만으로는 구분 불가 → G4/G5 와 사람 검토로 보완.
- G3 의 분해능: 포트·일 150 미만은 SKIP 으로 막지만, **150 ~ 300 포트·일 구간에서는 인시던트 1~2건 차이로 PASS/FAIL 경계가 흔들릴 수 있다**(1건 = 0.0033 ~ 0.0067, 하한 0.01). 이 구간의 FAIL/PASS 는 value 의 인시던트 수를 보고 사람이 판단한다.
- 포트 홀드아웃은 망 전체 공통 패턴에 대해서는 완전한 미관측 데이터가 아니다(2.1).
- 학습 구간 28일은 알람 이력 보존(30일)에 묶여 있다. 더 긴 구간이 필요하면 `RETENTION_DAYS` 연장 또는 알람 구간 별도 보관이 필요(범위 밖).
- 드리프트 기준값은 **새로 학습한 모델부터** 포트 분포로 저장된다. 현재 v1 은 legacy 라서 이 설계 적용 후에도 드리프트 재학습이 자동으로 일어나지 않는다(의도된 동작). P1-3 승격 후부터 유효.
- 알람 정책(`ALERT_POLICY`)은 모델과 짝이다(`performance_report.md` 6장). 게이트는 **현재 운영 정책**으로 두 모델을 비교하므로, 정책 프리셋을 바꾸는 결정(P1-3)과 함께 재평가해야 한다.

**auto 모드 사용 조건 (제안)**: candidate 모드에서 드리프트 후보가 3회 이상 생성되고, 그 게이트 판정과 사람의 승격 판단이 모두 일치했을 때 검토. 그 전에는 켜지 않는다.

---

## 9. 위험
| 위험 | 영향 | 완화 |
|---|---|---|
| 게이트 추론 시간 (2,000포트 × 3일 × 2모델) | 학습 스레드 지연 | `gate_max_ports` 샘플링, CPU 에서 측정 후 조정. 실패 시 `ERROR` 로 기록하고 수동 재실행 |
| 의심 구간 규칙이 정상 변동까지 제거 | 노이즈 환경 오탐 증가 (lessons #24 재발) | 6.1 의 nuisance 제외율 ≤ 10% 기준, `rule_min_consecutive` |
| alerting 함수 이동 중 동작 변경 | 운영 알람 변화 | 골든 테스트·튜닝 동등성 테스트를 이동 전후 그대로 통과 |
| 레지스트리 스키마 확장 | Consumer 로드 실패 | 새 키는 선택 필드, `reload_model` 은 기존 키만 사용 (T-I1) |
| 상태 파일과 레지스트리 불일치 (재시작/동시성) | 쿨다운 오판 | 상태 파일은 참고용; 대기 후보 판단은 레지스트리에서 직접 파생 |

## 10. 결정 사항 — 확정(2026-10-06, 사용자)
| # | 항목 | 결정 |
|---|---|---|
| 1 | 5장 기본값 (`train_days=28`, `drift_persist_checks=3`, `cooldown_hours=72`, G3 상한 0.05 등) | **5장 값 그대로 확정** |
| 2 | 카나리(G4) 도입 여부 | **도입** — `validation/cli/build_canary.py`(2.10) 구현 대상. 순환 완화는 규칙(B) + G4 + G5 |
| 3 | 포트 홀드아웃 방식(2.1) | **채택** (대안이던 시간 분할 + 학습 구간 연장은 쓰지 않음) |
| 4 | 수동 학습(UI)에도 의심 구간 제외 기본 적용 | **적용** — `exclude_suspect=false` 로만 끔 |
| 5 | 6.1 규칙(B) 차선안 | **수용** — `rule_error_ge / rule_rx_drop_db / rule_traffic_ratio / rule_min_consecutive = 10 / 2.0 / 0.3 / 6` (5장). 6.1 선택 기준은 완화하지 않았고 경계선 결과(검증 시드 장애 제외율 평균 90.6%, 3시드 중 2 PASS)임을 유지 |

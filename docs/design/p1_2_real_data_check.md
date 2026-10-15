# P1-2 설계: 실데이터 검증 도구 (`real_data_check`)

> 작성: 2026-10-15 (architect) · 근거: plan.md Backlog **P1-2** 행(2026-10-15 재점검 지점), `docs/report_package.md` 4.2·4.3·6.2, `docs/target_definition.md` 2~5장·7장, 2026-10-15 사용자가 실DB 에서 직접 확인한 사실(lead 전달, 아래 2.1), lessons #14·#18·#22·#23·#26·#27·#30·#38·#40·#44
> 상태: **설계 개정 1 (2026-10-15) — 사용자 답 반영, Q1~Q9 처리(14장)**. 코드 없음(구현은 dev). 확정: 성능 `occur_date` = 15분 구간의 **끝**(Q7), 조인 키 = **`(ip_addr, cid, lid)`**·address 는 진단용 보조 열(Q5), 알람 정답 처리 F1~F8 **사전 고정**(Q4), 판정 기간 기본안 = **최근 3개월 봉인**(Q9), 권한 검사 fail closed(Q8). 남은 결정은 14.2(쓰기 계정 예외 플래그, 결과 DB 분리 — 섀도 계획 단계).
> 범위: **읽기 전용** 실DB 조회 → 로컬 스냅샷 → 오프라인 분석. 운영 코드(`src`)의 동작·`MODEL_CONFIG`·활성 모델·알람 정책은 바꾸지 않는다. 학습(실데이터로 재학습)은 범위 밖.
> 표기: **[확인]** = 코드(위치 명시) 또는 사용자가 실DB 에서 확인한 사실(2.1), **[추정]** = 검증하지 않은 판단, **[확인 필요]** = 실DB·현장에서 확인해야 하며 설계에서는 설정으로 열어 둔 것, **[제안]** = 이 설계가 정한 값(승인 대상).

---

## 0. 요약

| 항목 | 내용 |
|---|---|
| 목적 | 테스트망(30대) 실DB 의 성능·알람 이력을 **읽기 전용**으로 꺼내, 타겟(`target_definition.md` 2장: CRC 점진 증가·트래픽 급감/단절, 광은 관찰용)에 대해 AI·고정 임계치·롤링 규칙을 같은 기간·같은 포트로 오프라인 재생하고, 합격 기준 C1~C6(`report_package.md` 4.2)의 계산 재료와 현장 검토 샘플을 만든다 |
| 새로 확인된 것 (2.1) | MySQL 5.7.43(CTE·윈도우 함수 없음), KST, 1년+ 보존. 성능·알람이 **시간별 테이블**(`cowptn_noti_pm_*`, `cowptn_noti_pm_optic_power_*`, `cowptn_noti_alarm_*`). ETH 는 `signal_type=6`·`time_type=1`, 광은 `signal_type=44`. 알람명 = `lpsztbl_<모델>_ALM_RSN[alarm_type − 367]`. EMS 알람 임계 설정·계획 작업 기록은 DB 에 없음 |
| 기존 코드에서 발견 (2.2) | ① `DBConnector()` 생성자는 `CREATE TABLE`·`ALTER TABLE` 을 실행한다 → 이 도구는 **쓰지 않는다**(별도 읽기 전용 연결). ② 운영 `fetch_optical` 은 `signal_type` 필터가 없고, `fetch_traffic`·`fetch_optical` 모두 `time_type` 필터가 없다 → 실DB 에 다른 행이 섞여 있으면 운영 입력이 오염된다(**운영 수정은 별도 결정, Q1**) |
| 구조 | 하위 명령 8개: `plan`(테이블 목록·예상 행 수, DB 데이터 미조회) → `extract`(읽기 전용 추출 → 스냅샷) → `join-check`(포트 조인 검증) → `alarms`(알람명 해석·에피소드) → `replay`(AI·베이스라인 오프라인 재생) → `report`(C1 원값·C3·C4b·C6·탐지 지연·유형별) → `sample`/`score-review`(현장 검토, C1·C2·C4a) + `live`(C5). 별도 1회성 변환기 `convert_alarm_tables`(C 소스 → JSON) |
| DB 접근 원칙 (4장) | **권한 검사 fail closed**: `SHOW GRANTS FOR CURRENT_USER()` 에 SELECT(와 USAGE) 외 권한이 있으면 실행 거부(사용자 계정은 현재 root → 읽기 전용 계정 생성 절차 4.8). 그 위에 SELECT/SHOW 만 허용하는 문장 가드 + `SET SESSION TRANSACTION READ ONLY` + 쿼리 시간 상한. 테이블명은 `information_schema.tables` 에서 찾은 이름만 정규식 검증 후 사용. 테이블별 `id` 키셋 청크 조회, 전체 행 상한 초과 시 중단. DB 는 추출 때 한 번만 읽고 분석은 스냅샷으로 반복 |
| 포트 조인 (6장) | 세 테이블(ETH·광·알람) 모두 **`(ip_addr, cid, lid)`** 로 조인(사용자 결정). address 의 `S.n-Pm` 은 **불일치 진단용 보조 열**(조인 키 아님). 알람 `cid=0` 또는 `lid=0` 은 **장비 단위 알람**으로 분리(판정 제외, HW 동시 발생·작업 대용 표시용). 모델별 cid/lid 체계 차이·`p5_cid/p5_lid` 일치 여부는 [확인 필요]로 측정. 안전 검사 J1(1:1)·J3(단절 구간 rx=0 일치율)·사용자 수동 확인 M1 유지 |
| 알람 정답 처리 (8장, **사전 고정**) | 같은 포트·같은 알람명의 깜빡임은 **간격 60분 이내 병합**(우리 인시던트 정의 `make_incidents` 의 1시간 묶음과 동일), 병합 후 **지속 15분 미만은 판정 제외·별도 집계**(성능 데이터 해상도), 같은 포트·같은 타겟 유형의 알람은 60분 이내면 **한 타겟 이벤트**로, 판정에는 **1차 등급 알람명만**, HW 동시 발생은 **표시만**(분모 유지) |
| 성능 시각 (확정) | 성능 `occur_date` = 15분 구간의 **끝**, 값은 `(occur_date − 15분, occur_date]` 구간 값(Q7, 사용자 확인) |
| 판정 기간 (9.7) | 기본안: 추출 기준일 직전 **3개월 봉인**(판정용), 그 이전 = 개발용. 알람 이벤트 **건수만** 보고 길이를 조정하는 규칙을 사전 고정, 판정 기간은 방법 결과를 보기 전에 잠금 |
| 운영 쓰기 경로 (2.2-9) | [확인] Consumer 는 처리마다 설정 DB 의 `anomaly_detection` 에 INSERT, `DBConnector` 생성자는 테이블 CREATE/ALTER, Consumer 시작 시·API 매일 03:00 에 오래된 행 DELETE. 읽는 DB = 쓰는 DB. → 이 도구는 해당 없음(읽기 전용). **섀도에서 결과 DB 를 EMS DB 와 분리할지는 섀도 계획 단계 결정**(14.2) |
| 판정 범위 | 판정 지표는 **트래픽 트랙만**(`--tracks traffic`, AI 는 트래픽 모델만 로드해 재생). 광은 그래프·요약만(판정 제외) |
| 구현 단위 (11장, 개정 1 반영) | U1 읽기 전용 DB·**권한 검사**·`plan`·계정 안내 → U2 `extract`·컬럼 매핑 → U3 알람명 변환기·해석 → U4 **키 조인**·행 분류·`join-check` → U5 알람 에피소드·작업 대용·`--count-only` → U6 오프라인 재생·`report`·**판정 기간 잠금** → U7 샘플·검토 채점·`live`(C5) → U8(선택) 그래프 |
| 결정 상태 (14장) | 처리됨: Q1(plan 결과 후 결정)·Q2(git 제외 JSON)·Q3(표 없는 모델 미분류)·Q4(F1~F8 고정)·Q5(키 조인)·Q7(구간 끝)·Q8(fail closed)·Q9(3개월 봉인 기본안). 남음(14.2): 쓰기 계정 예외 플래그 둘지(권고: 두지 않음), 섀도 결과 DB 분리 방식(섀도 계획 단계), Q1 의 실제 판단(plan 실행 후), M1 수동 확인 실시 |

---

## 1. 배경과 범위

- **배경**: P1-2 는 원래 "실 이력 접근 불가를 가정하고 도구를 먼저 준비"하는 과제였다(plan.md). 2026-10-14 타겟 재정의로 "타겟 확정 후 시작"으로 미뤘고, 2026-10-15 사용자가 테스트망 실DB 의 테이블·컬럼·알람 체계를 직접 확인해 주어(2.1) 입력 스키마가 대부분 확정되었다. 정답(장애 티켓)은 여전히 없고, **EMS 알람 이력이 약한 정답**이다(`report_package.md` C3).
- **범위 안**: 읽기 전용 추출, 스냅샷 관리, 알람명 해석, 포트 조인 검증, 알람 에피소드 정의, AI·베이스라인 오프라인 재생, C1(원값)·C3·C4·C6·탐지 지연·유형별 집계, 현장 검토 샘플과 채점(C1·C2·C4a), 라이브 결과의 C5.
- **범위 밖**: 실데이터 재학습·모델 교체, 알람 정책 변경, 운영 `src` 의 수정(2.2 의 fetch 필터 문제는 Q1 로 분리), RCA 룰 조정, 합격 기준 값 자체의 변경(P0-3 몫). 섀도 운영 절차 문서화는 `report_package.md` 5장을 따르고 이 설계는 도구만 다룬다.
- **원칙**: 이 도구는 `validation/` 에 둔다(`src` 는 import 하지 않음, `tests/test_architecture.py`). 정답(EMS 알람)은 평가에만 쓰고 학습 경로에 넣지 않는다(lessons #22).

## 2. 입력 사실

### 2.1 실DB 사실 [확인 — 2026-10-15 사용자가 실DB 에서 직접 확인, lead 전달]

> architect 는 DB 에 접속하지 않았고 접속 정보도 받지 않았다. 아래는 lead 가 전달한 사용자 확인 내용이다.

| # | 사실 |
|---|---|
| D1 | MySQL **5.7.43** — CTE(`WITH`)·윈도우 함수 없음, 5.7 문법만. 시간대 **KST(UTC+9)**. 성능 데이터 **1년 이상** 보존 |
| D2 | 시간별 분할 테이블: 이더넷 성능 `cowptn_noti_pm_YYYY_MM_DD_HH`, 광 성능 `cowptn_noti_pm_optic_power_YYYY_MM_DD_HH`, 알람 이력 `cowptn_noti_alarm_YYYY_MM_DD_HH` → 기간 조회는 `information_schema.tables` 로 존재하는 시간 테이블을 찾아 순회 |
| D3 | 이더넷 성능 컬럼: `id, system_type, meanid, sid, cid(슬롯), lid(포트), augid, au3id, tugid, tuid, ip_addr, address`(예 `'CA216-UT7320-S.1 -P1'` = 국소명-시스템이름-슬롯-포트)`, signal_type`(**6 인 행만 ETH 트래픽 성능**)`, unit_type, time_type`(**1 = 15분**)`, bbe_in_error`(= 에러 패킷)`, es`(= tx 패킷)`, ses`(= rx 패킷)`, flag, occur_date`(15분 간격). `uas·ofs_in_discard·pospj_out_error·negpj_out_discard·rx_cos1` 은 ETH 트래픽에서 채워지지 않음. **tx·rx·error 는 15분 구간별 값(누적값 아님)** |
| D4 | 광 성능 컬럼: D3 의 공통 컬럼 + `signal_type`(**44 인 행만 광 파워**)`, rx_min/max/avg/accu_power, tx_min/max/avg/accu_power, occur_date` |
| D5 | 알람 이력 컬럼: `id, msgid, tid`(국소명 예 `'CA237'`)`, system_type, meanid, sid, cid, lid, augid, au3id, tugid, tuid, p5_meanid, p5_cid, p5_lid, p5_augid, p5_au3id, p5_tugid, p5_tuid, ip_addr, address`(예 `'S.2 -P1  '` — 국소·시스템 이름 없이 슬롯-포트만, **뒤 공백 있음**)`, signal_type`(알람에선 안 씀)`, severity_type`(2 = CRITICAL, 3 = MAJOR, 4 = MINOR)`, srv_affect_type, unit_type, alarm_type, gen_data, occur_date, release_date`(**진행 중이면 NULL**) |
| D6 | 알람명 = `code = alarm_type − 367` 로 `system_type` 에 해당하는 모델의 문자열 표 `lpsztbl_<모델>_ALM_RSN[code]` 조회(범위 밖이면 `ALM-ID-<n>` 류). **CE7080 은 일반 표 `lpsztbl_CE7080_ALM_RSN` 만 사용**(`_4STEP` 표는 무시 — 이 환경은 KO 벤더 아님). 예: system_type 31(CE7080) alarm_type 600/599/598 → code 233/232/231 → HI-PWR/HI-PWR3/HI-PWR2(광 수신 전력 높음), release 가 **10~14초 간격으로 깜빡임**(짧은 알람 반복) |
| D7 | `system_type` enum(10진): 7=CE7200, 8=CE7300, 9=CE7400, 10=CE7300S, 11=CE740C, 12=CE73K0, 13=CE730M, 14=CE7510, 15=CE7533, 16=CE7511, 17=CE7534, 18=CE7515, 20=CE7611, 21=CE7610, 22=CE8300, 23=CE8310, 24=CE740E, 25=CE5100, 26=CE5200, 27=CE5300, 28=CE74C2, 29=CE5500, 30=CE841P, 31=CE7080, 32=CE1830, 33=CE7700Q, 34=CE7700, 35=CE7700Q2, 42=CE7700Q2V, 43=CE7510Q, 48=CE8600 |
| D8 | 알람 이름 표가 **있는** 모델: 8300, 8600, 841P, 7200, 7300(**7300S 도 동일 표**), 730M, 7080, 7700Q/Q2/Q2V, 7700, 7510Q, 73K0, 7400, 740C, 74C2, 740E, 7533, 7534, 7510, 7511, 7611, 7610, 5100, 5200, 7515. **없는** 모델: CE8310, CE5300, CE5500, CE1830 등 |
| D9 | 광 임계치 초과 알람(OPT-PWR-LOW/HIGH 계열, HI-PWR/LO-PWR 계열)은 EMS 로 올라온다 → 광 관찰용 정답 후보 |
| D10 | DB 에 **EMS 알람 임계 설정**과 **계획 작업(점검) 기록**은 **없다** |
| D11 | (2026-10-15 추가, Q7) 성능 `occur_date` 는 **15분 구간의 끝**이고, 값은 그 앞 15분 구간 `(occur_date − 15분, occur_date]` 의 값이다 |
| D12 | (2026-10-15 추가, Q8) 사용자가 쓰는 DB 계정은 **root**(쓰기 가능) |
| D13 | (2026-10-15 추가, Q5) 사용자 판단: 세 테이블(성능 ETH·광·알람)은 모두 **`ip_addr, cid, lid`** 로 맞추면 된다(address 는 보지 않는다). 이전에 의심한 cid 불일치(성능 cid=1 ↔ S.1 대 알람 cid=3 ↔ S.2)는 **서로 다른 장비 모델(system_type 12 와 31)의 샘플을 비교한 것**이라 근거가 약하다(lead 정정) |

### 2.2 기존 코드 사실 [확인 — 2026-10-15, 코드로 확인]

1. **`DBConnector` 는 읽기 전용이 아니다**: 생성자가 `initialize_pool()` 다음 `_ensure_anomaly_table()` 을 호출하고(`src/data/db_connector.py:16-17`), 여기서 `CREATE TABLE IF NOT EXISTS anomaly_detection`(`:47`)과 필요 시 `ALTER TABLE`(`:115`)을 실행한다. → `real_data_check` 는 `DBConnector()` 를 생성하지 않는다. 접속 설정만 같은 방식(`src.config.DB_CONFIG`, 예시 파일 기준 환경변수 `DB_HOST`·`DB_USER`·`DB_PASS`·`DB_NAME`·`DB_PORT` — `src/config.py.example:10-16`)을 따른다.
2. **운영 조회의 필터** (`db_connector.py:127-189`):
   - `fetch_traffic`: `signal_type = SIGNAL_TYPES['ETH']`(예시 파일 값 6, `config.py.example:68-71`)만 걸고 **`time_type` 필터 없음**. 컬럼 매핑 `bbe_in_error → error_packet, es → tx_packet, ses → rx_packet`(`:145`) — D3 와 일치.
   - `fetch_optical`: **`signal_type` 필터도 `time_type` 필터도 없음**(`:177-180`). `tx_avg_power, rx_avg_power` 만 읽음.
   - 두 함수 모두 시간 테이블을 `start.floor('h') ~ end.floor('h')` 로만 순회하고(`:131`) 테이블마다 `SHOW TABLES LIKE` 를 실행(`:151`). 테이블명은 f-string(값은 바인딩).
   - → 실DB 의 광 테이블에 `signal_type ≠ 44` 행이, 또는 어느 테이블에 `time_type ≠ 1` 행이 있으면 **운영 입력이 오염된다** [추정: 실제로 섞여 있는지는 미확인]. 이 도구는 필터를 걸고(4.4), 섞인 행 수를 `plan`/`extract` 에서 보고한다. 운영 코드 정정은 Q1.
3. **트랙별 알람 플래그**: `is_traffic_anomaly`·`is_optical_anomaly` 는 댐프닝 전 값이고, 최종 `is_anomaly` 는 두 트랙을 OR 한 뒤(`src/pipeline/inference.py:374-375`) 댐프닝을 거친 값이다(`:540`). 즉 라이브 `anomaly_detection` 의 `is_anomaly` 로는 트래픽만의 최종 알람을 분리할 수 없다 → 판정은 **트래픽 모델만 로드한 오프라인 재생**으로 한다(`AnomalyDetector(tracks=('traffic',))`, 생성자 인자 `inference.py:18`, 게이트가 이미 이 방식으로 사용 `src/models/promotion_gate.py:361`).
4. **인시던트 정의**: `make_incidents`(`validation/evaluation/metrics.py:47`)는 포트별 알람 간격 4스텝(60분) 이하를 한 인시던트로 묶는다. 입력에 `nuisance` 컬럼이 필요하다(`:58`, 실데이터에는 없으므로 `False` 로 채움).
5. **베이스라인** (`validation/evaluation/baselines.py`): `fixed_threshold` 는 에러 ≥ 100 **또는 광 수신 < −15 dBm**(`:20-24`), `rolling_rule` 은 에러 ≥ 20 **또는 광 3dB 하락** 또는 트래픽 합계 < 24h 중앙값의 15%(`:27-48`). 둘 다 광 조건이 섞여 있어, 광이 판정 제외인 이번 타겟에서는 광 조건을 끄는 옵션이 필요하다(요구사항 R6).
6. **시뮬레이션 DB 스키마는 실DB 와 다르다**: `validation/simulator/db_manager.py` 의 테이블은 광 테이블에 `signal_type` 이 없고, `time_type`·`address` 도 없다(`:38-60`, INSERT `:92`·`:110`). → 이 도구를 `cowptn_test` 에 돌리려면 스키마 프로필 `sim`(필터 컬럼이 없으면 생략) 이 필요하다(4.4).
7. **Wilson 구간 구현은 저장소에 없다** (grep 결과 0건) → 새로 만든다(U7).
8. `validation/runs/` 는 git 제외(`.gitignore:45`). 실DB 산출물(IP·국소명 포함)은 여기에만 둔다.
9. **운영 파이프라인의 DB 쓰기 경로** (Q6, 2026-10-15 코드로 확인):
   - **INSERT**: Kafka Consumer 는 메시지 처리마다 `self.db.save_results(results)` 로 설정 DB 의 `anomaly_detection` 에 upsert 한다(`src/pipeline/kafka_consumer.py:143`, `INSERT … ON DUPLICATE KEY UPDATE` `db_connector.py:253-257`).
   - **CREATE/ALTER**: `DBConnector` 생성자가 `_ensure_anomaly_table()` 을 호출해 테이블이 없으면 `CREATE TABLE`, RCA 컬럼이 없으면 `ALTER TABLE … ADD COLUMN` 을 실행한다(`db_connector.py:16-17`, `:41-125`). 생성자는 Consumer(`kafka_consumer.py:36`), Producer(`kafka_producer.py:29`), API(`src/api/main.py:36`), 드리프트 모니터(`drift_monitor.py:82`), 학습 데이터 수집기(`data_collector.py:11`)에서 호출된다 — **Producer 처럼 읽기만 하는 서비스도 기동 시 DDL 을 실행할 수 있다.**
   - **DELETE**: Consumer 시작 시 1회 `delete_old_data(RETENTION_DAYS)`(`kafka_consumer.py:158`), API 의 매일 03:00 유지보수 작업에서도 같은 함수(`src/api/main.py:147`, 스케줄 `:164`). 대상은 `anomaly_detection` 의 오래된 행뿐(`db_connector.py:308`).
   - **읽는 DB 와 쓰는 DB 가 같다**: 성능 조회(`fetch_traffic/optical`)와 결과 저장이 같은 `DB_CONFIG` 연결 풀을 쓴다(`db_connector.py:14`, 테이블명은 스키마 한정 없이 사용).
   - **이력**: 이 구조는 Docker·Kafka 도입(2026-09-21, `5fbb61d`) 이전부터 있었다 — `git log -S` 기준 `_ensure_anomaly_table` 첫 등장 `66d60f9`(2026-05-06), `save_results`·`delete_old_data` 첫 등장 `b4b7a23`(2026-05-06). (lead 전달 "2026-05-08, Phase 6" 의 `75f563d` 는 그 뒤 수정 커밋이다.)
   - → **섀도·운영에서 설정 DB 를 EMS 실DB 로 두면** EMS DB 에 `anomaly_detection` 테이블이 생기고, 컬럼 추가와 행 삭제(그 테이블 한정)가 일어난다. P1-2 도구는 `DBConnector` 를 쓰지 않는 읽기 전용이라 해당 없다. 분리 여부는 14.2-B(섀도 계획 단계 결정).

### 2.3 [확인 필요] — 실DB·현장에서 확인할 것 (설계에서는 설정으로 열어 둠)

| # | 항목 | 설계의 처리 | 확인 방법 |
|---|---|---|---|
| K1 | **같은 장비 모델 안에서** 성능과 알람의 `(cid, lid)` 가 같은 물리 포트를 가리키는지 (사용자 판단은 "같다", D13 — 이전 의심 사례는 다른 모델끼리 비교라 근거 약함) | 조인 키 = `(ip_addr, cid, lid)` (6장). address `S.n-Pm` 은 진단 열 | 6.3 J1~J3 + 사용자 수동 확인 M1 |
| K1a | **장비 모델(system_type)마다 cid/lid 번호 체계가 다른지** (예: 슬롯 번호가 0/1 시작, 성능 cid 와 address 슬롯 번호의 관계) | 모델별로 따로 측정·보고, 조인 규칙은 모든 모델 동일(모델별 보정 없음) | `plan`·`join-check` 의 모델별 `(cid, lid) ↔ S.n-Pm` 교차표 |
| K1b | 알람의 **`p5_cid`·`p5_lid` 가 `cid`·`lid` 와 항상 같은지** (다르면 무엇을 뜻하는지 모름 — 보호 경로 등 [추정]) | 조인에 쓰지 않음. 행별 일치 여부를 스냅샷에 열로 보관 | `join-check` 의 모델별 일치율·불일치 표본 20행 |
| K2 | ~~성능 `occur_date` 가 구간 시작인지 끝인지~~ → **해소(D11: 끝)** | 15분 구간 `(occur_date − 15분, occur_date]` 로 고정(9.5) | — |
| K3 | 시간 테이블과 `occur_date` 의 경계(00:00 행이 `_HH` 어느 테이블에 들어가는지) | 요청 구간 앞뒤로 1시간 테이블을 더 읽고 `occur_date` 로 거른다(4.3) | `plan` 의 경계 점검 출력 |
| K4 | 알람 `release_date` 가 **발생 행을 갱신**하는지(같은 `id`), 해제가 **새 행**인지 | 갱신으로 가정. 같은 키·같은 `occur_date` 의 중복 행이 있으면 경고·집계 | `extract` 의 중복 보고 |
| K5 | 시작이 구간 이전인 진행 중 알람이 어느 테이블에 있는지 | 알람은 요청 시작보다 `--alarm-lookback-hours`(기본 72 [제안]) 앞 테이블까지 읽고 `release_date ≥ start OR release_date IS NULL` 로 거름 | 추출 후 lookback 경계에 걸린 알람 수 보고 |
| K6 | 광 테이블의 `time_type` 사용 여부 | ETH 와 같이 `time_type = 1` 필터(컬럼이 있으면) | `plan` 의 `time_type` 분포 |
| K7 | 같은 `(ip_addr, cid, lid, occur_date)` 에 하위 ID(`augid` 등)가 다른 여러 ETH 행이 있는지 | 중복 수 보고. 분석은 운영과 같게 마지막 행 유지(`data_processor.py:105` `keep='last'`) | `extract` 중복 보고 |
| K8 | 광 파워 단위(dBm 인지, 0.1 dBm 정수인지), 무신호 값(−40 등) | 값 범위 요약을 보고하고 변환하지 않음 | `extract` 값 범위 |
| K9 | 알람과 성능의 `ip_addr` 표기가 같은지(공백·0 패딩) | 앞뒤 공백만 제거 후 문자열 비교 | J4 의 장비별 매칭률(알람 ip_addr 중 성능에 있는 비율) |
| K10 | 범위 밖 알람명 표기(`ALM-ID-<n>` 의 n 이 code 인지 alarm_type 인지) | 원값 `alarm_type`·`code` 를 모두 저장, 표시명 형식은 설정. **판정에는 영향 없음**(범위 밖은 항상 "미분류", 7.3) | 사용자: EMS 화면 표기 |
| K11 | 7700Q/Q2/Q2V 가 한 표를 공유하는지(D8 표기), 7300S 외 별칭 | 모델→표 별칭을 JSON 에 명시(7.2), 변환기가 enum 과 표 이름을 대조해 별칭 없는 모델 목록을 출력 | C 소스 |
| K12 | 활성 AE 모델이 무슨 데이터로 학습되었는지(시뮬레이터 vs 실데이터 스케일) | 재생 결과에 모델 버전·학습 메타를 기록. 스케일 불일치로 알람이 폭증·전무하면 그대로 보고(튜닝하지 않음) | 레지스트리 메타 |

---

## 3. 요구사항 재점검 — 9건 (2026-10-15 [확인] 사실 반영)

> 출처: plan.md P1-2 행의 재점검 지점(2026-10-15)과 `target_definition.md` 5·7장, `report_package.md` 6.2.

| # | 요구사항 | 상태 | 해소된 것 | 남은 것 |
|---|---|---|---|---|
| R1 | **트랙 필터** | **해소(설계)** | ETH = `signal_type 6`·`time_type 1`, 광 = `signal_type 44` [확인 D3·D4]. `--tracks traffic`(판정 기본), 광은 관찰 출력만(9.1) | 운영 fetch 의 같은 필터 부재(2.2-2) → Q1 |
| R2 | **EMS 알람명 → 타겟 유형 매핑 입력** | **부분 해소** | 알람 테이블·컬럼, 이름 조회 규칙(`alarm_type − 367`, 모델별 표, CE7080 `_4STEP` 무시) [확인 D5·D6]. 이름 키 매핑 CSV(7.4)와 C 소스 변환기(7.1) 설계 | C 소스 원문 수령·변환(사용자 보유), 표 없는 모델의 처리(7.3 규칙 확정 Q3), 별칭(K11), **실제 발생 알람명·건수는 이 도구 `alarms --summary` 로 처음 얻음**(질문지 Q1-1 의 답을 도구가 냄), 알람 발생 조건·임계(질문지 Q1-3)는 **DB 에 없음 [확인 D10]** → 여전히 현장 답변 필요 |
| R3 | **리드타임 1h/3h/6h 분할 출력** | **해소(설계)** | C3 선행 창 안에서 누적 비율 `lead ≥ 1h / ≥ 3h / ≥ 6h` 와 구간별 분포(9.4). 데이터 의존 없음 | — |
| R4 | **단절 탐지 지연** | **해소(설계)** — 정의 의존 부분만 남음 | 정의(9.5): 트래픽단절 이벤트 시각 `t0` → 첫 알람까지 분(벽시계)과 스텝(첫 관측 가능 구간 기준), 중앙값과 ≤15/30/60분 비율. `metrics.py:96` 의 `delay_min` 과 같은 의미. 성능 시각 기준 확정(D11: 구간 끝) | ETH-NO-RX-TRAFFIC 의 발생 조건(질문지 Q1-3) — 해석에만 영향 |
| R5 | **계획 작업 이력 입력** | **부분 해소** | DB 에 없음 [확인 D10] → (a) 선택 CSV 입력, (b) **작업성 알람 대용**(LINK-FAIL-MANUAL, BOOTING, RESET, SW-UPDATED, OPT-REMOVE 등, `target_definition.md` 5.3) — 8.6 | 대용 알람이 실제로 작업을 나타내는지 [추정], 테스트망에서 그 알람명이 실제로 나는지(R2 와 같은 경로로 확인), C6 의 망 이벤트일은 원칙상 운영 측 기록이라 대용 결과는 **참고 병기만**(9.6) |
| R6 | **베이스라인 광 조건 제거 옵션** | **해소(설계)** | `fixed_threshold`·`rolling_rule` 에 `use_optical`(기본 True = 현행) 추가, 판정 실행은 False(9.3) | 고정 임계치에는 트래픽 급감 조건이 원래 없음 — 추가하지 않는다(베이스라인을 결과 보고 바꾸지 않음, 한정으로 기록) |
| R7 | **순간 단절 픽스처** | **해소(설계)** | 시뮬레이터에 순간 단절이 없음 [확인 `target_definition.md` 3.2] → **테스트 픽스처로만** 만든다(12장 F-1). 시뮬레이터 `scenario_generator` 는 바꾸지 않는다(기존 시드 출력 보존) | 성능 수치의 근거로는 쓰지 않음(도구 동작 검증용) |
| R8 | **유형별 집계** | **해소(설계)** | 모든 이벤트·인시던트 출력에 타겟 유형 열(CRC증가/트래픽단절/광관찰/제외/해당없음/미분류)과 등급(1차/보조/사용 안 함)을 붙이고 유형별로 따로 집계(9.4~9.6) | — |
| R9 | **입력 스키마** | **대부분 해소** | 테이블 이름 규칙·컬럼·코드값·시간대·성능 시각 기준 [확인 D1~D7·D11], 조인 키 `(ip_addr, cid, lid)` [사용자 결정 D13] | K1·K1a·K1b(조인 안전 검사)·K3·K4·K6·K7·K8·K9 — 모두 `plan`/`extract`/`join-check` 출력으로 확인 가능하게 설계 |

`report_package.md` 6.2 의 기존 출력 항목(C1 인시던트·C6·C4a 오프라인 재생·층화 샘플·검토 채점·C3/C4b·C5·대안 정책 재생)은 9장에서 모두 다룬다. 단 **대안 정책 오프라인 재생**은 U6 에 인자만 두고(정책 dict 입력), 정책을 고르는 절차는 이 설계의 범위가 아니다.

---

## 4. 데이터 접근 설계 (읽기 전용)

### 4.1 연결과 읽기 전용 보장

- 새 모듈 `validation/realdata/db_readonly.py`. `DBConnector` 를 생성하지 않는다(2.2-1). 접속 정보는 `src.config.DB_CONFIG` 를 import 해서 쓴다(운영과 같은 설정 방식 — 이 설계는 설정 파일 내용을 다루지 않는다). 단 **계정·비밀번호만** 도구 전용 환경변수 `RDC_DB_USER`·`RDC_DB_PASS` 가 있으면 그 값으로 바꾼다 → 운영 설정을 건드리지 않고 읽기 전용 계정(4.8)으로 실행할 수 있다.
- **권한 검사 (fail closed, 연결 직후·세션 설정 전, 모든 하위 명령 공통)**: `SHOW GRANTS FOR CURRENT_USER()` 의 각 줄을 파싱한다.
  - 허용: `GRANT USAGE ON *.* …`(권한 없음 표시), `GRANT SELECT ON <db>.* …`·`GRANT SELECT ON <db>.<table> …`(SELECT **단독**).
  - 거부: 그 밖의 모든 권한 단어(`ALL PRIVILEGES`, `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `DROP`, `ALTER`, `INDEX`, `SUPER`, `FILE`, `PROCESS`, `RELOAD`, `PROXY`, `EXECUTE`, `CREATE TEMPORARY TABLES`, `LOCK TABLES`, `EVENT`, `TRIGGER` 등)과 `WITH GRANT OPTION` 이 **한 줄이라도** 있으면 실행을 거부하고 4.8 의 계정 생성 안내를 출력한다. 화이트리스트 방식이라 모르는 권한 단어도 거부된다.
  - `SHOW GRANTS` 자체가 실패하거나 파싱할 수 없는 줄이 있으면 거부(판단 불가 = 거부).
  - root(현재 사용자 계정, D12)는 `ALL PRIVILEGES … WITH GRANT OPTION` 이므로 **거부된다**. 이것이 의도된 동작이다.
  - 결과(허용된 grant 줄 목록, 계정명 — 비밀번호 제외)를 매니페스트에 기록.
- 연결 직후 세션 설정: `SET SESSION TRANSACTION READ ONLY`, `SET SESSION MAX_EXECUTION_TIME = <ms>`(5.7.8+ 의 SELECT 시간 상한, 기본 60,000 [제안]), `SET SESSION time_zone` 은 **바꾸지 않는다**(DB 의 naive datetime 을 그대로 KST 로 해석, 4.6).
- **문장 가드**: 모든 쿼리는 `ro_query(sql, params)` 한 곳을 지난다. 주석 제거 후 첫 토큰이 `SELECT`·`SHOW` 가 아니거나 `;` 로 여러 문장이 이어지면 실행 전에 예외. 세션 설정 3개만 연결 함수 내부에서 예외로 허용.
- 3중 방어의 역할: **권한 검사**(계정 자체가 쓸 수 없음을 확인) → 세션 읽기 전용(서버가 쓰기 문장을 거부) → 문장 가드(도구 코드가 쓰기 문장을 만들지 않음). 앞의 둘이 서버 쪽 보장이고 문장 가드는 실수 방지용이다.
- 접속·조회 실패는 빈 결과로 넘기지 않고 예외로 중단(lessons #14).

### 4.2 테이블 탐색

- 종류별로 `information_schema.tables` 를 한 번씩 조회: `SELECT table_name, table_rows FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name LIKE %s`(접두어 바인딩).
- 가져온 이름을 Python 정규식으로 분류·검증(**`LIKE 'cowptn_noti_pm_%'` 는 광 테이블도 잡으므로 정규식이 최종 판정**):
  - ETH: `^cowptn_noti_pm_(\d{4})_(\d{2})_(\d{2})_(\d{2})$`
  - 광: `^cowptn_noti_pm_optic_power_(\d{4})_(\d{2})_(\d{2})_(\d{2})$`
  - 알람: `^cowptn_noti_alarm_(\d{4})_(\d{2})_(\d{2})_(\d{2})$`
- 정규식을 통과한 이름만 백틱으로 감싸 SQL 에 넣는다(값은 항상 `%s` 바인딩). 사용자 입력이 테이블명이 되는 경로는 없다.
- 요청 구간의 시간 목록과 대조해 **없는 시간 테이블 목록**(데이터 공백)을 출력한다(C5c 와 별개인 원본 공백).

### 4.3 기간·경계

- 구간은 반열림 `[start, end)`(KST naive). 성능 테이블은 `floor(start) − 1h ~ floor(end − 1s) + 1h` 의 시간 테이블을 읽고 `occur_date >= start AND occur_date < end` 로 거른다(K3 대비).
- 알람은 `floor(start) − lookback ~ floor(end − 1s) + 1h` 를 읽고 `occur_date < end AND (release_date IS NULL OR release_date >= start)` 로 거른다(K5).
- `plan` 은 경계 시간 테이블 2개에서 `MIN/MAX(occur_date)` 를 읽어 테이블 시간과 `occur_date` 의 관계를 출력한다(K3 확인용).

### 4.4 필터와 스키마 프로필

| 종류 | 필터 (기본, `--schema real`) | `--schema sim` (`cowptn_test`) |
|---|---|---|
| ETH | `signal_type = 6 AND time_type = 1` | `signal_type = 6` (time_type 컬럼 없음) |
| 광 | `signal_type = 44 AND time_type = 1` (K6) | 필터 없음 (컬럼 없음, 2.2-6) |
| 알람 | 없음 (`signal_type` 은 알람에서 안 씀, D5) | 알람 테이블 없음 → 알람 기능 SKIP |

- 프로필은 테이블별로 `information_schema.columns` 를 조회해 **필터 컬럼이 있는지** 확인한다. `real` 에서 필터 컬럼이 없으면 그 테이블을 오류로 중단(조용히 필터를 빼지 않음). `sim` 은 없을 때만 생략하고 생략 사실을 매니페스트에 남긴다.
- `plan` 은 표본 테이블(구간 첫·마지막 시간)에서 `signal_type`·`time_type` 별 행 수를 `GROUP BY` 로 보고한다 → 2.2-2 의 오염 여부를 수치로 확인.
- 포트 필터: `--ports <CSV>`(열 `ip_addr`, 선택 `cid`, `lid`). `ip_addr` 은 SQL `IN (%s, …)` 바인딩(1회 1,000개 이하로 분할), `cid/lid` 는 Python 에서 거른다(세 테이블 같은 규칙). 알람의 장비 단위 행(`cid=0` 또는 `lid=0`, 6.1)은 포트 필터와 무관하게 **필터한 장비(ip_addr)의 것은 모두** 남긴다(HW 동시 발생·작업 대용 표시에 필요).

### 4.5 청크·행 수 상한

- 테이블마다 키셋 페이지: `SELECT … FROM \`t\` WHERE <필터> AND id > %s ORDER BY id LIMIT %s`(기본 청크 50,000 [제안]). `OFFSET` 은 쓰지 않는다. `id` 가 없거나 PK 가 아니면(확인: `information_schema.columns` / `statistics`) 그 테이블은 `LIMIT` 없이 1회 조회하되 테이블별 행 상한을 적용.
- **전체 행 상한** `--max-rows`(기본 10,000,000 [제안: 1,500포트 × 96 × 약 70일 규모]): `plan` 단계에서 `table_rows`(InnoDB 추정치)의 합이 상한을 넘으면 `extract` 를 시작하지 않고 구간·포트를 줄이라고 안내. 추출 중 실제 누적이 상한을 넘어도 즉시 중단하고 부분 스냅샷을 "불완전"으로 표시.
- 추출은 일 단위 파일로 흘려 쓴다(메모리에 전체를 올리지 않음). `stop_checker` 패턴(lessons #7)으로 중단 가능.

### 4.6 시간대

- DB 의 `occur_date`·`release_date` 는 **KST naive** 로 가정하고 변환하지 않는다 [확인 D1]. 스냅샷·출력의 모든 시각 열은 KST naive 이며 매니페스트에 `"tz": "Asia/Seoul (naive)"` 를 기록.
- `plan` 은 최신 테이블의 `MAX(occur_date)` 와 실행 호스트의 현재 시각 차이를 출력하고, 차이가 8~10시간이면 "시간대 의심" 경고(lessons #27 — 컨테이너 UTC 문제 재발 방지). 판정은 하지 않음.

### 4.7 스냅샷 (DB 는 한 번만 읽는다)

- 위치: `validation/runs/p1_2/<run_id>/raw/` (git 제외, 2.2-8). 형식: 종류·일자별 `csv.gz`(새 의존성 없음 — parquet 은 의존성 확인 후 선택 사항).
- `manifest.json`: 실행 시각, 도구 버전(git 커밋), DB 이름(호스트·계정은 기록하지 않음), 구간, 포트 필터, 스키마 프로필, 읽은 테이블 목록과 테이블별 행 수, 없는 시간 테이블, 필터 밖 행 수(2.2-2), 중복 키 수(K7), 값 범위(K8), lookback 경계 알람 수(K5), 완료/불완전 여부.
- 이후 하위 명령은 모두 `--run <run_id>` 의 스냅샷만 읽는다. 분석을 바꿔 다시 돌려도 DB 를 다시 읽지 않는다.

### 4.8 운영 절차 — 읽기 전용 계정 만들기 (도구 실행 전 1회, 사용자/DBA)

> 도구는 root 로는 실행되지 않는다(4.1). 아래 절차를 **도구 문서**(dev 가 `validation/README.md` 의 `real_data_check` 절과 `real_data_check.py --help` 의 거부 메시지에 같은 내용)로도 제공한다.

1. 관리자 계정(root)으로 EMS DB 서버에 접속해 계정을 만든다. **호스트는 도구를 실행하는 PC 의 IP(또는 좁은 대역)로 제한**하고 `'%'` 는 쓰지 않는다.
   ```sql
   -- MySQL 5.7. <EMS_DB> = 성능·알람 테이블이 있는 스키마 이름, <분석PC_IP> = 도구 실행 PC
   CREATE USER 'ptn_ro'@'<분석PC_IP>' IDENTIFIED BY '<충분히 긴 비밀번호>';
   GRANT SELECT ON `<EMS_DB>`.* TO 'ptn_ro'@'<분석PC_IP>';
   SHOW GRANTS FOR 'ptn_ro'@'<분석PC_IP>';
   -- 기대 출력 2줄: GRANT USAGE ON *.* TO ...  /  GRANT SELECT ON `<EMS_DB>`.* TO ...
   ```
   - 같은 대역의 여러 PC 가 필요하면 `'ptn_ro'@'10.1.2.%'` 처럼 좁은 대역만.
   - `information_schema` 는 따로 권한을 줄 필요가 없다(SELECT 권한이 있는 스키마의 테이블만 보인다 — MySQL 5.7 동작, 실DB 에서 `plan` 으로 확인).
   - 결과 테이블(`anomaly_detection`)을 라이브 C5 용으로 읽어야 하고 그것이 다른 스키마에 있으면 그 스키마에도 `GRANT SELECT` 만 추가한다.
2. 도구 실행 PC 에서 계정을 환경변수로 준다(비밀번호를 파일·저장소·명령 이력에 남기지 않는다): `RDC_DB_USER=ptn_ro`, `RDC_DB_PASS=…` (4.1). 호스트·포트·DB 이름은 기존 설정 방식 그대로.
3. `real_data_check.py plan …` 를 실행해 권한 검사 통과(매니페스트의 grant 줄)를 확인한다.
4. 계정을 지울 때: `DROP USER 'ptn_ro'@'<분석PC_IP>';`
5. 쓰기 계정 예외 플래그를 둘지는 14.2-A (권고: 두지 않음).

---

## 5. 컬럼 매핑과 결측 처리

### 5.1 모델 입력 ↔ DB 컬럼

| 모델 입력 (`FEATURE_GROUPS`, `config.py.example:38-41`) | DB 테이블·컬럼 | 필터 | 비고 |
|---|---|---|---|
| `tx_packet` | `cowptn_noti_pm_*`.`es` | `signal_type=6`, `time_type=1` | [확인 D3] 15분 구간 값. 운영 매핑 `db_connector.py:145` 와 같음 |
| `rx_packet` | `cowptn_noti_pm_*`.`ses` | 같음 | 같음 |
| `error_packet` | `cowptn_noti_pm_*`.`bbe_in_error` | 같음 | 같음 |
| `tx_avg_power` | `cowptn_noti_pm_optic_power_*`.`tx_avg_power` | `signal_type=44`(, `time_type=1`) | 단위 K8. min/max/accu 는 스냅샷에만 보관(관찰용), 모델 입력 아님 |
| `rx_avg_power` | `cowptn_noti_pm_optic_power_*`.`rx_avg_power` | 같음 | 같음 |
| 포트 키 `ip_addr, cid, lid` | 공통 컬럼 | — | 운영과 같은 키. 알람 조인은 6장 |
| (조인·표시용) `address`, `system_type` | 공통 컬럼 | — | 스냅샷에 보관. 모델 입력 아님 |

- ETH 에서 채워지지 않는 컬럼(`uas, ofs_in_discard, pospj_out_error, negpj_out_discard, rx_cos1`, D3)은 읽지 않는다.

### 5.2 결측·포트 없음

| 상황 | 처리 | 근거 |
|---|---|---|
| 행은 있는데 값이 NULL | NaN 유지(0 으로 채우지 않음). 이후 운영과 같은 `DataProcessor` 경로(보간 1스텝, NaN 윈도우 제외) | lessons #18, `data_processor.py:134-139` |
| 값이 0 | **유효값**(단절 신호) | lessons #18 |
| 시간 테이블 자체가 없음 | 그 시간의 모든 포트가 결측. 공백 목록에 기록 | 4.2 |
| 테이블은 있는데 특정 포트 행이 없음 | 그 포트·시각 결측(NaN). 단절과 수집 실패를 구분할 수 없으므로 J3 에서 "행 없음"을 따로 센다 | [추정] 링크 다운 시 행이 안 생길 수 있음 |
| 포트가 ETH 에만 있고 광에 없음 | 정상(광은 2~3대분뿐, `target_definition.md` 2장). 판정에 영향 없음 | — |
| 포트가 광에만 있음 | 관찰 출력에만 나타남 | — |
| 알람 포트가 성능 스냅샷에 없음 | 이벤트를 "포트 없음"으로 분류, C3·탐지 지연 분모에서 제외하고 건수 보고 | 9.4 |
| 15분 격자에 안 맞는 `occur_date` | 운영과 같이 `round('15min')`(`data_processor.py:102`), 반올림 이동이 있었던 행 수 보고 | — |
| 비정상 큰 값(카운터 이상) | 변환·제거하지 않고 1e9 초과 행 수 보고(운영 학습 필터 `data_processor.py:89-90` 참고 기준) | 결과를 보고 정제 규칙을 고르지 않음 |

---

## 6. 포트 조인 (성능 ↔ 알람)

### 6.1 조인 키와 행 분류 (2026-10-15 개정 — 사용자 결정 D13)

- **조인 키 = `(ip_addr, cid, lid)`** — 성능 ETH·광·알람 세 테이블 모두 같다. `ip_addr` 은 앞뒤 공백만 제거(K9), `cid`·`lid` 는 정수. 운영의 트래픽·광 병합 키(`kafka_producer.py:60`)와 같다.
- **알람 행 분류**:
  | 조건 | 분류 | 용도 |
  |---|---|---|
  | `cid > 0` 이고 `lid > 0` | **포트 알람** | 타겟 이벤트(8장) 후보 |
  | `cid = 0` 또는 `lid = 0` | **장비 단위 알람**(하위 구분 열: `cid=0` → 장비, `cid>0, lid=0` → 슬롯) | **판정 제외**. HW 동시 발생 표시(F8)·작업 대용 범위(8.6)에만 사용 |
  | `cid`·`lid` 가 NULL·음수 | **키 이상** | 판정 제외, 건수·표본 보고 |
  - 0 이 "포트 없음"을 뜻한다는 것은 lead 전달(사용자 판단)이며, 모델별로 그런지는 J2 에서 address 진단 열과 대조해 확인한다 [확인 필요].
- **address 진단 열(조인 키 아님)**: `addr_port = norm_port(address)` — 정규식 `S\.\s*(\d+)\s*-\s*P\s*(\d+)` 의 **마지막** 일치를 `S.<n>-P<m>` 으로(성능 `'CA216-UT7320-S.1 -P1'` → `S.1-P1`, 알람 `'S.2 -P1  '` → `S.2-P1`), 일치가 없으면 `S.<n>` 또는 빈 값. 세 테이블 모두 스냅샷에 보관하고 **불일치 진단(J2)에만** 쓴다.
- 알람 `p5_cid`·`p5_lid` 는 조인에 쓰지 않고 `p5_same = (p5_cid = cid AND p5_lid = lid)` 열만 남긴다(K1b).

### 6.2 이전 안(address 조인)에서 바꾼 이유

- 초안은 "성능 cid=1 ↔ S.1, 알람 cid=3 ↔ S.2" 관찰을 근거로 address 조인을 택했으나, 그 두 샘플은 **서로 다른 장비 모델(system_type 12 = CE73K0 와 31 = CE7080)** 의 것이라 같은 포트의 불일치 근거가 되지 못한다(D13). 사용자는 세 테이블을 `(ip_addr, cid, lid)` 로 맞추면 된다고 판단했다. 이 설계는 그 판단을 1차 키로 쓰되, 틀렸을 경우를 잡는 **안전 검사(6.3)는 유지**한다.

### 6.3 안전 검사 J1~J4 와 판정 규칙 (키만 교체, 기준값 유지)

| 단계 | 내용 | 출력 |
|---|---|---|
| **J1** 키 일관성 | 성능(ETH) 안에서 장비별로 `(cid, lid) ↔ addr_port` 가 1:1 인지. ETH 와 광 키의 겹침(광이 있는 장비에서 광 키 중 ETH 키에 있는 비율) | 1:1 포트 비율, 1:1 아닌 포트 목록, ETH·광 키 겹침률 |
| **J2** 진단 대조 (판정 아님) | 키로 조인된 알람·성능 쌍에서 `addr_port` 가 같은지 **system_type 별** 교차표, 장비 단위 알람(`cid=0`/`lid=0`)의 address 형태, `p5_same` 비율(K1a·K1b) | 모델별 불일치율·패턴(예: 슬롯 번호가 일정하게 어긋남), 불일치 표본 20행 |
| **J3** 물리 일치 검사 | 단절 1차 포트 알람(ETH-LINK-FAIL, ETH-NO-RX-TRAFFIC, CLI-LINK-FAIL, OTU-ETH-LINKDOWN, OPT-LOS 등, 8장 병합 후 지속 ≥ 30분) 에피소드마다, 에피소드 구간에 **완전히 포함된 15분 구간**(`(T − 15분, T]` ⊂ 구간, D11)에서 `rx_packet == 0 또는 행 없음` 비율. (a) **키 조인**, (c) 대조군 = 같은 장비의 다른 임의 포트(고정 난수 시드 0), 진단용 (d) `addr_port` 조인(참고) | 조인별 일치율(에피소드 평균), "0" 과 "행 없음" 내역 |
| **J4** 커버리지 | 포트 알람 중 같은 키의 성능 포트가 있는 비율(모델별) | 매칭률, 못 찾은 알람 표본 20건 |

**판정 (사전 고정 — 기준값은 초안 그대로, 대상 조인만 키 조인으로)**:
- J3 대상 에피소드 **5건 미만** → **판정 불가**: 키 조인을 쓰되 C3·탐지 지연 출력 전부에 `join_status = unverified` 를 표시하고 C3 판정에 쓰지 않는다(참고치).
- (a) 키 조인이 **J1 1:1 비율 ≥ 95%**, **J3 일치율 ≥ 0.8**, **(a) − (c) ≥ 0.5** 를 모두 만족 → `verified`.
- 미충족 → `failed`: C3·탐지 지연은 산출하되 판정에 쓰지 않고 **멈춰서 보고**한다. 이때 (d) address 조인이 기준을 만족해도 **자동으로 바꾸지 않는다**(사용자 결정 사항, D13 과 반대 결과이므로).
- J2 의 불일치는 판정에 쓰지 않는다(address 는 조인 키가 아님). 단 특정 모델에서 불일치가 체계적이면(예: 그 모델 쌍의 대부분) 그 모델만 따로 J3 일치율을 보고한다.
- 근거와 한정: 0.8 은 "단절 알람 구간의 대부분에서 수신이 0 이거나 수집이 끊긴다"는 [추정] 에 기댄 값이며, 단절 알람의 발생 조건(질문지 Q1-3)을 모르므로 근거가 약하다. 대조군 차이 0.5 는 우연 일치 배제용 여유. 결과를 본 뒤 값을 바꾸지 않는다(바꾸려면 다른 기간 데이터로 다시 검증하고 변경 이력에 남김, lessons #37).

**M1 수동 확인 (사용자, 유지)**: J 실행 전에 사용자가 EMS 화면에서 **3개 포트**(서로 다른 장비 모델 2종 이상)를 골라, 같은 물리 포트가 성능 화면과 알람 화면에서 각각 어떤 `(cid, lid)`(와 address)로 보이는지 적어 준다. `join_manual.csv`(`ip_addr, system_type, perf_cid, perf_lid, alarm_cid, alarm_lid, note`)로 입력해 키가 같은지 표시한다. **한 건이라도 키가 다르면 J 판정과 무관하게 멈춘다.**

---

## 7. 알람명 해석

### 7.1 1회성 변환기 `validation/cli/convert_alarm_tables.py`

- 입력: 사용자가 가진 C 소스 텍스트 파일(1개 이상). 출력: 알람명 JSON 1개.
- 파싱 대상:
  1. `enum_sys_id` enum → `{값: 모델명}`. 값이 생략된 항목은 C 규칙대로 직전 값 + 1. 주석(`//`, `/* */`) 제거.
  2. 배열 `lpsztbl_<MODEL>_ALM_RSN[]` → 문자열 목록(인덱스 = 배열 위치). 이름이 정확히 `^lpsztbl_([A-Za-z0-9]+)_ALM_RSN$` 인 배열만 채택하고, 그 밖의 `lpsztbl_*`(예: `lpsztbl_CE7080_ALM_RSN_4STEP`)은 **무시 목록**으로 출력(D6).
  3. 원소가 문자열 리터럴이 아니면(`NULL`, 매크로, 지정 초기화자 `[n] = …`) **변환 실패로 중단**하고 위치를 출력(조용히 건너뛰면 인덱스가 밀림). 인접 문자열 연결(`"A" "B"`)은 C 규칙대로 합침. 빈 문자열 `""` 은 허용하고 "빈 이름"으로 표시.
- 대조 검사: 파싱한 enum 이 2.1 D7 의 목록과 다르면 경고(값·이름 차이 표). D8 의 "표 있는 모델" 중 배열이 없는 모델, 배열은 있는데 enum 에 없는 이름을 출력.
- 출력 JSON 구조:
  ```json
  {"source_sha256": ["..."], "generated_at": "...", "code_offset": 367,
   "sys_id": {"31": "CE7080", "10": "CE7300S", "...": "..."},
   "tables": {"CE7080": ["...", "..."], "CE7300": ["..."]},
   "table_alias": {"CE7300S": "CE7300"},
   "ignored_arrays": ["lpsztbl_CE7080_ALM_RSN_4STEP"],
   "models_without_table": ["CE8310", "CE5300", "CE5500", "CE1830"]}
  ```
  `table_alias` 는 변환기 인자 `--alias CE7300S=CE7300`(D8 확인분)로만 넣는다(이름 유사성으로 추측하지 않음). 7700Q/Q2/Q2V 의 공유 여부는 K11 — C 소스에 배열이 각각 있으면 각각 쓰고, 없으면 사용자 확인 후 별칭 인자로.
- **저장 위치 (Q2, 2026-10-15 확정)**: **git 제외** — `validation/runs/ref/alarm_names.json`(이미 `.gitignore:45` 의 `validation/runs/` 아래라 설정 변경 불필요). 이유: 원문이 장비 소스 코드 일부이고 저장소 공유 범위에 넣어도 되는지 확인되지 않음, C 소스만 있으면 재생성 가능(`source_sha256` 로 동일성 확인). 저장소에는 변환기와 **합성 C 소스 픽스처**(테스트용 가짜 모델 2개)만 둔다. 저장소에 넣어도 된다고 사용자가 정하면 `validation/realdata/ref/alarm_names.json` 으로 옮기고 sha 를 테스트로 고정.

### 7.2 이름 조회 규칙

```
model  = sys_id.get(str(system_type))            # 없으면 → UNKNOWN-SYS-<system_type>
table  = tables.get(table_alias.get(model, model))
code   = alarm_type − 367
name   = table[code]   if table and 0 <= code < len(table) and table[code] != ""
```

| 경우 | `alarm_name` | `name_status` | 타겟 유형 |
|---|---|---|---|
| 정상 조회 | 표의 문자열(앞뒤 공백 제거) | `ok` | 7.4 매핑 |
| code 가 범위 밖·음수 | 표시명 형식 설정값(기본 `ALM-ID-{code}`, K10) | `out_of_range` | **미분류** |
| 표의 값이 빈 문자열 | `ALM-EMPTY-{code}` | `empty` | **미분류** |
| system_type 이 enum 에 없음 | `UNKNOWN-SYS-{system_type}-{alarm_type}` | `unknown_model` | **미분류** |
| 모델은 있으나 **표 없음**(CE8310·CE5300·CE5500·CE1830 등) | `NOTABLE-{model}-{alarm_type}` | `no_table` | **미분류** |

### 7.3 표 없는 모델의 처리 규칙 [확정 2026-10-15 — Q3]

- 표 없는 모델의 알람은 **이름을 추측하지 않는다**(다른 모델 표를 빌려 쓰지 않음 — 같은 인덱스가 모델마다 다른 이름이라는 사실 `target_definition.md` 5.1 규칙 1).
- C3·탐지 지연의 분자·분모에서 **제외**하고, 모델별 `alarm_type` 별 건수를 `unmapped_alarms.csv` 로 따로 낸다. 그 장비의 포트에서 나온 **AI 인시던트는 그대로 집계**(C1·C2 는 알람명과 무관).
- 표 없는 모델이 테스트망 30대 중 몇 대인지 `alarms --summary` 첫 줄에 출력 → 비중이 크면(예: 장비의 1/3 이상) C3 판정 보류 사유로 보고(판정 규칙 변경은 아님).
- 보완 경로: 사용자가 이름을 알게 되면 수동 매핑 파일 `alarm_name_override.csv`(`model, alarm_type, alarm_name`)로 넣을 수 있다. 수동 매핑은 `name_status = override` 로 구분 표시.

### 7.4 알람명 → 타겟 유형 매핑 입력

- 파일 `validation/realdata/alarm_target_map.csv`(저장소 포함 — 이름은 이미 `target_definition.md` 5.2 에 공개된 것만): 열 `pattern, target_type, grade, source`. `target_type ∈ {CRC증가, 트래픽단절, 광관찰, 제외, 해당없음}`, `grade ∈ {1차, 보조, 사용안함, 관찰참고}`, `source` = 5.2 의 표 ID(A~D).
- 같은 이름이 두 유형이면 행을 두 개(예: OPT-LOS → 트래픽단절/1차, 광관찰/관찰참고).
- 패턴: 정확한 이름 또는 `*` 글롭(예 `FAN-*`). 5.2 의 범위 표기(`OPT-PWR-LOW-1~4`)는 CSV 에서 개별 이름으로 펼친다. **정확한 이름 > 글롭**, 한 이름에 글롭이 둘 이상 걸리면 **로드 시 오류**(모호성은 데이터를 보기 전에 해소).
- 매핑에 없는 이름은 **미분류**. `alarms --summary` 가 미분류 이름·건수 상위 목록을 내어 매핑 갱신(architect, `target_definition.md` 5장)의 입력으로 쓴다. 매핑 갱신은 결과를 본 뒤 하는 일이므로, **갱신 전 실행 결과를 판정 기간 결과로 쓰지 않는다**(9.7).

---

## 8. 알람 정답 처리 [사전 고정 — 2026-10-15 확정(Q4: F1~F7 권고값 그대로, F8 은 표시만)]

> 대상은 6.1 의 **포트 알람**(`cid > 0`, `lid > 0`)이다. 장비 단위 알람은 F8·8.6 에만 쓴다.
> 아래 규칙은 데이터를 보기 전에 고정한다. 바꾸려면 (1) 이유, (2) 변경 전 결과를 남기고, (3) 판정에는 변경 후 **새 기간** 데이터만 쓴다(lessons #30·#37).

| ID | 규칙 | 값 | 근거 | 한정 |
|---|---|---|---|---|
| **F1** 깜빡임 병합 | 같은 `(ip_addr, cid, lid, alarm_name)` 의 발생을 시간순으로 놓고, 다음 `occur_date` ≤ 이전 병합 구간 끝(`release_date`) + **60분** 이면 한 **알람 에피소드**로 병합 | 60분 | 우리 알람 인시던트 정의(`make_incidents` gap 4스텝 = 60분, 2.2-4)와 같은 규칙으로 EMS 쪽도 센다 → C3 의 "이벤트 수"와 C1 의 "인시던트 수"가 같은 묶음 단위. D6 의 10~14초 깜빡임은 한 에피소드가 됨 | 1시간 안에 실제로 두 번 난 장애는 1건으로 셈(과소 계수). 감도 참고로 15분·4시간 병합 결과의 건수만 병기(판정에 안 씀) |
| **F2** 최소 지속 | 병합 후 에피소드 길이(첫 `occur_date` ~ 마지막 `release_date`) **< 15분** 이면 판정 이벤트에서 제외, `short_episodes` 로 따로 집계 | 15분 | 성능 데이터 해상도가 15분이라, 15분보다 짧은 사건은 성능 데이터에 안 보일 수 있음(어느 방법도 원리적으로 잡기 어려움) | EMS 에는 보이나 15분 데이터에 안 보이는 실제 순간 장애를 버림 → C3 는 "15분 데이터에 보일 수 있는 이벤트"에 대한 비율로 해석 |
| **F3** 진행 중 | `release_date IS NULL` → 구간 끝을 스냅샷 `end` 로 자르고 `ongoing = True`. F2 판단은 자른 길이로 | — | D5 | 길이가 과소 추정될 수 있음 |
| **F4** 타겟 이벤트 | 같은 포트·같은 **타겟 유형**의 1차 등급 에피소드가 서로 겹치거나 60분 이내면 한 **타겟 이벤트**. `t0` = 가장 이른 `occur_date`, 구성 알람명 목록 보존 | 60분 | 한 번의 단절에 ETH-LINK-FAIL·OPT-LOS·ETH-NO-RX-TRAFFIC 이 함께 나면 3건이 아니라 1건이어야 함 | 서로 다른 원인이 우연히 겹치면 1건으로 셈 |
| **F5** 판정 대상 | C3·C4b·탐지 지연의 분모 = 타겟 유형이 **CRC증가·트래픽단절**이고 등급 **1차**인 타겟 이벤트. 보조 등급은 같은 표를 따로 산출(참고) | — | `target_definition.md` 5.3 | 1차 후보는 이름 기반 [추정] |
| **F6** 작업 의심 제외 | 트래픽단절 이벤트 `t0` 의 ±60분 안에 같은 범위(8.6)의 작업 대용 알람 또는 작업 CSV 기록이 있으면 판정에서 제외, `work_suspect` 로 따로 집계 | ±60분 | 계획 작업에 의한 정상 단절은 장애가 아님(`report_package.md` 4.3 정상 변동) | 대용 알람의 의미는 [추정]. 작업이 아닌데 재부팅된 실제 장애도 빠짐 |
| **F7** 심각도 | 심각도로 거르지 않음. `severity_type` 2/3/4 = CRITICAL/MAJOR/MINOR, 그 밖의 값은 `기타-<n>` 으로 기록하고 모든 출력에 심각도별 열 | — | ETH-ERR 등의 심각도 의미를 모름(D10) | — |
| **F8** 제외 유형과 동시 발생 | 트래픽단절 이벤트와 ±15분 안에 같은 장비의 HW·전원 제외 유형 알람(UNIT-FAIL 등, `target_definition.md` 5.2 D — 포트 알람이든 장비 단위 알람이든)이 있으면 `hw_coincident = True` 로 **표시만** 하고 **분모에서 빼지 않는다**(2026-10-15 확정). 표시 건수와 표시 이벤트를 뺀 경우의 C3 를 참고로 병기 | ±15분 | `target_definition.md` 5.2 (D) 는 "분모에서 뺄 후보"였으나, 결과를 보기 전에 빼는 근거가 없어 표시만 하기로 확정 | HW 고장으로 인한 단절이 분모에 남아 C3 가 낮게 나올 수 있음 — 참고 병기로 크기를 보임 |

### 8.6 작업 대용 알람과 작업 CSV

- 대용 알람 목록(설정 파일, 기본값 = `target_definition.md` 5.3·5.2 D 의 작업성 이름): `LINK-FAIL-MANUAL`, `BOOTING`, `RESET`, `SW-UPDATED`, `OS-MISMATCH`, `FPGA-MISMATCH`, `CU-MISMATCH`, `OPT-REMOVE`, `OPT-MIS`, `UNIT-REM`, `UNIT-MIS`.
- 범위 규칙(6.1 의 키로 결정, 알람명과 무관): 포트 알람(`cid>0, lid>0`) → 그 포트, 슬롯 단위(`cid>0, lid=0`) → 같은 `(ip_addr, cid)` 의 모든 포트, 장비 단위(`cid=0`) → 그 장비의 모든 포트 [추정: 재부팅 등은 범위 전체에 영향, `lid=0` 이 슬롯을 뜻하는지는 J2 로 확인].
- 작업 CSV(선택, `--maintenance <CSV>`): 열 `start, end, ip_addr, cid(선택), lid(선택), note`. 비어 있으면 장비(또는 슬롯) 전체. 작업 CSV 가 있으면 대용 알람보다 우선하지 않고 **합집합**.
- 출력: 작업 의심 창 목록, 대용 알람별 건수(테스트망에서 실제로 나는지 확인 — R5 의 남은 질문).

---

## 9. 분석 출력

### 9.1 트랙 필터

- `--tracks traffic`(기본): AI 재생은 `AnomalyDetector(tracks=('traffic',))`, 베이스라인은 `use_optical=False`. C1~C6 판정 표는 이 결과로만.
- `--observe-optical`: 광 스냅샷이 있는 포트(2~3대)에 대해 광 모델 재생 결과·광 원값·광 알람(D9: OPT-PWR-*, HI-PWR*/LO-PWR*, OPT-LOS) 에피소드를 같은 시간축 CSV 로 낸다(그래프는 U8). **판정 표에는 넣지 않는다.**

### 9.2 오프라인 재생 (AI)

- 스냅샷 → 운영과 같은 `AnomalyDetector.detect(df_traffic=…, latest_only=False)`(평가 도구와 같은 호출, `validation/cli/evaluate_model.py:58-61`). 모델 버전·정책은 활성(기본) 또는 `--versions`·`--policy <json>` 로 지정하고 매니페스트에 기록. 대안 정책 재생은 `--policy` 를 바꿔 여러 번 실행.
- 재생 시작 후 포트마다 첫 `required_rows` 구간은 워밍업이라 결과가 없다(C5 분모 정의와 같음, `report_package.md` C5). 재생 구간은 판정 구간보다 **2일 앞**에서 시작하고 앞 2일 결과는 버린다 [제안: 동적 임계치·롤링 규칙 24h 이력 확보].
- 결과 행: `occur_date, ip_addr, cid, lid, alarm(is_anomaly), severity, alarm_level, traffic_score`. 인시던트는 `make_incidents(gap_steps=4)`(`nuisance=False`).

### 9.3 베이스라인 재생

- `fixed_threshold`, `rolling_rule` 을 같은 스냅샷·같은 구간·같은 포트로. 광 조건 제거 옵션: 두 함수에 `use_optical: bool = True` 인자를 추가(기본 = 현행 동작, 기존 테스트 불변), 판정 실행은 `False`. `False` 이면 광 열이 없어도 동작해야 한다.
- 베이스라인 임계값(에러 100 / 20, 트래픽 15%)은 **바꾸지 않는다**(시뮬레이터에 맞춰졌을 가능성은 `target_definition.md` 4장의 한정으로 둠). 바꾼 버전이 필요하면 별도 이름으로 추가하고 판정용 베이스라인과 섞지 않는다.

### 9.4 약한 정답 지표 (C3·C4b, 1h/3h/6h)

- 이벤트: 8장의 판정 대상 타겟 이벤트(F5, `join_status` 포함). 방법(AI·고정·롤링)마다:
  - **C3 선행 탐지**: `t0` 이전 `[t0 − 24h, t0 − 15분]` 에 같은 포트 알람이 있었던 이벤트 비율(`report_package.md` C3 그대로). 이벤트 < 20 이면 `판정 보류`.
  - **1h/3h/6h 분할 (R3)**: 선행 탐지된 이벤트의 리드 `t0 − 첫 선행 알람` 으로 `lead ≥ 1h`, `≥ 3h`, `≥ 6h` 비율(분모 = 전체 판정 이벤트, 누적) + 구간별 건수 `[15분,1h) [1h,3h) [3h,6h) [6h,24h]`.
  - C4b: AI C3 > 고정 C3 여부(C3 판정 시에만).
- 유형별(R8): CRC증가·트래픽단절 각각과 합계. 보조 등급 표는 별도(참고). 표 없는 모델·포트 없음·짧은 에피소드·작업 의심·조인 미검증 건수를 같은 표 아래에 고정 순서로 출력.
- **순환 주의 표시**: ETH-ERR 의 발생 조건을 모르는 동안(질문지 Q1-3) CRC증가의 C3 표에는 "EMS 임계와 규칙 임계가 같은 카운터일 수 있음(`target_definition.md` 5.3)" 문구를 고정 출력.

### 9.5 단절 탐지 지연 (R4)

- 시각 기준(D11 확정): 성능 행 `occur_date = T` 는 `(T − 15분, T]` 구간 값이고, 그 행으로 낸 알람의 시각도 `T` 다. 사건 시각 `t0` 를 처음 담을 수 있는 구간의 끝은 `bin_end(t0) = floor15(t0) + 15분`(`t0` 가 정각 경계이면 다음 구간).
- 트래픽단절 타겟 이벤트마다 같은 포트에서:
  - **탐지**: `bin_end(t0) ≤ occur_date ≤ t0 + 6h` 의 첫 알람. **벽시계 지연** = `occur_date − t0`(분, 최소값은 `bin_end(t0) − t0` 으로 0 초과 15분 이하 — 15분 데이터의 구조적 하한), **스텝 지연** = `(occur_date − bin_end(t0)) / 15분`(0 = 사건이 처음 보이는 구간에서 바로 탐지).
  - **직전 알람**: `t0 − 15분 ≤ occur_date < bin_end(t0)` 의 알람은 사건을 담지 않은 데이터로 낸 알람이다. 탐지로 치되(스텝 지연 0) `pre15 = True` 로 따로 센다(C3 선행 창 `≤ t0 − 15분` 과 겹치지 않음).
  - 방법별 벽시계 지연 중앙값, `≤ 15 / 30 / 60분` 비율, 스텝 지연 분포(0/1/2/3+), 6h 안에 없음(미탐) 비율.
- C3 선행 창(9.4) `[t0 − 24h, t0 − 15분]` 도 알람 시각 = 구간 끝 기준으로 그대로 쓴다(그 알람의 데이터는 모두 `t0 − 15분` 이전).
- 단절은 "예측"이 아니라 "빠른 탐지" 대상이라(`target_definition.md` 3.2) C3 와 함께 보되 별개 표로 낸다.

### 9.6 정답 없이 계산하는 지표 (C1 원값·C6·겹침)

- C1 원값: 방법별 알람 인시던트/포트·일(분모 = 스냅샷에서 재생 결과가 있는 포트·일). C1 의 "추정 오탐 부하"는 C2 가 나온 뒤 `score-review` 에서(9.8).
- C6: 일별 인시던트 수, 망 이벤트일 = 작업 CSV 의 날(원칙). 작업 대용 알람이 있는 날은 **참고 열로만** 병기(대용은 운영 측 기록이 아니므로 판정에는 CSV 만, `report_package.md` C6). 보류 조건(망 이벤트 없는 날 < 10, 일별 중앙값 < 10)을 그대로 적용 — 30대 규모에서는 보류될 가능성이 있다(plan.md P0-3 재점검 지점).
- 겹침: AI 와 규칙의 인시던트 겹침/차집합(같은 포트, 시간 겹침), "규칙이 못 잡은 AI 인시던트 수".
- 유형별: AI 인시던트를 같은 포트 ±60분의 EMS 에피소드 타겟 유형으로 태깅한 분포(태깅은 참고, 판정 아님).

### 9.7 실행 구분 — 개발 실행과 판정 실행

- 실데이터도 개발(규칙·매핑을 맞추는 데 쓴 기간)과 판정(그 뒤 새 기간)을 나눈다(lessons #30).
- **기간 분할 기본안 (Q9, 2026-10-15)**: 기준일 `D` = 첫 `extract` 실행일(KST 자정 기준, 매니페스트에 기록 후 고정). **판정 기간 = `[D − 3개월, D)` — 봉인**, **개발 기간 = 그 이전 보존분 전체**(1년 보존이면 약 9개월).
- **봉인의 뜻**: 판정 기간에 대해서는 잠금 전까지 **AI·베이스라인 재생과 그 결과(C1~C6, 인시던트, 샘플)를 만들지 않는다.** 허용되는 것은 원본 추출과 아래 3단계의 **알람 이벤트 건수 집계**뿐이다.
- **길이 조정 규칙 (방법 결과를 보기 전에 판정 기간을 정함)**:
  1. 개발 기간에서 `alarms --summary` → 알람명·매핑 CSV·작업 대용 목록을 확정한다(개발 기간 데이터만 보고).
  2. 확정된 매핑·F1~F8 로 **판정 후보 구간의 월별 판정 이벤트 수**(F5: CRC증가·트래픽단절 1차, F2·F6 제외 후)만 센다(`alarms --count-only`, 재생 없음, 알람 테이블만 사용).
  3. 판정 기간 = `D` 에서 거슬러 **판정 이벤트 합계가 20건 이상**(C3 판정 조건)이 되는 가장 짧은 **월 단위** 구간, 단 최소 3개월·최대 6개월. 6개월로도 20건 미만이면 3개월로 두고 "C3 판정 보류 예상"을 미리 기록한다. 판정 기간이 3개월보다 길어지면 그만큼 개발 기간이 줄어든다.
  4. 정해진 판정 기간·`D`·설정 해시를 잠금 파일에 쓰고(아래), 그 뒤에야 개발 기간 재생을 시작한다. 잠금 뒤 판정 기간은 **바꾸지 않는다**.
  - 근거: 길이를 이벤트 **수**로만 정하면 방법 성능을 보지 않고 C3 판정 가능성을 확보할 수 있다. 한정: 2단계에서 판정 구간의 알람 분포(어느 달에 단절이 많았는지)를 보게 되므로 완전한 봉인은 아니다. 최근 3개월이 계절·망 구성 변화로 개발 기간과 다를 수 있다(섀도 데이터와 같은 최근성을 위한 선택).
- 섀도 운영 판정 기간(`report_package.md` 5.3)은 이 과거 이력 판정과 **별개**이며, 섀도 기간 데이터는 시작 전에 봉인된 새 기간이다.
- 매니페스트에 `purpose: dev|judge` 를 필수로 받고, `judge` 실행은 **설정 파일(매핑 CSV·작업 대용 목록·F1~F8 값·조인 판정 결과)의 해시를 함께 기록**한다. 같은 판정 기간에 설정 해시가 다른 `judge` 실행이 이미 있으면 거부(측정 전 거부, lessons #38 — 잠금 파일 방식은 P1-3·P1-6 과 같게 `validation/runs/p1_2/locks/`).
- 섀도 판정 기간의 정의·연장 규칙은 `report_package.md` 5.3 을 따른다(이 도구는 기간을 입력받을 뿐).

### 9.8 현장 검토 샘플과 채점 (C1·C2·C4a)

- `sample`: `report_package.md` 4.3 그대로 — 방법별 주간 할당량(개발 30/10/10, 판정 50/15/10), 심각도 층화 무작위(시드 인자 필수, 기록), 포트당 최대 3건, 세 방법을 섞은 블라인드 ID, 같은 포트·시간 겹침 인시던트는 1행으로 라벨 공유(행이 대표하는 방법 목록은 별도 비공개 키 파일에만). 검토 열: `분류(유효-장애/유효-예방/정상 변동/오탐/판단 불가), 근거 메모`. 참고 정보 열: 같은 포트 ±24h EMS 알람명 목록, 작업 의심 여부(8.6) — **분류를 미리 채우지 않는다**.
- `score-review`: 검토 CSV + 비공개 키 → 방법별 유효율 점추정·Wilson 95%(z = 1.96), 판단 불가 비율(30% 초과 경고), 판정 가능 샘플 수(≥ 100 여부), C1 = 인시던트/포트·일 × (1 − C2 점추정), C4a = 인시던트 수 × 유효율 구간으로 충족/동률/미충족.

### 9.9 C5 (라이브 결과, `live`)

- 읽기 전용으로 `anomaly_detection` 을 조회(C5 는 라이브 가동 구간만, `report_package.md` C5). C5a = 저장 행 / 처리 대상 행(스냅샷에 `DataProcessor` 를 적용해 윈도우가 생기는 행), C5b = `detect_time − occur_date` 분포·30분 이내 비율, C5c 근거 = 결과가 1시간 이상 빈 구간.
- 라이브와 오프라인 재생의 알람 일치율도 출력(배치-스트리밍 동등성의 실데이터 재확인, lessons #26). 단 2.2-2 의 필터 차이가 있으면 입력이 달라 일치하지 않을 수 있다 — Q1 결정 전에는 참고치.
- 이 하위 명령도 읽기 전용(권한 검사 4.1 동일)이다. `anomaly_detection` 의 위치는 섀도 결과 DB 결정(14.2-B)에 따라 달라지므로 `--result-db` 접속 대상을 따로 받는다(기본 = 같은 설정).

---

## 10. 사전 고정 기준 요약 (2026-10-15 확정 — lead 지시로 사용자 답 반영)

| 항목 | 값 | 위치 |
|---|---|---|
| 권한 검사 | SELECT·USAGE 외 권한 또는 `WITH GRANT OPTION` 이 있으면 거부, `SHOW GRANTS` 실패도 거부 | 4.1 |
| 조인 키 | `(ip_addr, cid, lid)` 세 테이블 공통, address 는 진단 열. 알람 `cid=0`/`lid=0` = 장비 단위(판정 제외) | 6.1 |
| 조인 안전 검사 | J3 대상 ≥ 5건, J1 1:1 ≥ 95%, J3 일치 ≥ 0.8, 대조군 차이 ≥ 0.5, 미충족이면 멈춤(address 로 자동 전환 없음), M1 키 불일치 1건이면 멈춤 | 6.3 |
| 깜빡임 병합 간격 | 60분 | F1 |
| 최소 지속 | 15분 | F2 |
| 타겟 이벤트 묶음 | 같은 포트·같은 유형 60분 | F4 |
| 판정 이벤트 | CRC증가·트래픽단절의 1차 등급 | F5 |
| 작업 의심 창 | ±60분 | F6 |
| HW 동시 발생 | ±15분, 표시만(분모 유지) | F8 |
| 선행 창 / 분할 | `[t0 − 24h, t0 − 15분]`, 누적 ≥1h/≥3h/≥6h | 9.4 |
| 탐지 지연 창 | 탐지 `[bin_end(t0), t0 + 6h]`, 직전 알람 `[t0 − 15분, bin_end(t0))` 는 `pre15` | 9.5 |
| 재생 워밍업 | 판정 구간 2일 앞부터 재생 | 9.2 |
| 성능 시각 기준 | `occur_date` = 구간 끝, 값 = `(T − 15분, T]` (D11 확정) | 9.5 |
| 판정 기간 | `[D − 3개월, D)` 봉인 기본, 판정 이벤트 ≥ 20 이 되는 최단 월 단위(3~6개월)로 조정, 잠금 후 불변 | 9.7 |
| 알람 lookback | 72시간 | K5 |
| 청크 / 행 상한 / 쿼리 시간 | 50,000 / 10,000,000 / 60초 | 4.1·4.5 |

변경 규칙: 판정 실행(`judge`) 이후에는 위 값을 그 기간에 대해 바꾸지 않는다. 바꾸면 새 기간으로만 판정하고 설계서 변경 이력에 남긴다.

---

## 11. 구현 단위 (dev, 순서대로; 각 단위 끝에 전체 pytest)

| 단위 | 내용 | 산출 파일 (제안) | 완료 조건 |
|---|---|---|---|
| **U1** 읽기 전용 DB·권한 검사·테이블 탐색·`plan` | 4.1 연결(`RDC_DB_USER/PASS` 덮어쓰기)·**권한 검사 fail closed**·문장 가드·세션 설정, 4.2 정규식 분류, 4.3 시간 범위 계산, 4.4 스키마 프로필(컬럼 존재 확인), `plan` 출력(테이블 목록·예상 행·없는 시간·`signal_type/time_type` 분포·경계·시간대 경고), 4.8 계정 생성 안내(거부 메시지 + `validation/README.md` 절) | `validation/realdata/db_readonly.py`, `tables.py`, `validation/cli/real_data_check.py`(하위 명령 골격), `validation/README.md`(절 추가) | 가짜 커서로 권한 검사(SELECT 단독 허용 / root `ALL PRIVILEGES … WITH GRANT OPTION`·`INSERT` 섞임·`PROXY` 줄·`SHOW GRANTS` 실패·모르는 권한 단어 → 거부)·가드·분류·범위·프로필 테스트 PASS. `cowptn_test` 확인은 SELECT 전용 테스트 계정으로 `--schema sim plan` 실행 출력 첨부 |
| **U2** `extract`·컬럼 매핑 | 4.5 키셋 청크·행 상한·중단, 4.6 시간대, 4.7 스냅샷·매니페스트, 5장 매핑·결측 보고(중복·값 범위·반올림·1e9 초과) | `validation/realdata/extract.py` | 가짜 커서 청크·상한·불완전 표시 테스트, `cowptn_test` 1일 추출이 기존 `fetch_traffic` 결과와 행 단위 일치(필터 차이 없는 sim 기준) |
| **U3** 알람명 변환기·해석 | 7.1 변환기(합성 C 픽스처), 7.2 조회 규칙, 7.3 표 없는 모델, 7.4 매핑 CSV 로더(정확 > 글롭, 모호성 오류) | `validation/cli/convert_alarm_tables.py`, `validation/realdata/alarm_names.py`, `validation/realdata/alarm_target_map.csv` | D6 예시(31/600 → code 233)를 합성 표로 재현, 범위 밖·빈 이름·표 없음·미지 모델·`_4STEP` 무시·`NULL` 원소 중단 테스트 |
| **U4** 포트 조인·`join-check` | 6.1 키 조인·알람 행 분류(포트/장비 단위/키 이상)·`addr_port` 진단 열·`p5_same`, 6.3 J1~J4(모델별 J2 교차표), M1 입력, 판정 규칙 | `validation/realdata/port_join.py` | 행 분류 테스트(`cid=0`·`lid=0`·NULL), `norm_port` 정규화 표 테스트(뒤 공백·중간 공백), 픽스처 F-5 로 J3 `verified`/`failed`·address 기준 통과 시에도 자동 전환 없음, 대상 < 5 판정 불가, M1 불일치 1건 → 멈춤 |
| **U5** 알람 에피소드·작업 대용·건수 | 8장 F1~F8(F8 표시만), 8.6 작업 대용(키 기반 범위)·CSV, `alarms --summary`(알람명·건수·모델·표 없는 모델 비중 — 질문지 Q1-1 의 답), `alarms --count-only`(9.7 월별 판정 이벤트 수, 재생 없음) | `validation/realdata/alarm_episodes.py` | 픽스처 F-2(깜빡임)·F-3(진행 중)·F-4(동시 다중 알람 → 1 이벤트)·작업 창(장비·슬롯·포트 범위)·F8 표시가 분모를 바꾸지 않음 테스트 |
| **U6** 오프라인 재생·`report`·기간 잠금 | 9.1~9.7: 트래픽 전용 재생, 베이스라인 `use_optical` 인자(기본 현행), C1 원값·C3·1h/3h/6h·C4b·탐지 지연(구간 끝 기준 `bin_end`·`pre15`·스텝 지연)·C6·겹침·유형별, 판정 기간 결정 규칙(3~6개월)·잠금·봉인(잠금 전 판정 기간 재생 거부) | `validation/realdata/replay.py`, `report.py`, `periods.py`, `validation/evaluation/baselines.py`(인자 추가) | 픽스처 F-1(순간 단절) 벽시계·스텝 지연 값 정확(정각 경계 `t0` 포함), 리드 분할 경계값(정확히 1h·3h·6h), 기간 규칙(월별 건수 표 → 기대 구간, 6개월 미달 → 3개월+보류 예상), 잠금 전 판정 기간 `replay` 거부, 기존 baselines 테스트·골든 불변 |
| **U7** 샘플·채점·`live` | 9.8 샘플·블라인드 키·`score-review`(Wilson), 9.9 C5(읽기 전용, `--result-db`) | `validation/realdata/review.py`, `live.py` | Wilson 알려진 값(n=100, 60 → 하한 약 0.502) 테스트, 할당·포트당 3건·라벨 공유 테스트, `live` 도 권한 검사 경로를 지남 |
| **U8** (선택) 그래프 | 광 관찰·검토 샘플별 48h 그래프(원값 + 알람 + EMS 에피소드) | `validation/realdata/plots.py` | 파일 생성 테스트만 |

- 운영 `src` 는 수정하지 않는다. **예외 후보 U0**(Q1: `plan` 결과에서 섞인 행이 확인되고 lead·사용자가 승인할 때만): 운영 `fetch_traffic`·`fetch_optical` 에 `time_type`·광 `signal_type` 필터 추가 — 운영 동작 변경이라 별도 설계·골든 확인 필요(시뮬레이션 DB 는 컬럼이 없어 프로필 처리 필요, 2.2-6).
- 첫 실DB 실행 순서(사용자가 실행, 결과는 lead 경유): **읽기 전용 계정 생성(4.8)** → `plan`(1주 구간, Q1 판단 재료) → `extract`(1일·장비 3대) → `join-check` + M1 → 전체 보존분 `extract` → 개발 기간 `alarms --summary`(매핑 확정) → 판정 후보 `alarms --count-only` → **판정 기간 잠금(9.7)** → 개발 기간 `replay`/`report` → 판정 기간 `judge` 1회.

## 12. 테스트 계획 (`tests/validation/realdata/`, `tests/validation/test_convert_alarm_tables.py`)

- **DB 를 쓰지 않는 단위 테스트**가 기본: 가짜 연결/커서(쿼리 문자열·파라미터 기록)로 가드·청크·필터 SQL 을 검증. MySQL 이 필요한 테스트는 만들지 않는다(로컬 `cowptn_test` 확인은 완료 보고의 실행 출력으로).
- 픽스처 (R7 포함, 모두 합성 데이터):
  - **F-1 순간 단절**: 한 포트에서 `t0` 부터 2스텝 `rx=tx=0`, 앞뒤는 정상 트래픽 + ETH-LINK-FAIL 20분 에피소드. AI 대신 "알람 열을 직접 준" 프레임으로 지연·C3 계산을 검증(모델 의존 없음).
  - **F-2 깜빡임**: 같은 알람 10~14초 간격 30회(D6 모양) → 에피소드 1개, 지속 = 첫 발생 ~ 마지막 해제.
  - **F-3 진행 중**: `release_date = NULL` → `ongoing`, 구간 끝에서 자름.
  - **F-4 동시 다중 알람**: ETH-LINK-FAIL + OPT-LOS + ETH-NO-RX-TRAFFIC 이 5분 안에 → 트래픽단절 타겟 이벤트 1개, OPT-LOS 는 광관찰에도 1개.
  - **F-5 조인 안전 검사**: (a) 키가 맞는 장비 — 단절 알람 구간에 그 포트 `rx=0`, 다른 포트 정상 → `verified`. (b) 키가 어긋난 장비(알람 `cid` 가 성능보다 2 큰 구성, `addr_port` 는 일치) → J3 `failed`·멈춤, address 기준 통과해도 자동 전환 없음, J2 에 모델별 체계 불일치로 보고. (c) 장비 단위 알람(`cid=0`, `lid=0`) → 판정 제외·F8 표시에만 사용. (d) `p5_cid ≠ cid` 행 → `p5_same=False` 집계.
  - **F-6 표 없는 모델·범위 밖 code·미지 system_type**.
  - **F-7 시간 테이블 결측·경계**: 없는 시간 테이블, 다음 시간 테이블에 들어 있는 정각 행.
  - **F-8 중복 키**: 같은 포트·시각 2행(하위 ID 다름).
- 회귀: `baselines` 기존 테스트·`tests/validation/test_evaluation_metrics.py` 불변, `tests/test_architecture.py`(src → validation import 금지) PASS.
- 각 단위 끝에 전체 pytest(컨테이너), reviewer 검토.

## 13. 위험·한정·미검증

1. **약한 정답의 한계**: EMS 알람은 장애 그 자체가 아니다. 발생 조건(D10: 임계 설정 없음)을 모르므로 C3 는 "EMS 가 알린 사건을 먼저 울렸는가"일 뿐이며, ETH-ERR 과 우리 규칙이 같은 카운터라면 순환이다(9.4 고정 문구).
2. **조인 근거가 약함**: 키 `(ip_addr, cid, lid)` 는 사용자 판단(D13)이고 실데이터로는 아직 확인하지 않았다. J3 의 0.8·0.5 는 단절 알람의 의미에 대한 [추정] 에 기댄다. J 결과가 `unverified/failed` 면 C3·탐지 지연은 판정에 쓰지 않는다. 장비 단위 알람을 `cid=0`/`lid=0` 으로 가르는 규칙도 모델별 확인 전이다(K1a).
3. **8장 규칙은 데이터 없이 정한 값**이다. 해상도·일관성 논리로 정했으나 실제 알람 패턴(깜빡임 주기, 지속 분포)에 맞는지 모른다. 감도(15분·4h 병합) 건수를 참고로 내고 판정에는 쓰지 않는다.
4. **활성 모델이 실데이터에 맞지 않을 수 있음**(K12). 재생 결과가 극단이면 그대로 보고하며 이 도구 안에서 튜닝·재학습하지 않는다(재학습은 별도 결정).
5. **운영 입력과 도구 입력의 차이**(2.2-2): Q1 결정 전에는 라이브(C5) 와 재생의 일치율이 낮아도 원인이 필터일 수 있다.
6. **30대 규모**: C2 샘플 ≥ 100, C3 이벤트 ≥ 20, C6 중앙값 ≥ 10 이 채워지지 않아 판정 보류가 날 수 있다(plan.md P0-3 재점검 지점). 도구는 보류를 보류로 낸다.
7. **광은 판정 제외**: 2~3대 관찰뿐. 광 알람(D9)이 있어도 통계 판정하지 않는다.
8. **데이터 취급**: 스냅샷·샘플에는 IP·국소명이 들어간다. `validation/runs/` (git 제외) 밖으로 복사하지 않는다. 알람명 JSON 저장 위치는 Q2.
9. **미검증**: 실DB 에서 한 번도 실행하지 않은 설계다. 2.3 의 K 항목은 모두 첫 실행 출력으로 확인해야 한다. MySQL 5.7.43 의 `SET SESSION TRANSACTION READ ONLY`·`MAX_EXECUTION_TIME`·`SHOW GRANTS FOR CURRENT_USER()` 출력 형식은 문서 기준이며 실DB 에서 확인하지 않았다.
10. **현재 계정은 root**(D12): 읽기 전용 계정을 만들기 전에는 도구가 실행되지 않는다(의도된 동작). 계정 생성은 사용자가 root 로 수행하는 DB 변경이며 도구 밖의 일이다.
11. **판정 기간 봉인은 완전하지 않다**(9.7 2단계에서 판정 구간의 월별 알람 수를 봄). 방법 결과(AI·베이스라인)는 잠금 전에 보지 않는 것까지가 보장 범위다.
12. **섀도·운영 결과 쓰기 경로**(2.2-9): 이 도구와 무관하지만, 섀도에서 설정 DB 를 EMS DB 로 두면 EMS DB 에 테이블 생성·컬럼 추가·행 삭제가 일어난다. 14.2-B 결정 전에는 운영 설정 DB 를 EMS DB 로 두지 않을 것을 권한다.

## 14. 결정 사항

### 14.1 처리된 결정 (2026-10-15, lead 지시로 사용자 답 반영)

| # | 질문 | 결정 | 반영 위치 |
|---|---|---|---|
| Q1 | 운영 fetch 의 `time_type`·광 `signal_type` 필터 부재(2.2-2) | **`plan` 결과를 본 뒤 결정**(섞인 행이 있으면 U0 별도 설계) | 11장 U0 |
| Q2 | 알람명 JSON 저장 위치 | **git 제외 `validation/runs/ref/alarm_names.json`** | 7.1 |
| Q3 | 표 없는 모델 처리 | **미분류·판정 제외·별도 집계** | 7.3 |
| Q4 | F1~F8 | **F1~F7 권고값으로 사전 고정, F8 은 표시만(분모 유지)** | 8장 |
| Q5 | 조인 | **키 `(ip_addr, cid, lid)`**, address 는 진단 열, 장비 단위 알람 분리, J1·J3·M1 유지(키만 교체) | 6장 |
| Q6 | 결과를 EMS DB 에 쓰는가 | 코드 사실을 2.2-9 에 기록. **P1-2 도구는 해당 없음**. 분리 방식은 섀도 계획 단계 결정 → 14.2-B | 2.2-9, 13-12 |
| Q7 | 성능 `occur_date` 기준 | **구간 끝**(D11) | 9.5 |
| Q8 | 계정 | 사용자 계정은 root → **권한 검사 fail closed** + 읽기 전용 계정 생성 절차 | 4.1, 4.8 |
| Q9 | 기간 분할 | **최근 3개월 봉인(판정) + 이전 개발**, 이벤트 건수로만 3~6개월 조정 후 잠금 | 9.7 |

### 14.2 남은 결정 (사용자 승인 사항)

**A. 쓰기 권한 계정 예외 플래그(`--allow-write-account`)를 둘 것인가**

| 선택지 | 내용 | 위험 |
|---|---|---|
| A1. 두지 않음 | 권한 검사에 예외 없음 | 읽기 전용 계정을 만들 수 없는 환경에서는 도구를 못 씀 |
| A2. 두되 강하게 제한 | 플래그 + 확인 문구 입력, `plan`·`extract` 만 허용, 매니페스트·출력에 경고 기록, 세션 읽기 전용·문장 가드는 그대로 | 세션 읽기 전용과 문장 가드는 남지만 **서버가 계정 권한으로 막는 마지막 방어선이 없어진다**. root 비밀번호를 분석 PC 환경변수에 두게 됨(유출 시 EMS DB 전체 권한). 예외가 습관이 되기 쉬움. 이 도구 코드 밖의 실수(같은 셸에서 다른 스크립트 실행 등)도 막지 못함 |

- **권고: A1(두지 않음)**. 사용자가 이미 root 로 접속할 수 있으므로 4.8 의 계정 생성(명령 2줄)은 바로 할 수 있고, 예외를 둘 이유(계정 생성 불가)가 현재 없다. 나중에 계정 생성이 불가능한 환경이 생기면 그때 A2 를 별도 승인으로 추가한다.

**B. 섀도·운영에서 결과 테이블(`anomaly_detection`)을 EMS DB 와 분리할 것인가 (섀도 운영 계획 단계 결정, P1-2 범위 밖)**

| 선택지 | 내용 | 필요한 변경 | 위험 |
|---|---|---|---|
| B1. 그대로(설정 DB = EMS DB) | 현행 코드 그대로 | 없음 | EMS DB 에 테이블 생성·컬럼 추가·행 삭제(2.2-9), 운영 서비스가 EMS DB 쓰기 권한 계정을 가져야 함. EMS 운영 측 승인 없이는 부적절 |
| B2. 같은 서버, 다른 스키마 | 읽기 = EMS 스키마(SELECT 전용 계정), 쓰기 = 결과 전용 스키마(그 스키마만 쓰기 권한) | `DBConnector` 의 읽기/쓰기 접속 설정 분리(현재 한 풀·스키마 미한정 테이블명, `db_connector.py:14`·`:146`) — `src` 변경, 설계·테스트 필요 | EMS 서버 부하·디스크를 결과 저장이 같이 씀. 서버 관리자 승인 필요 |
| B3. 다른 서버/인스턴스 | 읽기 = EMS DB(SELECT 전용), 쓰기 = 별도 MySQL(분석 서버 등) | B2 와 같은 설정 분리 | 운영할 DB 가 하나 늘어남 |
| B4. 라이브 섀도 없이 오프라인 재생만 | P1-2 도구로 주기적 재생(C1~C4·C6) | 없음 | **C5(처리율·지연·중단)는 실환경에서 측정 불가** — 섀도 계획의 C5 를 다른 방식으로 대체해야 함 |

- **권고: B3**(안 되면 B2). EMS DB 에는 SELECT 전용 계정만 쓰고 결과는 분리하는 것이 최소 권한 원칙에 맞고, 운영 쪽 변경(읽기/쓰기 접속 분리)은 섀도 시작 전 별도 설계·구현 항목으로 둔다. 결정 전에는 **운영 설정 DB 를 EMS DB 로 두지 않는다**(13-12). 2.2-9 의 사실(Producer 처럼 읽기만 하는 서비스도 생성자에서 DDL 을 실행)은 B2·B3 설계 때 함께 고칠 대상이다.

**C. 실행 후에 정해지는 것 (지금 결정할 수 없음)**
- Q1 의 실제 판단(`plan` 출력 후), M1 수동 확인 3포트(사용자, 첫 `join-check` 전), 판정 기간 길이(9.7 2~3단계 건수 후), 알람명 별칭(K11, C 소스 변환 후).

## 15. 변경 이력
- 2026-10-15: 초안 작성(architect, `cd6634c`). 입력: 사용자 실DB 확인 사실(2.1, lead 전달), 기존 코드 확인(2.2).
- 2026-10-15 개정 1(architect, 실데이터 미개봉 상태): 사용자 답 반영 — Q7 성능 시각 = 구간 끝(D11, 9.5 탐지 지연 정의 `bin_end`·`pre15`·스텝 지연), Q8 root 계정 → 권한 검사 fail closed·읽기 전용 계정 절차(4.1·4.8), **Q5 조인 키를 address 정규화에서 `(ip_addr, cid, lid)` 로 변경**(6장 재작성 — 초안의 근거 사례가 다른 모델끼리의 비교였음, D13), 장비 단위 알람 분리, K1a·K1b 추가, Q4 F1~F8 확정(F8 표시만), Q9 3개월 봉인·건수 기반 조정 규칙(9.7), Q6 운영 쓰기 경로 코드 사실(2.2-9)과 결과 DB 분리 선택지(14.2-B), Q1~Q3 권고대로. 구현 단위 U1·U4~U7 갱신.

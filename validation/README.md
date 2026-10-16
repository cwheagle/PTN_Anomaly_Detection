# validation/ — 솔루션 밖의 검증 도구

이 폴더는 **솔루션(`src/`, `ui/`)이 아닙니다.** 시뮬레이션, 평가, 실험을 위한 도구이며 Docker 이미지에 포함되지 않습니다.

## 규칙
- `validation/` → `src/` 의존은 **허용** (예: 평가가 `AnomalyDetector`를 호출)
- `src/` → `validation/` 의존은 **금지** (`tests/test_architecture.py` 가 검사)
- 평가용 정답/시뮬레이터 산출물을 운영 코드(학습 파이프라인 등)가 읽으면 안 됨 (평가 정답 유출 방지)

## 구성
| 경로 | 내용 |
|---|---|
| `simulator/scenario_generator.py` | **평가용** 시드 고정 시나리오 생성기 (DB 불필요, 스텝 단위 정답 포함) |
| `simulator/data_generator.py`, `simulator_rca_live.py` | 15분마다 실시간 데이터를 DB에 주입 (Kafka 파이프라인 시연용). 정답지가 없어 **평가에는 사용 금지** |
| `simulator/config_sim.py`, `db_manager.py` | 시뮬레이션 DB 설정(환경변수 `SIM_DB_*`)과 테이블 관리 |
| `evaluation/metrics.py` | 이벤트 단위 지표 (조기 탐지율, 리드타임, 오탐 인시던트, 이벤트 F1, AUPRC) |
| `evaluation/baselines.py` | 자명한 베이스라인 (항상 알람, 고정 임계, 포트별 롤링 규칙) |
| `cli/evaluate_model.py` | 모델 평가 (생성 → 추론 → 지표 → 베이스라인 비교) |
| `cli/train_isolated.py` | 활성 모델을 건드리지 않는 격리 재학습 실험 |
| `cli/compare_reports.py` | 여러 평가 리포트 비교표 |
| `cli/seed_db.py` | 시나리오 생성기 데이터를 시뮬레이션 DB에 주입 (E2E 시연용, 이름에 `test` 없는 DB는 거부) |
| `cli/run_training.py`, `run_inference_check.py`, `run_full_cycle.py` | 수동 실행 러너 (**`run_training`은 실제 `models/` 를 갱신함**) |
| `runs/` | 실행 산출물 (git 제외, 시드로 재생성 가능) |

## 자주 쓰는 명령 (프로젝트 루트 어디서든 실행 가능)
```bash
python validation/cli/evaluate_model.py                       # 기본 평가 (seed=7, 60포트, 14일, 약 2~3분)
python validation/cli/evaluate_model.py --clean               # 정상 노이즈 제거 버전
python validation/cli/train_isolated.py --variant noisy --out validation/runs/models_noisy
python validation/cli/evaluate_model.py --models-dir validation/runs/models_noisy --tag noisymodel
python validation/cli/compare_reports.py                      # 리포트 비교표
python validation/cli/seed_db.py --days 3                     # E2E 시연용 데이터를 DB에 주입 (SIM_DB_NAME 으로 대상 지정)
```

## real_data_check (P1-2 실데이터 검증 도구, U1·U2: `plan`, `extract`)
설계서 `docs/design/p1_2_real_data_check.md`. 실DB 를 **읽기 전용**으로 조회해 로컬 스냅샷(`validation/runs/p1_2/<run_id>/`, git 제외)을 만든다. `src/config.py`·`.env` 는 읽지 않고 환경변수/CLI 인자로 접속한다.

```bash
export DB_HOST=... DB_NAME=... RDC_DB_USER=ptn_ro RDC_DB_PASS=...      # 비밀번호는 환경변수로만
python validation/cli/real_data_check.py plan    --start 2026-10-05 --end 2026-10-12   # 행 추출 없음: 테이블·예상 행·경계·필터 분포·시간대·결과 테이블 충돌
python validation/cli/real_data_check.py extract --start 2026-07-10 --end 2026-10-13 [--ports ports.csv] [--max-rows 2000000]
```
- **권한 검사는 fail closed**: `SHOW GRANTS` 에 SELECT/USAGE 외 권한(root 의 `ALL PRIVILEGES`, `WITH GRANT OPTION` 포함)이 있거나 판단 불가면 거부한다. 테스트망에서만 `--allow-write-account` 를 명령줄에 명시해 진행할 수 있다(환경변수로는 켜지지 않음, 매니페스트에 `privilege_check: override` 기록). 읽기 전용 세션(`SET SESSION TRANSACTION READ ONLY`)과 SELECT/SHOW 문장 가드는 플래그와 무관하게 유지된다.
- 접속 환경변수: `DB_HOST`·`DB_PORT`·`DB_NAME`, 계정은 `RDC_DB_USER`/`RDC_DB_PASS` 가 없으면 **`DB_USER`/`DB_PASS` 로 폴백**한다(운영 설정 환경변수를 그대로 쓰는 셸이면 의도치 않게 그 계정이 쓰일 수 있으니 확인).
- 문장 가드는 SELECT/SHOW 만 통과시키지만 `SELECT ... INTO OUTFILE/DUMPFILE` 은 막지 않는다. 이는 서버 권한(FILE)으로 막아야 하며, 그래서 읽기 전용 계정이 권장 경로다. 증분 추출은 구현하지 않는다(스냅샷은 추출 시점 상태로 고정).
- **읽기 전용 계정 만들기(권장, 관리자 1회, MySQL 5.7)**:
  ```sql
  CREATE USER 'ptn_ro'@'<분석PC_IP>' IDENTIFIED BY '<충분히 긴 비밀번호>';   -- '%' 호스트 금지
  GRANT SELECT ON `<EMS_DB>`.* TO 'ptn_ro'@'<분석PC_IP>';
  SHOW GRANTS FOR 'ptn_ro'@'<분석PC_IP>';   -- GRANT USAGE ON *.* / GRANT SELECT ON `<EMS_DB>`.*
  -- 삭제: DROP USER 'ptn_ro'@'<분석PC_IP>';
  ```
- `--schema sim`: 시뮬레이터 DB(`cowptn_test`) 용. 필터 컬럼(`time_type` 등)·알람 테이블이 없으면 생략하고 매니페스트에 남긴다(`real` 은 없으면 중단).
- 시각이 테이블 시각 ±1.5시간 밖인 행(쓰레기 `occur_date`)은 분석에서 빼고 `quarantine/` 에 보관한다(설계서 4.3a). 종료 코드: 0 완료, 2 인자 오류, 3 권한 거부, 4 중단/불완전.

### real_data_check 판정 실행 주의 (U6~U7)
- **판정 재생이 실패하면 표식이 소진되어 재시도할 수 없다.** `periods lock` 이후 `report --purpose judge` 는 측정 전에 `locks/judge_*.run.json` 표식을 배타적으로 만들고, 재생 중 오류(모델 로드 실패 등)가 나도 표식은 남는다. 복구 규칙은 설계 9.7.6 의 결정 사항이라 구현하지 않았다(architect 결정 대기).
- 판정에서는 `--policy` 를 쓸 수 없고, 레지스트리 모델의 `trained_at`·정책·가중치(`model/scaler/meta` sha256)가 잠금 때와 같아야 한다. 테스트에서 `detector` 를 직접 주입하면 이 해시 검사는 생략된다(CLI 는 항상 레지스트리 경로).

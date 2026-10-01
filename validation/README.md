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

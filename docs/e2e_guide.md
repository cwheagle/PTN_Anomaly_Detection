# E2E 점검 가이드 (Docker)

Producer → Kafka → Consumer(Redis 윈도우 + 모델 추론) → MySQL → API → UI 전체 경로를 실제 컨테이너로 검증하는 절차입니다.
로컬(Windows/Linux x86)에서 가짜 데이터로 수행하며, 마지막의 `e2e_verify.py` 가 결과를 PASS/FAIL 로 판정합니다.

## 0. 준비물
| 항목 | 확인 방법 |
|---|---|
| Docker Desktop (Linux containers 모드) | `docker version` 에서 Server 가 `linux/...` |
| 로컬 MySQL, 시뮬레이션 DB(`cowptn_test`) | `root` 가 외부 호스트(`'%'`)에서 접속 가능해야 함 (컨테이너는 `host.docker.internal` 로 접속) |
| 프로젝트 루트 `.env` | `PTN_PLATFORM=linux/amd64`, `DB_NAME=cowptn_test` (템플릿: `.env.example`) |
| `src/config.py`, `models/` | 이미지에 포함됨 (gitignore 대상이므로 로컬에 있어야 함) |
| 비어 있는 호스트 포트 | 80, 8000, 9092, 2181, 6379 |

> 시간대: compose 가 api/consumer/producer 에 `TZ=Asia/Seoul` 을 설정합니다(`PTN_TZ` 로 변경). DB 타임스탬프의 시간대와 반드시 같아야 합니다.

## 1. 시연 데이터 주입
현재 시점 근처에 장애가 있는 가짜 데이터를 DB 에 **추가**합니다 (기존 데이터는 삭제하지 않음, DB 이름에 `test` 가 없으면 거부).
```bash
python validation/cli/seed_db.py --days 1 --nodes 3 --warmup-days 0.4 --mean-gap-days 0.4 --yes --out validation/runs/e2e_seed
```
- 30포트 × 96스텝. 정답은 `validation/runs/e2e_seed/` (`episodes.csv`, `labels.csv`).
- Consumer 는 포트마다 **27건**(`DataProcessor.required_rows`)을 모아야 첫 결과를 내므로, 최소 28스텝 이상 필요합니다.

## 2. 이미지 빌드 및 기동
```bash
docker compose up -d --build        # 첫 빌드는 torch 때문에 수 GB, 수 분 소요
docker compose ps                   # 7개 서비스 Up 확인
```
알람 재알림 간격을 바꾸려면: `ALARM_RENOTIFY_MINUTES=60 docker compose up -d --build` (기본 15 = 스텝마다, 0 = 진입 시 1회).

## 3. 알람 스트림 수신 시작 (별도 터미널)
```bash
curl -s -N http://localhost:8000/api/stream/alarms > validation/runs/e2e_sse.log
```

## 4. Producer 로 데이터 흘리기
라이브 모드는 "최근 15분"만 읽으므로, 시연 데이터는 `sim` 모드로 시간 범위를 지정해 재생합니다 (시연 데이터의 시작/끝 시각에 맞출 것).
```bash
docker compose exec -T producer python -u src/pipeline/kafka_producer.py --mode sim --start "2026-09-30 14:00:00" --end "2026-10-01 14:00:00"
```
Consumer 가 따라잡을 때까지 기다립니다 (랙 0):
```bash
docker compose exec -T kafka kafka-consumer-groups --bootstrap-server kafka:29092 --describe --group ptn_inference_group
```
> Git Bash(Windows)에서 `docker compose exec` 에 `/` 로 시작하는 경로를 넘기면 경로가 변환됩니다. `MSYS_NO_PATHCONV=1` 을 앞에 붙이세요.

## 5. 자동 판정
```bash
python validation/cli/e2e_verify.py --sse-log validation/runs/e2e_sse.log --seed-dir validation/runs/e2e_seed --renotify 15
```
판정 항목: 서비스 응답, 포트당 결과 행 수, ALARM 이벤트 수(재알림 설정과 일치), CLEAR 가 실제 복구 시점과 일치, 오해제 없음.
(`--renotify` 는 스택에 설정한 `ALARM_RENOTIFY_MINUTES` 와 같아야 함)

## 6. 수동 확인 (선택)
| 확인 | 명령 |
|---|---|
| RCA 룰 + Consumer 리로드 | UTF-8 JSON 파일로 `POST /api/rca/rules` → consumer 로그에 `Received RCA rules reload request` |
| 재학습 → 후보 저장 | `POST /api/model/train?ft=optical&train_start=...&train_end=...` (`{"epochs":2}`) → `GET /api/model/versions?ft=optical` 에 `v2 candidate`, **활성은 v1 유지, consumer 리로드 없음** |
| 승격 경고 / 승격 / 롤백 | `POST /api/model/promote?ft=optical&version=v2` → 비정상 후보면 409 + 경고, `force=true` 로 승격 시 consumer 로그에 `(Re)loading optical model from models/optical_ae_v2.pth`, `POST /api/model/rollback?ft=optical` → v1 복귀 |
| Redis 윈도우 | `docker compose exec -T redis redis-cli LLEN "<ip>:<cid>:<lid>"` → 27 |
| UI | 브라우저에서 `http://localhost` (정적 파일/프록시는 HTTP 로 확인 가능) |

> **한글 JSON 주의**: Windows 터미널에서 `curl -d '{"diagnosis":"한글"}'` 는 인코딩이 깨져 파싱 오류가 납니다 (서버 문제 아님). UTF-8 파일을 `--data-binary @file` 로 보내세요.

## 7. 정리
```bash
docker compose down -v    # 컨테이너와 이 프로젝트의 볼륨(ptn_models, ptn_rca_rules 등) 삭제. 이미지/호스트 파일은 유지
```
- 재학습 테스트로 볼륨 안에 시험용 모델(v2)이 생기므로, 반드시 `-v` 로 정리해야 다음 실행에 영향이 없습니다.
- 호스트의 `models/` 는 이 절차로 바뀌지 않습니다 (컨테이너는 독립된 볼륨을 사용).
- 룰/모델 파일을 바꾼 뒤에는 기존 볼륨이 이미지 내용으로 갱신되지 않으므로 `down -v` 가 필요합니다.

## 알려진 한계 (이 절차가 검증하지 않는 것)
- arm64 이미지 빌드/실행, GPU 사용 (GPU 서버에서 별도 확인)
- 브라우저에서의 UI 화면 동작 (HTTP 수준만 확인)
- 모델 성능 (이 E2E 는 배선과 로직의 검증이며, 성능은 `validation/cli/evaluate_model.py` 로 평가)
- Kafka 가 늦게 뜰 때 시작 직후 `Connection refused` 로그가 나올 수 있음 (자동 재연결로 무해)

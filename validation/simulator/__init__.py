"""
Simulator Package (솔루션 밖의 검증용 가짜 데이터 생성)

- scenario_generator.py : [평가용] 시드 고정 시나리오 생성기 (DB 불필요, 스텝 단위 정답 포함)
- config_sim.py / db_manager.py : 가짜 데이터를 넣을 DB 설정(환경변수 SIM_DB_*)과 테이블 관리
- data_generator.py / simulator_rca_live.py : 15분마다 실시간 데이터를 DB에 주입 (Kafka 파이프라인 시연용)

과거 이력 DB 주입은 validation/cli/seed_db.py 를 사용한다.
"""

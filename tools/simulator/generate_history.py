import time
from datetime import datetime, timedelta
from config_sim import SIM_CONFIG, DB_CONFIG_SIM
from db_manager import SimulatorDBManager
from data_generator import DataGenerator

def generate_history():
    print("=== PTN Local Simulator: Historical Data Generation ===")
    db = SimulatorDBManager()
    gen = DataGenerator()
    
    end_dt = datetime.now().replace(second=0, microsecond=0)
    # 15분 단위 정렬
    end_dt = end_dt - timedelta(minutes=end_dt.minute % SIM_CONFIG['interval_minutes'])
    
    start_dt = end_dt - timedelta(days=SIM_CONFIG['history_days'])
    
    print(f"Generating data from {start_dt} to {end_dt} (Interval: 15m)")
    
    current_dt = start_dt
    total_steps = int((end_dt - start_dt).total_seconds() / (SIM_CONFIG['interval_minutes'] * 60))
    step = 0
    
    start_time_perf = time.time()
    
    # 캐싱 (매 15분마다 테이블 DDL을 호출하지 않기 위함)
    last_table_hr = None
    traffic_table = None
    optical_table = None
    
    while current_dt <= end_dt:
        hr_str = current_dt.strftime('%Y_%m_%d_%H')
        
        # 정각(시간 변경)마다 테이블 갱신 및 생성 확인
        if hr_str != last_table_hr:
            traffic_table = db.ensure_traffic_table(current_dt)
            optical_table = db.ensure_optical_table(current_dt)
            last_table_hr = hr_str
            
        # 중복 방지 로직: 이미 데이터가 있는지 검사
        if not db.check_data_exists(traffic_table, current_dt):
            traffic_data, optical_data = gen.generate_snapshot(current_dt)
            
            db.insert_traffic(traffic_table, traffic_data)
            db.insert_optical(optical_table, optical_data)
        else:
            pass # 중복 데이터 건너뜀
        
        step += 1
        if step % 100 == 0:
            print(f"Progress: {step}/{total_steps} ({(step/total_steps)*100:.1f}%) - Current: {current_dt}")
            
        current_dt += timedelta(minutes=SIM_CONFIG['interval_minutes'])
        
    elapsed = time.time() - start_time_perf
    print(f"Generation Complete! Inserted {step} snapshots across {SIM_CONFIG['nodes']} nodes.")
    print(f"Time taken: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    # 실행 시 사용자에게 확인을 받음
    print(f"[Warning] This will generate {SIM_CONFIG['history_days']} days of data on the configured DB.")
    print("Config DB:", DB_CONFIG_SIM['database'], "@", DB_CONFIG_SIM['host'])
    res = input("Proceed? (y/n): ")
    if res.lower() == 'y':
        generate_history()
    else:
        print("Cancelled.")

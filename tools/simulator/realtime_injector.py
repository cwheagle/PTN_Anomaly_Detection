import time
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timedelta
from config_sim import SIM_CONFIG
from db_manager import SimulatorDBManager
from data_generator import DataGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

db = SimulatorDBManager()
gen = DataGenerator()

def job_inject_data():
    now = datetime.now().replace(second=0, microsecond=0)
    
    # 15분 단위로 강제 정렬하여 DB에 넣기 위함 (스케줄러가 약간 늦게 돌더라도 정각 유지)
    minutes = now.minute - (now.minute % SIM_CONFIG['interval_minutes'])
    current_dt = now.replace(minute=minutes)
    
    logging.info(f"Injecting mock data for timestamp: {current_dt}")
    
    traffic_table = db.ensure_traffic_table(current_dt)
    optical_table = db.ensure_optical_table(current_dt)
    
    # 중복 시간 방지
    if db.check_data_exists(traffic_table, current_dt):
        logging.info(f"Data already exists for timestamp: {current_dt}. Skipping injection.")
        return
        
    traffic_data, optical_data = gen.generate_snapshot(current_dt)
    
    db.insert_traffic(traffic_table, traffic_data)
    db.insert_optical(optical_table, optical_data)
    
    logging.info(f"Successfully injected {len(traffic_data)} records to {traffic_table}")

def start_injector():
    scheduler = BlockingScheduler()
    # 15분 마다 실행 (0, 15, 30, 45 분)
    scheduler.add_job(job_inject_data, 'cron', minute=f"*/{SIM_CONFIG['interval_minutes']}")
    
    logging.info(f"Starting Real-time Mock Injector. (Interval: {SIM_CONFIG['interval_minutes']}m)")
    logging.info("Press Ctrl+C to stop.")
    
    # 구동 즉시 한 번 실행
    job_inject_data()
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Injector stopped.")

if __name__ == "__main__":
    start_injector()

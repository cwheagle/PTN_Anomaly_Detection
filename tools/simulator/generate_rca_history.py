import time
import os
import pandas as pd
from datetime import datetime, timedelta
from config_sim import SIM_CONFIG, DB_CONFIG_SIM
from db_manager import SimulatorDBManager
from simulator_rca_live import RCATrendDataGenerator

def generate_rca_history():
    print("=== PTN Local Simulator: RCA Ground Truth Data Generation ===")
    db = SimulatorDBManager()
    
    # 완전히 깨끗한 데이터를 위해 기존 DB 초기화 여부 확인
    res = input("Do you want to RESET(Drop & Recreate) the existing database for clean data? (y/n): ")
    if res.lower() == 'y':
        print("[*] Resetting Database...")
        conn = db.get_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(f"DROP DATABASE IF EXISTS {DB_CONFIG_SIM['database']}")
            cursor.close()
            conn.close()
        db._ensure_database()
        print("[SUCCESS] Database has been reset to a clean state.")
    
    # We use a modified generator that logs ground truth
    class GroundTruthGenerator(RCATrendDataGenerator):
        def __init__(self):
            super().__init__()
            self.ground_truth = []
            
        def trigger_trend_anomaly(self, dt, node_key):
            if node_key not in self.trend_states:
                scenario = self.rca_scenarios[0] if 'optical' in node_key else self.rca_scenarios[1] # Mix randomly
                import random
                scenario = random.choice(self.rca_scenarios)
                
                max_steps = 20 # 5 hours
                self.trend_states[node_key] = {
                    'scenario': scenario,
                    'start_time': dt,
                    'step_count': 0,
                    'max_steps': max_steps
                }
                
                # TTF = 0 occurs exactly at start_time + (max_steps * 15 minutes)
                failure_time = dt + timedelta(minutes=max_steps * SIM_CONFIG['interval_minutes'])
                
                ip, cid, lid = node_key.split('_')
                self.ground_truth.append({
                    'ip_addr': ip,
                    'cid': int(cid),
                    'lid': int(lid),
                    'scenario': scenario,
                    'start_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'failure_time': failure_time.strftime('%Y-%m-%d %H:%M:%S')
                })
                # print(f"[Ground Truth] {node_key} will fail at {failure_time} due to {scenario}")

    gen = GroundTruthGenerator()
    
    end_dt = datetime.now().replace(second=0, microsecond=0)
    end_dt = end_dt - timedelta(minutes=end_dt.minute % SIM_CONFIG['interval_minutes'])
    start_dt = end_dt - timedelta(days=SIM_CONFIG['history_days'])
    
    print(f"Generating RCA data from {start_dt} to {end_dt} (Interval: 15m)")
    
    current_dt = start_dt
    total_steps = int((end_dt - start_dt).total_seconds() / (SIM_CONFIG['interval_minutes'] * 60))
    step = 0
    
    start_time_perf = time.time()
    
    # DB Tables
    last_table_hr = None
    traffic_table = None
    optical_table = None
    
    while current_dt <= end_dt:
        hr_str = current_dt.strftime('%Y_%m_%d_%H')
        
        if hr_str != last_table_hr:
            traffic_table = db.ensure_traffic_table(current_dt)
            optical_table = db.ensure_optical_table(current_dt)
            last_table_hr = hr_str
            
        if not db.check_data_exists(traffic_table, current_dt):
            traffic_data, optical_data = gen.generate_snapshot(current_dt)
            db.insert_traffic(traffic_table, traffic_data)
            db.insert_optical(optical_table, optical_data)
            
        step += 1
        if step % 200 == 0:
            print(f"Progress: {step}/{total_steps} ({(step/total_steps)*100:.1f}%) - Current: {current_dt}")
            
        current_dt += timedelta(minutes=SIM_CONFIG['interval_minutes'])
        
    elapsed = time.time() - start_time_perf
    print(f"Generation Complete! Inserted {step} snapshots.")
    
    # Save Ground Truth
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(curr_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    gt_df = pd.DataFrame(gen.ground_truth)
    gt_path = os.path.join(data_dir, 'eval_dataset.csv')
    gt_df.to_csv(gt_path, index=False)
    
    print(f"Time taken: {elapsed:.2f} seconds.")
    print(f"[SUCCESS] Ground Truth perfectly generated: saved to {gt_path} ({len(gt_df)} anomalies generated)")

if __name__ == "__main__":
    print(f"[Warning] This will generate {SIM_CONFIG['history_days']} days of data on the configured DB.")
    print("Config DB:", DB_CONFIG_SIM['database'], "@", DB_CONFIG_SIM['host'])
    res = input("Proceed? (y/n): ")
    if res.lower() == 'y':
        generate_rca_history()
    else:
        print("Cancelled.")

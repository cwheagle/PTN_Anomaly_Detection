import time
import logging
import random
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from config_sim import SIM_CONFIG, SIGNAL_TYPES
from db_manager import SimulatorDBManager
from data_generator import DataGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class RCATrendDataGenerator(DataGenerator):
    def __init__(self):
        super().__init__()
        self.rca_scenarios = ['crc_error', 'optical_degradation']
        # 오버라이드: { 'node_key': {'scenario': 'crc_error', 'start_time': dt, 'step_count': 0, 'max_steps': 20} }
        self.trend_states = {}
    
    def trigger_trend_anomaly(self, dt, node_key):
        if node_key not in self.trend_states:
            scenario = random.choice(self.rca_scenarios)
            self.trend_states[node_key] = {
                'scenario': scenario,
                'start_time': dt,
                'step_count': 0,
                'max_steps': 20 # 5시간(20스텝) 동안 서서히 악화
            }
            logging.info(f"[!] RCA Scenario '{scenario}' TRIGGERED on port {node_key}")

    def get_traffic_value(self, dt, node_key, is_anomaly=False):
        # 기본 주기 트래픽 생성 (기존 랜덤 spike는 무시하기 위해 is_anomaly 강제 해제)
        tx, rx, errors = super().get_traffic_value(dt, node_key, is_anomaly=False)
        
        if node_key in self.trend_states:
            state = self.trend_states[node_key]
            progress = min(1.0, state['step_count'] / state['max_steps'])
            
            if state['scenario'] == 'crc_error':
                # 트래픽 유지 또는 약간 감소, 에러 패킷 기하급수적 폭증
                tx = int(tx * (1.0 - 0.2 * progress))
                rx = int(rx * (1.0 - 0.2 * progress))
                errors = int(5000 * (progress ** 2))
                
        return tx, rx, errors

    def get_optical_value(self, dt, node_key, is_anomaly=False):
        profile = self.port_profiles[node_key]
        tx = random.gauss(profile['optical_tx_mean'], profile['optical_noise_std'])
        rx = random.gauss(profile['optical_rx_mean'], profile['optical_noise_std'])
        
        if node_key in self.trend_states:
            state = self.trend_states[node_key]
            progress = min(1.0, state['step_count'] / state['max_steps'])
            
            if state['scenario'] == 'optical_degradation':
                # RX 수신 파워가 서서히 떨어짐 (최대 -25dBm 하락시켜 장애 레벨 도달)
                rx -= (25.0 * progress)
                tx -= (2.0 * progress)
                
        return round(tx, 2), round(rx, 2)

    def generate_snapshot(self, dt):
        traffic_data = []
        optical_data = []
        
        # 1. 시나리오 상태 업데이트 (1스텝 증가)
        for key in list(self.trend_states.keys()):
            self.trend_states[key]['step_count'] += 1
            # max_steps(20)에 도달하면 최고 심각도(progress=1.0) 상태로 유지됨

        for n in range(self.config['nodes']):
            ip = self.generate_ip(n)
            for p in range(self.config['ports_per_node']):
                cid = p
                lid = p + 1
                node_key = f"{ip}_{cid}_{lid}"
                
                # 매우 낮은 확률(약 0.05%)로 15분마다 포트 하나씩 RCA 시나리오에 빠짐
                # 100개 포트 기준 15분마다 약 1건의 장애 발생 (테스트용 황금 비율 1%)
                if random.random() < 0.01: 
                    self.trigger_trend_anomaly(dt, node_key)
                elif random.random() < 0.005: # 장비 교체 등 자연 복구 확률
                    self.reset_trend(node_key)
                
                tx_pkt, rx_pkt, err_pkt = self.get_traffic_value(dt, node_key, False)
                tx_opt, rx_opt = self.get_optical_value(dt, node_key, False)
                
                traffic_data.append((dt, ip, cid, lid, SIGNAL_TYPES['ETH'], tx_pkt, rx_pkt, err_pkt))
                optical_data.append((dt, ip, cid, lid, tx_opt, rx_opt))
                
        return traffic_data, optical_data

db = SimulatorDBManager()
gen = RCATrendDataGenerator()

def job_inject_data():
    now = datetime.now().replace(second=0, microsecond=0)
    minutes = now.minute - (now.minute % SIM_CONFIG['interval_minutes'])
    current_dt = now.replace(minute=minutes)
    
    logging.info(f"Injecting RCA Live Mock data for timestamp: {current_dt}")
    
    traffic_table = db.ensure_traffic_table(current_dt)
    optical_table = db.ensure_optical_table(current_dt)
    
    if db.check_data_exists(traffic_table, current_dt):
        logging.info(f"Data already exists for timestamp: {current_dt}. Skipping injection.")
        return
        
    traffic_data, optical_data = gen.generate_snapshot(current_dt)
    
    db.insert_traffic(traffic_table, traffic_data)
    db.insert_optical(optical_table, optical_data)
    
    logging.info(f"Successfully injected {len(traffic_data)} records. (Active RCA Trend Ports: {len(gen.trend_states)})")

def start_injector():
    scheduler = BlockingScheduler()
    # 15분 마다 실행
    scheduler.add_job(job_inject_data, 'cron', minute=f"*/{SIM_CONFIG['interval_minutes']}")
    
    logging.info(f"Starting Live RCA Simulator... (100 ports, Interval: {SIM_CONFIG['interval_minutes']}m)")
    logging.info("Press Ctrl+C to stop.")
    
    # 구동 즉시 1회 실행
    job_inject_data()
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("RCA Simulator stopped.")

if __name__ == "__main__":
    start_injector()

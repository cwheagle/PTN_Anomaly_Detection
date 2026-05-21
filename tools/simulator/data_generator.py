import random
import math
from datetime import datetime, timedelta
from config_sim import SIM_CONFIG, SIGNAL_TYPES

class DataGenerator:
    def __init__(self):
        self.config = SIM_CONFIG
        self.trend_states = {}  # { 'ip_cid_lid': { 'type': 'optical/traffic', 'start_time': dt, 'degradation_factor': float } }
        self.port_profiles = self._generate_port_profiles()

    def _generate_port_profiles(self):
        """각 포트(링크)마다 고유한 트래픽/광파워 특성을 생성합니다."""
        profiles = {}
        for n in range(self.config['nodes']):
            ip = self.generate_ip(n)
            for p in range(self.config['ports_per_node']):
                cid = p
                lid = p + 1
                node_key = f"{ip}_{cid}_{lid}"
                
                # 포트별 특성 부여 (Core 링크는 트래픽 높음, Access 링크는 낮음 등)
                traffic_scale = random.uniform(0.1, 2.0)
                # 포트별 광파워 베이스라인 (거리에 따라 다름)
                tx_base = random.uniform(-6.0, -1.0)
                rx_base = tx_base - random.uniform(1.0, 4.0) # Rx는 Tx보다 보통 약간 낮음
                
                profiles[node_key] = {
                    'traffic_tx_mean': self.config['traffic_base']['tx_mean'] * traffic_scale,
                    'traffic_rx_mean': self.config['traffic_base']['rx_mean'] * traffic_scale,
                    'traffic_noise_std': self.config['traffic_base']['noise_std'] * traffic_scale,
                    'optical_tx_mean': tx_base,
                    'optical_rx_mean': rx_base,
                    'optical_noise_std': self.config['optical_base']['noise_std'] * random.uniform(0.8, 1.5),
                    'phase_shift': random.uniform(-2, 2) # 트래픽 피크 시간이 포트마다 약간씩 다름 (-2~+2시간)
                }
        return profiles

    def generate_ip(self, node_idx):
        return f"192.168.100.{10 + node_idx}"

    def get_traffic_value(self, dt, node_key, is_anomaly=False):
        """
        시간대별 주기성(Sinusoidal)을 갖는 트래픽 생성
        포트별 프로필 반영.
        """
        profile = self.port_profiles[node_key]
        
        # 주기 함수: 24시간 주기. 14시(기본) + phase_shift 시 피크
        hour = dt.hour + dt.minute / 60.0
        shifted_hour = hour - profile['phase_shift']
        phase = (shifted_hour - 14) / 24.0 * 2 * math.pi
        cycle_multiplier = (math.cos(phase) + 1.5) / 2.5 # 0.2 ~ 1.0 비율
        
        base_tx = profile['traffic_tx_mean'] * cycle_multiplier
        base_rx = profile['traffic_rx_mean'] * cycle_multiplier
        
        tx = max(0, int(random.gauss(base_tx, profile['traffic_noise_std'])))
        rx = max(0, int(random.gauss(base_rx, profile['traffic_noise_std'])))
        errors = 0
        
        if is_anomaly:
            # 다양한 트래픽 이상 시나리오
            anomaly_type = random.choice([
                "spike_up",      # 갑작스러운 트래픽 폭증 (DDoS, 혼잡)
                "drop_down",     # 갑작스러운 트래픽 급감 (단선, 라우팅 장애)
                "error_burst"    # 에러 패킷 폭증 (케이블 불량, 포트 장애)
            ])
            
            if anomaly_type == "spike_up":
                tx = int(tx * random.uniform(3.0, 5.0))
                rx = int(rx * random.uniform(3.0, 5.0))
            elif anomaly_type == "drop_down":
                tx = int(tx * random.uniform(0.01, 0.2)) # 트래픽 80~99% 감소
                rx = int(rx * random.uniform(0.01, 0.2))
            elif anomaly_type == "error_burst":
                errors = int(random.uniform(500, 10000))
                # 에러 폭증 시 실제 처리되는 정상 패킷은 약간 감소할 수 있음
                tx = int(tx * random.uniform(0.5, 0.8))
                rx = int(rx * random.uniform(0.5, 0.8))
                
        # 복합 장애 시나리오 강제 주입
        if getattr(self, '_force_both_anomaly', False):
            errors = int(random.uniform(2000, 15000))
            tx = int(tx * random.uniform(0.0, 0.1))
            rx = int(rx * random.uniform(0.0, 0.1))
                
        return tx, rx, errors

    def get_optical_value(self, dt, node_key, is_anomaly=False):
        """
        광파워 생성 (포트별 프로필 반영)
        """
        profile = self.port_profiles[node_key]
        tx = random.gauss(profile['optical_tx_mean'], profile['optical_noise_std'])
        rx = random.gauss(profile['optical_rx_mean'], profile['optical_noise_std'])
        
        # Trend (점진적 열화) 반영
        if node_key in self.trend_states:
            state = self.trend_states[node_key]
            elapsed_hours = (dt - state['start_time']).total_seconds() / 3600.0
            if elapsed_hours > 0:
                # 시간에 따라 서서히 광파워 감소 (예: 시간당 0.1 dBm 하락)
                drop = elapsed_hours * state['degradation_factor']
                tx -= drop
                rx -= (drop * random.uniform(0.8, 1.2)) # Rx도 같이 나빠지는 경향
                
        if is_anomaly:
            # 급격한 스파이크 (Optical Signal Loss 등)
            tx -= random.uniform(5.0, 15.0)
            rx -= random.uniform(5.0, 15.0)
            
        # 복합 장애 시나리오 강제 주입
        if getattr(self, '_force_both_anomaly', False):
            tx -= random.uniform(15.0, 30.0)
            rx -= random.uniform(15.0, 30.0)
            
        return round(tx, 2), round(rx, 2)

    def trigger_trend_anomaly(self, dt, node_key):
        """점진적 열화 시나리오 시작"""
        if node_key not in self.trend_states:
            self.trend_states[node_key] = {
                'start_time': dt,
                'degradation_factor': random.uniform(0.05, 0.2) # 시간당 0.05 ~ 0.2 dBm 하락
            }

    def reset_trend(self, node_key):
        """장비 교체 등으로 열화 상태 리셋"""
        if node_key in self.trend_states:
            del self.trend_states[node_key]

    def generate_snapshot(self, dt):
        """특정 시점(dt)의 모든 노드 데이터(정상+이상 혼합) 생성"""
        traffic_data = []
        optical_data = []
        
        for n in range(self.config['nodes']):
            ip = self.generate_ip(n)
            for p in range(self.config['ports_per_node']):
                cid = p
                lid = p + 1
                node_key = f"{ip}_{cid}_{lid}"
                
                # 랜덤으로 Trend 시작/리셋 결정
                if random.random() < self.config['anomaly_probabilities']['trend_start']:
                    self.trigger_trend_anomaly(dt, node_key)
                elif random.random() < 0.005: # 아주 낮은 확률로 복구
                    self.reset_trend(node_key)
                
                is_traffic_anomaly = random.random() < self.config['anomaly_probabilities']['spike']
                is_optical_anomaly = random.random() < self.config['anomaly_probabilities']['spike']
                is_both_anomaly = random.random() < self.config.get('anomaly_probabilities', {}).get('spike_both', 0.02)
                
                if is_both_anomaly:
                    self._force_both_anomaly = True
                else:
                    self._force_both_anomaly = False
                
                tx_pkt, rx_pkt, err_pkt = self.get_traffic_value(dt, node_key, is_traffic_anomaly)
                tx_opt, rx_opt = self.get_optical_value(dt, node_key, is_optical_anomaly)
                
                self._force_both_anomaly = False
                
                # Traffic Data (occur_date, ip_addr, cid, lid, signal_type, es, ses, bbe_in_error)
                traffic_data.append((dt, ip, cid, lid, SIGNAL_TYPES['ETH'], tx_pkt, rx_pkt, err_pkt))
                # Optical Data (occur_date, ip_addr, cid, lid, tx_avg_power, rx_avg_power)
                optical_data.append((dt, ip, cid, lid, tx_opt, rx_opt))
                
        return traffic_data, optical_data

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from mysql.connector import Error, pooling
from src.config import DB_CONFIG, SIGNAL_TYPES

# pandas의 SQLAlchemy 권고 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

class DBConnector:
    def __init__(self):
        self.config = DB_CONFIG
        self.pool = None
        self.initialize_pool()
        self._ensure_anomaly_table()

    def initialize_pool(self):
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="ptn_pool", pool_size=5,
                host=self.config['host'], user=self.config['user'],
                password=self.config['password'], database=self.config['database'],
                port=self.config['port']
            )
        except Error as e:
            print(f"[DB] Connection pool error: {e}")

    def get_connection(self):
        try:
            return self.pool.get_connection()
        except:
            return None

    def _ensure_anomaly_table(self):
        """탐지 결과 저장 테이블 자동 생성"""
        conn = self.get_connection()
        if not conn: return
        cursor = conn.cursor()
        create_query = """
            CREATE TABLE IF NOT EXISTS anomaly_detection (
                id INT AUTO_INCREMENT PRIMARY KEY,
                occur_date DATETIME NOT NULL,
                ip_addr VARCHAR(50) NOT NULL,
                cid INT,
                lid INT,
                -- 원본 성능 데이터 (그래프용)
                tx_packet BIGINT DEFAULT 0,
                rx_packet BIGINT DEFAULT 0,
                error_packet BIGINT DEFAULT 0,
                tx_avg_power FLOAT DEFAULT 0.0,
                rx_avg_power FLOAT DEFAULT 0.0,
                -- 트래픽 트랙 상세
                traffic_score FLOAT DEFAULT 0.0,
                traffic_severity FLOAT DEFAULT 0.0,
                traffic_slope FLOAT DEFAULT 0.0,
                traffic_threshold FLOAT DEFAULT 0.0,
                is_traffic_anomaly TINYINT(1) DEFAULT 0,
                -- 광성능 트랙 상세
                optical_score FLOAT DEFAULT 0.0,
                optical_severity FLOAT DEFAULT 0.0,
                optical_slope FLOAT DEFAULT 0.0,
                optical_threshold FLOAT DEFAULT 0.0,
                is_optical_anomaly TINYINT(1) DEFAULT 0,
                -- 통합 결과
                anomaly_score FLOAT DEFAULT 0.0,
                severity FLOAT DEFAULT 0.0,
                slope FLOAT DEFAULT 0.0,
                slope_label VARCHAR(20) DEFAULT 'STABLE',
                threshold FLOAT DEFAULT 0.0,
                is_anomaly TINYINT(1) DEFAULT 0,
                alarm_level INT DEFAULT 0,
                alarm_label VARCHAR(20) DEFAULT 'NORMAL',
                -- 예지 정비
                ttf_minutes FLOAT DEFAULT NULL,
                expected_fatal_time DATETIME DEFAULT NULL,
                -- 상세 정보
                anomaly_reason TEXT,
                detect_time DATETIME,
                UNIQUE KEY uk_anomaly (occur_date, ip_addr, cid, lid),
                INDEX idx_occur_date (occur_date),
                INDEX idx_is_anomaly (is_anomaly),
                INDEX idx_severity (severity)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        try:
            cursor.execute(create_query)
            conn.commit()
        except Error as e:
            print(f"[DB] Table creation error: {e}")
        finally:
            cursor.close()
            conn.close()

    def fetch_traffic(self, start_time, end_time, stop_checker=None):
        """이더넷 트래픽 성능 데이터 조회"""
        hours = pd.date_range(start=start_time, end=end_time, freq='h')
        conn = self.get_connection()
        if not conn: return None
        all_dfs = []
        try:
            for hr in hours:
                # 중지 요청 확인
                if stop_checker and stop_checker():
                    print("[DB] fetch_traffic interrupted by user.")
                    break

                table = f"cowptn_noti_pm_{hr.strftime('%Y_%m_%d_%H')}"
                query = f"""
                    SELECT occur_date, ip_addr, cid, lid,
                           bbe_in_error as error_packet, es as tx_packet, ses as rx_packet
                    FROM {table}
                    WHERE signal_type = {SIGNAL_TYPES['ETH']}
                      AND occur_date BETWEEN '{start_time}' AND '{end_time}'
                """
                cursor = conn.cursor()
                cursor.execute(f"SHOW TABLES LIKE '{table}'")
                if cursor.fetchone():
                    df = pd.read_sql(query, conn)
                    if not df.empty: all_dfs.append(df)
                cursor.close()
            return pd.concat(all_dfs, ignore_index=True) if all_dfs else None
        finally:
            conn.close()

    def fetch_optical(self, start_time, end_time, stop_checker=None):
        """광파워 성능 데이터 조회"""
        hours = pd.date_range(start=start_time, end=end_time, freq='h')
        conn = self.get_connection()
        if not conn: return None
        all_dfs = []
        try:
            for hr in hours:
                # 중지 요청 확인
                if stop_checker and stop_checker():
                    print("[DB] fetch_optical interrupted by user.")
                    break

                table = f"cowptn_noti_pm_optic_power_{hr.strftime('%Y_%m_%d_%H')}"
                query = f"""
                    SELECT occur_date, ip_addr, cid, lid, tx_avg_power, rx_avg_power
                    FROM {table}
                    WHERE occur_date BETWEEN '{start_time}' AND '{end_time}'
                """
                cursor = conn.cursor()
                cursor.execute(f"SHOW TABLES LIKE '{table}'")
                if cursor.fetchone():
                    df = pd.read_sql(query, conn)
                    if not df.empty: all_dfs.append(df)
                cursor.close()
            return pd.concat(all_dfs, ignore_index=True) if all_dfs else None
        finally:
            conn.close()

    def _clean_value(self, val, default=0):
        """NaN, inf 등의 값을 DB에 안전한 None(NULL) 또는 기본값으로 변환"""
        if pd.isna(val) or (isinstance(val, float) and (np.isinf(val) or np.isnan(val))):
            return None if default is None else default
        return val

    def save_results(self, df):
        """탐지 결과 저장 (안전한 값 변환 및 정합성 보장)"""
        if df is None or df.empty: return
        conn = self.get_connection()
        if not conn: return
        cursor = conn.cursor()
        
        # 명시적 컬럼 정의 (DB 스키마와 1:1 매핑)
        cols = [
            'occur_date', 'ip_addr', 'cid', 'lid', 
            'tx_packet', 'rx_packet', 'error_packet', 'tx_avg_power', 'rx_avg_power',
            'traffic_score', 'traffic_severity', 'traffic_slope', 'traffic_threshold', 'is_traffic_anomaly',
            'optical_score', 'optical_severity', 'optical_slope', 'optical_threshold', 'is_optical_anomaly',
            'anomaly_score', 'severity', 'slope', 'slope_label', 'threshold', 'is_anomaly', 
            'alarm_level', 'alarm_label', 'ttf_minutes', 'expected_fatal_time', 
            'anomaly_reason', 'detect_time'
        ]
        
        placeholders = ", ".join(["%s"] * len(cols))
        query = f"INSERT IGNORE INTO anomaly_detection ({', '.join(cols)}) VALUES ({placeholders})"
        
        try:
            data = []
            for _, row in df.iterrows():
                row_data = []
                for col in cols:
                    if col == 'detect_time':
                        row_data.append(datetime.now())
                        continue
                        
                    val = row.get(col)
                    
                    # 타입별 정밀 처리
                    if col == 'ttf_minutes':
                        row_data.append(float(val) if pd.notna(val) else None)
                    elif col in ['occur_date', 'expected_fatal_time']:
                        row_data.append(val if pd.notna(val) else None)
                    elif col in ['ip_addr', 'slope_label', 'alarm_label', 'anomaly_reason']:
                        row_data.append(str(val) if pd.notna(val) else "")
                    elif col in ['tx_packet', 'rx_packet', 'error_packet', 'is_anomaly', 
                               'is_traffic_anomaly', 'is_optical_anomaly', 'cid', 'lid', 'alarm_level']:
                        row_data.append(int(self._clean_value(val, 0)))
                    else: # 점수, 임계치 등 나머지 실수형
                        row_data.append(float(self._clean_value(val, 0.0)))
                
                data.append(tuple(row_data))
                
            cursor.executemany(query, data)
            conn.commit()
            print(f"    [DB] Successfully saved {len(data)} results.")
        except Error as e:
            print(f"    [DB] Save error: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

    def delete_old_data(self, days):
        """보관 주기 지난 데이터 삭제"""
        conn = self.get_connection()
        if not conn: return
        cursor = conn.cursor()
        try:
            cursor.execute(f"DELETE FROM anomaly_detection WHERE occur_date < DATE_SUB(NOW(), INTERVAL {days} DAY)")
            conn.commit()
        finally:
            cursor.close()
            conn.close()

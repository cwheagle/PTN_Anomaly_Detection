import mysql.connector
from mysql.connector import Error, pooling
import pandas as pd
import warnings
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

    def fetch_traffic(self, start_time, end_time):
        """이더넷 트래픽 성능 데이터 조회"""
        hours = pd.date_range(start=start_time, end=end_time, freq='h')
        conn = self.get_connection()
        if not conn: return None
        all_dfs = []
        try:
            for hr in hours:
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

    def fetch_optical(self, start_time, end_time):
        """광파워 성능 데이터 조회"""
        hours = pd.date_range(start=start_time, end=end_time, freq='h')
        conn = self.get_connection()
        if not conn: return None
        all_dfs = []
        try:
            for hr in hours:
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

    def save_results(self, df):
        """탐지 결과 저장"""
        if df is None or df.empty: return
        conn = self.get_connection()
        if not conn: return
        cursor = conn.cursor()
        query = """
            INSERT IGNORE INTO anomaly_detection (
                occur_date, ip_addr, cid, lid, 
                tx_packet, rx_packet, error_packet, tx_avg_power, rx_avg_power,
                traffic_score, traffic_severity, traffic_slope, traffic_threshold, is_traffic_anomaly,
                optical_score, optical_severity, optical_slope, optical_threshold, is_optical_anomaly,
                anomaly_score, severity, slope, slope_label, threshold, is_anomaly, 
                alarm_level, alarm_label, ttf_minutes, expected_fatal_time, 
                anomaly_reason, detect_time
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()) 
        """
        try:
            data = []
            for _, row in df.iterrows():
                # NaN 처리
                ttf = float(row.get('ttf_minutes')) if pd.notnull(row.get('ttf_minutes')) else None
                fatal_time = row.get('expected_fatal_time') if pd.notnull(row.get('expected_fatal_time')) else None
                
                data.append((
                    row['occur_date'], row['ip_addr'], row['cid'], row['lid'],
                    int(row.get('tx_packet', 0)), int(row.get('rx_packet', 0)), int(row.get('error_packet', 0)),
                    float(row.get('tx_avg_power', 0)), float(row.get('rx_avg_power', 0)),
                    float(row.get('traffic_score', 0)), float(row.get('traffic_severity', 0)), 
                    float(row.get('traffic_slope', 0)), float(row.get('traffic_threshold', 0)), int(row.get('is_traffic_anomaly', 0)),
                    float(row.get('optical_score', 0)), float(row.get('optical_severity', 0)), 
                    float(row.get('optical_slope', 0)), float(row.get('optical_threshold', 0)), int(row.get('is_optical_anomaly', 0)),
                    float(row.get('anomaly_score', 0)), float(row.get('severity', 0)), 
                    float(row.get('slope', 0)), row.get('slope_label', 'STABLE'),
                    float(row.get('threshold', 0)), int(row.get('is_anomaly', 0)),
                    int(row.get('alarm_level', 0)), row.get('alarm_label', 'NORMAL'),
                    ttf, fatal_time, row.get('anomaly_reason', '')
                ))
            cursor.executemany(query, data)
            conn.commit()
        except Error as e:
            print(f"[DB] Save error: {e}")
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

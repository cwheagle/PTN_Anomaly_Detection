import mysql.connector
from mysql.connector import Error, pooling
import pandas as pd
from src.config import DB_CONFIG, SIGNAL_TYPES

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
            CREATE TABLE IF NOT EXISTS anomaly_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                occur_date DATETIME NOT NULL,
                ip_addr VARCHAR(50) NOT NULL,
                cid INT,
                lid INT,
                anomaly_score FLOAT,
                threshold FLOAT,
                is_anomaly TINYINT(1),
                anomaly_reason TEXT,
                detect_time DATETIME,
                UNIQUE KEY uk_anomaly (occur_date, ip_addr, cid, lid),
                INDEX idx_occur_date (occur_date),
                INDEX idx_is_anomaly (is_anomaly)
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
            INSERT IGNORE INTO anomaly_results (
                occur_date, ip_addr, cid, lid, 
                anomaly_score, threshold, is_anomaly, anomaly_reason, detect_time
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        try:
            data = [(row['occur_date'], row['ip_addr'], row['cid'], row['lid'], 
                     float(row['anomaly_score']), float(row.get('threshold', 0)), 
                     int(row['is_anomaly']), row.get('anomaly_reason', '')) 
                    for _, row in df.iterrows()]
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
            cursor.execute(f"DELETE FROM anomaly_results WHERE occur_date < DATE_SUB(NOW(), INTERVAL {days} DAY)")
            conn.commit()
        finally:
            cursor.close()
            conn.close()

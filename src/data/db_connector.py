import mysql.connector
from mysql.connector import Error, pooling
import pandas as pd
import time
from src.config import DB_CONFIG

class DBConnector:
    def __init__(self):
        self.config = DB_CONFIG
        self.pool = None
        self._initialize_pool()

    def _initialize_pool(self):
        """MySQL Connection Pool을 초기화합니다."""
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="ptn_pool",
                pool_size=5,
                host=self.config['host'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                port=self.config['port']
            )
            print("Connection pool initialized.")
        except Error as e:
            print(f"Failed to initialize connection pool: {e}")

    def get_connection(self):
        """풀에서 커넥션을 가져옵니다. 실패 시 재시도합니다."""
        retries = 3
        while retries > 0:
            try:
                if self.pool:
                    return self.pool.get_connection()
            except Error:
                retries -= 1
                time.sleep(1)
        return None

    def fetch_performance_data(self, minutes=180):
        """최근 n분 동안의 성능 데이터를 조회합니다."""
        conn = self.get_connection()
        if not conn:
            return None

        query = f"""
            SELECT 
                collect_time, equipment_id, port_id,
                tx_error, rx_error, tx_bps, rx_bps, tx_pps, rx_pps, tx_power, rx_power
            FROM performance_metrics
            WHERE collect_time >= NOW() - INTERVAL {minutes} MINUTE
            ORDER BY equipment_id, port_id, collect_time ASC
        """
        
        try:
            # pd.read_sql은 커넥션 객체를 직접 받음
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        finally:
            if conn and conn.is_connected():
                conn.close()

    def save_anomaly_results(self, df_results):
        """이상 탐지 결과(RCA 포함)를 DB에 저장합니다."""
        if df_results is None or df_results.empty:
            return

        conn = self.get_connection()
        if not conn:
            return

        cursor = conn.cursor()
        # anomaly_reason 칼럼이 추가된 쿼리
        insert_query = """
            INSERT INTO anomaly_results (
                collect_time, equipment_id, port_id, 
                anomaly_score, threshold, is_anomaly, anomaly_reason, detect_time
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """
        
        try:
            data_to_insert = []
            for _, row in df_results.iterrows():
                data_to_insert.append((
                    row['collect_time'],
                    row['equipment_id'],
                    row['port_id'],
                    float(row['anomaly_score']),
                    float(row['threshold']),
                    int(row['is_anomaly']),
                    row.get('anomaly_reason', '') # RCA 결과 포함
                ))
            
            cursor.executemany(insert_query, data_to_insert)
            conn.commit()
            print(f"Successfully saved {len(data_to_insert)} results to DB.")
        except Error as e:
            print(f"Error saving results: {e}")
            conn.rollback()
        finally:
            cursor.close()
            if conn and conn.is_connected():
                conn.close()

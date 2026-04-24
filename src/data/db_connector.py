import mysql.connector
from mysql.connector import Error
import pandas as pd
from src.config import DB_CONFIG

class DBConnector:
    def __init__(self):
        self.config = DB_CONFIG
        self.connection = None

    def connect(self):
        """데이터베이스 연결을 수행합니다."""
        try:
            self.connection = mysql.connector.connect(
                host=self.config['host'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                port=self.config['port']
            )
            if self.connection.is_connected():
                return True
        except Error as e:
            print(f"Error while connecting to MySQL: {e}")
            return False

    def fetch_performance_data(self, minutes=180):
        """
        최근 n분 동안의 성능 및 파워 데이터를 조회합니다.
        window_size를 고려하여 기본 180분(3시간) 데이터를 가져옵니다.
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None

        query = f"""
            SELECT 
                collect_time, equipment_id, port_id,
                tx_packet, rx_packet, tx_error, rx_error,
                tx_bps, rx_bps, tx_pps, rx_pps,
                tx_power, rx_power
            FROM performance_metrics
            WHERE collect_time >= NOW() - INTERVAL {minutes} MINUTE
            ORDER BY equipment_id, port_id, collect_time ASC
        """
        
        try:
            df = pd.read_sql(query, self.connection)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def save_anomaly_results(self, df_results):
        """
        이상 탐지 결과를 anomaly_results 테이블에 저장합니다.
        """
        if df_results is None or df_results.empty:
            return

        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return

        cursor = self.connection.cursor()
        insert_query = """
            INSERT INTO anomaly_results (
                collect_time, equipment_id, port_id, 
                anomaly_score, threshold, is_anomaly, detect_time
            ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
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
                    int(row['is_anomaly'])
                ))
            
            cursor.executemany(insert_query, data_to_insert)
            self.connection.commit()
            print(f"Successfully saved {len(data_to_insert)} results to DB.")
        except Error as e:
            print(f"Error saving results: {e}")
            self.connection.rollback()
        finally:
            cursor.close()

    def disconnect(self):
        """데이터베이스 연결을 해제합니다."""
        if self.connection and self.connection.is_connected():
            self.connection.close()

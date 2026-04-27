import mysql.connector
from mysql.connector import Error, pooling
import pandas as pd
import time
from src.config import DB_CONFIG

class DBConnector:
    def __init__(self):
        self.config = DB_CONFIG
        self.pool = None
        self.initialize_pool()

    def initialize_pool(self):
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

    def fetch_table_data(self, conn, table_name, query):
        """테이블 존재 여부 확인 후 데이터를 조회합니다."""
        cursor = conn.cursor()
        try:
            cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            if cursor.fetchone():
                return pd.read_sql(query, conn)
        except Exception as e:
            print(f"Query Error on {table_name}: {e}")
        finally:
            cursor.close()
        return pd.DataFrame()

    def fetch_real_data(self, start_time, end_time):
        """
        운영 DB에서 PM 및 광파워 데이터를 조회하고 병합합니다.
        """
        from src.config import SIGNAL_TYPES
        hours = pd.date_range(start=start_time, end=end_time, freq='h')
        all_dfs = []
        conn = self.get_connection()
        if not conn: return None

        try:
            for hr in hours:
                suffix = hr.strftime("%Y_%m_%d_%H")
                pm_table = f"cowptn_noti_pm_{suffix}"
                opt_table = f"cowptn_noti_pm_optic_power_{suffix}"
                
                # 1. PM 데이터 조회
                q_pm = f"""
                    SELECT occur_date, ip_addr, cid, lid,
                           bbe_in_error as error_packet, es as tx_packet, ses as rx_packet
                    FROM {pm_table}
                    WHERE signal_type = {SIGNAL_TYPES['ETH']}
                      AND occur_date BETWEEN '{start_time}' AND '{end_time}'
                """
                df_pm = self.fetch_table_data(conn, pm_table, q_pm)

                # 2. 광파워 데이터 조회
                q_opt = f"""
                    SELECT occur_date, ip_addr, cid, lid, tx_avg_power, rx_avg_power
                    FROM {opt_table}
                    WHERE occur_date BETWEEN '{start_time}' AND '{end_time}'
                """
                df_opt = self.fetch_table_data(conn, opt_table, q_opt)

                # 3. 데이터 병합 (Outer Merge)
                if df_pm.empty and df_opt.empty:
                    continue
                
                # 시간 정규화 및 병합
                if not df_pm.empty:
                    df_pm['occur_date_key'] = pd.to_datetime(df_pm['occur_date']).dt.round('1min')
                if not df_opt.empty:
                    df_opt['occur_date_key'] = pd.to_datetime(df_opt['occur_date']).dt.round('1min')

                if not df_pm.empty and not df_opt.empty:
                    df_merged = pd.merge(
                        df_pm, df_opt, on=['occur_date_key', 'ip_addr', 'cid', 'lid'], 
                        how='outer', suffixes=('', '_opt_raw')
                    )
                    df_merged['occur_date'] = df_merged['occur_date_key']
                    df_merged = df_merged.drop(columns=['occur_date_key', 'occur_date_opt_raw'], errors='ignore')
                else:
                    df_merged = df_pm if not df_pm.empty else df_opt
                    df_merged['occur_date'] = df_merged['occur_date_key']
                    df_merged = df_merged.drop(columns=['occur_date_key'], errors='ignore')

                all_dfs.append(df_merged)

            if not all_dfs: return None
            return pd.concat(all_dfs, ignore_index=True).sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])
        finally:
            if conn and conn.is_connected(): conn.close()

    def save_anomaly_results(self, df_results):
        """탐지 결과를 DB에 저장합니다."""
        if df_results is None or df_results.empty: return
        conn = self.get_connection()
        if not conn: return
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO anomaly_results (
                occur_date, ip_addr, cid, lid, 
                anomaly_score, threshold, is_anomaly, anomaly_reason, detect_time
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        try:
            data = [(row['occur_date'], row['ip_addr'], row['cid'], row['lid'], 
                     float(row['anomaly_score']), float(row['threshold']), 
                     int(row['is_anomaly']), row.get('anomaly_reason', '')) 
                    for _, row in df_results.iterrows()]
            cursor.executemany(insert_query, data)
            conn.commit()
            print(f"Successfully saved {len(data)} results to DB.")
        except Error as e:
            print(f"Error saving results: {e}")
            conn.rollback()
        finally:
            cursor.close()
            if conn and conn.is_connected(): conn.close()

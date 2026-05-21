import mysql.connector
from mysql.connector import Error
from config_sim import DB_CONFIG_SIM

class SimulatorDBManager:
    def __init__(self):
        self.config = DB_CONFIG_SIM
        self._ensure_database()
        
    def _ensure_database(self):
        try:
            temp_config = self.config.copy()
            db_name = temp_config.pop('database')
            conn = mysql.connector.connect(**temp_config)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
            conn.commit()
            cursor.close()
            conn.close()
        except Error as e:
            print(f"[SimulatorDB] Could not auto-create database: {e}")
        
    def get_connection(self):
        try:
            return mysql.connector.connect(**self.config)
        except Error as e:
            print(f"[SimulatorDB] Connection error: {e}")
            return None

    def ensure_traffic_table(self, dt):
        """특정 시간의 트래픽 테이블을 생성합니다."""
        table_name = f"cowptn_noti_pm_{dt.strftime('%Y_%m_%d_%H')}"
        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                occur_date DATETIME NOT NULL,
                ip_addr VARCHAR(50) NOT NULL,
                cid INT,
                lid INT,
                signal_type INT,
                es BIGINT DEFAULT 0,          -- tx_packet
                ses BIGINT DEFAULT 0,         -- rx_packet
                bbe_in_error BIGINT DEFAULT 0, -- error_packet
                INDEX idx_occur_date (occur_date),
                INDEX idx_ip_addr (ip_addr)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        self._execute_ddl(query)
        return table_name

    def ensure_optical_table(self, dt):
        """특정 시간의 광파워 테이블을 생성합니다."""
        table_name = f"cowptn_noti_pm_optic_power_{dt.strftime('%Y_%m_%d_%H')}"
        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                occur_date DATETIME NOT NULL,
                ip_addr VARCHAR(50) NOT NULL,
                cid INT,
                lid INT,
                tx_avg_power FLOAT DEFAULT 0.0,
                rx_avg_power FLOAT DEFAULT 0.0,
                INDEX idx_occur_date (occur_date),
                INDEX idx_ip_addr (ip_addr)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        self._execute_ddl(query)
        return table_name

    def _execute_ddl(self, query):
        conn = self.get_connection()
        if not conn: return
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
        except Error as e:
            print(f"[SimulatorDB] DDL execution error: {e}")
        finally:
            cursor.close()
            conn.close()

    def insert_traffic(self, table_name, data):
        """트래픽 데이터 일괄 삽입"""
        conn = self.get_connection()
        if not conn: return
        query = f"""
            INSERT INTO {table_name} (occur_date, ip_addr, cid, lid, signal_type, es, ses, bbe_in_error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            cursor = conn.cursor()
            cursor.executemany(query, data)
            conn.commit()
        except Error as e:
            print(f"[SimulatorDB] Insert traffic error: {e}")
        finally:
            cursor.close()
            conn.close()

    def insert_optical(self, table_name, data):
        """광파워 데이터 일괄 삽입"""
        conn = self.get_connection()
        if not conn: return
        query = f"""
            INSERT INTO {table_name} (occur_date, ip_addr, cid, lid, tx_avg_power, rx_avg_power)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        try:
            cursor = conn.cursor()
            cursor.executemany(query, data)
            conn.commit()
        except Error as e:
            print(f"[SimulatorDB] Insert optical error: {e}")
        finally:
            cursor.close()
            conn.close()

    def check_data_exists(self, table_name, occur_date):
        """특정 시간의 데이터가 이미 존재하는지 확인합니다."""
        conn = self.get_connection()
        if not conn: return False
        try:
            cursor = conn.cursor()
            cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            if not cursor.fetchone():
                return False
            cursor.execute(f"SELECT 1 FROM {table_name} WHERE occur_date = %s LIMIT 1", (occur_date,))
            result = cursor.fetchone()
            return result is not None
        except Error as e:
            return False
        finally:
            if conn:
                cursor.close()
                conn.close()

import os
import json
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from src.data.db_connector import DBConnector
from src.config import PATHS

class DriftMonitor:
    def __init__(self, drift_factor=1.5):
        self.db = DBConnector()
        self.drift_factor = drift_factor
        
    def _get_active_meta_path(self, ft: str):
        p = PATHS.get(ft, {})
        if not p: return ""
        base_model = p.get('model', '')
        if not base_model: return ""
        
        model_dir = os.path.dirname(base_model)
        registry_path = os.path.join(model_dir, f"{ft}_registry.json")
        actual_meta = base_model.replace('.pth', '.json')
        
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                active_ver = registry.get("active_version")
                if active_ver:
                    for v in registry.get("versions", []):
                        if v["version"] == active_ver and "config_path" in v:
                            actual_meta = os.path.join(model_dir, v["config_path"])
                            break
            except: pass
        return actual_meta

    def _get_baseline_mse(self, feature_type):
        """저장된 메타데이터에서 기준 MSE(val_loss 또는 threshold/10)를 가져옴"""
        return self._get_single_baseline(feature_type)

    def _get_single_baseline(self, ft):
        meta_path = self._get_active_meta_path(ft)
        if not meta_path or not os.path.exists(meta_path):
            return None
            
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        val_loss = meta.get("final_val_loss")
        if val_loss is not None and val_loss > 0:
            return val_loss
        
        # val_loss가 없을 경우(단순 테스트 등) 임계치의 특정 비율을 baseline으로 간주 (Heuristic)
        return max(meta.get("threshold", 0.1) / 5.0, 1e-6)

    def check_drift(self):
        """DB를 조회하여 최근 24시간 동안 Data Drift가 발생했는지 검사"""
        conn = self.db.get_connection()
        if not conn:
            return {"status": "error", "message": "DB Connection failed"}
            
        now = datetime.now()
        yesterday_str = (now - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        
        result = {
            "traffic": {"status": "normal", "mean_mse": 0.0, "baseline_mse": 0.0, "drift_ratio": 0.0},
            "optical": {"status": "normal", "mean_mse": 0.0, "baseline_mse": 0.0, "drift_ratio": 0.0},
            "drift_detected": False,
            "drifted_tracks": []
        }
        
        try:
            # 최근 24시간 평균 MSE 추출
            query = f"""
                SELECT 
                    AVG(traffic_score) as avg_traffic_mse,
                    AVG(optical_score) as avg_optical_mse,
                    COUNT(*) as sample_count
                FROM anomaly_detection
                WHERE occur_date >= '{yesterday_str}'
            """
            df = pd.read_sql(query, conn)
            
            if df.empty or df['sample_count'].iloc[0] == 0:
                result["message"] = "No recent data available for drift check."
                return result 
                
            avg_traffic = df['avg_traffic_mse'].iloc[0] or 0.0
            avg_optical = df['avg_optical_mse'].iloc[0] or 0.0
            
            # 1. Traffic 평가
            baseline_t = self._get_baseline_mse("traffic")
            if baseline_t:
                ratio_t = float(avg_traffic) / baseline_t
                result["traffic"].update({
                    "mean_mse": float(avg_traffic), 
                    "baseline_mse": float(baseline_t), 
                    "drift_ratio": ratio_t
                })
                if ratio_t > self.drift_factor:
                    result["traffic"]["status"] = "drifted"
                    result["drift_detected"] = True
                    result["drifted_tracks"].append("traffic")
                    
            # 2. Optical 평가
            baseline_o = self._get_baseline_mse("optical")
            if baseline_o:
                ratio_o = float(avg_optical) / baseline_o
                result["optical"].update({
                    "mean_mse": float(avg_optical), 
                    "baseline_mse": float(baseline_o), 
                    "drift_ratio": ratio_o
                })
                if ratio_o > self.drift_factor:
                    result["optical"]["status"] = "drifted"
                    result["drift_detected"] = True
                    result["drifted_tracks"].append("optical")
                    
            result["timestamp"] = now.strftime('%Y-%m-%d %H:%M:%S')
            return result
            
        except Exception as e:
            print(f"[!] Drift Check Error: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            conn.close()

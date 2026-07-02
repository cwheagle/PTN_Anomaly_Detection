import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# 프로젝트 루트 디렉토리를 path에 추가 (scripts 폴더에서 실행 시 패키지 인식 목적)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
os.chdir(root_dir)

from src.data.db_connector import DBConnector
from src.pipeline.inference import AnomalyDetector

def evaluate():
    print("="*50)
    print("   Phase 10: TTF-Aware Offline Model Evaluation   ")
    print("="*50)
    
    # 1. 정답지(Ground Truth) 로드 (시뮬레이터가 만든 경로)
    gt_path = os.path.join(root_dir, 'tools', 'simulator', 'data', 'eval_dataset.csv')
    if not os.path.exists(gt_path):
        print(f"[!] Ground Truth file not found at {gt_path}. Please run generate_rca_history.py first.")
        return
        
    gt_df = pd.read_csv(gt_path)
    gt_df['start_time'] = pd.to_datetime(gt_df['start_time'])
    gt_df['failure_time'] = pd.to_datetime(gt_df['failure_time'])
    print(f"[*] Loaded Ground Truth: {len(gt_df)} anomaly scenarios found.")
    
    # 2. DB에서 전체 평가 데이터 로드
    db = DBConnector()
    start_time = gt_df['start_time'].min() - timedelta(hours=1)
    end_time = gt_df['failure_time'].max() + timedelta(hours=1)
    
    start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"[*] Fetching historical DB data from {start_str} to {end_str}...")
    df_t = db.fetch_traffic(start_str, end_str)
    df_o = db.fetch_optical(start_str, end_str)
    
    if df_t.empty and df_o.empty:
        print("[!] DB returned empty datasets. Ensure you have generated history data.")
        return
        
    # 3. 모델 추론 진행
    print(f"[*] Running AI Inference on {len(df_t)} traffic records and {len(df_o)} optical records...")
    detector = AnomalyDetector()
    
    # latest_only=False 를 주어 과거 전체에 대한 추론 수행
    results = detector.detect(df_traffic=df_t, df_optical=df_o, latest_only=False)
    
    if results is None or results.empty:
        print("[!] No results returned from inference.")
        return
        
    print(f"[*] Inference complete. Total prediction points: {len(results)}")
    
    # 4. TTF-Aware 평가 로직 (채점)
    print("\n[*] Evaluating TTF-Aware Accuracy...")
    
    alarms = results[results['is_anomaly'] == True].copy()
    
    tp_count = 0
    fp_count = 0
    matched_gt = set()
    ttf_errors = []
    
    no_ttf_alarms = 0
    
    alarms['occur_date'] = pd.to_datetime(alarms['occur_date'])
    alarms['expected_fatal_time'] = pd.to_datetime(alarms['expected_fatal_time'])
    
    for _, alarm in alarms.iterrows():
        ip = alarm['ip_addr']
        cid = alarm['cid']
        lid = alarm['lid']
        alarm_time = alarm['occur_date']
        pred_fatal = alarm['expected_fatal_time']
        
        if pd.isna(pred_fatal):
            # 예상 장애 시간이 없다는 건 (딥러닝은 이상이라 판단했으나, RCA에서 악화 추세가 없다고 본 경우)
            # 예지 알람으로는 부적합하므로 TTF 평가에서는 스킵 (사용자님 의견 수용)
            no_ttf_alarms += 1
            continue
            
        # 매칭되는 정답지 검색: 같은 포트이면서, 장애 진행 시간(start~failure) 내에 발생한 알람
        match = gt_df[(gt_df['ip_addr'] == ip) & 
                      (gt_df['cid'] == cid) & 
                      (gt_df['lid'] == lid) & 
                      (gt_df['start_time'] <= alarm_time) & 
                      (gt_df['failure_time'] >= alarm_time)]
                      
        if not match.empty:
            # True Positive (정답)
            gt = match.iloc[0]
            matched_gt.add(gt.name) # 장애 시나리오 감지 성공 기록
            
            # 예측한 잔여 시간(TTF)과 실제 장애 시간의 오차(MAE) 계산
            actual_fatal = gt['failure_time']
            error_minutes = abs((pred_fatal - actual_fatal).total_seconds()) / 60.0
            ttf_errors.append(error_minutes)
            tp_count += 1
        else:
            # False Positive (오탐 - 정상이거나 장애 범위 밖인데 알람 띄움)
            fp_count += 1
            
    # False Negative (미탐 - 장애가 발생했는데 단 한 번도 알람을 띄우지 못한 시나리오 수)
    fn_count = len(gt_df) - len(matched_gt)
    
    # 5. 리포트 산출
    # Precision: 알람 중 진짜 장애의 비율
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    
    # Scenario Recall: 전체 장애 시나리오 중 하나라도 알람을 띄우는 데 성공한 비율
    recall = len(matched_gt) / len(gt_df) if len(gt_df) > 0 else 0
    
    # TTF-Aware F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # MAE (Mean Absolute Error)
    mae = sum(ttf_errors) / len(ttf_errors) if ttf_errors else 0
    
    print("\n" + "="*50)
    print("    [Phase 10] TTF-Aware Evaluation Report    ")
    print("="*50)
    print(f"[*] Total Evaluated Scenarios : {len(gt_df)}")
    print(f"[*] Total AI Alarms Fired     : {len(alarms)}")
    print(f"   - Valid TTF Predicted    : {len(alarms) - no_ttf_alarms}")
    print(f"   - No TTF (Flat Trend)    : {no_ttf_alarms} (Ignored)")
    print("-" * 50)
    print(f"[*] True Positives (Valid)    : {tp_count}")
    print(f"[*] False Positives (False)   : {fp_count}")
    print(f"[*] False Negatives (Missed)  : {fn_count}")
    print("-" * 50)
    print(f"[+] Precision : {precision:.4f} (정밀도)")
    print(f"[+] Recall    : {recall:.4f} (재현율, 탐지 성공률)")
    print(f"[*] F1-Score  : {f1:.4f} (TTF-Aware 종합 점수)")
    print(f"[~] TTF MAE    : {mae:.2f} minutes (예측 잔여 수명 평균 오차)")
    print("="*50)
    print("평가 완료. 결과를 확인해 주세요!")
    
if __name__ == "__main__":
    evaluate()

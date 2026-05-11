import sys
import os
import asyncio
import json
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Body, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np

# 프로젝트 루트 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.pipeline.scheduler import PTNAnomalyScheduler
from src.data.db_connector import DBConnector

# 글로벌 객체
scheduler_instance = None
db = DBConnector()
event_queues = set() # SSE 클라이언트들을 위한 큐 집합

async def broadcast_alarm(alarm_data: dict):
    """모든 연결된 SSE 클라이언트에게 알람 전송"""
    if not event_queues:
        return
    
    message = json.dumps(alarm_data)
    for queue in event_queues:
        await queue.put(message)

def alarm_callback(anomalies_df):
    """스케줄러에서 호출할 콜백 함수: Critical 알람 필터링 및 브로드캐스트"""
    criticals = anomalies_df[anomalies_df['alarm_label'] == 'CRITICAL']
    if not criticals.empty:
        loop = asyncio.get_event_loop()
        for _, row in criticals.iterrows():
            alarm_info = {
                "event_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "ip_addr": row['ip_addr'],
                "slot_id": int(row['cid']),
                "port_id": int(row['lid']),
                "severity": row['alarm_label'],
                "reason": row['anomaly_reason']
            }
            # 동기 환경에서 비동기 함수 호출 (스케줄러가 동기로 동작할 경우를 대비)
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcast_alarm(alarm_info), loop)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 생명주기 관리"""
    global scheduler_instance
    print("[*] PTN Anomaly Detection Service Initializing...")
    
    # 스케줄러 객체 생성 및 가동
    scheduler_instance = PTNAnomalyScheduler()
    # 런타임에 콜백 주입
    if hasattr(scheduler_instance, 'set_callback'):
        scheduler_instance.set_callback(alarm_callback)
    
    scheduler_instance.start()
    
    yield
    
    print("[*] Service shutting down...")
    if scheduler_instance:
        scheduler_instance.scheduler.shutdown()

app = FastAPI(
    title="PTN Anomaly Detection API",
    description="REST API for PTN Anomaly Detection & Predictive Maintenance",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"service": "PTN Anomaly Detection API", "status": "online"}

@app.get("/api/anomalies")
async def get_anomalies(
    severity_min: int = Query(0, description="Minimum alarm level (0-3)"),
    severity_max: int = Query(3, description="Maximum alarm level (0-3)"),
    rising_only: bool = Query(False, description="Whether to filter only rising trends")
):
    """이상탐지 현황 조회 (필터 옵션에 따른 가변 조회)"""
    conn = db.get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        # 기본 조건: 가장 최신 배치의 데이터
        where_clauses = ["occur_date = (SELECT MAX(occur_date) FROM anomaly_detection)"]
        
        # 등급 필터링
        where_clauses.append(f"alarm_level BETWEEN {severity_min} AND {severity_max}")
        
        # 추세 필터링
        if rising_only:
            where_clauses.append("slope_label = 'RISING'")
            
        where_query = " AND ".join(where_clauses)
        
        query = f"""
            SELECT occur_date, ip_addr, cid as slot_id, lid as port_id, 
                   severity, alarm_level, alarm_label, slope, slope_label, 
                   ttf_minutes, expected_fatal_time, anomaly_reason
            FROM anomaly_detection
            WHERE {where_query}
            ORDER BY severity DESC, slope DESC
        """
        df = pd.read_sql(query, conn)
        
        # 날짜 컬럼들을 읽기 좋은 문자열 형식으로 변환 ('T' 제거)
        if not df.empty:
            for col in ['occur_date', 'expected_fatal_time']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df.replace({np.nan: None}).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/stream/alarms")
async def stream_alarms(request: Request):
    """실시간 알림 SSE 엔드포인트"""
    queue = asyncio.Queue()
    event_queues.add(queue)
    
    async def event_generator():
        try:
            while True:
                # 클라이언트 연결 종료 확인
                if await request.is_disconnected():
                    break
                
                data = await queue.get()
                yield f"event: alarm\ndata: {data}\n\n"
        finally:
            event_queues.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """스케줄러 현재 상태 조회"""
    if not scheduler_instance:
        return {"status": "stopped", "next_run_time": None}
        
    # APScheduler states: 0: STOPPED, 1: RUNNING, 2: PAUSED
    state = scheduler_instance.scheduler.state
    status_str = "running" if state == 1 else "stopped"
    
    next_run = None
    if status_str == "running":
        jobs = scheduler_instance.scheduler.get_jobs()
        if jobs and jobs[0].next_run_time:
            next_run = jobs[0].next_run_time.strftime('%Y-%m-%d %H:%M:%S')
            
    return {"status": status_str, "next_run_time": next_run}

@app.post("/api/scheduler/status")
async def control_scheduler(data: dict = Body(...)):
    """스케줄러 제어"""
    action = data.get("action")
    if not scheduler_instance:
        raise HTTPException(status_code=500, detail="Scheduler not initialized")
        
    if action == "stop":
        scheduler_instance.scheduler.pause()
    elif action == "start":
        # Fresh Start (즉시 실행 및 15분 주기 리셋)
        scheduler_instance.restart()
        if scheduler_instance.scheduler.state == 2:
            scheduler_instance.scheduler.resume()
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
        
    # 제어 후 최신 상태 즉시 반환
    return await get_scheduler_status()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

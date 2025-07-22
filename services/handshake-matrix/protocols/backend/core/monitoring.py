# backend/core/monitoring.py
from fastapi import FastAPI, Response
import psutil
import time
from typing import Dict, Any

app = FastAPI()

class MetricsCollector:
    def __init__(self):
        self.request_count = 0
        self.start_time = time.time()
        
    def collect_metrics(self) -> Dict[str, Any]:
        return {
            "requests_total": self.request_count,
            "uptime_seconds": time.time() - self.start_time,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }

metrics = MetricsCollector()

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    metrics.request_count += 1
    return response

@app.get("/health")
async def health_check():
    metrics_data = metrics.collect_metrics()
    if metrics_data["cpu_percent"] > 90 or metrics_data["memory_percent"] > 90:
        return Response(status_code=503)
    return {"status": "healthy", "metrics": metrics_data}
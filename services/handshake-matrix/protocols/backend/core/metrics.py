# backend/core/metrics.py
from typing import Dict, Any
from datetime import datetime, timedelta
import psutil
import asyncio
from collections import deque
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, history_minutes: int = 60):
        self.system_metrics = deque(maxlen=history_minutes * 60)  # 1 hour of per-second metrics
        self.business_metrics = {}
        self.last_collection = datetime.utcnow()
        
    async def collect_system_metrics(self):
        """Collect system-level metrics"""
        while True:
            try:
                metrics = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage_percent': psutil.disk_usage('/').percent,
                    'open_files': len(psutil.Process().open_files()),
                    'connections': len(psutil.Process().connections())
                }
                self.system_metrics.append(metrics)
                await asyncio.sleep(1)  # Collect every second
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {str(e)}")
                await asyncio.sleep(5)  # Back off on error

    def update_business_metrics(self, campaign_id: str, metrics: Dict[str, float]):
        """Update business metrics for a campaign"""
        self.business_metrics[campaign_id] = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics
        }

    def get_metrics_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        now = datetime.utcnow()
        
        # Calculate system metrics averages
        if self.system_metrics:
            avg_metrics = {
                'cpu_percent': sum(m['cpu_percent'] for m in self.system_metrics) / len(self.system_metrics),
                'memory_percent': sum(m['memory_percent'] for m in self.system_metrics) / len(self.system_metrics)
            }
        else:
            avg_metrics = {'cpu_percent': 0, 'memory_percent': 0}

        return {
            'timestamp': now.isoformat(),
            'system': {
                'current': dict(self.system_metrics[-1]) if self.system_metrics else {},
                'averages': avg_metrics
            },
            'business': self.business_metrics,
            'health_status': self._calculate_health_status(avg_metrics)
        }

    def _calculate_health_status(self, avg_metrics: Dict[str, float]) -> str:
        """Calculate overall health status based on metrics"""
        if avg_metrics['cpu_percent'] > 90 or avg_metrics['memory_percent'] > 90:
            return 'critical'
        elif avg_metrics['cpu_percent'] > 75 or avg_metrics['memory_percent'] > 75:
            return 'warning'
        return 'healthy'
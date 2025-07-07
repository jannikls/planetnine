#!/usr/bin/env python
"""
Progress tracking and error recovery system for long-running Planet Nine searches.
Provides real-time monitoring, checkpointing, and recovery capabilities.
"""

import time
import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import psutil
import signal
import sys
from loguru import logger
import pickle

from src.config import RESULTS_DIR


@dataclass
class ProgressCheckpoint:
    """Checkpoint data for recovery."""
    timestamp: datetime
    completed_regions: List[str]
    failed_regions: List[str]
    current_region: Optional[str]
    total_candidates: int
    processing_time: float
    memory_usage_mb: float
    cpu_percent: float


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_free_gb: float
    active_processes: int


class ProgressTracker:
    """Track progress of large-scale searches with recovery capabilities."""
    
    def __init__(self, search_id: str, checkpoint_interval: int = 300):
        self.search_id = search_id
        self.checkpoint_interval = checkpoint_interval  # seconds
        self.start_time = time.time()
        
        # Setup directories
        self.progress_dir = RESULTS_DIR / "progress_tracking" / search_id
        self.progress_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize tracking files
        self.progress_db = self.progress_dir / "progress.db"
        self.checkpoint_file = self.progress_dir / "latest_checkpoint.pkl"
        self.metrics_file = self.progress_dir / "system_metrics.json"
        
        # Initialize database
        self._init_progress_database()
        
        # Tracking state
        self.completed_regions = set()
        self.failed_regions = set()
        self.current_region = None
        self.total_candidates = 0
        self.is_running = False
        self.metrics_thread = None
        self.checkpoint_thread = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _init_progress_database(self):
        """Initialize progress tracking database."""
        conn = sqlite3.connect(self.progress_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                search_id TEXT,
                event_type TEXT,
                region_id TEXT,
                message TEXT,
                candidates_found INTEGER,
                processing_time REAL,
                memory_mb REAL,
                cpu_percent REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                search_id TEXT,
                cpu_percent REAL,
                memory_percent REAL,
                memory_used_gb REAL,
                disk_free_gb REAL,
                active_processes INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                search_id TEXT,
                region_id TEXT,
                error_type TEXT,
                error_message TEXT,
                stack_trace TEXT,
                recovery_attempted BOOLEAN DEFAULT FALSE
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start_tracking(self):
        """Start progress tracking and monitoring."""
        logger.info(f"Starting progress tracking for search {self.search_id}")
        self.is_running = True
        
        # Log search start
        self._log_event("SEARCH_START", message="Large-scale search initiated")
        
        # Start background threads
        self.metrics_thread = threading.Thread(target=self._monitor_system_metrics, daemon=True)
        self.checkpoint_thread = threading.Thread(target=self._periodic_checkpoint, daemon=True)
        
        self.metrics_thread.start()
        self.checkpoint_thread.start()
        
    def stop_tracking(self):
        """Stop progress tracking."""
        logger.info("Stopping progress tracking")
        self.is_running = False
        
        # Save final checkpoint
        self._save_checkpoint()
        
        # Log search completion
        total_time = time.time() - self.start_time
        self._log_event("SEARCH_END", 
                       message=f"Search completed. Total time: {total_time/3600:.1f}h")
        
    def update_region_start(self, region_id: str):
        """Update when starting a new region."""
        self.current_region = region_id
        self._log_event("REGION_START", region_id=region_id, 
                       message=f"Started processing region {region_id}")
        
    def update_region_complete(self, region_id: str, candidates_found: int, 
                              processing_time: float):
        """Update when completing a region."""
        self.completed_regions.add(region_id)
        self.total_candidates += candidates_found
        self.current_region = None
        
        self._log_event("REGION_COMPLETE", region_id=region_id,
                       message=f"Completed region {region_id}",
                       candidates_found=candidates_found,
                       processing_time=processing_time)
        
        logger.success(f"Region {region_id} complete: {candidates_found} candidates, "
                      f"{processing_time/60:.1f}m")
        
    def update_region_failed(self, region_id: str, error_message: str, 
                           error_type: str = "PROCESSING_ERROR"):
        """Update when a region fails."""
        self.failed_regions.add(region_id)
        self.current_region = None
        
        self._log_error(region_id, error_type, error_message)
        self._log_event("REGION_FAILED", region_id=region_id,
                       message=f"Failed region {region_id}: {error_message}")
        
        logger.error(f"Region {region_id} failed: {error_message}")
        
    def _log_event(self, event_type: str, region_id: str = None, message: str = "",
                   candidates_found: int = 0, processing_time: float = 0):
        """Log progress event to database."""
        conn = sqlite3.connect(self.progress_db)
        cursor = conn.cursor()
        
        metrics = self._get_current_metrics()
        
        cursor.execute("""
            INSERT INTO search_progress 
            (search_id, event_type, region_id, message, candidates_found, 
             processing_time, memory_mb, cpu_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.search_id, event_type, region_id, message, candidates_found,
            processing_time, metrics['memory_mb'], metrics['cpu_percent']
        ))
        
        conn.commit()
        conn.close()
        
    def _log_error(self, region_id: str, error_type: str, error_message: str,
                   stack_trace: str = ""):
        """Log error to database."""
        conn = sqlite3.connect(self.progress_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO error_log 
            (search_id, region_id, error_type, error_message, stack_trace)
            VALUES (?, ?, ?, ?, ?)
        """, (self.search_id, region_id, error_type, error_message, stack_trace))
        
        conn.commit()
        conn.close()
        
    def _get_current_metrics(self) -> Dict:
        """Get current system metrics."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            return {
                'memory_mb': memory.used / 1024 / 1024,
                'memory_percent': memory.percent,
                'cpu_percent': cpu_percent,
                'disk_free_gb': psutil.disk_usage('/').free / 1024 / 1024 / 1024
            }
        except:
            return {'memory_mb': 0, 'memory_percent': 0, 'cpu_percent': 0, 'disk_free_gb': 0}
        
    def _monitor_system_metrics(self):
        """Monitor system metrics in background thread."""
        while self.is_running:
            try:
                metrics = SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=psutil.cpu_percent(interval=1),
                    memory_percent=psutil.virtual_memory().percent,
                    memory_used_gb=psutil.virtual_memory().used / 1024 / 1024 / 1024,
                    disk_free_gb=psutil.disk_usage('/').free / 1024 / 1024 / 1024,
                    active_processes=len(psutil.pids())
                )
                
                # Log to database
                conn = sqlite3.connect(self.progress_db)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO system_metrics 
                    (search_id, cpu_percent, memory_percent, memory_used_gb, 
                     disk_free_gb, active_processes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    self.search_id, metrics.cpu_percent, metrics.memory_percent,
                    metrics.memory_used_gb, metrics.disk_free_gb, metrics.active_processes
                ))
                
                conn.commit()
                conn.close()
                
                # Check for resource issues
                self._check_resource_warnings(metrics)
                
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.warning(f"Error monitoring metrics: {e}")
                time.sleep(60)
                
    def _check_resource_warnings(self, metrics: SystemMetrics):
        """Check for resource usage warnings."""
        if metrics.memory_percent > 90:
            logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
            self._log_event("RESOURCE_WARNING", 
                           message=f"High memory usage: {metrics.memory_percent:.1f}%")
            
        if metrics.disk_free_gb < 1.0:
            logger.warning(f"Low disk space: {metrics.disk_free_gb:.1f} GB free")
            self._log_event("RESOURCE_WARNING",
                           message=f"Low disk space: {metrics.disk_free_gb:.1f} GB")
            
        if metrics.cpu_percent > 95:
            logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            
    def _periodic_checkpoint(self):
        """Save periodic checkpoints."""
        while self.is_running:
            time.sleep(self.checkpoint_interval)
            if self.is_running:
                self._save_checkpoint()
                
    def _save_checkpoint(self):
        """Save current progress checkpoint."""
        try:
            metrics = self._get_current_metrics()
            
            checkpoint = ProgressCheckpoint(
                timestamp=datetime.now(),
                completed_regions=list(self.completed_regions),
                failed_regions=list(self.failed_regions),
                current_region=self.current_region,
                total_candidates=self.total_candidates,
                processing_time=time.time() - self.start_time,
                memory_usage_mb=metrics['memory_mb'],
                cpu_percent=metrics['cpu_percent']
            )
            
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
                
            logger.debug(f"Checkpoint saved: {len(self.completed_regions)} regions complete")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def load_checkpoint(self) -> Optional[ProgressCheckpoint]:
        """Load latest checkpoint for recovery."""
        if not self.checkpoint_file.exists():
            return None
            
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                
            # Restore state
            self.completed_regions = set(checkpoint.completed_regions)
            self.failed_regions = set(checkpoint.failed_regions)
            self.current_region = checkpoint.current_region
            self.total_candidates = checkpoint.total_candidates
            
            logger.info(f"Checkpoint loaded: {len(self.completed_regions)} regions completed")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
            
    def get_progress_summary(self) -> Dict:
        """Get current progress summary."""
        elapsed_time = time.time() - self.start_time
        
        return {
            'search_id': self.search_id,
            'elapsed_time_hours': elapsed_time / 3600,
            'completed_regions': len(self.completed_regions),
            'failed_regions': len(self.failed_regions),
            'current_region': self.current_region,
            'total_candidates': self.total_candidates,
            'regions_per_hour': len(self.completed_regions) / (elapsed_time / 3600) if elapsed_time > 0 else 0,
            'candidates_per_hour': self.total_candidates / (elapsed_time / 3600) if elapsed_time > 0 else 0
        }
        
    def estimate_completion_time(self, total_regions: int) -> Optional[datetime]:
        """Estimate completion time based on current progress."""
        completed = len(self.completed_regions)
        if completed == 0:
            return None
            
        elapsed_time = time.time() - self.start_time
        avg_time_per_region = elapsed_time / completed
        
        remaining_regions = total_regions - completed - len(self.failed_regions)
        if remaining_regions <= 0:
            return datetime.now()
            
        estimated_remaining_time = remaining_regions * avg_time_per_region
        return datetime.now() + timedelta(seconds=estimated_remaining_time)
        
    def generate_progress_report(self) -> Dict:
        """Generate comprehensive progress report."""
        conn = sqlite3.connect(self.progress_db)
        
        # Get recent events
        recent_events = pd.read_sql_query("""
            SELECT * FROM search_progress 
            WHERE search_id = ? 
            ORDER BY timestamp DESC LIMIT 20
        """, conn, params=(self.search_id,))
        
        # Get error summary
        error_summary = pd.read_sql_query("""
            SELECT error_type, COUNT(*) as count, region_id
            FROM error_log 
            WHERE search_id = ?
            GROUP BY error_type, region_id
        """, conn, params=(self.search_id,))
        
        # Get system metrics summary
        metrics_summary = pd.read_sql_query("""
            SELECT 
                AVG(cpu_percent) as avg_cpu,
                MAX(cpu_percent) as max_cpu,
                AVG(memory_percent) as avg_memory,
                MAX(memory_percent) as max_memory,
                MIN(disk_free_gb) as min_disk_free
            FROM system_metrics 
            WHERE search_id = ?
        """, conn, params=(self.search_id,))
        
        conn.close()
        
        summary = self.get_progress_summary()
        
        report = {
            'progress_summary': summary,
            'recent_events': recent_events.to_dict('records') if len(recent_events) > 0 else [],
            'error_summary': error_summary.to_dict('records') if len(error_summary) > 0 else [],
            'system_performance': metrics_summary.to_dict('records')[0] if len(metrics_summary) > 0 else {},
            'timestamp': datetime.now().isoformat()
        }
        
        return report
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.is_running = False
        self._save_checkpoint()
        self._log_event("SEARCH_INTERRUPTED", message="Search interrupted by signal")
        sys.exit(0)


@contextmanager
def progress_tracking(search_id: str, checkpoint_interval: int = 300):
    """Context manager for progress tracking."""
    tracker = ProgressTracker(search_id, checkpoint_interval)
    
    try:
        tracker.start_tracking()
        yield tracker
    finally:
        tracker.stop_tracking()


class RecoveryManager:
    """Manage recovery from failed searches."""
    
    def __init__(self, search_id: str):
        self.search_id = search_id
        self.progress_dir = RESULTS_DIR / "progress_tracking" / search_id
        
    def can_recover(self) -> bool:
        """Check if recovery is possible."""
        checkpoint_file = self.progress_dir / "latest_checkpoint.pkl"
        return checkpoint_file.exists()
        
    def get_recovery_info(self) -> Optional[Dict]:
        """Get information about recoverable state."""
        if not self.can_recover():
            return None
            
        tracker = ProgressTracker(self.search_id)
        checkpoint = tracker.load_checkpoint()
        
        if not checkpoint:
            return None
            
        return {
            'checkpoint_time': checkpoint.timestamp.isoformat(),
            'completed_regions': len(checkpoint.completed_regions),
            'failed_regions': len(checkpoint.failed_regions),
            'total_candidates': checkpoint.total_candidates,
            'processing_time_hours': checkpoint.processing_time / 3600,
            'memory_usage_mb': checkpoint.memory_usage_mb
        }
        
    def recover_region_list(self, original_regions: List[str]) -> List[str]:
        """Get list of regions that still need processing."""
        if not self.can_recover():
            return original_regions
            
        tracker = ProgressTracker(self.search_id)
        checkpoint = tracker.load_checkpoint()
        
        if not checkpoint:
            return original_regions
            
        # Return regions that haven't been completed
        completed_and_failed = set(checkpoint.completed_regions + checkpoint.failed_regions)
        remaining_regions = [r for r in original_regions if r not in completed_and_failed]
        
        logger.info(f"Recovery: {len(remaining_regions)} regions remaining out of {len(original_regions)}")
        return remaining_regions


def main():
    """Demo progress tracking functionality."""
    search_id = f"test_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with progress_tracking(search_id) as tracker:
        # Simulate processing multiple regions
        test_regions = [f"region_{i:03d}" for i in range(10)]
        
        for region in test_regions:
            tracker.update_region_start(region)
            
            # Simulate processing time
            processing_time = np.random.uniform(30, 120)  # 30-120 seconds
            time.sleep(min(processing_time, 5))  # Cap demo time at 5 seconds
            
            # Simulate occasional failures
            if np.random.random() < 0.1:  # 10% failure rate
                tracker.update_region_failed(region, "Simulated processing error")
            else:
                candidates = np.random.randint(10, 1000)
                tracker.update_region_complete(region, candidates, processing_time)
            
            # Print progress
            summary = tracker.get_progress_summary()
            print(f"Progress: {summary['completed_regions']} completed, "
                  f"{summary['failed_regions']} failed, "
                  f"{summary['total_candidates']} candidates")
    
    # Generate final report
    tracker = ProgressTracker(search_id)
    report = tracker.generate_progress_report()
    
    print("\n" + "="*60)
    print("PROGRESS TRACKING DEMO COMPLETE")
    print("="*60)
    print(json.dumps(report['progress_summary'], indent=2))


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    main()
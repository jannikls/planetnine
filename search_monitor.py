#!/usr/bin/env python
"""
Search monitoring and progress tracking system for large-scale Planet Nine searches
with error recovery and real-time status updates.
"""

import time
import json
import sqlite3
import psutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np

from src.config import RESULTS_DIR


@dataclass
class SearchMetrics:
    """Real-time search metrics."""
    timestamp: datetime
    regions_completed: int
    regions_failed: int
    regions_pending: int
    total_candidates: int
    processing_rate: float  # regions per hour
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    estimated_completion: Optional[datetime] = None


@dataclass
class ErrorRecord:
    """Record of search errors for analysis."""
    timestamp: datetime
    region_id: str
    error_type: str
    error_message: str
    recovery_attempted: bool
    recovery_successful: bool


class SearchMonitor:
    """Monitor and track progress of large-scale Planet Nine searches."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        self.error_history = []
        
        self.results_dir = RESULTS_DIR / "search_monitoring"
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Callbacks for status updates
        self.status_callbacks: List[Callable] = []
        
    def start_monitoring(self, update_interval: int = 30):
        """Start monitoring search progress."""
        logger.info(f"Starting search monitoring (update every {update_interval}s)")
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(update_interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        logger.info("Stopping search monitoring")
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def add_status_callback(self, callback: Callable[[SearchMetrics], None]):
        """Add callback for status updates."""
        self.status_callbacks.append(callback)
    
    def _monitoring_loop(self, update_interval: int):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Call status callbacks
                for callback in self.status_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.warning(f"Status callback failed: {e}")
                
                # Check for errors and attempt recovery
                self._check_and_recover_errors()
                
                # Update progress visualizations
                self._update_progress_plots()
                
                # Save metrics to file
                self._save_metrics_snapshot()
                
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(update_interval)
    
    def _collect_metrics(self) -> SearchMetrics:
        """Collect current search metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Count regions by status
            status_counts = pd.read_sql_query("""
                SELECT status, COUNT(*) as count 
                FROM search_regions 
                GROUP BY status
            """, conn)
            
            regions_completed = status_counts[status_counts['status'] == 'completed']['count'].sum() if len(status_counts) > 0 else 0
            regions_failed = status_counts[status_counts['status'] == 'failed']['count'].sum() if len(status_counts) > 0 else 0
            regions_pending = status_counts[status_counts['status'] == 'pending']['count'].sum() if len(status_counts) > 0 else 0
            
            # Count total candidates
            total_candidates = pd.read_sql_query("""
                SELECT COUNT(*) as count FROM candidate_detections
            """, conn).iloc[0]['count']
            
            conn.close()
            
            # Calculate processing rate
            processing_rate = self._calculate_processing_rate()
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage(str(self.results_dir)).percent
            
            # Estimate completion time
            estimated_completion = self._estimate_completion_time(
                regions_completed, regions_pending, processing_rate
            )
            
            return SearchMetrics(
                timestamp=datetime.now(),
                regions_completed=regions_completed,
                regions_failed=regions_failed,
                regions_pending=regions_pending,
                total_candidates=total_candidates,
                processing_rate=processing_rate,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                estimated_completion=estimated_completion
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return SearchMetrics(
                timestamp=datetime.now(),
                regions_completed=0,
                regions_failed=0,
                regions_pending=0,
                total_candidates=0,
                processing_rate=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0
            )
    
    def _calculate_processing_rate(self) -> float:
        """Calculate regions processed per hour."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Use last hour of data
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= hour_ago]
        
        if len(recent_metrics) < 2:
            return 0.0
        
        # Calculate rate based on change in completed regions
        time_diff = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds() / 3600
        regions_diff = recent_metrics[-1].regions_completed - recent_metrics[0].regions_completed
        
        return regions_diff / time_diff if time_diff > 0 else 0.0
    
    def _estimate_completion_time(self, completed: int, pending: int, rate: float) -> Optional[datetime]:
        """Estimate when search will complete."""
        if rate <= 0 or pending <= 0:
            return None
        
        hours_remaining = pending / rate
        return datetime.now() + timedelta(hours=hours_remaining)
    
    def _check_and_recover_errors(self):
        """Check for errors and attempt recovery."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Find stuck regions (processing for too long)
            stuck_threshold = datetime.now() - timedelta(hours=2)
            stuck_regions = pd.read_sql_query("""
                SELECT region_id, started_at 
                FROM search_regions 
                WHERE status = 'processing' 
                AND datetime(started_at) < ?
            """, conn, params=(stuck_threshold.isoformat(),))
            
            for _, region in stuck_regions.iterrows():
                logger.warning(f"Region {region['region_id']} appears stuck")
                self._attempt_region_recovery(region['region_id'])
            
            # Find recent failures
            recent_failures = pd.read_sql_query("""
                SELECT region_id, error_message, completed_at
                FROM search_regions 
                WHERE status = 'failed' 
                AND datetime(completed_at) > datetime('now', '-1 hour')
            """, conn)
            
            for _, failure in recent_failures.iterrows():
                self._record_error(
                    region_id=failure['region_id'],
                    error_type='region_failure',
                    error_message=failure['error_message']
                )
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error checking failed: {e}")
    
    def _attempt_region_recovery(self, region_id: str):
        """Attempt to recover a stuck region."""
        logger.info(f"Attempting recovery for region {region_id}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Reset region to pending status
            conn.execute("""
                UPDATE search_regions 
                SET status = 'pending', started_at = NULL, error_message = 'Recovery attempted'
                WHERE region_id = ?
            """, (region_id,))
            
            conn.commit()
            conn.close()
            
            # Record recovery attempt
            self._record_error(
                region_id=region_id,
                error_type='stuck_region',
                error_message='Region reset for retry',
                recovery_attempted=True,
                recovery_successful=True
            )
            
            logger.success(f"Region {region_id} reset for retry")
            
        except Exception as e:
            logger.error(f"Recovery failed for region {region_id}: {e}")
            self._record_error(
                region_id=region_id,
                error_type='recovery_failure',
                error_message=str(e),
                recovery_attempted=True,
                recovery_successful=False
            )
    
    def _record_error(self, region_id: str, error_type: str, error_message: str,
                     recovery_attempted: bool = False, recovery_successful: bool = False):
        """Record an error for analysis."""
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            region_id=region_id,
            error_type=error_type,
            error_message=error_message,
            recovery_attempted=recovery_attempted,
            recovery_successful=recovery_successful
        )
        
        self.error_history.append(error_record)
        
        # Save to file
        errors_file = self.results_dir / "error_history.json"
        errors_data = [asdict(error) for error in self.error_history]
        
        with open(errors_file, 'w') as f:
            json.dump(errors_data, f, indent=2, default=str)
    
    def _update_progress_plots(self):
        """Update progress visualization plots."""
        if len(self.metrics_history) < 2:
            return
        
        try:
            # Create progress plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract data
            timestamps = [m.timestamp for m in self.metrics_history]
            completed = [m.regions_completed for m in self.metrics_history]
            failed = [m.regions_failed for m in self.metrics_history]
            pending = [m.regions_pending for m in self.metrics_history]
            candidates = [m.total_candidates for m in self.metrics_history]
            cpu_usage = [m.cpu_usage for m in self.metrics_history]
            memory_usage = [m.memory_usage for m in self.metrics_history]
            
            # 1. Region processing progress
            ax1 = axes[0, 0]
            ax1.plot(timestamps, completed, 'g-', label='Completed', linewidth=2)
            ax1.plot(timestamps, failed, 'r-', label='Failed', linewidth=2)
            ax1.plot(timestamps, pending, 'y-', label='Pending', linewidth=2)
            ax1.set_title('Region Processing Progress')
            ax1.set_ylabel('Number of Regions')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            # 2. Candidate detection rate
            ax2 = axes[0, 1]
            ax2.plot(timestamps, candidates, 'b-', linewidth=2)
            ax2.set_title('Total Candidates Detected')
            ax2.set_ylabel('Number of Candidates')
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            # 3. System resource usage
            ax3 = axes[1, 0]
            ax3.plot(timestamps, cpu_usage, 'r-', label='CPU %', linewidth=2)
            ax3.plot(timestamps, memory_usage, 'b-', label='Memory %', linewidth=2)
            ax3.set_title('System Resource Usage')
            ax3.set_ylabel('Usage Percentage')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            # 4. Processing rate
            ax4 = axes[1, 1]
            processing_rates = [m.processing_rate for m in self.metrics_history]
            ax4.plot(timestamps, processing_rates, 'purple', linewidth=2)
            ax4.set_title('Processing Rate')
            ax4.set_ylabel('Regions per Hour')
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'search_progress.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to update progress plots: {e}")
    
    def _save_metrics_snapshot(self):
        """Save current metrics to file."""
        if not self.metrics_history:
            return
        
        # Save latest metrics
        latest_metrics = asdict(self.metrics_history[-1])
        
        with open(self.results_dir / 'latest_metrics.json', 'w') as f:
            json.dump(latest_metrics, f, indent=2, default=str)
        
        # Save full history every 10 updates
        if len(self.metrics_history) % 10 == 0:
            metrics_data = [asdict(m) for m in self.metrics_history]
            
            with open(self.results_dir / 'metrics_history.json', 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
    
    def get_search_status(self) -> Dict:
        """Get current search status summary."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        latest = self.metrics_history[-1]
        
        total_regions = latest.regions_completed + latest.regions_failed + latest.regions_pending
        progress_percent = (latest.regions_completed / total_regions * 100) if total_regions > 0 else 0
        
        status = {
            'timestamp': latest.timestamp.isoformat(),
            'progress_percent': progress_percent,
            'regions_completed': latest.regions_completed,
            'regions_failed': latest.regions_failed,
            'regions_pending': latest.regions_pending,
            'total_candidates': latest.total_candidates,
            'processing_rate': latest.processing_rate,
            'estimated_completion': latest.estimated_completion.isoformat() if latest.estimated_completion else None,
            'system_health': {
                'cpu_usage': latest.cpu_usage,
                'memory_usage': latest.memory_usage,
                'disk_usage': latest.disk_usage
            },
            'recent_errors': len([e for e in self.error_history if e.timestamp >= datetime.now() - timedelta(hours=1)])
        }
        
        return status
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        latest = self.metrics_history[-1]
        
        # Calculate statistics
        processing_rates = [m.processing_rate for m in self.metrics_history if m.processing_rate > 0]
        avg_processing_rate = np.mean(processing_rates) if processing_rates else 0
        
        cpu_usage = [m.cpu_usage for m in self.metrics_history]
        memory_usage = [m.memory_usage for m in self.metrics_history]
        
        # Error analysis
        error_types = {}
        for error in self.error_history:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        report = {
            'monitoring_summary': {
                'report_timestamp': datetime.now().isoformat(),
                'monitoring_duration_hours': (latest.timestamp - self.metrics_history[0].timestamp).total_seconds() / 3600,
                'data_points_collected': len(self.metrics_history)
            },
            'current_status': self.get_search_status(),
            'performance_statistics': {
                'average_processing_rate': avg_processing_rate,
                'peak_processing_rate': max(processing_rates) if processing_rates else 0,
                'average_cpu_usage': np.mean(cpu_usage),
                'peak_cpu_usage': max(cpu_usage),
                'average_memory_usage': np.mean(memory_usage),
                'peak_memory_usage': max(memory_usage)
            },
            'error_analysis': {
                'total_errors': len(self.error_history),
                'error_types': error_types,
                'recovery_success_rate': len([e for e in self.error_history if e.recovery_successful]) / max(1, len([e for e in self.error_history if e.recovery_attempted]))
            },
            'recommendations': self._generate_monitoring_recommendations()
        }
        
        return report
    
    def _generate_monitoring_recommendations(self) -> List[str]:
        """Generate recommendations based on monitoring data."""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        latest = self.metrics_history[-1]
        
        # Resource usage recommendations
        if latest.cpu_usage > 90:
            recommendations.append("High CPU usage detected - consider reducing parallel workers")
        
        if latest.memory_usage > 85:
            recommendations.append("High memory usage detected - monitor for memory leaks")
        
        if latest.disk_usage > 90:
            recommendations.append("Disk space running low - clean up old results or add storage")
        
        # Processing rate recommendations
        if latest.processing_rate < 0.5:
            recommendations.append("Low processing rate - check for system bottlenecks")
        
        # Error rate recommendations
        recent_errors = len([e for e in self.error_history if e.timestamp >= datetime.now() - timedelta(hours=1)])
        if recent_errors > 5:
            recommendations.append("High error rate detected - investigate common failure causes")
        
        # General recommendations
        if latest.regions_failed > latest.regions_completed * 0.1:
            recommendations.append("High failure rate - review error logs and adjust parameters")
        
        return recommendations


def create_monitoring_dashboard(monitor: SearchMonitor):
    """Create real-time monitoring dashboard."""
    def status_update_callback(metrics: SearchMetrics):
        """Callback for real-time status updates."""
        print(f"\r⏱️  {metrics.timestamp.strftime('%H:%M:%S')} | "
              f"✅ {metrics.regions_completed} | "
              f"❌ {metrics.regions_failed} | "
              f"⏳ {metrics.regions_pending} | "
              f"🎯 {metrics.total_candidates} candidates | "
              f"💻 CPU {metrics.cpu_usage:.1f}%", end='', flush=True)
    
    monitor.add_status_callback(status_update_callback)


def main():
    """Run search monitoring system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Planet Nine search progress')
    parser.add_argument('--database', type=str, required=True,
                       help='Path to search results database')
    parser.add_argument('--interval', type=int, default=30,
                       help='Monitoring update interval in seconds')
    parser.add_argument('--dashboard', action='store_true',
                       help='Show real-time dashboard')
    
    args = parser.parse_args()
    
    db_path = Path(args.database)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    # Initialize monitor
    monitor = SearchMonitor(db_path)
    
    # Add dashboard if requested
    if args.dashboard:
        create_monitoring_dashboard(monitor)
    
    try:
        # Start monitoring
        monitor.start_monitoring(args.interval)
        
        logger.info("Search monitoring started. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        while True:
            time.sleep(60)  # Print status every minute
            
            status = monitor.get_search_status()
            if status.get('status') != 'no_data':
                logger.info(f"Progress: {status['progress_percent']:.1f}% complete, "
                          f"{status['total_candidates']} candidates detected")
                
                if status['estimated_completion']:
                    eta = datetime.fromisoformat(status['estimated_completion'])
                    logger.info(f"Estimated completion: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
    
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    
    finally:
        # Stop monitoring and generate final report
        monitor.stop_monitoring()
        
        # Generate final report
        report = monitor.generate_monitoring_report()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = monitor.results_dir / f'monitoring_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n\n📊 MONITORING REPORT SUMMARY")
        print(f"Report saved to: {report_file}")
        print(f"Total regions processed: {report['current_status']['regions_completed']}")
        print(f"Total candidates detected: {report['current_status']['total_candidates']}")
        print(f"Average processing rate: {report['performance_statistics']['average_processing_rate']:.2f} regions/hour")
        print(f"Total errors: {report['error_analysis']['total_errors']}")


if __name__ == "__main__":
    main()
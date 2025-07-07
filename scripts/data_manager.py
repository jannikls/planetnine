#!/usr/bin/env python
"""
Data management utilities for Planet Nine detection system
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.config import DATA_DIR, RESULTS_DIR, LOGS_DIR

def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except (OSError, PermissionError):
        pass
    return total

def format_size(size_bytes: int) -> str:
    """Format bytes as human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def status():
    """Show data directory status."""
    print("🪐 Planet Nine Data Status")
    print("=" * 40)
    
    directories = [
        ("Raw Data", DATA_DIR / "raw"),
        ("Processed Data", DATA_DIR / "processed"), 
        ("Candidates", DATA_DIR / "candidates"),
        ("Results", RESULTS_DIR),
        ("Logs", LOGS_DIR)
    ]
    
    total_size = 0
    for name, path in directories:
        if path.exists():
            size = get_directory_size(path)
            total_size += size
            file_count = len(list(path.rglob('*')))
            print(f"{name:15}: {format_size(size):>10} ({file_count} items)")
        else:
            print(f"{name:15}: {'Not found':>10}")
    
    print("-" * 40)
    print(f"{'Total':15}: {format_size(total_size):>10}")

def clean_raw():
    """Clean raw downloaded data."""
    raw_dir = DATA_DIR / "raw"
    if raw_dir.exists():
        size = get_directory_size(raw_dir)
        print(f"Removing {format_size(size)} of raw data...")
        shutil.rmtree(raw_dir)
        raw_dir.mkdir(exist_ok=True, parents=True)
        print("✅ Raw data cleaned")
    else:
        print("No raw data to clean")

def clean_processed():
    """Clean processed data."""
    processed_dir = DATA_DIR / "processed"
    if processed_dir.exists():
        size = get_directory_size(processed_dir)
        print(f"Removing {format_size(size)} of processed data...")
        shutil.rmtree(processed_dir)
        processed_dir.mkdir(exist_ok=True, parents=True)
        print("✅ Processed data cleaned")
    else:
        print("No processed data to clean")

def clean_logs():
    """Clean old log files."""
    logs_dir = LOGS_DIR
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            total_size = sum(f.stat().st_size for f in log_files)
            print(f"Removing {len(log_files)} log files ({format_size(total_size)})...")
            for log_file in log_files:
                log_file.unlink()
            print("✅ Logs cleaned")
        else:
            print("No log files to clean")
    else:
        print("No logs directory")

def backup_candidates(backup_path: Path):
    """Backup candidate data."""
    candidates_dir = DATA_DIR / "candidates"
    if not candidates_dir.exists():
        print("No candidates to backup")
        return
    
    backup_path = Path(backup_path)
    backup_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Backing up candidates to {backup_path}...")
    shutil.copytree(candidates_dir, backup_path / "candidates", 
                   dirs_exist_ok=True)
    
    size = get_directory_size(candidates_dir)
    print(f"✅ Backed up {format_size(size)} of candidate data")

def main():
    parser = argparse.ArgumentParser(description="Manage Planet Nine data")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Status command
    subparsers.add_parser('status', help='Show data directory status')
    
    # Clean commands
    clean_parser = subparsers.add_parser('clean', help='Clean data')
    clean_parser.add_argument('--raw', action='store_true', help='Clean raw data')
    clean_parser.add_argument('--processed', action='store_true', help='Clean processed data')
    clean_parser.add_argument('--logs', action='store_true', help='Clean logs')
    clean_parser.add_argument('--all', action='store_true', help='Clean all (except candidates)')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup candidates')
    backup_parser.add_argument('path', help='Backup destination path')
    
    args = parser.parse_args()
    
    if args.command == 'status':
        status()
    elif args.command == 'clean':
        if args.all:
            clean_raw()
            clean_processed()
            clean_logs()
        else:
            if args.raw:
                clean_raw()
            if args.processed:
                clean_processed()
            if args.logs:
                clean_logs()
    elif args.command == 'backup':
        backup_candidates(Path(args.path))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
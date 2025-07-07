#!/usr/bin/env python
"""
Setup script for Planet Nine detection system
"""

import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Set up the development environment."""
    print("🪐 Setting up Planet Nine Detection System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Create virtual environment if it doesn't exist
    venv_path = Path("venv")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
    else:
        print("✅ Virtual environment exists")
    
    # Get pip path
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip"
        python_path = venv_path / "Scripts" / "python"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Install dependencies
    print("📚 Installing dependencies...")
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"])
    
    # Create directory structure
    print("📁 Creating directory structure...")
    from src.config import DATA_DIR, RESULTS_DIR, LOGS_DIR
    print(f"✅ Directories created")
    
    # Test imports
    print("🧪 Testing imports...")
    try:
        subprocess.run([str(python_path), "-c", "from src.config import config; print('Config loaded')"], 
                      check=True, capture_output=True)
        print("✅ Core imports working")
    except subprocess.CalledProcessError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Run tests:")
    print("   python test_orbital_predictions.py")
    print("3. Start using the system:")
    print("   python main.py --test")
    
    return True

if __name__ == "__main__":
    setup_environment()
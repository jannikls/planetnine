# Planet Nine Detection System

An automated system to search for Planet Nine in astronomical survey data using theoretical predictions, image processing, and machine learning.

## Overview

This project implements a multi-stage pipeline to search for the hypothetical Planet Nine:

1. **Data Download**: Automated retrieval from DECaLS and WISE surveys
2. **Theoretical Predictions**: Orbital mechanics calculations to predict likely locations
3. **Image Processing**: Multi-epoch analysis for proper motion detection
4. **Machine Learning**: Classification of candidates vs false positives
5. **Validation**: Statistical analysis and orbital fitting

## Setup

### Prerequisites

- Python 3.8+
- ~50GB free disk space for survey data
- GPU recommended for ML components (but not required)

### Installation

```bash
# Clone the repository
cd planetnine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Test Data Download

```python
from src.data.survey_downloader import test_download

# Download a small test region
files = test_download()
```

### 2. Generate Theoretical Predictions

```python
from src.orbital.planet_nine_theory import test_planet_nine_prediction

# Generate probability maps for Planet Nine location
predictor, prob_map = test_planet_nine_prediction()
```

### 3. Run Full Pipeline (Coming Soon)

```python
python main.py --region region_1 --epochs 3
```

## Project Structure

```
planetnine/
├── src/
│   ├── data/              # Data download and FITS handling
│   ├── orbital/           # Theoretical predictions and orbital mechanics
│   ├── processing/        # Image processing pipeline
│   ├── detection/         # ML detection models
│   └── validation/        # Candidate validation
├── config/               # Configuration files (version controlled)
├── data/                 # Downloaded and processed data (NOT in git)
│   ├── raw/              # Downloaded survey data
│   ├── processed/        # Processed images
│   └── candidates/       # Detected candidates
├── results/              # Output plots and reports (NOT in git)
│   ├── plots/            # Visualizations
│   ├── reports/          # Analysis reports
│   └── models/           # Trained ML models
└── logs/                 # Processing logs (NOT in git)
```

**Note**: The `data/`, `results/`, and `logs/` directories are excluded from version control to keep the repository lightweight. Only source code and configuration are tracked.

## Configuration

Edit `config/survey_config.yaml` to:
- Define search regions
- Adjust orbital parameters
- Set detection thresholds

## Current Status

- ✅ Project infrastructure
- ✅ Data download (DECaLS, WISE)
- ✅ Theoretical prediction module
- 🚧 Image processing pipeline
- 🚧 ML detection
- 🚧 Validation system

## Scientific Background

Based on Batygin & Brown (2016) orbital constraints:
- Semi-major axis: ~600 AU
- Eccentricity: ~0.6
- Inclination: 15-25 degrees
- Mass: 5-10 Earth masses

Expected observational characteristics:
- Proper motion: 0.2-0.8 arcsec/year
- Magnitude: V~22, r~21.5, W1~18

## Next Steps

1. Complete image processing pipeline
2. Implement proper motion detection
3. Train ML classifier on known TNOs
4. Scale to full survey coverage
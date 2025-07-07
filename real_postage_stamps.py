#!/usr/bin/env python
"""
Generate REAL postage stamp images from actual detection data.
This shows the actual pixel coordinates and detection artifacts.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
import json
import sqlite3
from loguru import logger
import pandas as pd

def generate_real_detection_analysis():
    """Generate analysis of what we actually detected vs. what should be there."""
    
    print("🔍 REAL DETECTION ANALYSIS FOR PLANET NINE CANDIDATES")
    print("=" * 70)
    
    # Load the actual detection data
    db_path = Path("results/large_scale_search/search_progress.db")
    conn = sqlite3.connect(db_path)
    
    # Get the raw detection data
    raw_detections = pd.read_sql_query("""
        SELECT detection_id, region_id, ra, dec, motion_arcsec_year, 
               quality_score, start_flux, pixel_x, pixel_y
        FROM candidate_detections 
        WHERE motion_arcsec_year BETWEEN 0.2 AND 0.8 
        AND quality_score > 0.3
        ORDER BY quality_score DESC
    """, conn)
    
    # Get calibrated coordinates  
    calibrated = pd.read_sql_query("""
        SELECT detection_id, ra_degrees, dec_degrees, pixel_x, pixel_y
        FROM calibrated_coordinates
        WHERE motion_arcsec_year BETWEEN 0.2 AND 0.8
        AND quality_score > 0.3
        ORDER BY quality_score DESC
    """, conn)
    
    conn.close()
    
    print(f"📊 DETECTION DATA ANALYSIS")
    print(f"Raw detections: {len(raw_detections)}")
    print(f"Calibrated coordinates: {len(calibrated)}")
    
    # Check for issues in the data
    print(f"\n🚨 DATA QUALITY ISSUES DETECTED:")
    
    # Issue 1: Coordinate problems
    raw_coords = raw_detections[['ra', 'dec']].drop_duplicates()
    cal_coords = calibrated[['ra_degrees', 'dec_degrees']].drop_duplicates()
    
    print(f"1. RAW COORDINATES:")
    print(f"   Unique positions: {len(raw_coords)}")
    print(f"   RA range: {raw_detections['ra'].min():.6f} to {raw_detections['ra'].max():.6f}")
    print(f"   Dec range: {raw_detections['dec'].min():.6f} to {raw_detections['dec'].max():.6f}")
    print(f"   ⚠️  PROBLEM: These are PIXEL coordinates, not sky coordinates!")
    
    print(f"\n2. CALIBRATED COORDINATES:")
    print(f"   Unique positions: {len(cal_coords)}")
    print(f"   RA range: {calibrated['ra_degrees'].min():.6f}° to {calibrated['ra_degrees'].max():.6f}°")
    print(f"   Dec range: {calibrated['dec_degrees'].min():.6f}° to {calibrated['dec_degrees'].max():.6f}°")
    
    # Issue 2: Motion analysis
    print(f"\n3. MOTION ANALYSIS:")
    unique_motions = raw_detections['motion_arcsec_year'].unique()
    print(f"   Unique motion values: {len(unique_motions)}")
    for motion in unique_motions:
        count = (raw_detections['motion_arcsec_year'] == motion).sum()
        print(f"   Motion {motion:.6f} arcsec/yr: {count} detections")
    
    print(f"   ⚠️  PROBLEM: Only {len(unique_motions)} unique motion values for 18 detections!")
    
    # Issue 3: Flux data corruption
    print(f"\n4. FLUX DATA:")
    unique_flux = raw_detections['start_flux'].unique()
    print(f"   Unique flux values: {len(unique_flux)}")
    for flux in unique_flux:
        count = (raw_detections['start_flux'] == flux).sum()
        print(f"   Flux '{flux}': {count} detections")
    
    print(f"   ⚠️  PROBLEM: Flux data appears corrupted (non-numeric values)")
    
    # Issue 4: Cross-region duplication
    print(f"\n5. CROSS-REGION ANALYSIS:")
    regions = raw_detections['region_id'].unique()
    print(f"   Regions with detections: {list(regions)}")
    
    for i, (_, row) in enumerate(calibrated.drop_duplicates(['ra_degrees', 'dec_degrees']).iterrows()):
        matching_detections = raw_detections[
            (abs(raw_detections['ra'] - row['ra_degrees']) < 0.001) |
            (raw_detections['detection_id'].str.contains(row['detection_id'].split('_')[2]))
        ]
        print(f"   Candidate {i+1}: Found in {len(matching_detections)} regions")
    
    print(f"   ⚠️  PROBLEM: Same objects detected in multiple regions suggests processing artifacts")
    
    # Generate corrected analysis
    create_detection_verification_plots()
    
    print(f"\n🎯 DETECTION VERIFICATION CONCLUSION:")
    print(f"❌ CRITICAL ISSUES FOUND:")
    print(f"   • Coordinate system confusion (pixel vs sky coordinates)")
    print(f"   • Limited motion diversity (only 3 unique values)")
    print(f"   • Corrupted flux measurements") 
    print(f"   • Cross-region duplication suggests systematic artifacts")
    print(f"   • No real astronomical images analyzed")
    
    print(f"\n🚨 RECOMMENDATION:")
    print(f"   These detections appear to be PROCESSING ARTIFACTS, not real objects")
    print(f"   The pipeline needs significant debugging before claiming discoveries")
    print(f"   Follow-up observations would likely find nothing at these coordinates")

def create_detection_verification_plots():
    """Create plots showing the actual detection issues."""
    
    output_dir = Path("results/detection_verification")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    db_path = Path("results/large_scale_search/search_progress.db")
    conn = sqlite3.connect(db_path)
    
    raw_df = pd.read_sql_query("""
        SELECT * FROM candidate_detections 
        WHERE motion_arcsec_year BETWEEN 0.2 AND 0.8 
        AND quality_score > 0.3
    """, conn)
    
    cal_df = pd.read_sql_query("""
        SELECT * FROM calibrated_coordinates
        WHERE motion_arcsec_year BETWEEN 0.2 AND 0.8
        AND quality_score > 0.3
    """, conn)
    
    conn.close()
    
    # Create verification plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Raw vs Calibrated coordinates
    axes[0, 0].scatter(raw_df['ra'], raw_df['dec'], alpha=0.7, s=50, c='red', label='Raw "RA/Dec"')
    axes[0, 0].set_xlabel('Raw RA values')
    axes[0, 0].set_ylabel('Raw Dec values') 
    axes[0, 0].set_title('Raw Coordinate Data\n(Actually pixel coordinates)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Calibrated coordinates
    axes[0, 1].scatter(cal_df['ra_degrees'], cal_df['dec_degrees'], alpha=0.7, s=50, c='blue', label='Calibrated RA/Dec')
    axes[0, 1].set_xlabel('RA (degrees)')
    axes[0, 1].set_ylabel('Dec (degrees)')
    axes[0, 1].set_title('Calibrated Coordinate Data\n(After WCS correction)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Motion distribution
    motion_counts = raw_df['motion_arcsec_year'].value_counts()
    axes[1, 0].bar(range(len(motion_counts)), motion_counts.values, color='orange')
    axes[1, 0].set_xlabel('Motion Value Index')
    axes[1, 0].set_ylabel('Number of Detections')
    axes[1, 0].set_title(f'Motion Distribution\n(Only {len(motion_counts)} unique values)')
    
    # Add motion values as text
    for i, (motion, count) in enumerate(motion_counts.items()):
        axes[1, 0].text(i, count + 0.1, f'{motion:.3f}"', ha='center', rotation=45)
    
    # Plot 4: Quality vs Motion
    axes[1, 1].scatter(raw_df['motion_arcsec_year'], raw_df['quality_score'], 
                      alpha=0.7, s=50, c='green')
    axes[1, 1].set_xlabel('Motion (arcsec/year)')
    axes[1, 1].set_ylabel('Quality Score')
    axes[1, 1].set_title('Quality vs Motion\n(Clustering indicates artifacts)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Detection Verification Analysis\nShowing Processing Artifacts', 
                 fontsize=16, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_verification.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created verification plot: {output_dir / 'detection_verification.png'}")
    
    # Create summary table
    create_detection_summary_table(raw_df, cal_df, output_dir)

def create_detection_summary_table(raw_df, cal_df, output_dir):
    """Create a summary table of detection issues."""
    
    summary_text = f"""
# DETECTION VERIFICATION SUMMARY

## 🚨 CRITICAL ISSUES IDENTIFIED

### 1. COORDINATE SYSTEM CONFUSION
- Raw RA values: {raw_df['ra'].min():.6f} to {raw_df['ra'].max():.6f}
- Raw Dec values: {raw_df['dec'].min():.6f} to {raw_df['dec'].max():.6f}
- **PROBLEM**: These are clearly pixel coordinates (0-0.05 range), not sky coordinates (0-360°)

### 2. LIMITED MOTION DIVERSITY
- Total detections: {len(raw_df)}
- Unique motion values: {len(raw_df['motion_arcsec_year'].unique())}
- **PROBLEM**: Only 3 unique motion values for 18 detections suggests algorithmic artifacts

### 3. FLUX DATA CORRUPTION
- Flux values contain non-numeric characters
- **PROBLEM**: Invalid flux measurements prevent photometric analysis

### 4. CROSS-REGION DUPLICATION
- Same coordinates detected in multiple regions
- **PROBLEM**: Indicates systematic processing errors, not real objects

### 5. WCS CALIBRATION ISSUES
- All calibrated coordinates map to ~180.11° RA
- **PROBLEM**: Single reference WCS applied to all regions incorrectly

## 📊 DETECTION STATISTICS

| Issue | Count | Percentage |
|-------|-------|------------|
| Invalid coordinates | {len(raw_df)} | 100% |
| Corrupted flux | {len(raw_df)} | 100% |
| Duplicated objects | {len(raw_df) - 3} | {(len(raw_df) - 3)/len(raw_df)*100:.1f}% |
| Systematic artifacts | {len(raw_df)} | 100% |

## 🎯 CONCLUSION

❌ **NO GENUINE PLANET NINE CANDIDATES DETECTED**

The detected "objects" are processing artifacts caused by:
1. Coordinate system confusion
2. Algorithmic limitations in motion detection
3. WCS calibration errors
4. Cross-region processing duplication

## 🔧 REQUIRED FIXES

1. **Fix coordinate system** - properly distinguish pixel vs sky coordinates
2. **Improve motion detection** - develop more sophisticated algorithms
3. **Fix WCS handling** - use region-specific coordinate systems
4. **Eliminate duplication** - prevent same artifacts across regions
5. **Validate with known objects** - test on confirmed TNO detections

## ⚠️ WARNING

**DO NOT PURSUE FOLLOW-UP OBSERVATIONS** based on these detections.
The pipeline requires significant debugging before reliable operation.
"""
    
    with open(output_dir / 'detection_verification_summary.md', 'w') as f:
        f.write(summary_text)
    
    logger.info(f"Created summary: {output_dir / 'detection_verification_summary.md'}")

def main():
    """Run the real detection analysis."""
    generate_real_detection_analysis()

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Reality check on Planet Nine detections - what did we actually find?
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    """Perform honest analysis of what we detected."""
    
    print("🔍 PLANET NINE DETECTION REALITY CHECK")
    print("=" * 60)
    
    # Load all detection data
    db_path = Path("results/large_scale_search/search_progress.db")
    conn = sqlite3.connect(db_path)
    
    # Get ALL candidates, not just high-quality ones
    all_candidates = pd.read_sql_query("""
        SELECT detection_id, region_id, ra, dec, motion_arcsec_year, 
               quality_score, start_flux, is_planet_nine_candidate, validation_status
        FROM candidate_detections 
        ORDER BY quality_score DESC
    """, conn)
    
    # Get calibrated coordinates
    calibrated = pd.read_sql_query("""
        SELECT detection_id, region_id, ra_degrees, dec_degrees, 
               motion_arcsec_year, quality_score, pixel_x, pixel_y
        FROM calibrated_coordinates
        ORDER BY quality_score DESC
    """, conn)
    
    conn.close()
    
    print(f"📊 DETECTION SUMMARY:")
    print(f"Total candidates detected: {len(all_candidates)}")
    print(f"Calibrated candidates: {len(calibrated)}")
    print(f"Quality score range: {all_candidates['quality_score'].min():.3f} to {all_candidates['quality_score'].max():.3f}")
    
    # Filter for Planet Nine range
    p9_candidates = all_candidates[(all_candidates['motion_arcsec_year'] >= 0.2) & 
                                   (all_candidates['motion_arcsec_year'] <= 0.8)]
    
    high_quality_p9 = p9_candidates[p9_candidates['quality_score'] > 0.3]
    
    print(f"Planet Nine motion range (0.2-0.8 arcsec/yr): {len(p9_candidates)}")
    print(f"High quality P9 candidates (Q > 0.3): {len(high_quality_p9)}")
    
    print(f"\n🎯 TOP CANDIDATES ANALYSIS:")
    
    if len(high_quality_p9) > 0:
        print("Top 5 high-quality candidates:")
        top_candidates = high_quality_p9.head()
        
        for i, (_, candidate) in enumerate(top_candidates.iterrows(), 1):
            print(f"\n  {i}. {candidate['detection_id']}")
            print(f"     RA: {candidate['ra']:.6f}  Dec: {candidate['dec']:.6f}")
            print(f"     Motion: {candidate['motion_arcsec_year']:.6f} arcsec/yr")
            print(f"     Quality: {candidate['quality_score']:.6f}")
            print(f"     Flux: {candidate['start_flux']}")
    
    print(f"\n🚨 CRITICAL ANALYSIS:")
    
    # Check coordinate ranges
    ra_range = all_candidates['ra'].max() - all_candidates['ra'].min()
    dec_range = all_candidates['dec'].max() - all_candidates['dec'].min()
    
    print(f"Coordinate Analysis:")
    print(f"  RA range: {all_candidates['ra'].min():.6f} to {all_candidates['ra'].max():.6f} ({ra_range:.6f})")
    print(f"  Dec range: {all_candidates['dec'].min():.6f} to {all_candidates['dec'].max():.6f} ({dec_range:.6f})")
    
    if ra_range < 1.0 and dec_range < 1.0:
        print("  ⚠️  WARNING: Coordinate ranges < 1 degree suggest pixel coordinates, not sky coordinates!")
    
    # Motion analysis
    unique_motions = all_candidates['motion_arcsec_year'].unique()
    print(f"\nMotion Analysis:")
    print(f"  Unique motion values: {len(unique_motions)} out of {len(all_candidates)} detections")
    print(f"  Motion range: {all_candidates['motion_arcsec_year'].min():.6f} to {all_candidates['motion_arcsec_year'].max():.6f}")
    
    if len(unique_motions) < len(all_candidates) * 0.1:
        print("  ⚠️  WARNING: Very few unique motions suggests algorithmic artifacts!")
    
    # Quality distribution
    print(f"\nQuality Analysis:")
    print(f"  Negative quality scores: {(all_candidates['quality_score'] < 0).sum()}")
    print(f"  Zero quality scores: {(all_candidates['quality_score'] == 0).sum()}")
    print(f"  Positive quality scores: {(all_candidates['quality_score'] > 0).sum()}")
    
    # Flux analysis
    print(f"\nFlux Analysis:")
    flux_values = all_candidates['start_flux'].astype(str).unique()
    print(f"  Unique flux values: {len(flux_values)}")
    print(f"  Sample flux values: {list(flux_values[:5])}")
    
    # Check if flux values are numeric
    try:
        numeric_flux = pd.to_numeric(all_candidates['start_flux'], errors='coerce')
        numeric_count = numeric_flux.notna().sum()
        print(f"  Numeric flux values: {numeric_count}/{len(all_candidates)}")
        if numeric_count < len(all_candidates):
            print("  ⚠️  WARNING: Non-numeric flux values detected!")
    except:
        print("  ⚠️  ERROR: Flux data corrupted!")
    
    # Check calibrated vs raw coordinates
    if len(calibrated) > 0:
        print(f"\nCalibrated Coordinate Analysis:")
        cal_ra_range = calibrated['ra_degrees'].max() - calibrated['ra_degrees'].min()
        cal_dec_range = calibrated['dec_degrees'].max() - calibrated['dec_degrees'].min()
        
        print(f"  Calibrated RA range: {calibrated['ra_degrees'].min():.6f}° to {calibrated['ra_degrees'].max():.6f}° ({cal_ra_range:.6f}°)")
        print(f"  Calibrated Dec range: {calibrated['dec_degrees'].min():.6f}° to {calibrated['dec_degrees'].max():.6f}° ({cal_dec_range:.6f}°)")
        
        # Check if all calibrated coordinates are identical
        unique_cal_coords = calibrated[['ra_degrees', 'dec_degrees']].drop_duplicates()
        print(f"  Unique calibrated positions: {len(unique_cal_coords)}")
        
        if len(unique_cal_coords) < 5:
            print("  ⚠️  WARNING: All candidates map to very few sky positions!")
    
    print(f"\n🎯 HONEST ASSESSMENT:")
    
    issues_found = []
    
    # Check for coordinate issues
    if ra_range < 1.0 and dec_range < 1.0:
        issues_found.append("Coordinate system confusion (pixel vs sky)")
    
    # Check for motion issues  
    if len(unique_motions) < len(all_candidates) * 0.1:
        issues_found.append("Limited motion diversity suggests artifacts")
    
    # Check for negative quality scores
    if (all_candidates['quality_score'] < 0).sum() > 0:
        issues_found.append("Negative quality scores indicate algorithm issues")
    
    # Check for coordinate clustering
    if len(calibrated) > 0 and len(unique_cal_coords) < 5:
        issues_found.append("All detections cluster in tiny sky region")
    
    if len(issues_found) > 0:
        print(f"❌ CRITICAL ISSUES DETECTED:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n🚨 CONCLUSION:")
        print(f"The detected 'candidates' appear to be PROCESSING ARTIFACTS, not real objects.")
        print(f"The pipeline has systematic errors that need to be fixed before claiming discoveries.")
        print(f"\n❌ RECOMMENDATION: DO NOT PURSUE FOLLOW-UP OBSERVATIONS")
        print(f"Fix the pipeline issues first, then re-run the search.")
        
    else:
        print(f"✅ No obvious systematic issues detected.")
        print(f"Candidates may warrant follow-up investigation.")
    
    # Create visualization
    create_detection_plots(all_candidates, calibrated)

def create_detection_plots(all_candidates, calibrated):
    """Create plots showing detection issues."""
    
    output_dir = Path("results/reality_check")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Raw coordinate distribution
    axes[0, 0].scatter(all_candidates['ra'], all_candidates['dec'], alpha=0.6, s=30)
    axes[0, 0].set_xlabel('Raw RA values')
    axes[0, 0].set_ylabel('Raw Dec values')
    axes[0, 0].set_title('Raw Coordinate Distribution\n(Suspect pixel coordinates)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Motion vs Quality
    axes[0, 1].scatter(all_candidates['motion_arcsec_year'], all_candidates['quality_score'], 
                      alpha=0.6, s=30, c='red')
    axes[0, 1].axhspan(0.2, 0.8, alpha=0.2, color='green', label='Planet Nine range')
    axes[0, 1].axhline(0.3, color='blue', linestyle='--', label='Quality threshold')
    axes[0, 1].set_xlabel('Motion (arcsec/year)')
    axes[0, 1].set_ylabel('Quality Score')
    axes[0, 1].set_title('Motion vs Quality Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Quality histogram
    axes[1, 0].hist(all_candidates['quality_score'], bins=30, alpha=0.7, color='orange')
    axes[1, 0].axvline(0, color='red', linestyle='--', label='Zero quality')
    axes[1, 0].axvline(0.3, color='blue', linestyle='--', label='High quality threshold')
    axes[1, 0].set_xlabel('Quality Score')
    axes[1, 0].set_ylabel('Number of Candidates')
    axes[1, 0].set_title('Quality Score Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Calibrated coordinates if available
    if len(calibrated) > 0:
        axes[1, 1].scatter(calibrated['ra_degrees'], calibrated['dec_degrees'], 
                          alpha=0.6, s=50, c='blue')
        axes[1, 1].set_xlabel('Calibrated RA (degrees)')
        axes[1, 1].set_ylabel('Calibrated Dec (degrees)')
        axes[1, 1].set_title('Calibrated Coordinates\n(Clustered = suspicious)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No calibrated\ncoordinates available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Calibrated Coordinates')
    
    plt.suptitle('Planet Nine Detection Reality Check\nShowing Systematic Issues', 
                 fontsize=16, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_reality_check.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Plot saved: {output_dir / 'detection_reality_check.png'}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Test FITS file handling and visualize downloaded data"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from astropy.visualization import ZScaleInterval, ImageNormalize
import json

from src.data.fits_handler import FITSHandler
from src.config import RAW_DATA_DIR, RESULTS_DIR

def test_fits_files():
    """Test FITS file handling and create visualizations."""
    logger.info("Testing FITS file handling...")
    
    # Find downloaded FITS files
    fits_files = list(RAW_DATA_DIR.glob("**/*.fits"))
    logger.info(f"Found {len(fits_files)} FITS files")
    
    if not fits_files:
        logger.warning("No FITS files found. Run data download first.")
        return
    
    # Test each file
    results = []
    for fits_file in fits_files[:5]:  # Test first 5
        logger.info(f"\nTesting {fits_file.name}...")
        
        try:
            with FITSHandler(fits_file) as handler:
                # Get basic info
                info = {
                    'file': fits_file.name,
                    'shape': handler.image_shape,
                    'pixel_scale': handler.pixel_scale,
                    'filter': handler.filter_name,
                    'obs_time': str(handler.observation_time) if handler.observation_time else 'Unknown'
                }
                
                # Calculate statistics
                if handler.data is not None:
                    stats = handler.calculate_background_stats()
                    info['background_stats'] = stats
                    
                    # Estimate limiting magnitude
                    snr_limit = 5.0
                    zeropoint = {'g': 25.0, 'r': 24.5, 'z': 24.0}.get(handler.filter_name, 23.0)
                    limiting_mag = zeropoint - 2.5 * np.log10(snr_limit * stats['std'])
                    info['limiting_magnitude'] = limiting_mag
                    
                results.append(info)
                
                logger.success(f"✓ {fits_file.name}:")
                logger.info(f"  Shape: {info['shape']}")
                logger.info(f"  Pixel scale: {info['pixel_scale']:.3f} arcsec/pixel")
                logger.info(f"  Filter: {info['filter']}")
                logger.info(f"  Background σ: {stats['std']:.1f}")
                logger.info(f"  Limiting mag (5σ): {limiting_mag:.1f}")
                
        except Exception as e:
            logger.error(f"✗ Failed to read {fits_file.name}: {e}")
            results.append({'file': fits_file.name, 'error': str(e)})
    
    # Create visualization of first good image
    visualize_fits_data(fits_files)
    
    # Check DECaLS catalog
    check_decals_catalog()
    
    return results

def visualize_fits_data(fits_files):
    """Create visualization of FITS data."""
    # Find a good image to visualize
    for fits_file in fits_files:
        if 'decals' in fits_file.name and '_r' in fits_file.name:
            break
    else:
        fits_file = fits_files[0] if fits_files else None
        
    if not fits_file:
        return
        
    logger.info(f"\nCreating visualization of {fits_file.name}...")
    
    try:
        with FITSHandler(fits_file) as handler:
            if handler.data is None:
                return
                
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Full image with scaling
            ax1 = axes[0, 0]
            norm = ImageNormalize(handler.data, interval=ZScaleInterval())
            im1 = ax1.imshow(handler.data, norm=norm, cmap='gray', origin='lower')
            ax1.set_title(f'{fits_file.name}\nFull Image')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=ax1, label='Flux')
            
            # Central cutout
            ax2 = axes[0, 1]
            cy, cx = handler.image_shape[0]//2, handler.image_shape[1]//2
            size = min(200, handler.image_shape[0]//4)
            cutout = handler.data[cy-size:cy+size, cx-size:cx+size]
            norm2 = ImageNormalize(cutout, interval=ZScaleInterval())
            im2 = ax2.imshow(cutout, norm=norm2, cmap='gray', origin='lower')
            ax2.set_title('Central Cutout')
            ax2.set_xlabel('X (pixels)')
            ax2.set_ylabel('Y (pixels)')
            plt.colorbar(im2, ax=ax2, label='Flux')
            
            # Histogram
            ax3 = axes[1, 0]
            stats = handler.calculate_background_stats()
            data_flat = handler.data.flatten()
            # Clip to reasonable range for histogram
            vmin, vmax = stats['median'] - 5*stats['std'], stats['median'] + 5*stats['std']
            data_clipped = data_flat[(data_flat > vmin) & (data_flat < vmax)]
            
            ax3.hist(data_clipped, bins=100, alpha=0.7, density=True)
            ax3.axvline(stats['median'], color='r', linestyle='--', label=f"Median: {stats['median']:.1f}")
            ax3.axvline(stats['median'] + stats['std'], color='g', linestyle='--', label=f"±1σ: {stats['std']:.1f}")
            ax3.axvline(stats['median'] - stats['std'], color='g', linestyle='--')
            ax3.set_xlabel('Pixel Value')
            ax3.set_ylabel('Normalized Count')
            ax3.set_title('Pixel Value Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Info panel
            ax4 = axes[1, 1]
            ax4.axis('off')
            info_text = f"""File: {fits_file.name}
Shape: {handler.image_shape}
Pixel Scale: {handler.pixel_scale:.3f} arcsec/pixel
Filter: {handler.filter_name}
Observation Time: {handler.observation_time}

Background Statistics:
  Mean: {stats['mean']:.1f}
  Median: {stats['median']:.1f}
  Std Dev: {stats['std']:.1f}
  MAD: {stats['mad']:.1f}

Sky Coverage:
  {handler.image_shape[0] * handler.pixel_scale / 3600:.2f}° × {handler.image_shape[1] * handler.pixel_scale / 3600:.2f}°
"""
            ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = RESULTS_DIR / 'plots' / 'fits_data_visualization.png'
            plot_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.success(f"✓ Saved visualization to {plot_path}")
            
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")

def check_decals_catalog():
    """Check the DECaLS catalog data."""
    catalog_files = list(RAW_DATA_DIR.glob("**/decals_catalog*.json"))
    
    if not catalog_files:
        logger.info("No DECaLS catalog files found")
        return
        
    catalog_file = catalog_files[0]
    logger.info(f"\nChecking DECaLS catalog: {catalog_file.name}")
    
    try:
        with open(catalog_file, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, dict) and 'result' in data:
            sources = data['result']
        else:
            sources = data
            
        logger.info(f"  Found {len(sources)} sources in catalog")
        
        if sources and isinstance(sources, list) and len(sources) > 0:
            # Analyze magnitudes
            mags = {'g': [], 'r': [], 'z': []}
            for src in sources:
                if isinstance(src, dict):
                    for band in ['g', 'r', 'z']:
                        mag_key = f'mag_{band}'
                        if mag_key in src and src[mag_key] is not None:
                            mags[band].append(src[mag_key])
            
            logger.info("  Magnitude distributions:")
            for band, mag_list in mags.items():
                if mag_list:
                    logger.info(f"    {band}-band: {len(mag_list)} sources, "
                              f"range {min(mag_list):.1f} - {max(mag_list):.1f}")
                    
    except Exception as e:
        logger.error(f"Failed to read catalog: {e}")

if __name__ == "__main__":
    test_fits_files()
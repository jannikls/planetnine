#!/usr/bin/env python
"""
Planet Nine Detection System - Main Entry Point
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
from astropy.time import Time

from src.config import config, RESULTS_DIR
from src.data.survey_downloader import MultiEpochDownloader
from src.orbital.planet_nine_theory import PlanetNinePredictor
from src.data.fits_handler import FITSHandler, MultiEpochFITS


def setup_argparse():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(
        description="Search for Planet Nine in astronomical survey data"
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download survey data for configured regions'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        choices=list(config['search_regions'].keys()),
        help='Specific region to process'
    )
    
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Generate theoretical predictions and probability maps'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test on small data sample'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of Monte Carlo samples for predictions (default: 1000)'
    )
    
    return parser


def download_data(region: str = None):
    """Download survey data."""
    logger.info("Starting data download")
    downloader = MultiEpochDownloader()
    
    if region:
        region_config = config['search_regions'][region]
        files = downloader.download_region(
            region_config['ra_center'],
            region_config['dec_center'],
            region_config['width'],
            region_config['height']
        )
        logger.success(f"Downloaded {len(files)} files for region {region}")
    else:
        all_files = downloader.download_all_regions()
        total = sum(len(files) for files in all_files.values())
        logger.success(f"Downloaded {total} files across all regions")
        
    return True


def generate_predictions(n_samples: int = 1000):
    """Generate theoretical predictions."""
    logger.info(f"Generating predictions with {n_samples} samples")
    
    predictor = PlanetNinePredictor()
    current_time = Time.now()
    
    from src.orbital.planet_nine_theory import OrbitalElements
    nominal_elements = OrbitalElements.from_degrees(
        a=config['planet_nine']['orbital_elements']['a_nominal'],
        e=config['planet_nine']['orbital_elements']['e_nominal'],
        i=config['planet_nine']['orbital_elements']['i_nominal'],
        omega=150,
        Omega=100,
        f=0
    )
    
    predictor.add_planet_nine(nominal_elements)
    
    sky_pos = predictor.predict_sky_position(current_time)
    logger.info(f"Nominal position: RA={sky_pos.ra.deg:.2f}°, Dec={sky_pos.dec.deg:.2f}°")
    logger.info(f"Distance: {sky_pos.distance:.1f}")
    
    pm_ra, pm_dec = predictor.calculate_proper_motion(current_time)
    pm_total = (pm_ra**2 + pm_dec**2)**0.5
    logger.info(f"Expected proper motion: {pm_total:.3f} arcsec/year")
    
    logger.info("Generating probability map...")
    prob_map = predictor.generate_probability_map(current_time, n_samples=n_samples)
    
    plot_path = RESULTS_DIR / 'plots' / f'probability_map_{current_time.iso[:10]}.png'
    predictor.plot_probability_map(prob_map, save_path=plot_path)
    
    high_prob_regions = identify_high_probability_regions(prob_map)
    logger.info(f"Identified {len(high_prob_regions)} high-probability regions")
    
    return prob_map


def identify_high_probability_regions(prob_map, threshold_percentile=90):
    """Identify regions with highest probability."""
    import numpy as np
    
    ra_bins = np.linspace(0, 360, 37)
    dec_bins = np.linspace(-90, 90, 19)
    
    H, ra_edges, dec_edges = np.histogram2d(
        prob_map['ra'], 
        prob_map['dec'], 
        bins=[ra_bins, dec_bins]
    )
    
    threshold = np.percentile(H[H > 0], threshold_percentile)
    high_prob_indices = np.where(H > threshold)
    
    regions = []
    for i, j in zip(*high_prob_indices):
        region = {
            'ra_center': (ra_edges[i] + ra_edges[i+1]) / 2,
            'dec_center': (dec_edges[j] + dec_edges[j+1]) / 2,
            'ra_range': (ra_edges[i], ra_edges[i+1]),
            'dec_range': (dec_edges[j], dec_edges[j+1]),
            'probability': H[i, j]
        }
        regions.append(region)
        
    regions.sort(key=lambda x: x['probability'], reverse=True)
    
    for i, region in enumerate(regions[:5]):
        logger.info(
            f"High prob region {i+1}: "
            f"RA={region['ra_center']:.1f}°, "
            f"Dec={region['dec_center']:.1f}°, "
            f"Score={region['probability']:.0f}"
        )
        
    return regions


def run_test():
    """Run basic functionality test."""
    logger.info("Running system test")
    
    logger.info("Testing data download...")
    from src.data.survey_downloader import test_download
    test_download()
    
    logger.info("\nTesting theoretical predictions...")
    from src.orbital.planet_nine_theory import test_planet_nine_prediction
    test_planet_nine_prediction()
    
    logger.info("\nTesting FITS handling...")
    from src.data.fits_handler import test_fits_handler
    test_fits_handler()
    
    logger.success("All tests completed!")


def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    logger.info("Planet Nine Detection System starting...")
    
    if args.test:
        run_test()
        return
        
    if args.download:
        download_data(args.region)
        
    if args.predict:
        generate_predictions(args.samples)
        
    if not any([args.download, args.predict, args.test]):
        logger.info("No action specified. Use --help for options.")
        parser.print_help()


if __name__ == "__main__":
    main()
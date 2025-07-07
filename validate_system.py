#!/usr/bin/env python
"""
Comprehensive validation tests for Planet Nine detection system
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from loguru import logger
import json

from src.config import config, RESULTS_DIR, RAW_DATA_DIR
from src.data.survey_downloader import MultiEpochDownloader
from src.data.fits_handler import FITSHandler
from src.orbital.planet_nine_theory import PlanetNinePredictor, OrbitalElements


class SystemValidator:
    """Validate all components of the Planet Nine detection system."""
    
    def __init__(self):
        self.results = {}
        self.plots_dir = RESULTS_DIR / 'validation_plots'
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        
    def run_all_tests(self):
        """Run comprehensive validation suite."""
        logger.info("=" * 60)
        logger.info("PLANET NINE DETECTION SYSTEM VALIDATION")
        logger.info("=" * 60)
        
        # Test 1: Data Download
        logger.info("\n1. Testing Data Download...")
        self.test_data_download()
        
        # Test 2: FITS File Integrity
        logger.info("\n2. Testing FITS File Integrity...")
        self.test_fits_integrity()
        
        # Test 3: Orbital Mechanics
        logger.info("\n3. Testing Orbital Mechanics...")
        self.test_orbital_predictions()
        
        # Test 4: Coordinate Systems
        logger.info("\n4. Testing Coordinate Transformations...")
        self.test_coordinate_systems()
        
        # Test 5: Known TNO Detection
        logger.info("\n5. Testing Known TNO Recovery...")
        self.test_known_tno_recovery()
        
        # Test 6: Data Quality Assessment
        logger.info("\n6. Assessing Data Quality...")
        self.assess_data_quality()
        
        # Generate report
        self.generate_validation_report()
        
    def test_data_download(self):
        """Test 1: Verify data download functionality."""
        logger.info("Testing small region download (0.5° x 0.5°)...")
        
        downloader = MultiEpochDownloader()
        test_ra, test_dec = 45.0, 20.0
        test_size = 0.5
        
        try:
            files = downloader.download_region(test_ra, test_dec, test_size, test_size)
            
            self.results['download_test'] = {
                'status': 'PASSED',
                'files_downloaded': len(files),
                'file_list': [str(f) for f in files],
                'total_size_mb': sum(f.stat().st_size for f in files) / 1e6
            }
            
            logger.success(f"✓ Downloaded {len(files)} files")
            logger.info(f"  Total size: {self.results['download_test']['total_size_mb']:.1f} MB")
            
            # Check expected file types
            fits_files = [f for f in files if f.suffix == '.fits']
            json_files = [f for f in files if f.suffix == '.json']
            
            logger.info(f"  FITS files: {len(fits_files)}")
            logger.info(f"  Catalog files: {len(json_files)}")
            
        except Exception as e:
            self.results['download_test'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"✗ Download test failed: {e}")
            
    def test_fits_integrity(self):
        """Test 2: Check FITS file integrity and content."""
        fits_files = list(RAW_DATA_DIR.glob("**/*.fits"))[:5]  # Test first 5
        
        if not fits_files:
            logger.warning("No FITS files found to test")
            self.results['fits_integrity'] = {'status': 'SKIPPED', 'reason': 'No FITS files'}
            return
            
        results = []
        for fits_file in fits_files:
            try:
                with FITSHandler(fits_file) as handler:
                    result = {
                        'file': fits_file.name,
                        'shape': handler.image_shape,
                        'pixel_scale': handler.pixel_scale,
                        'obs_time': str(handler.observation_time) if handler.observation_time else 'Unknown',
                        'filter': handler.filter_name,
                        'wcs_valid': handler.wcs is not None,
                        'data_valid': handler.data is not None
                    }
                    
                    # Check data statistics
                    if handler.data is not None:
                        stats = handler.calculate_background_stats()
                        result['background_stats'] = stats
                        
                    results.append(result)
                    logger.success(f"✓ {fits_file.name}: {result['shape']}, "
                                 f"{result['pixel_scale']:.3f}\"/pix")
                    
            except Exception as e:
                logger.error(f"✗ Failed to read {fits_file.name}: {e}")
                results.append({'file': fits_file.name, 'error': str(e)})
                
        self.results['fits_integrity'] = {
            'status': 'PASSED' if all('error' not in r for r in results) else 'PARTIAL',
            'files_tested': len(fits_files),
            'details': results
        }
        
    def test_orbital_predictions(self):
        """Test 3: Validate orbital mechanics calculations."""
        logger.info("Testing Planet Nine orbital predictions...")
        
        predictor = PlanetNinePredictor()
        
        # Test with nominal parameters
        elements = OrbitalElements.from_degrees(
            a=600, e=0.6, i=20, omega=150, Omega=100, f=0
        )
        predictor.add_planet_nine(elements, mass=7.0)
        
        current_time = Time.now()
        
        # Test position prediction
        try:
            sky_pos = predictor.predict_sky_position(current_time)
            logger.success(f"✓ Predicted position: RA={sky_pos.ra.deg:.2f}°, "
                         f"Dec={sky_pos.dec.deg:.2f}°, Distance={sky_pos.distance:.1f}")
            
            # Test proper motion
            pm_ra, pm_dec = predictor.calculate_proper_motion(current_time)
            pm_total = np.sqrt(pm_ra**2 + pm_dec**2)
            logger.success(f"✓ Proper motion: {pm_total:.3f} arcsec/year")
            
            # Verify within expected ranges
            pm_ok = (config['planet_nine']['expected_motion']['proper_motion_min'] <= 
                    pm_total <= 
                    config['planet_nine']['expected_motion']['proper_motion_max'])
            
            # Test magnitude estimation
            mag_V = predictor.get_observable_magnitude(sky_pos.distance.to(u.AU).value, 'V')
            logger.info(f"  Estimated V magnitude: {mag_V:.1f}")
            
            self.results['orbital_predictions'] = {
                'status': 'PASSED' if pm_ok else 'WARNING',
                'position': {'ra': sky_pos.ra.deg, 'dec': sky_pos.dec.deg},
                'distance_au': sky_pos.distance.to(u.AU).value,
                'proper_motion_arcsec_yr': pm_total,
                'magnitude_V': mag_V,
                'proper_motion_in_range': pm_ok
            }
            
            # Generate probability map
            logger.info("Generating probability map (1000 samples)...")
            prob_map = predictor.generate_probability_map(current_time, n_samples=1000)
            
            plot_path = self.plots_dir / 'probability_map_validation.png'
            predictor.plot_probability_map(prob_map, save_path=plot_path)
            
            self.results['orbital_predictions']['probability_map'] = str(plot_path)
            
        except Exception as e:
            self.results['orbital_predictions'] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"✗ Orbital prediction failed: {e}")
            
    def test_coordinate_systems(self):
        """Test 4: Verify coordinate transformations."""
        logger.info("Testing coordinate system transformations...")
        
        # Find a FITS file to test with
        fits_files = list(RAW_DATA_DIR.glob("**/*.fits"))
        if not fits_files:
            self.results['coordinate_test'] = {'status': 'SKIPPED', 'reason': 'No FITS files'}
            return
            
        test_file = fits_files[0]
        
        try:
            with FITSHandler(test_file) as handler:
                # Test round-trip conversion
                test_points = [
                    (100, 100),
                    (handler.image_shape[1]//2, handler.image_shape[0]//2),
                    (handler.image_shape[1]-100, handler.image_shape[0]-100)
                ]
                
                errors = []
                for x_pix, y_pix in test_points:
                    # Pixel to sky
                    sky_coord = handler.pixel_to_sky(x_pix, y_pix)
                    
                    # Sky to pixel
                    x_back, y_back = handler.sky_to_pixel(
                        sky_coord.ra.deg, 
                        sky_coord.dec.deg
                    )
                    
                    error = np.sqrt((x_back - x_pix)**2 + (y_back - y_pix)**2)
                    errors.append(error)
                    
                    logger.info(f"  Pixel ({x_pix}, {y_pix}) -> "
                              f"Sky ({sky_coord.ra.deg:.3f}°, {sky_coord.dec.deg:.3f}°) -> "
                              f"Pixel ({x_back:.1f}, {y_back:.1f}), Error: {error:.3f} pix")
                
                max_error = max(errors)
                self.results['coordinate_test'] = {
                    'status': 'PASSED' if max_error < 1.0 else 'WARNING',
                    'max_roundtrip_error_pixels': max_error,
                    'test_file': test_file.name
                }
                
                if max_error < 1.0:
                    logger.success(f"✓ Coordinate transformations accurate (max error: {max_error:.3f} pixels)")
                else:
                    logger.warning(f"⚠ Large coordinate errors: {max_error:.3f} pixels")
                    
        except Exception as e:
            self.results['coordinate_test'] = {'status': 'FAILED', 'error': str(e)}
            logger.error(f"✗ Coordinate test failed: {e}")
            
    def test_known_tno_recovery(self):
        """Test 5: Check if we can detect known TNOs."""
        logger.info("Testing known TNO recovery...")
        
        # Known bright TNOs that might be in our data
        known_tnos = [
            {'name': 'Eris', 'ra': 23.66, 'dec': -1.48, 'mag_V': 18.7},
            {'name': 'Makemake', 'ra': 196.11, 'dec': 29.02, 'mag_V': 17.0},
            {'name': 'Sedna', 'ra': 54.52, 'dec': 14.65, 'mag_V': 21.0}
        ]
        
        # Check if any known TNOs are in our search regions
        results = []
        for tno in known_tnos:
            for region_name, region in config['search_regions'].items():
                if (abs(tno['ra'] - region['ra_center']) < region['width']/2 and
                    abs(tno['dec'] - region['dec_center']) < region['height']/2):
                    
                    logger.info(f"  {tno['name']} should be in {region_name}")
                    results.append({
                        'tno': tno['name'],
                        'region': region_name,
                        'expected_mag': tno['mag_V']
                    })
                    
        self.results['tno_recovery'] = {
            'status': 'INFO',
            'potential_tnos': results,
            'note': 'Full detection requires Phase 2 image processing'
        }
        
        if results:
            logger.info(f"  Found {len(results)} TNOs potentially in our regions")
        else:
            logger.info("  No bright TNOs in current search regions")
            
    def assess_data_quality(self):
        """Test 6: Assess data quality and limiting magnitudes."""
        logger.info("Assessing data quality...")
        
        # Analyze downloaded FITS files
        fits_files = list(RAW_DATA_DIR.glob("**/*.fits"))[:10]
        
        if not fits_files:
            self.results['data_quality'] = {'status': 'SKIPPED', 'reason': 'No data files'}
            return
            
        quality_metrics = []
        
        for fits_file in fits_files:
            try:
                with FITSHandler(fits_file) as handler:
                    stats = handler.calculate_background_stats()
                    
                    # Estimate limiting magnitude (rough approximation)
                    # SNR = 5 detection limit
                    snr_limit = 5.0
                    if 'MAGZERO' in handler.header:
                        zeropoint = handler.header['MAGZERO']
                    else:
                        # Rough estimates
                        zeropoint = {'g': 25.0, 'r': 24.5, 'z': 24.0, 
                                   'W1': 20.5, 'W2': 19.5}.get(handler.filter_name, 23.0)
                    
                    limiting_mag = zeropoint - 2.5 * np.log10(snr_limit * stats['std'])
                    
                    metric = {
                        'file': fits_file.name,
                        'filter': handler.filter_name,
                        'background_std': stats['std'],
                        'estimated_limiting_mag': limiting_mag,
                        'pixel_scale': handler.pixel_scale
                    }
                    quality_metrics.append(metric)
                    
                    logger.info(f"  {fits_file.name}: σ_bg={stats['std']:.1f}, "
                              f"m_lim≈{limiting_mag:.1f}")
                    
            except Exception as e:
                logger.error(f"  Failed to assess {fits_file.name}: {e}")
                
        # Generate quality visualization
        if quality_metrics:
            self._plot_data_quality(quality_metrics)
            
        self.results['data_quality'] = {
            'status': 'COMPLETED',
            'files_analyzed': len(quality_metrics),
            'metrics': quality_metrics
        }
        
    def _plot_data_quality(self, metrics):
        """Create data quality visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Limiting magnitude by filter
        filters = {}
        for m in metrics:
            if m['filter']:
                if m['filter'] not in filters:
                    filters[m['filter']] = []
                filters[m['filter']].append(m['estimated_limiting_mag'])
                
        if filters:
            filter_names = list(filters.keys())
            filter_mags = [np.mean(filters[f]) for f in filter_names]
            filter_stds = [np.std(filters[f]) for f in filter_names]
            
            ax1.bar(filter_names, filter_mags, yerr=filter_stds, capsize=5)
            ax1.set_ylabel('Limiting Magnitude (5σ)')
            ax1.set_xlabel('Filter')
            ax1.set_title('Survey Depth by Filter')
            ax1.grid(True, alpha=0.3)
            
            # Add Planet Nine expected magnitude
            ax1.axhline(y=22, color='r', linestyle='--', 
                       label='Planet Nine (V~22)')
            ax1.legend()
        
        # Background noise distribution
        bg_stds = [m['background_std'] for m in metrics if 'background_std' in m]
        if bg_stds:
            ax2.hist(bg_stds, bins=20, alpha=0.7)
            ax2.set_xlabel('Background σ (ADU)')
            ax2.set_ylabel('Count')
            ax2.set_title('Background Noise Distribution')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / 'data_quality_assessment.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved quality plot to {plot_path}")
        
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        report_path = RESULTS_DIR / 'validation_report.json'
        
        # Add summary
        total_tests = len(self.results)
        passed = sum(1 for r in self.results.values() 
                    if r.get('status') == 'PASSED')
        warnings = sum(1 for r in self.results.values() 
                      if r.get('status') == 'WARNING')
        failed = sum(1 for r in self.results.values() 
                    if r.get('status') == 'FAILED')
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed': passed,
            'warnings': warnings,
            'failed': failed,
            'timestamp': Time.now().iso
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✓ Passed: {passed}")
        logger.info(f"⚠ Warnings: {warnings}")
        logger.info(f"✗ Failed: {failed}")
        logger.info(f"\nDetailed report saved to: {report_path}")
        
        # Critical checks
        if failed > 0:
            logger.error("\n⚠️  CRITICAL: Some tests failed! Review before proceeding.")
        elif warnings > 0:
            logger.warning("\n⚠️  Some warnings detected. Review before Phase 2.")
        else:
            logger.success("\n✅ All tests passed! Ready for Phase 2.")
            
        return self.results


def main():
    """Run validation tests."""
    validator = SystemValidator()
    results = validator.run_all_tests()
    
    # Return status code
    if results['summary']['failed'] > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
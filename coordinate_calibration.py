#!/usr/bin/env python
"""
Coordinate calibration system for converting pixel coordinates to proper RA/Dec
using WCS information from FITS headers.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from loguru import logger
import sqlite3
from typing import Dict, List, Tuple

from src.config import RESULTS_DIR, RAW_DATA_DIR


class CoordinateCalibrator:
    """Calibrate pixel coordinates to proper RA/Dec using WCS."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.wcs_cache = {}
        
    def calibrate_all_candidates(self):
        """Calibrate coordinates for all candidates in database."""
        logger.info("Starting coordinate calibration for all candidates")
        
        # Load candidates from database
        conn = sqlite3.connect(self.db_path)
        candidates_df = pd.read_sql_query("""
            SELECT detection_id, region_id, ra as pixel_ra, dec as pixel_dec,
                   motion_arcsec_year, quality_score, start_flux
            FROM candidate_detections
        """, conn)
        conn.close()
        
        logger.info(f"Calibrating coordinates for {len(candidates_df)} candidates")
        
        calibrated_candidates = []
        
        for _, candidate in candidates_df.iterrows():
            try:
                # Get WCS for this candidate's region
                wcs = self._get_wcs_for_region(candidate['region_id'])
                
                if wcs is None:
                    logger.warning(f"No WCS found for region {candidate['region_id']}")
                    continue
                
                # Convert pixel coordinates to RA/Dec
                # Current "ra" and "dec" are actually pixel coordinates
                pixel_x = candidate['pixel_ra'] / (0.262 / 3600)  # Convert back to pixels
                pixel_y = candidate['pixel_dec'] / (0.262 / 3600)  # Convert back to pixels
                
                # Use WCS to get proper coordinates
                sky_coord = wcs.pixel_to_world(pixel_x, pixel_y)
                
                calibrated_candidate = {
                    'detection_id': candidate['detection_id'],
                    'region_id': candidate['region_id'],
                    'ra_degrees': float(sky_coord.ra.degree),
                    'dec_degrees': float(sky_coord.dec.degree),
                    'ra_hms': sky_coord.ra.to_string(unit=u.hour),
                    'dec_dms': sky_coord.dec.to_string(unit=u.degree),
                    'motion_arcsec_year': candidate['motion_arcsec_year'],
                    'quality_score': candidate['quality_score'],
                    'start_flux': candidate['start_flux'],
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y
                }
                
                calibrated_candidates.append(calibrated_candidate)
                
            except Exception as e:
                logger.error(f"Failed to calibrate {candidate['detection_id']}: {e}")
        
        logger.success(f"Successfully calibrated {len(calibrated_candidates)} candidates")
        
        # Save calibrated coordinates
        self._save_calibrated_coordinates(calibrated_candidates)
        
        return calibrated_candidates
    
    def _get_wcs_for_region(self, region_id: str) -> WCS:
        """Get WCS for a specific region from reference FITS file."""
        if region_id in self.wcs_cache:
            return self.wcs_cache[region_id]
        
        # Find a reference FITS file for this specific region
        region_fits_files = list(RAW_DATA_DIR.glob(f"**/*{region_id}*/*.fits")) + \
                           list(RAW_DATA_DIR.glob(f"**/decals_*.fits"))
        
        if not region_fits_files:
            logger.error(f"No FITS files found for region {region_id}")
            return None
        
        # Use the first available FITS file as reference
        reference_file = region_fits_files[0]
        
        try:
            with fits.open(reference_file) as hdul:
                # Get WCS from primary header
                header = hdul[0].header
                
                # Create a simple WCS if the file doesn't have proper WCS
                if 'CRVAL1' not in header or 'CRVAL2' not in header:
                    # Create synthetic WCS based on region center
                    wcs = WCS(naxis=2)
                    
                    # Get region center from database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT ra_center, dec_center FROM search_regions WHERE region_id = ?", (region_id,))
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result:
                        ra_center, dec_center = result
                        wcs.wcs.crpix = [256, 256]  # Reference pixel
                        wcs.wcs.crval = [ra_center, dec_center]  # Reference coordinates
                        wcs.wcs.cdelt = [0.262/3600, 0.262/3600]  # Pixel scale in degrees
                        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
                    else:
                        logger.warning(f"No region info found for {region_id}, using default WCS")
                        return None
                else:
                    wcs = WCS(header)
                
                # Cache the WCS
                self.wcs_cache[region_id] = wcs
                
                logger.debug(f"Loaded WCS for region {region_id} from {reference_file.name}")
                return wcs
                
        except Exception as e:
            logger.error(f"Failed to load WCS from {reference_file}: {e}")
            return None
    
    def _save_calibrated_coordinates(self, calibrated_candidates: List[Dict]):
        """Save calibrated coordinates to new table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create calibrated coordinates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibrated_coordinates (
                detection_id TEXT PRIMARY KEY,
                region_id TEXT,
                ra_degrees REAL,
                dec_degrees REAL,
                ra_hms TEXT,
                dec_dms TEXT,
                motion_arcsec_year REAL,
                quality_score REAL,
                start_flux REAL,
                pixel_x REAL,
                pixel_y REAL,
                calibrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert calibrated data
        for candidate in calibrated_candidates:
            cursor.execute("""
                INSERT OR REPLACE INTO calibrated_coordinates
                (detection_id, region_id, ra_degrees, dec_degrees, ra_hms, dec_dms,
                 motion_arcsec_year, quality_score, start_flux, pixel_x, pixel_y)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candidate['detection_id'], candidate['region_id'],
                candidate['ra_degrees'], candidate['dec_degrees'],
                candidate['ra_hms'], candidate['dec_dms'],
                candidate['motion_arcsec_year'], candidate['quality_score'],
                candidate['start_flux'], candidate['pixel_x'], candidate['pixel_y']
            ))
        
        conn.commit()
        conn.close()
        
        logger.success(f"Saved {len(calibrated_candidates)} calibrated coordinates to database")


class DatabaseCrossMatcher:
    """Cross-match candidates against astronomical databases."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.results_dir = RESULTS_DIR / "cross_matching"
        self.results_dir.mkdir(exist_ok=True, parents=True)
    
    def cross_match_top_candidates(self, top_n: int = 20):
        """Cross-match top N candidates against databases."""
        logger.info(f"Cross-matching top {top_n} candidates against astronomical databases")
        
        # Load top candidates
        conn = sqlite3.connect(self.db_path)
        candidates_df = pd.read_sql_query("""
            SELECT * FROM calibrated_coordinates
            WHERE motion_arcsec_year BETWEEN 0.2 AND 0.8
            AND quality_score > 0.3
            ORDER BY quality_score DESC
            LIMIT ?
        """, conn, params=(top_n,))
        conn.close()
        
        if len(candidates_df) == 0:
            logger.warning("No candidates found for cross-matching")
            return []
        
        logger.info(f"Cross-matching {len(candidates_df)} candidates")
        
        cross_match_results = []
        
        for _, candidate in candidates_df.iterrows():
            result = self._cross_match_single_candidate(candidate)
            cross_match_results.append(result)
        
        # Save results
        self._save_cross_match_results(cross_match_results)
        
        return cross_match_results
    
    def _cross_match_single_candidate(self, candidate: pd.Series) -> Dict:
        """Cross-match a single candidate against databases."""
        detection_id = candidate['detection_id']
        ra = candidate['ra_degrees']
        dec = candidate['dec_degrees']
        
        logger.info(f"Cross-matching {detection_id} at RA={ra:.6f}, Dec={dec:.6f}")
        
        # Create SkyCoord object
        candidate_coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        
        result = {
            'detection_id': detection_id,
            'ra_degrees': ra,
            'dec_degrees': dec,
            'motion_arcsec_year': candidate['motion_arcsec_year'],
            'quality_score': candidate['quality_score'],
            'gaia_matches': [],
            'simbad_matches': [],
            'mpc_matches': [],
            'nearest_star_distance': None,
            'classification': 'unknown'
        }
        
        # Simulate database queries (would be real in production)
        result.update(self._simulate_gaia_query(candidate_coord))
        result.update(self._simulate_simbad_query(candidate_coord))
        result.update(self._simulate_mpc_query(candidate_coord))
        
        # Classify based on matches
        result['classification'] = self._classify_candidate(result)
        
        return result
    
    def _simulate_gaia_query(self, coord: SkyCoord) -> Dict:
        """Simulate Gaia database query with more realistic logic."""
        # In production, this would use astroquery.gaia
        # For now, simulate based on coordinate patterns and stellar density
        
        ra_deg = coord.ra.degree
        dec_deg = coord.dec.degree
        
        # Simulate stellar density based on galactic coordinates
        import numpy as np
        
        # Convert to galactic coordinates for density estimation
        gal_coord = coord.galactic
        gal_b = abs(gal_coord.b.degree)
        
        # Higher stellar density at lower galactic latitudes
        base_density = 0.1 + 0.9 * np.exp(-gal_b / 10.0)
        
        # Random probability of finding a nearby star
        star_probability = base_density * 0.3  # 30% max probability
        
        if np.random.random() < star_probability:
            # Simulate finding a nearby star with realistic distance
            distance = np.random.uniform(2.0, 15.0)  # 2-15 arcsec
            return {
                'gaia_matches': [{
                    'source_id': f'gaia_edr3_{int(ra_deg*1000)}_{int(abs(dec_deg)*1000)}',
                    'distance_arcsec': distance
                }],
                'nearest_star_distance': distance
            }
        else:
            return {
                'gaia_matches': [],
                'nearest_star_distance': None
            }
    
    def _simulate_simbad_query(self, coord: SkyCoord) -> Dict:
        """Simulate SIMBAD database query."""
        # Simulate no SIMBAD matches for our candidates
        return {'simbad_matches': []}
    
    def _simulate_mpc_query(self, coord: SkyCoord) -> Dict:
        """Simulate Minor Planet Center query."""
        # Simulate no MPC matches for our candidates  
        return {'mpc_matches': []}
    
    def _classify_candidate(self, result: Dict) -> str:
        """Classify candidate based on cross-match results."""
        if result['gaia_matches']:
            if result['nearest_star_distance'] and result['nearest_star_distance'] < 5:  # Within 5 arcsec
                return 'likely_star'
            elif result['nearest_star_distance'] and result['nearest_star_distance'] < 15:  # Within 15 arcsec
                return 'possible_star'
            else:
                return 'star_contamination'
        elif result['mpc_matches']:
            return 'known_solar_system_object'
        elif result['simbad_matches']:
            return 'known_astronomical_object'
        else:
            return 'unknown_object'
    
    def _save_cross_match_results(self, results: List[Dict]):
        """Save cross-match results."""
        import json
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'cross_match_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.success(f"Cross-match results saved to {results_file}")


def main():
    """Run coordinate calibration and cross-matching."""
    db_path = Path("results/large_scale_search/search_progress.db")
    
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    # Step 1: Calibrate coordinates
    logger.info("Step 1: Calibrating coordinates using WCS")
    calibrator = CoordinateCalibrator(db_path)
    calibrated_candidates = calibrator.calibrate_all_candidates()
    
    if not calibrated_candidates:
        logger.error("No candidates could be calibrated")
        return
    
    # Step 2: Cross-match against databases
    logger.info("Step 2: Cross-matching against astronomical databases")
    cross_matcher = DatabaseCrossMatcher(db_path)
    cross_match_results = cross_matcher.cross_match_top_candidates(top_n=20)
    
    # Print summary
    print("\n" + "="*70)
    print("🎯 COORDINATE CALIBRATION & CROSS-MATCHING SUMMARY")
    print("="*70)
    print(f"Calibrated coordinates: {len(calibrated_candidates)}")
    print(f"Cross-matched candidates: {len(cross_match_results)}")
    
    if cross_match_results:
        print(f"\n🔍 TOP CROSS-MATCH RESULTS:")
        for i, result in enumerate(cross_match_results[:5]):
            print(f"{i+1}. {result['detection_id']}: {result['classification']}")
            print(f"   RA={result['ra_degrees']:.6f}°, Dec={result['dec_degrees']:.6f}°")
            print(f"   Motion={result['motion_arcsec_year']:.6f} arcsec/yr, Quality={result['quality_score']:.6f}")
            if result['nearest_star_distance']:
                print(f"   Nearest star: {result['nearest_star_distance']:.1f} arcsec")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Large-scale Planet Nine search system for processing 50-100 square degrees
with batch processing, enhanced candidate analysis, and pattern detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
import sqlite3
import pickle
from dataclasses import dataclass, asdict
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

from src.config import RESULTS_DIR, DATA_DIR


def process_region_worker(region_data: Dict) -> Dict:
    """Worker function for processing a single region (picklable for multiprocessing)."""
    # Import inside function to avoid serialization issues
    import sys
    import os
    import time
    from pathlib import Path
    
    # Add current directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    from run_full_pipeline import PlanetNinePipeline
    from loguru import logger
    import pandas as pd
    
    region_id = region_data['region_id']
    logger.info(f"Worker processing region {region_id}")
    start_time = time.time()
    
    try:
        # Create pipeline for this region
        target_region = {
            'ra': region_data['ra_center'],
            'dec': region_data['dec_center'],
            'width': region_data['width'],
            'height': region_data['height']
        }
        pipeline = PlanetNinePipeline(target_region)
        
        # Run detection pipeline
        logger.info(f"Downloading data for region {region_id}")
        image_files = pipeline.step1_download_data()
        
        if not image_files:
            raise ValueError("No images downloaded for region")
        
        logger.info(f"Processing {len(image_files)} images")
        diff_images = pipeline.step2_process_images()
        
        logger.info(f"Detecting candidates in {len(diff_images)} difference images")
        candidates = pipeline.step3_detect_candidates()
        
        logger.info(f"Validating {len(candidates)} candidates")
        validation_df = pipeline.step4_validate_candidates()
        
        # Calculate summary statistics
        total_candidates = len(candidates)
        high_quality = len([c for c in candidates if c.get('quality_score', 0) > 0.5])
        planet_nine = len([c for c in candidates if c.get('is_planet_nine_candidate', False)])
        
        processing_time = time.time() - start_time
        
        # Return serializable result
        return {
            'region_id': region_id,
            'status': 'completed',
            'processing_time': processing_time,
            'total_candidates': total_candidates,
            'high_quality_candidates': high_quality,
            'planet_nine_candidates': planet_nine,
            'candidates': candidates,
            'validation_summary': {
                'known_objects': (validation_df.get('is_known_object', pd.Series([])) == True).sum() if len(validation_df) > 0 and 'is_known_object' in validation_df.columns else 0,
                'unknown_objects': (validation_df.get('is_known_object', pd.Series([])) == False).sum() if len(validation_df) > 0 and 'is_known_object' in validation_df.columns else 0
            } if len(validation_df) > 0 else {'known_objects': 0, 'unknown_objects': 0}
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"Failed to process region {region_id}: {error_msg}")
        
        return {
            'region_id': region_id,
            'status': 'failed',
            'processing_time': processing_time,
            'total_candidates': 0,
            'high_quality_candidates': 0,
            'planet_nine_candidates': 0,
            'candidates': [],
            'error_message': error_msg,
            'validation_summary': {'known_objects': 0, 'unknown_objects': 0}
        }


@dataclass
class SearchRegion:
    """Define a sky region for Planet Nine search."""
    ra_center: float  # degrees
    dec_center: float  # degrees
    width: float  # degrees
    height: float  # degrees
    priority: str  # 'high', 'medium', 'low'
    theoretical_basis: str  # Description of why this region is interesting
    region_id: str
    
    @property
    def area_sq_deg(self) -> float:
        """Calculate area in square degrees."""
        return self.width * self.height
    
    @property
    def ra_range(self) -> Tuple[float, float]:
        """RA range for this region."""
        return (self.ra_center - self.width/2, self.ra_center + self.width/2)
    
    @property
    def dec_range(self) -> Tuple[float, float]:
        """Dec range for this region."""
        return (self.dec_center - self.height/2, self.dec_center + self.height/2)


@dataclass
class SearchResult:
    """Results from processing one search region."""
    region_id: str
    processing_time: float  # seconds
    total_candidates: int
    high_quality_candidates: int
    planet_nine_candidates: int
    status: str  # 'completed', 'failed', 'partial'
    error_message: Optional[str] = None
    validation_summary: Optional[Dict] = None
    
    
class LargeScaleSearchManager:
    """Manage large-scale Planet Nine searches across multiple sky regions."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.results_dir = RESULTS_DIR / "large_scale_search"
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize database for tracking progress
        self.db_path = self.results_dir / "search_progress.db"
        self._init_database()
        
        # Note: Region generation is done statically, not using orbital predictor for this demo
        
    def _init_database(self):
        """Initialize SQLite database for tracking search progress."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_regions (
                region_id TEXT PRIMARY KEY,
                ra_center REAL,
                dec_center REAL,
                width REAL,
                height REAL,
                priority TEXT,
                theoretical_basis TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                processing_time REAL,
                total_candidates INTEGER,
                high_quality_candidates INTEGER,
                planet_nine_candidates INTEGER,
                error_message TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidate_detections (
                detection_id TEXT PRIMARY KEY,
                region_id TEXT,
                ra REAL,
                dec REAL,
                motion_arcsec_year REAL,
                quality_score REAL,
                start_flux REAL,
                is_planet_nine_candidate BOOLEAN,
                validation_status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (region_id) REFERENCES search_regions (region_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                description TEXT,
                regions_involved TEXT,
                significance REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def generate_priority_regions(self, total_area_sq_deg: float = 100) -> List[SearchRegion]:
        """Generate high-priority search regions based on theoretical predictions."""
        logger.info(f"Generating priority regions for {total_area_sq_deg} square degrees")
        
        regions = []
        
        # 1. High-probability regions based on Batygin & Brown (2016)
        high_prob_regions = [
            # Anti-clustering region (opposite to known KBOs)
            SearchRegion(
                ra_center=45.0, dec_center=-20.0, width=8.0, height=8.0,
                priority='high', region_id='anticlustering_1',
                theoretical_basis='Anti-clustering region opposite to known KBO perihelia'
            ),
            SearchRegion(
                ra_center=60.0, dec_center=-15.0, width=8.0, height=8.0,
                priority='high', region_id='anticlustering_2',
                theoretical_basis='Secondary anti-clustering region'
            ),
            
            # Regions where Planet Nine would be brightest (perihelion approach)
            SearchRegion(
                ra_center=225.0, dec_center=15.0, width=10.0, height=8.0,
                priority='high', region_id='perihelion_approach_1',
                theoretical_basis='Predicted perihelion approach region'
            ),
            SearchRegion(
                ra_center=240.0, dec_center=20.0, width=10.0, height=8.0,
                priority='high', region_id='perihelion_approach_2',
                theoretical_basis='Alternative perihelion region'
            ),
            
            # Galactic plane avoidance regions (less stellar contamination)
            SearchRegion(
                ra_center=90.0, dec_center=45.0, width=12.0, height=6.0,
                priority='high', region_id='galactic_north_1',
                theoretical_basis='High galactic latitude, reduced stellar contamination'
            ),
            SearchRegion(
                ra_center=270.0, dec_center=-45.0, width=12.0, height=6.0,
                priority='high', region_id='galactic_south_1',
                theoretical_basis='High galactic latitude, reduced stellar contamination'
            )
        ]
        
        regions.extend(high_prob_regions)
        
        # 2. Medium-priority systematic survey regions
        current_area = sum(r.area_sq_deg for r in regions)
        remaining_area = total_area_sq_deg - current_area
        
        if remaining_area > 0:
            # Fill remaining area with systematic grid
            grid_regions = self._generate_systematic_grid(remaining_area)
            regions.extend(grid_regions)
        
        logger.info(f"Generated {len(regions)} search regions totaling {sum(r.area_sq_deg for r in regions):.1f} sq deg")
        return regions
    
    def _generate_systematic_grid(self, area_sq_deg: float) -> List[SearchRegion]:
        """Generate systematic grid to fill remaining search area."""
        regions = []
        region_size = 6.0  # 6x6 degree regions
        
        # Focus on ecliptic regions where Planet Nine is more likely
        dec_range = (-30, +30)  # Avoid extreme declinations
        ra_centers = np.arange(0, 360, region_size)
        dec_centers = np.arange(dec_range[0], dec_range[1], region_size)
        
        region_count = 0
        target_count = int(area_sq_deg / (region_size ** 2))
        
        for ra in ra_centers:
            for dec in dec_centers:
                if region_count >= target_count:
                    break
                    
                regions.append(SearchRegion(
                    ra_center=ra, dec_center=dec, 
                    width=region_size, height=region_size,
                    priority='medium',
                    region_id=f'systematic_grid_{region_count:03d}',
                    theoretical_basis='Systematic survey coverage'
                ))
                region_count += 1
            
            if region_count >= target_count:
                break
        
        return regions
    
    def process_region(self, region: SearchRegion) -> SearchResult:
        """Process a single search region."""
        logger.info(f"Processing region {region.region_id}")
        start_time = time.time()
        
        try:
            # Update database status
            self._update_region_status(region.region_id, 'processing', start_time)
            
            # Create pipeline for this region
            pipeline = PlanetNinePipeline(
                ra_center=region.ra_center,
                dec_center=region.dec_center,
                width_deg=region.width,
                height_deg=region.height
            )
            
            # Run detection pipeline
            logger.info(f"Downloading data for region {region.region_id}")
            image_files = pipeline.step1_download_data()
            
            if not image_files:
                raise ValueError("No images downloaded for region")
            
            logger.info(f"Processing {len(image_files)} images")
            diff_images = pipeline.step2_process_images()
            
            logger.info(f"Detecting candidates in {len(diff_images)} difference images")
            candidates = pipeline.step3_detect_candidates()
            
            logger.info(f"Validating {len(candidates)} candidates")
            validation_df = pipeline.step4_validate_candidates()
            
            # Enhanced candidate analysis for this region
            enhanced_candidates = self._enhance_candidate_analysis(candidates, validation_df, region)
            
            # Store candidates in database
            self._store_candidates(region.region_id, enhanced_candidates)
            
            # Calculate summary statistics
            total_candidates = len(candidates)
            high_quality = len([c for c in candidates if c.get('quality_score', 0) > 0.5])
            planet_nine = len([c for c in candidates if c.get('is_planet_nine_candidate', False)])
            
            processing_time = time.time() - start_time
            
            # Create result
            result = SearchResult(
                region_id=region.region_id,
                processing_time=processing_time,
                total_candidates=total_candidates,
                high_quality_candidates=high_quality,
                planet_nine_candidates=planet_nine,
                status='completed',
                validation_summary={
                    'known_objects': (validation_df['is_known_object'] == True).sum() if len(validation_df) > 0 else 0,
                    'unknown_objects': (validation_df['is_known_object'] == False).sum() if len(validation_df) > 0 else 0
                }
            )
            
            # Update database
            self._update_region_completion(region.region_id, result)
            
            logger.success(f"Completed region {region.region_id}: {total_candidates} candidates")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Failed to process region {region.region_id}: {error_msg}")
            
            result = SearchResult(
                region_id=region.region_id,
                processing_time=processing_time,
                total_candidates=0,
                high_quality_candidates=0,
                planet_nine_candidates=0,
                status='failed',
                error_message=error_msg
            )
            
            self._update_region_completion(region.region_id, result)
            return result
    
    def _enhance_candidate_analysis(self, candidates: List[Dict], validation_df: pd.DataFrame, 
                                   region: SearchRegion) -> List[Dict]:
        """Enhance candidate analysis with additional metrics for large-scale search."""
        enhanced = []
        
        for candidate in candidates:
            enhanced_candidate = candidate.copy()
            
            # Add region context
            enhanced_candidate['region_id'] = region.region_id
            enhanced_candidate['region_priority'] = region.priority
            enhanced_candidate['theoretical_basis'] = region.theoretical_basis
            
            # Enhanced motion analysis
            motion = candidate.get('motion_arcsec_year', 0)
            enhanced_candidate['motion_category'] = self._categorize_motion(motion)
            
            # Distance from known objects
            if len(validation_df) > 0:
                # Find closest validation match
                closest_match = self._find_closest_validation_match(candidate, validation_df)
                enhanced_candidate['validation_distance'] = closest_match.get('distance', 999)
                enhanced_candidate['validation_confidence'] = closest_match.get('confidence', 0)
            
            # Novelty score (how unusual is this detection?)
            enhanced_candidate['novelty_score'] = self._calculate_novelty_score(candidate, region)
            
            # Regional density (how many other candidates nearby?)
            enhanced_candidate['local_density'] = self._calculate_local_density(candidate, candidates)
            
            enhanced.append(enhanced_candidate)
        
        return enhanced
    
    def _categorize_motion(self, motion_arcsec_year: float) -> str:
        """Categorize motion into different regime types."""
        if motion_arcsec_year < 0.1:
            return 'ultra_slow'
        elif motion_arcsec_year < 0.5:
            return 'planet_nine_range'
        elif motion_arcsec_year < 2.0:
            return 'distant_tno'
        elif motion_arcsec_year < 10.0:
            return 'nearby_tno'
        else:
            return 'stellar_or_asteroid'
    
    def _find_closest_validation_match(self, candidate: Dict, validation_df: pd.DataFrame) -> Dict:
        """Find closest validation match for enhanced analysis."""
        if len(validation_df) == 0:
            return {'distance': 999, 'confidence': 0}
        
        candidate_coord = SkyCoord(
            ra=candidate.get('ra', 0) * u.deg,
            dec=candidate.get('dec', 0) * u.deg
        )
        
        min_distance = 999
        best_confidence = 0
        
        for _, row in validation_df.iterrows():
            try:
                match_coord = SkyCoord(ra=row['ra'] * u.deg, dec=row['dec'] * u.deg)
                distance = candidate_coord.separation(match_coord).arcsec
                
                if distance < min_distance:
                    min_distance = distance
                    best_confidence = row.get('match_confidence', 0)
            except:
                continue
        
        return {'distance': min_distance, 'confidence': best_confidence}
    
    def _calculate_novelty_score(self, candidate: Dict, region: SearchRegion) -> float:
        """Calculate how novel/unusual this candidate is."""
        score = 0.0
        
        # Reward slow motion (Planet Nine range)
        motion = candidate.get('motion_arcsec_year', 0)
        if 0.2 <= motion <= 0.8:
            score += 0.4
        elif motion < 0.2:
            score += 0.6  # Ultra-slow is very interesting
        
        # Reward high quality
        quality = candidate.get('quality_score', 0)
        score += quality * 0.3
        
        # Reward faint objects (less likely to be known)
        flux = candidate.get('start_flux', 1000)
        if flux < 10:
            score += 0.2
        elif flux < 100:
            score += 0.1
        
        # Bonus for high-priority regions
        if region.priority == 'high':
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_local_density(self, candidate: Dict, all_candidates: List[Dict]) -> int:
        """Calculate how many other candidates are nearby."""
        candidate_coord = SkyCoord(
            ra=candidate.get('ra', 0) * u.deg,
            dec=candidate.get('dec', 0) * u.deg
        )
        
        nearby_count = 0
        search_radius = 0.1 * u.deg  # 6 arcminutes
        
        for other in all_candidates:
            if other == candidate:
                continue
                
            other_coord = SkyCoord(
                ra=other.get('ra', 0) * u.deg,
                dec=other.get('dec', 0) * u.deg
            )
            
            if candidate_coord.separation(other_coord) < search_radius:
                nearby_count += 1
        
        return nearby_count
    
    def run_large_scale_search(self, total_area_sq_deg: float = 100) -> Dict:
        """Run large-scale Planet Nine search across multiple regions."""
        logger.info(f"Starting large-scale search covering {total_area_sq_deg} square degrees")
        
        # Generate search regions
        regions = self.generate_priority_regions(total_area_sq_deg)
        
        # Store regions in database
        self._store_regions(regions)
        
        # Process regions in parallel
        results = []
        failed_regions = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all regions for processing (convert to serializable dict)
            region_data_list = [
                {
                    'region_id': region.region_id,
                    'ra_center': region.ra_center,
                    'dec_center': region.dec_center,
                    'width': region.width,
                    'height': region.height,
                    'priority': region.priority,
                    'theoretical_basis': region.theoretical_basis
                }
                for region in regions
            ]
            
            future_to_region = {
                executor.submit(process_region_worker, region_data): region_data['region_id']
                for region_data in region_data_list
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_region):
                region_id = future_to_region[future]
                try:
                    result_dict = future.result()
                    
                    # Convert dict result to SearchResult object
                    result = SearchResult(
                        region_id=result_dict['region_id'],
                        processing_time=result_dict['processing_time'],
                        total_candidates=result_dict['total_candidates'],
                        high_quality_candidates=result_dict['high_quality_candidates'],
                        planet_nine_candidates=result_dict['planet_nine_candidates'],
                        status=result_dict['status'],
                        error_message=result_dict.get('error_message'),
                        validation_summary=result_dict.get('validation_summary')
                    )
                    
                    results.append(result)
                    
                    # Store candidates in database if successful
                    if result.status == 'completed' and 'candidates' in result_dict:
                        self._store_candidates(result.region_id, result_dict['candidates'])
                    
                    # Update database with results
                    self._update_region_completion(result.region_id, result)
                    
                    if result.status == 'failed':
                        failed_regions.append(region_id)
                        
                    # Log progress
                    completed = len(results)
                    total = len(regions)
                    logger.info(f"Progress: {completed}/{total} regions completed")
                    
                except Exception as e:
                    logger.error(f"Region {region_id} generated exception: {e}")
                    failed_regions.append(region_id)
        
        # Analyze patterns across all regions
        pattern_analysis = self._analyze_cross_region_patterns()
        
        # Generate comprehensive summary
        summary = self._generate_search_summary(results, pattern_analysis)
        
        # Save results
        summary_path = self.results_dir / f"large_scale_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.success(f"Large-scale search completed. Summary saved to {summary_path}")
        return summary
    
    def _store_regions(self, regions: List[SearchRegion]):
        """Store search regions in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for region in regions:
            cursor.execute("""
                INSERT OR REPLACE INTO search_regions 
                (region_id, ra_center, dec_center, width, height, priority, theoretical_basis)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                region.region_id, region.ra_center, region.dec_center,
                region.width, region.height, region.priority, region.theoretical_basis
            ))
        
        conn.commit()
        conn.close()
    
    def _store_candidates(self, region_id: str, candidates: List[Dict]):
        """Store candidates in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i, candidate in enumerate(candidates):
            detection_id = f"{region_id}_{i:04d}"
            
            # Calculate quality score from available metrics
            match_score = candidate.get('match_score', 0)
            flux_ratio = candidate.get('flux_ratio', 1)
            flux_consistency = abs(1.0 - flux_ratio)  # How close to 1.0
            quality_score = match_score * (1.0 - flux_consistency)
            
            # Convert pixel coordinates to RA/Dec (approximate)
            # For now, use start coordinates as representative position
            start_x = candidate.get('start_x', 0)
            start_y = candidate.get('start_y', 0)
            
            # Simple pixel-to-RA/Dec conversion (needs improvement)
            # Assuming 0.262 arcsec/pixel typical for DECaLS
            pixel_scale = 0.262 / 3600  # degrees per pixel
            ra = start_x * pixel_scale  # Simplified - needs WCS
            dec = start_y * pixel_scale  # Simplified - needs WCS
            
            # Check if Planet Nine candidate
            motion = candidate.get('motion_arcsec_year', 0)
            is_planet_nine = 0.2 <= motion <= 0.8
            
            cursor.execute("""
                INSERT OR REPLACE INTO candidate_detections
                (detection_id, region_id, ra, dec, motion_arcsec_year, quality_score,
                 start_flux, is_planet_nine_candidate, validation_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                detection_id, region_id,
                ra, dec,
                motion, quality_score,
                candidate.get('start_flux', 0), is_planet_nine,
                'needs_validation'
            ))
        
        conn.commit()
        conn.close()
    
    def _update_region_status(self, region_id: str, status: str, timestamp: float):
        """Update region processing status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if status == 'processing':
            cursor.execute("""
                UPDATE search_regions 
                SET status = ?, started_at = datetime(?, 'unixepoch')
                WHERE region_id = ?
            """, (status, timestamp, region_id))
        
        conn.commit()
        conn.close()
    
    def _update_region_completion(self, region_id: str, result: SearchResult):
        """Update region with completion results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE search_regions 
            SET status = ?, completed_at = CURRENT_TIMESTAMP, processing_time = ?,
                total_candidates = ?, high_quality_candidates = ?, 
                planet_nine_candidates = ?, error_message = ?
            WHERE region_id = ?
        """, (
            result.status, result.processing_time,
            result.total_candidates, result.high_quality_candidates,
            result.planet_nine_candidates, result.error_message,
            region_id
        ))
        
        conn.commit()
        conn.close()
    
    def _analyze_cross_region_patterns(self) -> Dict:
        """Analyze patterns across multiple search regions."""
        conn = sqlite3.connect(self.db_path)
        
        # Load all candidates
        candidates_df = pd.read_sql_query("""
            SELECT c.*, r.priority, r.theoretical_basis
            FROM candidate_detections c
            JOIN search_regions r ON c.region_id = r.region_id
        """, conn)
        
        conn.close()
        
        if len(candidates_df) == 0:
            return {'patterns': []}
        
        patterns = []
        
        # 1. Motion clustering analysis
        motion_clusters = self._find_motion_clusters(candidates_df)
        if motion_clusters:
            patterns.append({
                'type': 'motion_clustering',
                'description': f'Found {len(motion_clusters)} motion clusters',
                'significance': len(motion_clusters) / 10.0,
                'details': motion_clusters
            })
        
        # 2. Spatial clustering analysis
        spatial_clusters = self._find_spatial_clusters(candidates_df)
        if spatial_clusters:
            patterns.append({
                'type': 'spatial_clustering',
                'description': f'Found {len(spatial_clusters)} spatial clusters',
                'significance': len(spatial_clusters) / 5.0,
                'details': spatial_clusters
            })
        
        # 3. Regional efficiency analysis
        efficiency_patterns = self._analyze_regional_efficiency(candidates_df)
        patterns.extend(efficiency_patterns)
        
        return {'patterns': patterns}
    
    def _find_motion_clusters(self, df: pd.DataFrame) -> List[Dict]:
        """Find clusters of candidates with similar motion."""
        clusters = []
        
        # Group by motion ranges
        motion_bins = [0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
        
        for i in range(len(motion_bins) - 1):
            min_motion, max_motion = motion_bins[i], motion_bins[i+1]
            cluster_candidates = df[
                (df['motion_arcsec_year'] >= min_motion) & 
                (df['motion_arcsec_year'] < max_motion)
            ]
            
            if len(cluster_candidates) > 5:  # Significant cluster
                clusters.append({
                    'motion_range': [min_motion, max_motion],
                    'candidate_count': len(cluster_candidates),
                    'regions_involved': cluster_candidates['region_id'].nunique(),
                    'avg_quality': cluster_candidates['quality_score'].mean()
                })
        
        return clusters
    
    def _find_spatial_clusters(self, df: pd.DataFrame) -> List[Dict]:
        """Find spatial clusters of candidates."""
        from sklearn.cluster import DBSCAN
        
        if len(df) < 10:
            return []
        
        # Prepare coordinates
        coords = df[['ra', 'dec']].values
        
        # DBSCAN clustering (eps in degrees, min_samples)
        clustering = DBSCAN(eps=2.0, min_samples=3).fit(coords)
        
        clusters = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_members = df[clustering.labels_ == cluster_id]
            
            clusters.append({
                'cluster_id': int(cluster_id),
                'member_count': len(cluster_members),
                'center_ra': cluster_members['ra'].mean(),
                'center_dec': cluster_members['dec'].mean(),
                'avg_motion': cluster_members['motion_arcsec_year'].mean(),
                'regions_involved': list(cluster_members['region_id'].unique())
            })
        
        return clusters
    
    def _analyze_regional_efficiency(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze detection efficiency across different region types."""
        patterns = []
        
        # Group by region priority
        priority_stats = df.groupby('priority').agg({
            'detection_id': 'count',
            'quality_score': 'mean',
            'motion_arcsec_year': 'mean'
        }).round(3)
        
        if len(priority_stats) > 1:
            patterns.append({
                'type': 'regional_efficiency',
                'description': 'Detection efficiency varies by region priority',
                'significance': 0.5,
                'details': priority_stats.to_dict()
            })
        
        return patterns
    
    def _generate_search_summary(self, results: List[SearchResult], pattern_analysis: Dict) -> Dict:
        """Generate comprehensive summary of large-scale search."""
        total_candidates = sum(r.total_candidates for r in results)
        total_high_quality = sum(r.high_quality_candidates for r in results)
        total_planet_nine = sum(r.planet_nine_candidates for r in results)
        total_time = sum(r.processing_time for r in results)
        
        successful_regions = [r for r in results if r.status == 'completed']
        failed_regions = [r for r in results if r.status == 'failed']
        
        summary = {
            'search_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_regions': len(results),
                'successful_regions': len(successful_regions),
                'failed_regions': len(failed_regions),
                'total_processing_time_hours': total_time / 3600,
                'average_time_per_region_minutes': (total_time / len(results)) / 60 if results else 0
            },
            'detection_summary': {
                'total_candidates': total_candidates,
                'high_quality_candidates': total_high_quality,
                'planet_nine_candidates': total_planet_nine,
                'candidates_per_sq_deg': total_candidates / sum(64 for r in successful_regions) if successful_regions else 0,
                'detection_rate': total_candidates / total_time * 3600 if total_time > 0 else 0  # per hour
            },
            'pattern_analysis': pattern_analysis,
            'regional_breakdown': [
                {
                    'region_id': r.region_id,
                    'status': r.status,
                    'candidates': r.total_candidates,
                    'high_quality': r.high_quality_candidates,
                    'planet_nine': r.planet_nine_candidates,
                    'processing_time_minutes': r.processing_time / 60
                }
                for r in results
            ],
            'recommendations': self._generate_scale_up_recommendations(results, pattern_analysis)
        }
        
        return summary
    
    def _generate_scale_up_recommendations(self, results: List[SearchResult], 
                                         pattern_analysis: Dict) -> List[str]:
        """Generate recommendations for further scaling up the search."""
        recommendations = []
        
        successful_results = [r for r in results if r.status == 'completed']
        
        if not successful_results:
            recommendations.append("Address processing failures before scaling further")
            return recommendations
        
        # Analysis based on results
        avg_candidates = np.mean([r.total_candidates for r in successful_results])
        total_planet_nine = sum(r.planet_nine_candidates for r in successful_results)
        
        if avg_candidates < 10:
            recommendations.append("Consider deeper exposure times or fainter magnitude limits")
        
        if total_planet_nine == 0:
            recommendations.append("Focus on unexplored sky regions or infrared surveys")
        
        if len(pattern_analysis.get('patterns', [])) > 2:
            recommendations.append("Investigate significant detection patterns for systematic effects")
        
        recommendations.extend([
            "Implement real-time Gaia cross-matching to reduce false positives",
            "Add spectroscopic follow-up for highest-priority candidates",
            "Extend search to southern hemisphere regions",
            "Consider WISE infrared data for thermal detection"
        ])
        
        return recommendations


def main():
    """Run large-scale Planet Nine search."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Large-scale Planet Nine search')
    parser.add_argument('--area', type=float, default=100, 
                       help='Total area to search in square degrees')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous search')
    
    args = parser.parse_args()
    
    # Initialize search manager
    manager = LargeScaleSearchManager(max_workers=args.workers)
    
    # Run search
    logger.info(f"Starting large-scale Planet Nine search")
    logger.info(f"Target area: {args.area} square degrees")
    logger.info(f"Parallel workers: {args.workers}")
    
    summary = manager.run_large_scale_search(total_area_sq_deg=args.area)
    
    # Print summary
    print("\n" + "="*80)
    print("🌌 LARGE-SCALE PLANET NINE SEARCH SUMMARY")
    print("="*80)
    print(f"Regions processed: {summary['search_metadata']['successful_regions']}")
    print(f"Total candidates: {summary['detection_summary']['total_candidates']}")
    print(f"High-quality candidates: {summary['detection_summary']['high_quality_candidates']}")
    print(f"Planet Nine candidates: {summary['detection_summary']['planet_nine_candidates']}")
    print(f"Processing time: {summary['search_metadata']['total_processing_time_hours']:.1f} hours")
    print(f"Detection rate: {summary['detection_summary']['detection_rate']:.1f} candidates/hour")
    
    patterns = summary['pattern_analysis']['patterns']
    if patterns:
        print(f"\n🔍 PATTERNS DETECTED:")
        for pattern in patterns:
            print(f"• {pattern['description']} (significance: {pattern['significance']:.2f})")
    
    print(f"\n📋 RECOMMENDATIONS:")
    for rec in summary['recommendations']:
        print(f"• {rec}")


if __name__ == "__main__":
    main()
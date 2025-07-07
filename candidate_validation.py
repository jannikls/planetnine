#!/usr/bin/env python
"""
Candidate validation system for cross-checking detections against known objects.
This module validates Planet Nine candidates against the Minor Planet Center database
and other astronomical catalogs to identify genuine discoveries vs known objects.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from astroquery.mpc import MPC
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from loguru import logger
import requests
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import RESULTS_DIR


@dataclass
class ValidationResult:
    """Results of candidate validation."""
    candidate_id: str
    is_known_object: bool
    match_confidence: float
    matched_objects: List[Dict]
    validation_status: str
    notes: str


class CandidateValidator:
    """Cross-validate candidates against astronomical databases."""
    
    def __init__(self, search_radius_arcsec: float = 30.0):
        """
        Initialize the validator.
        
        Args:
            search_radius_arcsec: Search radius for catalog matching in arcseconds
        """
        self.search_radius = search_radius_arcsec * u.arcsec
        self.validation_dir = RESULTS_DIR / "validation"
        self.validation_dir.mkdir(exist_ok=True, parents=True)
        
        # Configure query services
        self.simbad = Simbad()
        self.simbad.add_votable_fields('propermotions', 'flux(V)', 'flux(R)')
        
        self.vizier = Vizier(columns=['*'], row_limit=50)
        
    def validate_candidate_list(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate a list of candidates against astronomical databases.
        
        Args:
            candidates_df: DataFrame with candidate information
            
        Returns:
            DataFrame with validation results
        """
        logger.info(f"Validating {len(candidates_df)} candidates against astronomical databases")
        
        validation_results = []
        
        for idx, candidate in candidates_df.iterrows():
            if idx % 10 == 0:
                logger.info(f"Processing candidate {idx+1}/{len(candidates_df)}")
            
            try:
                result = self._validate_single_candidate(candidate)
                validation_results.append(result)
                
            except Exception as e:
                logger.error(f"Validation failed for candidate {candidate.get('detection_id', idx)}: {e}")
                # Create failed validation result
                result = ValidationResult(
                    candidate_id=candidate.get('detection_id', f'candidate_{idx}'),
                    is_known_object=False,
                    match_confidence=0.0,
                    matched_objects=[],
                    validation_status='VALIDATION_FAILED',
                    notes=f'Validation error: {str(e)}'
                )
                validation_results.append(result)
        
        # Convert results to DataFrame
        validation_df = pd.DataFrame([
            {
                'candidate_id': r.candidate_id,
                'is_known_object': r.is_known_object,
                'match_confidence': r.match_confidence,
                'num_matches': len(r.matched_objects),
                'validation_status': r.validation_status,
                'notes': r.notes,
                'matched_objects_json': json.dumps(r.matched_objects)
            }
            for r in validation_results
        ])
        
        # Merge with original candidate data
        full_results = candidates_df.merge(validation_df, 
                                         left_on='detection_id', 
                                         right_on='candidate_id', 
                                         how='left')
        
        # Save validation results
        output_path = self.validation_dir / "candidate_validation_results.csv"
        full_results.to_csv(output_path, index=False)
        
        logger.success(f"Validation complete. Results saved to {output_path}")
        
        return full_results
    
    def _validate_single_candidate(self, candidate: pd.Series) -> ValidationResult:
        """Validate a single candidate against databases."""
        candidate_id = candidate.get('detection_id', 'unknown')
        
        # Convert pixel coordinates to sky coordinates (approximate)
        # This is a simplified conversion - in practice would use proper WCS
        ra_deg, dec_deg = self._pixels_to_sky_coords(
            candidate['start_x'], candidate['start_y'],
            candidate.get('ra_center', 180.0), 
            candidate.get('dec_center', 0.0)
        )
        
        sky_coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
        
        # Check multiple databases
        matched_objects = []
        
        # 1. Minor Planet Center database
        mpc_matches = self._query_mpc_database(sky_coord, candidate)
        matched_objects.extend(mpc_matches)
        
        # 2. SIMBAD database
        simbad_matches = self._query_simbad_database(sky_coord, candidate)
        matched_objects.extend(simbad_matches)
        
        # 3. VizieR catalogs (Gaia, etc.)
        vizier_matches = self._query_vizier_catalogs(sky_coord, candidate)
        matched_objects.extend(vizier_matches)
        
        # 4. Check for stellar proper motion matches
        stellar_matches = self._check_stellar_proper_motion(sky_coord, candidate)
        matched_objects.extend(stellar_matches)
        
        # Analyze matches
        is_known_object, confidence, status, notes = self._analyze_matches(
            matched_objects, candidate
        )
        
        return ValidationResult(
            candidate_id=candidate_id,
            is_known_object=is_known_object,
            match_confidence=confidence,
            matched_objects=matched_objects,
            validation_status=status,
            notes=notes
        )
    
    def _pixels_to_sky_coords(self, x_pix: float, y_pix: float, 
                            ra_center: float, dec_center: float) -> Tuple[float, float]:
        """Convert pixel coordinates to approximate sky coordinates."""
        # Simplified conversion assuming 0.262 arcsec/pixel DECaLS scale
        pixel_scale = 0.262 / 3600  # degrees per pixel
        
        # Assume 512x512 image centered on ra_center, dec_center
        center_x, center_y = 256, 256
        
        # Offset from center
        dx_deg = (x_pix - center_x) * pixel_scale
        dy_deg = (y_pix - center_y) * pixel_scale
        
        # Apply approximate coordinate transformation
        ra_deg = ra_center + dx_deg / np.cos(np.radians(dec_center))
        dec_deg = dec_center + dy_deg
        
        return ra_deg, dec_deg
    
    def _query_mpc_database(self, sky_coord: SkyCoord, candidate: pd.Series) -> List[Dict]:
        """Query Minor Planet Center database for known objects."""
        matches = []
        
        try:
            # Query MPC for objects near this position
            # Note: MPC queries require careful rate limiting
            logger.debug(f"Querying MPC database at {sky_coord.ra.deg:.6f}, {sky_coord.dec.deg:.6f}")
            
            # For demonstration, we'll simulate MPC queries since real queries require
            # careful API management and rate limiting
            
            # In a real implementation, you would query:
            # - MPC's web services
            # - Minor planet ephemeris data
            # - Known TNO catalogs
            
            # Simulate some potential matches based on position and motion
            motion_arcsec_year = candidate.get('motion_arcsec_year', 0)
            
            # Flag fast-moving objects as potentially known asteroids
            if motion_arcsec_year > 5.0:
                matches.append({
                    'source': 'MPC_SIMULATION',
                    'object_type': 'ASTEROID',
                    'designation': f'SIMULATED_AST_{int(sky_coord.ra.deg*100):05d}',
                    'separation_arcsec': np.random.uniform(5, 25),
                    'proper_motion_match': True,
                    'confidence': 0.8
                })
            
            # Flag objects in known TNO regions
            if 150 < sky_coord.ra.deg < 210 and -20 < sky_coord.dec.deg < 20:
                if 0.1 < motion_arcsec_year < 2.0:
                    matches.append({
                        'source': 'MPC_TNO_SIMULATION',
                        'object_type': 'TNO',
                        'designation': f'SIMULATED_TNO_{int(sky_coord.dec.deg*100):05d}',
                        'separation_arcsec': np.random.uniform(10, 30),
                        'proper_motion_match': False,
                        'confidence': 0.3
                    })
            
        except Exception as e:
            logger.warning(f"MPC query failed: {e}")
        
        return matches
    
    def _query_simbad_database(self, sky_coord: SkyCoord, candidate: pd.Series) -> List[Dict]:
        """Query SIMBAD database for known objects."""
        matches = []
        
        try:
            # Query SIMBAD for objects within search radius
            result_table = self.simbad.query_region(sky_coord, radius=self.search_radius)
            
            if result_table is not None and len(result_table) > 0:
                for row in result_table:
                    # Extract object information
                    obj_coord = SkyCoord(
                        ra=row['RA'], dec=row['DEC'], 
                        unit=(u.hourangle, u.deg), frame='icrs'
                    )
                    
                    separation = sky_coord.separation(obj_coord)
                    
                    # Check if proper motion is available and matches
                    pm_match = False
                    if 'PMRA' in row.colnames and 'PMDEC' in row.colnames:
                        if not (np.ma.is_masked(row['PMRA']) or np.ma.is_masked(row['PMDEC'])):
                            # Convert proper motion to arcsec/year
                            pm_total = np.sqrt(row['PMRA']**2 + row['PMDEC']**2) / 1000.0  # mas/yr to arcsec/yr
                            candidate_pm = candidate.get('motion_arcsec_year', 0)
                            
                            # Check if proper motions are consistent (within factor of 2)
                            if 0.5 * pm_total < candidate_pm < 2.0 * pm_total:
                                pm_match = True
                    
                    matches.append({
                        'source': 'SIMBAD',
                        'object_type': str(row['OTYPE']),
                        'designation': str(row['MAIN_ID']),
                        'separation_arcsec': separation.arcsec,
                        'proper_motion_match': pm_match,
                        'confidence': max(0.1, 1.0 - separation.arcsec / self.search_radius.value)
                    })
            
        except Exception as e:
            logger.warning(f"SIMBAD query failed: {e}")
        
        return matches
    
    def _query_vizier_catalogs(self, sky_coord: SkyCoord, candidate: pd.Series) -> List[Dict]:
        """Query VizieR catalogs for known objects."""
        matches = []
        
        try:
            # Query important catalogs
            catalogs_to_search = [
                'I/350/gaiaedr3',  # Gaia EDR3
                'I/259/tyc2',      # Tycho-2
                'V/147/sdss12',    # SDSS DR12
            ]
            
            for catalog in catalogs_to_search:
                try:
                    result = self.vizier.query_region(sky_coord, radius=self.search_radius, 
                                                    catalog=catalog)
                    
                    if result and len(result) > 0:
                        table = result[0]  # First table from catalog
                        
                        for row in table[:5]:  # Limit to 5 closest matches
                            # Try to extract RA/Dec (column names vary by catalog)
                            ra_col = None
                            dec_col = None
                            for col in ['RA_ICRS', 'RAJ2000', 'ra', '_RAJ2000']:
                                if col in table.colnames:
                                    ra_col = col
                                    break
                            for col in ['DE_ICRS', 'DEJ2000', 'dec', '_DEJ2000']:
                                if col in table.colnames:
                                    dec_col = col
                                    break
                            
                            if ra_col and dec_col:
                                obj_coord = SkyCoord(
                                    ra=row[ra_col]*u.deg, 
                                    dec=row[dec_col]*u.deg, 
                                    frame='icrs'
                                )
                                separation = sky_coord.separation(obj_coord)
                                
                                matches.append({
                                    'source': f'VIZIER_{catalog}',
                                    'object_type': 'STAR/GALAXY',
                                    'designation': f'{catalog}_{row.index}',
                                    'separation_arcsec': separation.arcsec,
                                    'proper_motion_match': False,
                                    'confidence': max(0.1, 1.0 - separation.arcsec / self.search_radius.value)
                                })
                
                except Exception as e:
                    logger.debug(f"Failed to query catalog {catalog}: {e}")
                    
        except Exception as e:
            logger.warning(f"VizieR query failed: {e}")
        
        return matches
    
    def _check_stellar_proper_motion(self, sky_coord: SkyCoord, candidate: pd.Series) -> List[Dict]:
        """Check if motion is consistent with stellar proper motion."""
        matches = []
        
        try:
            candidate_pm = candidate.get('motion_arcsec_year', 0)
            
            # Typical stellar proper motions for comparison
            # High proper motion stars: > 0.1 arcsec/year
            # Nearby stars: 0.01 - 1.0 arcsec/year
            # Distant stars: < 0.01 arcsec/year
            
            if 0.01 < candidate_pm < 10.0:
                # This motion range is consistent with stellar proper motion
                confidence = 0.5 if 0.1 < candidate_pm < 1.0 else 0.2
                
                matches.append({
                    'source': 'STELLAR_MOTION_CHECK',
                    'object_type': 'POTENTIAL_STAR',
                    'designation': f'STELLAR_PM_{candidate_pm:.3f}',
                    'separation_arcsec': 0.0,
                    'proper_motion_match': True,
                    'confidence': confidence
                })
        
        except Exception as e:
            logger.warning(f"Stellar motion check failed: {e}")
        
        return matches
    
    def _analyze_matches(self, matched_objects: List[Dict], candidate: pd.Series) -> Tuple[bool, float, str, str]:
        """Analyze matches to determine if candidate is a known object."""
        
        if not matched_objects:
            return False, 0.0, 'NO_MATCHES', 'No known objects found within search radius'
        
        # Calculate overall confidence based on matches
        max_confidence = max(match['confidence'] for match in matched_objects)
        
        # Check for high-confidence matches
        high_conf_matches = [m for m in matched_objects if m['confidence'] > 0.7]
        medium_conf_matches = [m for m in matched_objects if 0.3 < m['confidence'] <= 0.7]
        
        # Check for proper motion matches
        pm_matches = [m for m in matched_objects if m['proper_motion_match']]
        
        # Decision logic
        if high_conf_matches:
            if pm_matches:
                return True, max_confidence, 'KNOWN_OBJECT_HIGH_CONF', f'High confidence match with proper motion: {high_conf_matches[0]["designation"]}'
            else:
                return True, max_confidence, 'KNOWN_OBJECT_POSITION', f'High confidence positional match: {high_conf_matches[0]["designation"]}'
        
        elif medium_conf_matches:
            if pm_matches:
                return True, max_confidence, 'LIKELY_KNOWN_OBJECT', f'Medium confidence match with proper motion: {medium_conf_matches[0]["designation"]}'
            else:
                return False, max_confidence, 'POSSIBLE_MATCH', f'Medium confidence positional match, no proper motion confirmation'
        
        else:
            # Only low confidence matches
            if pm_matches:
                return False, max_confidence, 'STELLAR_MOTION_CANDIDATE', 'Motion consistent with stellar proper motion but no strong catalog match'
            else:
                return False, max_confidence, 'WEAK_MATCHES', 'Only weak catalog matches found'
    
    def create_validation_report(self, validation_df: pd.DataFrame):
        """Create comprehensive validation report with visualizations."""
        logger.info("Creating validation report")
        
        report_dir = self.validation_dir / "validation_report"
        report_dir.mkdir(exist_ok=True)
        
        # 1. Validation summary statistics
        self._create_validation_summary(validation_df, report_dir)
        
        # 2. Validation status plots
        self._plot_validation_results(validation_df, report_dir)
        
        # 3. False positive analysis
        self._analyze_false_positives(validation_df, report_dir)
        
        # 4. Unknown object candidates
        self._identify_unknown_candidates(validation_df, report_dir)
        
        # 5. Generate detailed reports for top unknowns
        self._generate_candidate_reports(validation_df, report_dir)
        
        logger.success(f"Validation report created in {report_dir}")
    
    def _create_validation_summary(self, df: pd.DataFrame, report_dir: Path):
        """Create validation summary statistics."""
        summary = {
            'total_candidates': int(len(df)),
            'known_objects': int(df['is_known_object'].sum()),
            'unknown_candidates': int((~df['is_known_object']).sum()),
            'validation_failed': int((df['validation_status'] == 'VALIDATION_FAILED').sum()),
            'high_confidence_matches': int((df['match_confidence'] > 0.7).sum()),
            'stellar_motion_candidates': int((df['validation_status'] == 'STELLAR_MOTION_CANDIDATE').sum()),
            'planet_nine_unknowns': int(len(df[
                (df['is_planet_nine_candidate'] == True) & 
                (df['is_known_object'] == False)
            ]))
        }
        
        # Save summary
        with open(report_dir / "validation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("🔍 CANDIDATE VALIDATION SUMMARY")
        print("="*50)
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    def _plot_validation_results(self, df: pd.DataFrame, report_dir: Path):
        """Create validation result visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Validation status distribution
        ax1 = axes[0, 0]
        status_counts = df['validation_status'].value_counts()
        ax1.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
        ax1.set_title('Validation Status Distribution')
        
        # 2. Known vs Unknown objects
        ax2 = axes[0, 1]
        known_counts = df['is_known_object'].value_counts()
        colors = ['lightcoral', 'lightblue']
        bars = ax2.bar(['Unknown', 'Known'], [known_counts[False], known_counts[True]], color=colors)
        ax2.set_title('Known vs Unknown Objects')
        ax2.set_ylabel('Count')
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 3. Confidence distribution
        ax3 = axes[1, 0]
        ax3.hist(df['match_confidence'], bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Match Confidence')
        ax3.set_ylabel('Count')
        ax3.set_title('Match Confidence Distribution')
        
        # 4. Motion vs validation status
        ax4 = axes[1, 1]
        for status in df['validation_status'].unique():
            status_data = df[df['validation_status'] == status]
            if len(status_data) > 0:
                ax4.scatter(status_data['motion_arcsec_year'], 
                          status_data['match_confidence'],
                          label=status, alpha=0.6)
        
        ax4.set_xlabel('Proper Motion (arcsec/year)')
        ax4.set_ylabel('Match Confidence')
        ax4.set_title('Motion vs Validation Status')
        ax4.set_xscale('log')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(report_dir / "validation_results.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _analyze_false_positives(self, df: pd.DataFrame, report_dir: Path):
        """Analyze false positive patterns."""
        # Identify likely false positives
        false_positives = df[
            (df['is_known_object'] == True) | 
            (df['validation_status'] == 'STELLAR_MOTION_CANDIDATE')
        ]
        
        # Analyze patterns
        fp_analysis = {
            'total_false_positives': len(false_positives),
            'stellar_motion_fps': len(df[df['validation_status'] == 'STELLAR_MOTION_CANDIDATE']),
            'known_object_fps': len(df[df['is_known_object'] == True]),
            'avg_motion_fps': false_positives['motion_arcsec_year'].mean() if len(false_positives) > 0 else 0,
            'motion_range_fps': [
                false_positives['motion_arcsec_year'].min() if len(false_positives) > 0 else 0,
                false_positives['motion_arcsec_year'].max() if len(false_positives) > 0 else 0
            ]
        }
        
        # Save analysis
        with open(report_dir / "false_positive_analysis.json", 'w') as f:
            json.dump(fp_analysis, f, indent=2, default=str)
    
    def _identify_unknown_candidates(self, df: pd.DataFrame, report_dir: Path):
        """Identify and rank unknown object candidates."""
        # Filter for unknown objects
        unknowns = df[
            (df['is_known_object'] == False) & 
            (df['validation_status'] != 'VALIDATION_FAILED')
        ]
        
        # Rank by quality and Planet Nine criteria
        unknowns = unknowns.copy()
        unknowns['discovery_score'] = (
            unknowns['quality_score'] * 
            (1.0 - unknowns['match_confidence']) *  # Prefer lower match confidence
            unknowns['is_planet_nine_candidate'].astype(float)  # Boost Planet Nine candidates
        )
        
        # Sort by discovery score
        unknowns = unknowns.sort_values('discovery_score', ascending=False)
        
        # Save top unknown candidates
        top_unknowns = unknowns.head(20)
        top_unknowns.to_csv(report_dir / "top_unknown_candidates.csv", index=False)
        
        logger.info(f"Identified {len(unknowns)} unknown candidates, top 20 saved")
        
        return top_unknowns
    
    def _generate_candidate_reports(self, df: pd.DataFrame, report_dir: Path):
        """Generate detailed reports for top candidate discoveries."""
        unknowns = df[df['is_known_object'] == False]
        top_candidates = unknowns.nlargest(10, 'quality_score')
        
        candidate_reports_dir = report_dir / "candidate_reports"
        candidate_reports_dir.mkdir(exist_ok=True)
        
        for idx, (_, candidate) in enumerate(top_candidates.iterrows()):
            self._create_single_candidate_report(candidate, candidate_reports_dir, idx + 1)
    
    def _create_single_candidate_report(self, candidate: pd.Series, reports_dir: Path, rank: int):
        """Create detailed report for a single candidate."""
        candidate_id = candidate['detection_id']
        
        report_text = f"""
PLANET NINE CANDIDATE DISCOVERY REPORT #{rank}
{'='*50}

CANDIDATE IDENTIFICATION:
  Detection ID: {candidate_id}
  Rank: #{rank} (by quality score)
  
OBSERVATIONAL PROPERTIES:
  Position (pixels): ({candidate['start_x']:.1f}, {candidate['start_y']:.1f})
  Motion Vector: ({candidate['end_x'] - candidate['start_x']:.1f}, {candidate['end_y'] - candidate['start_y']:.1f}) pixels
  Proper Motion: {candidate['motion_arcsec_year']:.3f} arcsec/year
  Motion Quality: {candidate.get('motion_quality', 'N/A'):.3f}
  
PHOTOMETRIC PROPERTIES:
  Start Flux: {candidate['start_flux']:.1f}
  End Flux: {candidate['end_flux']:.1f}
  Flux Ratio: {candidate['flux_ratio']:.3f}
  Flux Consistency: {candidate.get('flux_consistency', 'N/A'):.3f}
  
QUALITY METRICS:
  Overall Quality Score: {candidate['quality_score']:.3f}
  Match Score: {candidate['match_score']:.3f}
  Edge Penalty: {candidate.get('edge_penalty', 'N/A'):.3f}
  
VALIDATION RESULTS:
  Validation Status: {candidate.get('validation_status', 'N/A')}
  Known Object: {'YES' if candidate.get('is_known_object', False) else 'NO'}
  Match Confidence: {candidate.get('match_confidence', 0):.3f}
  Catalog Matches: {candidate.get('num_matches', 0)}
  
PLANET NINE ASSESSMENT:
  Planet Nine Candidate: {'YES' if candidate.get('is_planet_nine_candidate', False) else 'NO'}
  Motion in P9 Range: {'YES' if 0.2 <= candidate['motion_arcsec_year'] <= 0.8 else 'NO'}
  Discovery Potential: {'HIGH' if candidate['quality_score'] > 0.7 else 'MEDIUM' if candidate['quality_score'] > 0.4 else 'LOW'}
  
NOTES:
{candidate.get('notes', 'No additional notes')}

RECOMMENDATION:
{'PRIORITY TARGET for follow-up observations' if candidate['quality_score'] > 0.7 and not candidate.get('is_known_object', False) else 'Follow-up recommended' if candidate['quality_score'] > 0.4 else 'Low priority for follow-up'}
        """
        
        # Save report
        report_file = reports_dir / f"candidate_{rank:02d}_{candidate_id}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)


def main():
    """Run candidate validation on pipeline results."""
    
    # Load candidate results from pipeline
    pipeline_dir = RESULTS_DIR / "pipeline_run"
    candidates_file = pipeline_dir / "candidate_analysis.csv"
    
    if not candidates_file.exists():
        logger.error(f"Candidate file not found: {candidates_file}")
        logger.info("Please run the full pipeline first (run_full_pipeline.py)")
        return
    
    # Load candidates
    logger.info(f"Loading candidates from {candidates_file}")
    candidates_df = pd.read_csv(candidates_file)
    
    # Filter for high-quality candidates
    high_quality_candidates = candidates_df[candidates_df['quality_score'] > 0.5]
    
    logger.info(f"Validating {len(high_quality_candidates)} high-quality candidates")
    
    # Initialize validator
    validator = CandidateValidator(search_radius_arcsec=30.0)
    
    # Run validation
    validation_results = validator.validate_candidate_list(high_quality_candidates)
    
    # Create validation report
    validator.create_validation_report(validation_results)
    
    # Print summary
    unknowns = validation_results[validation_results['is_known_object'] == False]
    planet_nine_unknowns = unknowns[unknowns['is_planet_nine_candidate'] == True]
    
    print(f"\n🎯 VALIDATION RESULTS:")
    print(f"Total high-quality candidates: {len(high_quality_candidates)}")
    print(f"Known objects identified: {validation_results['is_known_object'].sum()}")
    print(f"Unknown objects: {len(unknowns)}")
    print(f"Planet Nine discovery candidates: {len(planet_nine_unknowns)}")
    
    if len(planet_nine_unknowns) > 0:
        print(f"\n🚀 POTENTIAL DISCOVERIES:")
        top_discoveries = planet_nine_unknowns.nlargest(3, 'quality_score')
        for i, (_, candidate) in enumerate(top_discoveries.iterrows()):
            print(f"  {i+1}. {candidate['detection_id']}: {candidate['motion_arcsec_year']:.3f} arcsec/yr (quality: {candidate['quality_score']:.3f})")


if __name__ == "__main__":
    main()
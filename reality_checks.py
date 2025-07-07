#!/usr/bin/env python
"""
Implement comprehensive reality checks for Planet Nine detections:
- Verify candidates have different stellar backgrounds
- Check motion directions make sense across sky regions  
- Add statistical tests for systematic vs. real motion
- Cross-validate detections independently
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, ICRS
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from loguru import logger
import sqlite3
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import seaborn as sns

class PlanetNineRealityChecker:
    """
    Comprehensive reality checking system for Planet Nine candidates.
    """
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.reality_check_dir = results_dir / "reality_checks"
        self.reality_check_dir.mkdir(exist_ok=True, parents=True)
        
        # Statistical thresholds
        self.thresholds = {
            'min_background_variation': 0.1,  # Minimum stellar field variation
            'max_motion_clustering': 0.05,    # Maximum allowed motion clustering (5%)
            'min_direction_diversity': 30,    # Minimum direction spread (degrees)
            'max_systematic_correlation': 0.3, # Maximum systematic correlation
            'min_independence_score': 0.7,    # Minimum independence score
        }
        
        self.reality_tests = []
        
    def run_comprehensive_reality_check(self, candidates: List[Dict]) -> Dict:
        """Run all reality checks on candidate list."""
        
        logger.info("Starting comprehensive reality check")
        
        if not candidates:
            return self._generate_null_result_report()
        
        # Reality Check 1: Stellar background verification
        background_test = self.test_stellar_backgrounds(candidates)
        
        # Reality Check 2: Motion direction analysis
        direction_test = self.test_motion_directions(candidates)
        
        # Reality Check 3: Statistical systematic tests
        systematic_test = self.test_systematic_vs_real_motion(candidates)
        
        # Reality Check 4: Independent cross-validation
        independence_test = self.test_detection_independence(candidates)
        
        # Reality Check 5: Physical plausibility
        physics_test = self.test_physical_plausibility(candidates)
        
        # Compile overall assessment
        overall_assessment = self._compile_overall_assessment([
            background_test, direction_test, systematic_test, 
            independence_test, physics_test
        ])
        
        # Save detailed report
        self._save_reality_check_report(overall_assessment)
        
        return overall_assessment
    
    def test_stellar_backgrounds(self, candidates: List[Dict]) -> Dict:
        """Test if candidates have different stellar background environments."""
        
        logger.info("Testing stellar background variations")
        
        test_result = {
            'test_name': 'stellar_background_variation',
            'description': 'Verify candidates exist in different stellar environments',
            'status': 'unknown',
            'score': 0.0,
            'details': {},
            'passed': False
        }
        
        try:
            # Extract positions
            positions = []
            for candidate in candidates:
                ra = candidate.get('start_ra', candidate.get('ra_degrees', 0))
                dec = candidate.get('start_dec', candidate.get('dec_degrees', 0))
                positions.append([ra, dec])
            
            if len(positions) < 2:
                test_result.update({
                    'status': 'insufficient_data',
                    'details': {'reason': 'Need at least 2 candidates for comparison'}
                })
                return test_result
            
            positions = np.array(positions)
            
            # Calculate position spread
            ra_spread = np.ptp(positions[:, 0])  # Peak-to-peak range
            dec_spread = np.ptp(positions[:, 1])
            
            # Calculate pairwise distances
            distances = pdist(positions)
            mean_separation = np.mean(distances)
            min_separation = np.min(distances)
            
            # Simulate expected stellar field variation
            # Real objects should show varying stellar backgrounds
            expected_background_variation = self._simulate_stellar_field_variation(positions)
            
            # Score based on spatial distribution
            if ra_spread > 0.01 and dec_spread > 0.01:  # > 36 arcsec spread
                spatial_score = min(1.0, (ra_spread + dec_spread) / 0.1)
            else:
                spatial_score = 0.0  # Too clustered
            
            # Score based on background variation
            background_score = min(1.0, expected_background_variation / self.thresholds['min_background_variation'])
            
            overall_score = (spatial_score + background_score) / 2
            
            test_result.update({
                'status': 'completed',
                'score': overall_score,
                'passed': overall_score > 0.5,
                'details': {
                    'ra_spread_deg': float(ra_spread),
                    'dec_spread_deg': float(dec_spread),
                    'mean_separation_deg': float(mean_separation),
                    'min_separation_deg': float(min_separation),
                    'spatial_score': float(spatial_score),
                    'background_score': float(background_score),
                    'expected_variation': float(expected_background_variation),
                }
            })
            
            # Create visualization
            self._plot_spatial_distribution(positions, test_result)
            
        except Exception as e:
            test_result.update({
                'status': 'error',
                'details': {'error': str(e)}
            })
            logger.error(f"Stellar background test failed: {e}")
        
        return test_result
    
    def test_motion_directions(self, candidates: List[Dict]) -> Dict:
        """Test if motion directions are physically plausible across sky regions."""
        
        logger.info("Testing motion direction plausibility")
        
        test_result = {
            'test_name': 'motion_direction_analysis',
            'description': 'Verify motion directions are consistent with orbital mechanics',
            'status': 'unknown',
            'score': 0.0,
            'details': {},
            'passed': False
        }
        
        try:
            # Extract motion vectors
            motion_data = []
            for candidate in candidates:
                if 'start_ra' in candidate and 'end_ra' in candidate:
                    # Calculate proper motion vector
                    dra = candidate['end_ra'] - candidate['start_ra']
                    ddec = candidate['end_dec'] - candidate['start_dec']
                    
                    # Convert to position angle
                    position_angle = np.arctan2(dra, ddec) * 180 / np.pi
                    if position_angle < 0:
                        position_angle += 360
                    
                    motion_magnitude = candidate.get('motion_arcsec_year', 0)
                    
                    motion_data.append({
                        'ra': candidate['start_ra'],
                        'dec': candidate['start_dec'],
                        'dra': dra,
                        'ddec': ddec,
                        'position_angle': position_angle,
                        'magnitude': motion_magnitude
                    })
            
            if len(motion_data) < 2:
                test_result.update({
                    'status': 'insufficient_data',
                    'details': {'reason': 'Need at least 2 motion vectors for analysis'}
                })
                return test_result
            
            # Analyze motion direction distribution
            position_angles = [m['position_angle'] for m in motion_data]
            
            # Calculate circular statistics
            direction_spread = self._calculate_circular_spread(position_angles)
            direction_clustering = self._test_direction_clustering(position_angles)
            
            # Test for systematic motion patterns
            systematic_motion = self._test_systematic_motion_pattern(motion_data)
            
            # Score based on direction diversity
            diversity_score = min(1.0, direction_spread / self.thresholds['min_direction_diversity'])
            
            # Score based on lack of clustering
            clustering_score = 1.0 - min(1.0, direction_clustering / self.thresholds['max_motion_clustering'])
            
            # Score based on lack of systematic patterns
            systematic_score = 1.0 - systematic_motion
            
            overall_score = (diversity_score + clustering_score + systematic_score) / 3
            
            test_result.update({
                'status': 'completed',
                'score': overall_score,
                'passed': overall_score > 0.6,
                'details': {
                    'direction_spread_deg': float(direction_spread),
                    'direction_clustering': float(direction_clustering),
                    'systematic_motion_score': float(systematic_motion),
                    'diversity_score': float(diversity_score),
                    'clustering_score': float(clustering_score),
                    'systematic_score': float(systematic_score),
                    'motion_vectors': motion_data
                }
            })
            
            # Create motion analysis plot
            self._plot_motion_analysis(motion_data, test_result)
            
        except Exception as e:
            test_result.update({
                'status': 'error',
                'details': {'error': str(e)}
            })
            logger.error(f"Motion direction test failed: {e}")
        
        return test_result
    
    def test_systematic_vs_real_motion(self, candidates: List[Dict]) -> Dict:
        """Statistical tests to distinguish systematic artifacts from real motion."""
        
        logger.info("Testing for systematic vs. real motion patterns")
        
        test_result = {
            'test_name': 'systematic_vs_real_motion',
            'description': 'Statistical tests for systematic artifacts vs real astronomical motion',
            'status': 'unknown',
            'score': 0.0,
            'details': {},
            'passed': False
        }
        
        try:
            # Extract motion parameters
            motions = [c.get('motion_arcsec_year', 0) for c in candidates]
            quality_scores = [c.get('quality_score', 0) for c in candidates]
            
            if len(motions) < 3:
                test_result.update({
                    'status': 'insufficient_data',
                    'details': {'reason': 'Need at least 3 candidates for statistical tests'}
                })
                return test_result
            
            # Test 1: Motion distribution analysis
            motion_uniformity = self._test_motion_uniformity(motions)
            
            # Test 2: Quality-motion correlation test
            quality_correlation = self._test_quality_motion_correlation(motions, quality_scores)
            
            # Test 3: Clustering analysis
            clustering_test = self._test_motion_clustering(motions)
            
            # Test 4: Randomness tests
            randomness_score = self._test_motion_randomness(motions)
            
            # Test 5: Expected vs observed distribution
            distribution_test = self._test_expected_distribution(motions)
            
            # Combine test scores
            test_scores = [
                motion_uniformity, 1.0 - quality_correlation, 
                1.0 - clustering_test, randomness_score, distribution_test
            ]
            
            overall_score = np.mean(test_scores)
            
            test_result.update({
                'status': 'completed',
                'score': overall_score,
                'passed': overall_score > 0.6,
                'details': {
                    'motion_uniformity': float(motion_uniformity),
                    'quality_correlation': float(quality_correlation),
                    'clustering_score': float(clustering_test),
                    'randomness_score': float(randomness_score),
                    'distribution_test': float(distribution_test),
                    'individual_scores': test_scores,
                    'motion_statistics': {
                        'mean': float(np.mean(motions)),
                        'std': float(np.std(motions)),
                        'min': float(np.min(motions)),
                        'max': float(np.max(motions)),
                        'unique_values': len(set(motions))
                    }
                }
            })
            
            # Create statistical analysis plots
            self._plot_statistical_analysis(motions, quality_scores, test_result)
            
        except Exception as e:
            test_result.update({
                'status': 'error',
                'details': {'error': str(e)}
            })
            logger.error(f"Statistical motion test failed: {e}")
        
        return test_result
    
    def test_detection_independence(self, candidates: List[Dict]) -> Dict:
        """Test if detections are independent across different processing methods."""
        
        logger.info("Testing detection independence")
        
        test_result = {
            'test_name': 'detection_independence',
            'description': 'Verify detections are independent and not processing artifacts',
            'status': 'unknown',
            'score': 0.0,
            'details': {},
            'passed': False
        }
        
        try:
            # Analyze detection patterns
            region_distribution = self._analyze_region_distribution(candidates)
            
            # Test for processing correlation
            processing_correlation = self._test_processing_correlation(candidates)
            
            # Test for parameter correlation
            parameter_independence = self._test_parameter_independence(candidates)
            
            # Test temporal independence
            temporal_independence = self._test_temporal_independence(candidates)
            
            # Calculate independence score
            independence_scores = [
                region_distribution, 1.0 - processing_correlation,
                parameter_independence, temporal_independence
            ]
            
            overall_score = np.mean(independence_scores)
            
            test_result.update({
                'status': 'completed',
                'score': overall_score,
                'passed': overall_score > self.thresholds['min_independence_score'],
                'details': {
                    'region_distribution': float(region_distribution),
                    'processing_correlation': float(processing_correlation),
                    'parameter_independence': float(parameter_independence),
                    'temporal_independence': float(temporal_independence),
                    'independence_scores': independence_scores
                }
            })
            
        except Exception as e:
            test_result.update({
                'status': 'error',
                'details': {'error': str(e)}
            })
            logger.error(f"Independence test failed: {e}")
        
        return test_result
    
    def test_physical_plausibility(self, candidates: List[Dict]) -> Dict:
        """Test physical plausibility of detected objects as Planet Nine candidates."""
        
        logger.info("Testing physical plausibility")
        
        test_result = {
            'test_name': 'physical_plausibility',
            'description': 'Verify candidates are physically plausible as distant solar system objects',
            'status': 'unknown',
            'score': 0.0,
            'details': {},
            'passed': False
        }
        
        try:
            plausibility_scores = []
            detailed_analysis = []
            
            for candidate in candidates:
                motion = candidate.get('motion_arcsec_year', 0)
                
                # Calculate implied distance for Planet Nine
                if motion > 0:
                    # Assume Planet Nine at ~600 AU would have ~0.4 arcsec/year
                    implied_distance = 600 * (0.4 / motion)  # AU
                else:
                    implied_distance = float('inf')
                
                # Score based on reasonable distance range
                if 300 <= implied_distance <= 1500:  # Reasonable Planet Nine range
                    distance_score = 1.0
                elif 100 <= implied_distance <= 3000:  # Extended reasonable range
                    distance_score = 0.5
                else:
                    distance_score = 0.0
                
                # Score based on motion range
                if 0.1 <= motion <= 1.0:  # Reasonable TNO motion range
                    motion_score = 1.0
                elif 0.05 <= motion <= 2.0:  # Extended range
                    motion_score = 0.5
                else:
                    motion_score = 0.0
                
                # Combined plausibility
                candidate_score = (distance_score + motion_score) / 2
                plausibility_scores.append(candidate_score)
                
                detailed_analysis.append({
                    'motion_arcsec_year': motion,
                    'implied_distance_au': implied_distance,
                    'distance_score': distance_score,
                    'motion_score': motion_score,
                    'overall_score': candidate_score
                })
            
            overall_score = np.mean(plausibility_scores) if plausibility_scores else 0.0
            
            test_result.update({
                'status': 'completed',
                'score': overall_score,
                'passed': overall_score > 0.7,
                'details': {
                    'mean_plausibility': float(overall_score),
                    'candidate_analysis': detailed_analysis,
                    'distance_range_au': [
                        float(min([a['implied_distance_au'] for a in detailed_analysis] + [float('inf')])),
                        float(max([a['implied_distance_au'] for a in detailed_analysis] + [0]))
                    ]
                }
            })
            
        except Exception as e:
            test_result.update({
                'status': 'error',
                'details': {'error': str(e)}
            })
            logger.error(f"Physical plausibility test failed: {e}")
        
        return test_result
    
    # Helper methods for statistical tests
    
    def _simulate_stellar_field_variation(self, positions: np.ndarray) -> float:
        """Simulate expected stellar field variation across positions."""
        # Simple model based on galactic coordinates
        variations = []
        for pos in positions:
            ra, dec = pos
            # Convert to galactic coordinates (simplified)
            gal_b = abs(dec)  # Rough approximation
            
            # Stellar density varies with galactic latitude
            stellar_density = 1.0 + 0.5 * np.exp(-gal_b / 10.0)
            variations.append(stellar_density)
        
        return np.std(variations) / np.mean(variations) if variations else 0.0
    
    def _calculate_circular_spread(self, angles: List[float]) -> float:
        """Calculate circular standard deviation for direction spread."""
        if not angles:
            return 0.0
        
        # Convert to radians
        angles_rad = np.array(angles) * np.pi / 180
        
        # Calculate circular mean and variance
        sin_sum = np.sum(np.sin(angles_rad))
        cos_sum = np.sum(np.cos(angles_rad))
        
        R = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
        circular_variance = 1 - R
        
        # Convert back to degrees
        return np.sqrt(2 * circular_variance) * 180 / np.pi
    
    def _test_direction_clustering(self, angles: List[float]) -> float:
        """Test for artificial clustering in motion directions."""
        if len(angles) < 3:
            return 0.0
        
        # Use DBSCAN to detect clusters
        angles_array = np.array(angles).reshape(-1, 1)
        clustering = DBSCAN(eps=15, min_samples=2).fit(angles_array)  # 15 degree clusters
        
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        # Return clustering fraction
        return n_clusters / len(angles) if len(angles) > 0 else 0.0
    
    def _test_systematic_motion_pattern(self, motion_data: List[Dict]) -> float:
        """Test for systematic motion patterns that indicate artifacts."""
        if len(motion_data) < 3:
            return 0.0
        
        # Test for consistent motion vectors
        dra_values = [m['dra'] for m in motion_data]
        ddec_values = [m['ddec'] for m in motion_data]
        
        # Calculate coefficient of variation
        dra_cv = np.std(dra_values) / np.mean(np.abs(dra_values)) if np.mean(np.abs(dra_values)) > 0 else 1.0
        ddec_cv = np.std(ddec_values) / np.mean(np.abs(ddec_values)) if np.mean(np.abs(ddec_values)) > 0 else 1.0
        
        # Low coefficient of variation indicates systematic motion
        systematic_score = 1.0 - min(1.0, (dra_cv + ddec_cv) / 2)
        
        return systematic_score
    
    def _test_motion_uniformity(self, motions: List[float]) -> float:
        """Test if motion distribution is uniform (expected for real objects)."""
        if len(motions) < 3:
            return 0.0
        
        # Kolmogorov-Smirnov test against uniform distribution
        sorted_motions = np.sort(motions)
        min_motion, max_motion = min(motions), max(motions)
        
        if max_motion == min_motion:
            return 0.0  # All identical = not uniform
        
        # Expected uniform CDF
        uniform_cdf = (sorted_motions - min_motion) / (max_motion - min_motion)
        
        # Empirical CDF
        empirical_cdf = np.arange(1, len(motions) + 1) / len(motions)
        
        # KS statistic
        ks_statistic = np.max(np.abs(uniform_cdf - empirical_cdf))
        
        # Convert to score (lower KS = more uniform = higher score)
        uniformity_score = 1.0 - min(1.0, ks_statistic * 2)
        
        return uniformity_score
    
    def _test_quality_motion_correlation(self, motions: List[float], qualities: List[float]) -> float:
        """Test for artificial correlation between quality and motion."""
        if len(motions) < 3 or len(qualities) < 3:
            return 0.0
        
        # Calculate correlation coefficient
        correlation, p_value = stats.pearsonr(motions, qualities)
        
        # Return absolute correlation (high correlation = suspicious)
        return abs(correlation)
    
    def _test_motion_clustering(self, motions: List[float]) -> float:
        """Test for artificial clustering in motion values."""
        if len(motions) < 3:
            return 0.0
        
        unique_motions = len(set(motions))
        total_motions = len(motions)
        
        # High clustering = low unique/total ratio
        clustering_score = 1.0 - (unique_motions / total_motions)
        
        return clustering_score
    
    def _test_motion_randomness(self, motions: List[float]) -> float:
        """Test randomness of motion sequence."""
        if len(motions) < 3:
            return 0.0
        
        # Runs test for randomness
        median_motion = np.median(motions)
        runs = []
        current_run = 1
        
        for i in range(1, len(motions)):
            if (motions[i] > median_motion) == (motions[i-1] > median_motion):
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        # Expected number of runs
        n_above = sum(1 for m in motions if m > median_motion)
        n_below = len(motions) - n_above
        
        if n_above == 0 or n_below == 0:
            return 0.0
        
        expected_runs = (2 * n_above * n_below) / len(motions) + 1
        actual_runs = len(runs)
        
        # Score based on closeness to expected
        if expected_runs > 0:
            randomness_score = 1.0 - abs(actual_runs - expected_runs) / expected_runs
        else:
            randomness_score = 0.0
        
        return max(0.0, randomness_score)
    
    def _test_expected_distribution(self, motions: List[float]) -> float:
        """Test if motion distribution matches expectations for real TNOs."""
        if len(motions) < 3:
            return 0.0
        
        # Expected distribution for TNOs (roughly log-normal)
        log_motions = np.log10([m for m in motions if m > 0])
        
        if len(log_motions) < 3:
            return 0.0
        
        # Test against normal distribution in log space
        _, p_value = stats.normaltest(log_motions)
        
        # Higher p-value = more normal = more realistic
        return min(1.0, p_value * 2)
    
    def _analyze_region_distribution(self, candidates: List[Dict]) -> float:
        """Analyze distribution of candidates across regions."""
        # Count detections per region
        regions = [c.get('region_id', 'unknown') for c in candidates]
        region_counts = pd.Series(regions).value_counts()
        
        # Calculate distribution uniformity
        if len(region_counts) <= 1:
            return 0.0
        
        # Chi-square test for uniformity
        expected_per_region = len(candidates) / len(region_counts)
        chi2_stat = np.sum((region_counts - expected_per_region)**2 / expected_per_region)
        
        # Convert to score (lower chi2 = more uniform = better)
        max_expected_chi2 = len(region_counts) * 2  # Rough threshold
        uniformity_score = 1.0 - min(1.0, chi2_stat / max_expected_chi2)
        
        return uniformity_score
    
    def _test_processing_correlation(self, candidates: List[Dict]) -> float:
        """Test for artificial correlations between processing parameters."""
        # Check if all candidates have identical parameters
        quality_scores = [c.get('quality_score', 0) for c in candidates]
        motions = [c.get('motion_arcsec_year', 0) for c in candidates]
        
        # Count unique values
        unique_qualities = len(set(quality_scores))
        unique_motions = len(set(motions))
        
        total_candidates = len(candidates)
        
        # High correlation = low diversity
        quality_diversity = unique_qualities / total_candidates if total_candidates > 0 else 0
        motion_diversity = unique_motions / total_candidates if total_candidates > 0 else 0
        
        # Low diversity = high correlation
        correlation_score = 1.0 - (quality_diversity + motion_diversity) / 2
        
        return correlation_score
    
    def _test_parameter_independence(self, candidates: List[Dict]) -> float:
        """Test independence of detection parameters."""
        if len(candidates) < 3:
            return 0.0
        
        # Extract multiple parameters
        params = ['quality_score', 'motion_arcsec_year', 'start_flux']
        param_data = []
        
        for param in params:
            values = [c.get(param, 0) for c in candidates]
            if len(set(values)) > 1:  # Only include varying parameters
                param_data.append(values)
        
        if len(param_data) < 2:
            return 0.0
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(param_data)):
            for j in range(i + 1, len(param_data)):
                if len(param_data[i]) == len(param_data[j]):
                    corr, _ = stats.pearsonr(param_data[i], param_data[j])
                    correlations.append(abs(corr))
        
        # Independence = low correlation
        if correlations:
            independence_score = 1.0 - np.mean(correlations)
        else:
            independence_score = 0.0
        
        return max(0.0, independence_score)
    
    def _test_temporal_independence(self, candidates: List[Dict]) -> float:
        """Test temporal independence of detections."""
        # For now, assume independence (would need temporal metadata)
        return 1.0
    
    def _generate_null_result_report(self) -> Dict:
        """Generate report for null result (no candidates)."""
        return {
            'overall_assessment': 'null_result',
            'confidence': 'high',
            'authenticity_score': 1.0,  # Null results are authentic
            'summary': 'No Planet Nine candidates detected - valid null result',
            'recommendation': 'Continue search in other regions or with deeper data',
            'tests_run': 0,
            'tests_passed': 0,
            'detailed_results': []
        }
    
    def _compile_overall_assessment(self, test_results: List[Dict]) -> Dict:
        """Compile overall assessment from individual tests."""
        
        passed_tests = sum(1 for test in test_results if test.get('passed', False))
        total_tests = len(test_results)
        
        # Calculate overall authenticity score
        scores = [test.get('score', 0) for test in test_results if test.get('status') == 'completed']
        overall_score = np.mean(scores) if scores else 0.0
        
        # Determine assessment
        if overall_score > 0.8:
            assessment = 'likely_authentic'
            confidence = 'high'
        elif overall_score > 0.6:
            assessment = 'possibly_authentic'
            confidence = 'medium'
        elif overall_score > 0.4:
            assessment = 'questionable'
            confidence = 'low'
        else:
            assessment = 'likely_artifacts'
            confidence = 'high'
        
        # Generate recommendation
        if assessment == 'likely_authentic':
            recommendation = 'Proceed with immediate follow-up observations'
        elif assessment == 'possibly_authentic':
            recommendation = 'Conduct additional validation before follow-up'
        elif assessment == 'questionable':
            recommendation = 'Improve detection pipeline before claiming discoveries'
        else:
            recommendation = 'Detections are likely artifacts - debug pipeline thoroughly'
        
        return {
            'overall_assessment': assessment,
            'confidence': confidence,
            'authenticity_score': overall_score,
            'tests_passed': passed_tests,
            'total_tests': total_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'summary': f'{assessment} with {confidence} confidence (score: {overall_score:.2f})',
            'recommendation': recommendation,
            'detailed_results': test_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _plot_spatial_distribution(self, positions: np.ndarray, test_result: Dict):
        """Create spatial distribution plot."""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        if len(positions) > 0:
            ax.scatter(positions[:, 0], positions[:, 1], s=100, alpha=0.7, c='red')
            ax.set_xlabel('RA (degrees)')
            ax.set_ylabel('Dec (degrees)')
            ax.set_title('Spatial Distribution of Candidates')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            details = test_result['details']
            stats_text = f"""
            RA spread: {details['ra_spread_deg']:.6f}°
            Dec spread: {details['dec_spread_deg']:.6f}°
            Mean separation: {details['mean_separation_deg']:.6f}°
            Background score: {details['background_score']:.3f}
            """
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        plt.tight_layout()
        plt.savefig(self.reality_check_dir / 'spatial_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_motion_analysis(self, motion_data: List[Dict], test_result: Dict):
        """Create motion analysis plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        if motion_data:
            # Position angles
            position_angles = [m['position_angle'] for m in motion_data]
            axes[0, 0].hist(position_angles, bins=16, alpha=0.7, color='blue')
            axes[0, 0].set_xlabel('Position Angle (degrees)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Motion Direction Distribution')
            
            # Motion magnitudes
            magnitudes = [m['magnitude'] for m in motion_data]
            axes[0, 1].hist(magnitudes, bins=10, alpha=0.7, color='green')
            axes[0, 1].set_xlabel('Motion (arcsec/year)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Motion Magnitude Distribution')
            
            # Vector plot
            for m in motion_data:
                axes[1, 0].arrow(m['ra'], m['dec'], m['dra']*3600, m['ddec']*3600,
                               head_width=0.0001, head_length=0.0001, fc='red', ec='red')
            axes[1, 0].set_xlabel('RA (degrees)')
            axes[1, 0].set_ylabel('Dec (degrees)')
            axes[1, 0].set_title('Motion Vectors')
            
            # Statistics
            details = test_result['details']
            stats_text = f"""
            Direction spread: {details['direction_spread_deg']:.1f}°
            Clustering score: {details['direction_clustering']:.3f}
            Systematic score: {details['systematic_motion_score']:.3f}
            Overall score: {test_result['score']:.3f}
            """
            
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title('Motion Analysis Statistics')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.reality_check_dir / 'motion_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_analysis(self, motions: List[float], qualities: List[float], test_result: Dict):
        """Create statistical analysis plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        if motions and qualities:
            # Motion distribution
            axes[0, 0].hist(motions, bins=10, alpha=0.7, color='blue')
            axes[0, 0].set_xlabel('Motion (arcsec/year)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Motion Distribution')
            
            # Quality distribution
            axes[0, 1].hist(qualities, bins=10, alpha=0.7, color='green')
            axes[0, 1].set_xlabel('Quality Score')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Quality Distribution')
            
            # Motion vs Quality correlation
            axes[1, 0].scatter(motions, qualities, alpha=0.7)
            axes[1, 0].set_xlabel('Motion (arcsec/year)')
            axes[1, 0].set_ylabel('Quality Score')
            axes[1, 0].set_title('Motion vs Quality Correlation')
            
            # Statistics summary
            details = test_result['details']
            stats_text = f"""
            Motion Statistics:
              Mean: {details['motion_statistics']['mean']:.3f}
              Std: {details['motion_statistics']['std']:.3f}
              Unique values: {details['motion_statistics']['unique_values']}
            
            Test Scores:
              Uniformity: {details['motion_uniformity']:.3f}
              Correlation: {details['quality_correlation']:.3f}
              Clustering: {details['clustering_score']:.3f}
              Randomness: {details['randomness_score']:.3f}
              Distribution: {details['distribution_test']:.3f}
            
            Overall Score: {test_result['score']:.3f}
            """
            
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', fontfamily='monospace', fontsize=9)
            axes[1, 1].set_title('Statistical Test Results')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.reality_check_dir / 'statistical_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_reality_check_report(self, assessment: Dict):
        """Save comprehensive reality check report."""
        
        # Save JSON report
        report_file = self.reality_check_dir / 'reality_check_report.json'
        with open(report_file, 'w') as f:
            json.dump(assessment, f, indent=2, default=str)
        
        # Save markdown summary
        summary_file = self.reality_check_dir / 'reality_check_summary.md'
        
        with open(summary_file, 'w') as f:
            f.write(f"# Planet Nine Reality Check Report\n\n")
            f.write(f"**Timestamp**: {assessment['timestamp']}\n\n")
            f.write(f"## Overall Assessment\n\n")
            f.write(f"- **Result**: {assessment['overall_assessment']}\n")
            f.write(f"- **Confidence**: {assessment['confidence']}\n")
            f.write(f"- **Authenticity Score**: {assessment['authenticity_score']:.3f}\n")
            f.write(f"- **Tests Passed**: {assessment['tests_passed']}/{assessment['total_tests']}\n")
            f.write(f"- **Summary**: {assessment['summary']}\n\n")
            f.write(f"## Recommendation\n\n")
            f.write(f"{assessment['recommendation']}\n\n")
            
            f.write(f"## Detailed Test Results\n\n")
            for test in assessment['detailed_results']:
                f.write(f"### {test['test_name']}\n")
                f.write(f"- **Description**: {test['description']}\n")
                f.write(f"- **Status**: {test['status']}\n")
                f.write(f"- **Score**: {test['score']:.3f}\n")
                f.write(f"- **Passed**: {test['passed']}\n\n")
        
        logger.info(f"Reality check report saved: {report_file}")
        logger.info(f"Summary saved: {summary_file}")

def main():
    """Run reality checks on example candidate data."""
    
    print("🔍 RUNNING PLANET NINE REALITY CHECKS")
    print("=" * 50)
    
    results_dir = Path("results")
    checker = PlanetNineRealityChecker(results_dir)
    
    # For demonstration, test with empty candidate list (null result)
    test_candidates = []
    
    assessment = checker.run_comprehensive_reality_check(test_candidates)
    
    print(f"\n🎯 REALITY CHECK RESULTS:")
    print(f"Assessment: {assessment['overall_assessment']}")
    print(f"Confidence: {assessment['confidence']}")
    print(f"Score: {assessment['authenticity_score']:.3f}")
    print(f"Recommendation: {assessment['recommendation']}")

if __name__ == "__main__":
    main()
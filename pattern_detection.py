#!/usr/bin/env python
"""
Pattern detection system for identifying systematic trends and anomalies
across multiple Planet Nine search regions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from loguru import logger
import sqlite3
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clip
from scipy import stats
from scipy.spatial.distance import pdist, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.config import RESULTS_DIR


@dataclass
class DetectionPattern:
    """Represents a detected pattern in the search results."""
    pattern_id: str
    pattern_type: str  # 'spatial_cluster', 'motion_cluster', 'temporal', 'anomaly'
    significance: float  # Statistical significance (0-1)
    description: str
    affected_regions: List[str]
    candidate_count: int
    properties: Dict
    created_at: datetime
    
    
@dataclass
class RegionalTrend:
    """Regional detection trends and statistics."""
    region_id: str
    detection_rate: float  # candidates per square degree
    avg_motion: float
    avg_quality: float
    stellar_contamination: float
    efficiency_score: float
    anomaly_indicators: List[str]


class PatternDetectionEngine:
    """Detect patterns and anomalies across Planet Nine search regions."""
    
    def __init__(self):
        self.results_dir = RESULTS_DIR / "pattern_detection"
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.patterns = []
        self.regional_trends = {}
        
    def analyze_search_patterns(self, candidates_df: pd.DataFrame, 
                              regions_df: pd.DataFrame) -> Dict:
        """Perform comprehensive pattern analysis across search regions."""
        logger.info(f"Analyzing patterns in {len(candidates_df)} candidates across {len(regions_df)} regions")
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'spatial_patterns': [],
            'motion_patterns': [],
            'temporal_patterns': [],
            'anomaly_patterns': [],
            'regional_trends': {},
            'cross_correlations': {},
            'recommendations': []
        }
        
        if len(candidates_df) == 0:
            logger.warning("No candidates provided for pattern analysis")
            return analysis_results
        
        # 1. Spatial clustering analysis
        spatial_patterns = self._detect_spatial_clusters(candidates_df)
        analysis_results['spatial_patterns'] = spatial_patterns
        
        # 2. Motion pattern analysis
        motion_patterns = self._detect_motion_patterns(candidates_df)
        analysis_results['motion_patterns'] = motion_patterns
        
        # 3. Temporal pattern analysis
        temporal_patterns = self._detect_temporal_patterns(candidates_df)
        analysis_results['temporal_patterns'] = temporal_patterns
        
        # 4. Anomaly detection
        anomaly_patterns = self._detect_anomalies(candidates_df)
        analysis_results['anomaly_patterns'] = anomaly_patterns
        
        # 5. Regional trend analysis
        regional_trends = self._analyze_regional_trends(candidates_df, regions_df)
        analysis_results['regional_trends'] = regional_trends
        
        # 6. Cross-correlation analysis
        correlations = self._analyze_cross_correlations(candidates_df, regions_df)
        analysis_results['cross_correlations'] = correlations
        
        # 7. Generate recommendations
        recommendations = self._generate_pattern_recommendations(analysis_results)
        analysis_results['recommendations'] = recommendations
        
        # Save results
        self._save_analysis_results(analysis_results)
        
        # Create visualizations
        self._create_pattern_visualizations(candidates_df, regions_df, analysis_results)
        
        logger.success(f"Pattern analysis complete. Found {len(spatial_patterns) + len(motion_patterns)} significant patterns")
        return analysis_results
    
    def _detect_spatial_clusters(self, df: pd.DataFrame) -> List[Dict]:
        """Detect spatial clusters of candidates."""
        if len(df) < 5:
            return []
        
        logger.info("Detecting spatial clusters")
        patterns = []
        
        # Prepare coordinates
        coords = df[['ra', 'dec']].values
        
        # DBSCAN clustering
        # Use different scales for different cluster sizes
        eps_values = [0.5, 1.0, 2.0, 5.0]  # degrees
        min_samples_values = [3, 5, 8]
        
        best_clusters = []
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
                
                # Evaluate clustering quality
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                
                if n_clusters > 0:
                    # Calculate silhouette score if possible
                    cluster_info = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'noise_points': (clustering.labels_ == -1).sum(),
                        'labels': clustering.labels_
                    }
                    best_clusters.append(cluster_info)
        
        # Select best clustering result
        if best_clusters:
            # Choose clustering with reasonable number of clusters (2-10)
            valid_clusters = [c for c in best_clusters if 2 <= c['n_clusters'] <= 10]
            
            if valid_clusters:
                best = max(valid_clusters, key=lambda x: x['n_clusters'] - x['noise_points']/len(df))
                labels = best['labels']
                
                # Analyze each cluster
                for cluster_id in set(labels):
                    if cluster_id == -1:  # Skip noise points
                        continue
                    
                    cluster_members = df[labels == cluster_id]
                    
                    if len(cluster_members) >= 3:
                        # Calculate cluster properties
                        center_ra = cluster_members['ra'].mean()
                        center_dec = cluster_members['dec'].mean()
                        spread = np.sqrt(cluster_members['ra'].var() + cluster_members['dec'].var())
                        avg_motion = cluster_members['motion_arcsec_year'].mean()
                        avg_quality = cluster_members['quality_score'].mean()
                        
                        # Calculate significance based on density and isolation
                        cluster_area = np.pi * spread**2
                        density = len(cluster_members) / cluster_area if cluster_area > 0 else 0
                        
                        # Background density
                        total_area = (df['ra'].max() - df['ra'].min()) * (df['dec'].max() - df['dec'].min())
                        background_density = len(df) / total_area if total_area > 0 else 0
                        
                        significance = min(1.0, density / background_density) if background_density > 0 else 0.5
                        
                        if significance > 0.3:  # Significant cluster
                            pattern = {
                                'pattern_id': f"spatial_cluster_{cluster_id}",
                                'cluster_id': int(cluster_id),
                                'member_count': len(cluster_members),
                                'center_ra': float(center_ra),
                                'center_dec': float(center_dec),
                                'spread_deg': float(spread),
                                'avg_motion': float(avg_motion),
                                'avg_quality': float(avg_quality),
                                'density': float(density),
                                'significance': float(significance),
                                'regions_involved': list(cluster_members.get('region_id', ['unknown']).unique())
                            }
                            patterns.append(pattern)
        
        return patterns
    
    def _detect_motion_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect patterns in proper motion distribution."""
        logger.info("Analyzing motion patterns")
        patterns = []
        
        motions = df['motion_arcsec_year'].dropna()
        if len(motions) < 10:
            return patterns
        
        # 1. Test for multi-modal distribution
        try:
            # Use Gaussian Mixture Model to detect modes
            from sklearn.mixture import GaussianMixture
            
            # Test different numbers of components
            bic_scores = []
            n_components_range = range(1, min(6, len(motions)//3))
            
            for n_components in n_components_range:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(motions.values.reshape(-1, 1))
                bic_scores.append(gmm.bic(motions.values.reshape(-1, 1)))
            
            if len(bic_scores) > 1:
                # Find optimal number of components (minimum BIC)
                optimal_components = n_components_range[np.argmin(bic_scores)]
                
                if optimal_components > 1:
                    # Fit optimal model
                    gmm = GaussianMixture(n_components=optimal_components, random_state=42)
                    gmm.fit(motions.values.reshape(-1, 1))
                    
                    # Extract component properties
                    means = gmm.means_.flatten()
                    weights = gmm.weights_
                    
                    pattern = {
                        'pattern_id': 'motion_multimodal',
                        'type': 'multimodal_distribution',
                        'n_components': int(optimal_components),
                        'component_means': [float(m) for m in means],
                        'component_weights': [float(w) for w in weights],
                        'significance': float(min(1.0, (optimal_components - 1) * 0.3)),
                        'description': f'Motion distribution shows {optimal_components} distinct modes'
                    }
                    patterns.append(pattern)
                    
        except ImportError:
            logger.warning("sklearn.mixture not available for motion pattern analysis")
        
        # 2. Test for Planet Nine motion excess
        planet_nine_range = (motions >= 0.2) & (motions <= 0.8)
        pn_fraction = planet_nine_range.sum() / len(motions)
        
        # Compare to expected background (based on TNO populations)
        expected_pn_fraction = 0.05  # Expected 5% in Planet Nine range
        
        if pn_fraction > expected_pn_fraction * 2:  # Significant excess
            significance = min(1.0, pn_fraction / expected_pn_fraction - 1)
            
            pattern = {
                'pattern_id': 'planet_nine_excess',
                'type': 'motion_range_excess',
                'observed_fraction': float(pn_fraction),
                'expected_fraction': float(expected_pn_fraction),
                'excess_factor': float(pn_fraction / expected_pn_fraction),
                'candidate_count': int(planet_nine_range.sum()),
                'significance': float(significance),
                'description': f'Excess of candidates in Planet Nine motion range'
            }
            patterns.append(pattern)
        
        # 3. Test for systematic motion directions (would indicate tracking errors)
        if 'motion_pa' in df.columns:  # Position angle of motion
            pa_values = df['motion_pa'].dropna()
            if len(pa_values) > 20:
                # Test for uniformity using Rayleigh test
                pa_radians = np.deg2rad(pa_values)
                mean_vector_length = np.abs(np.mean(np.exp(1j * pa_radians)))
                
                if mean_vector_length > 0.3:  # Significant clustering
                    pattern = {
                        'pattern_id': 'systematic_motion_direction',
                        'type': 'directional_bias',
                        'mean_vector_length': float(mean_vector_length),
                        'preferred_direction': float(np.angle(np.mean(np.exp(1j * pa_radians)), deg=True)),
                        'significance': float(mean_vector_length),
                        'description': 'Systematic bias in motion directions detected'
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_temporal_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect temporal patterns in detections."""
        logger.info("Analyzing temporal patterns")
        patterns = []
        
        # For this implementation, we'll look for patterns in processing order
        # In a real implementation, this would use observation timestamps
        
        if 'created_at' in df.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            
            # Group by time periods
            df_with_time = df.dropna(subset=['created_at'])
            
            if len(df_with_time) > 20:
                # Hourly detection rates
                df_with_time['hour'] = df_with_time['created_at'].dt.hour
                hourly_counts = df_with_time.groupby('hour').size()
                
                # Test for temporal clustering
                if len(hourly_counts) > 1:
                    # Chi-square test for uniformity
                    expected = len(df_with_time) / 24
                    chi2_stat, p_value = stats.chisquare(hourly_counts.reindex(range(24), fill_value=0), 
                                                        f_exp=[expected]*24)
                    
                    if p_value < 0.05:  # Significant temporal pattern
                        pattern = {
                            'pattern_id': 'temporal_clustering',
                            'type': 'hourly_variation',
                            'chi2_statistic': float(chi2_stat),
                            'p_value': float(p_value),
                            'significance': float(min(1.0, -np.log10(p_value) / 3)),
                            'peak_hours': [int(h) for h in hourly_counts.nlargest(3).index],
                            'description': 'Significant temporal clustering in detection times'
                        }
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalous candidates and patterns."""
        logger.info("Detecting anomalies")
        patterns = []
        
        if len(df) < 10:
            return patterns
        
        # 1. Statistical outliers in motion
        motions = df['motion_arcsec_year'].dropna()
        if len(motions) > 5:
            # Use sigma clipping to find outliers
            motion_clipped = sigma_clip(motions, sigma=3, maxiters=2)
            outliers = ~motion_clipped.mask
            
            extreme_slow = motions[motions < 0.05]  # Ultra-slow motion
            extreme_fast = motions[motions > 20.0]   # Very fast motion
            
            if len(extreme_slow) > 0:
                pattern = {
                    'pattern_id': 'ultra_slow_motion',
                    'type': 'motion_anomaly',
                    'candidate_count': len(extreme_slow),
                    'motion_range': [float(extreme_slow.min()), float(extreme_slow.max())],
                    'significance': float(min(1.0, len(extreme_slow) / len(motions) * 10)),
                    'description': f'Ultra-slow motion candidates (<0.05 arcsec/yr): {len(extreme_slow)} found'
                }
                patterns.append(pattern)
            
            if len(extreme_fast) > 0:
                pattern = {
                    'pattern_id': 'very_fast_motion',
                    'type': 'motion_anomaly', 
                    'candidate_count': len(extreme_fast),
                    'motion_range': [float(extreme_fast.min()), float(extreme_fast.max())],
                    'significance': float(min(1.0, len(extreme_fast) / len(motions) * 5)),
                    'description': f'Very fast motion candidates (>20 arcsec/yr): {len(extreme_fast)} found'
                }
                patterns.append(pattern)
        
        # 2. Quality score anomalies
        qualities = df['quality_score'].dropna()
        if len(qualities) > 10:
            # Find exceptionally high quality detections
            high_quality = qualities[qualities > 0.9]
            
            if len(high_quality) > 0:
                expected_high_quality = len(qualities) * 0.05  # Expect 5% high quality
                
                if len(high_quality) > expected_high_quality * 2:
                    pattern = {
                        'pattern_id': 'exceptional_quality',
                        'type': 'quality_anomaly',
                        'candidate_count': len(high_quality),
                        'observed_fraction': float(len(high_quality) / len(qualities)),
                        'significance': float(min(1.0, len(high_quality) / expected_high_quality - 1)),
                        'description': f'Unusual number of exceptional quality detections: {len(high_quality)}'
                    }
                    patterns.append(pattern)
        
        # 3. Flux anomalies
        if 'start_flux' in df.columns:
            fluxes = df['start_flux'].dropna()
            if len(fluxes) > 10:
                # Very faint objects
                very_faint = fluxes[fluxes < 1.0]
                
                if len(very_faint) > 0:
                    pattern = {
                        'pattern_id': 'very_faint_objects',
                        'type': 'flux_anomaly',
                        'candidate_count': len(very_faint),
                        'flux_range': [float(very_faint.min()), float(very_faint.max())],
                        'significance': float(min(1.0, len(very_faint) / len(fluxes) * 5)),
                        'description': f'Very faint objects detected: {len(very_faint)} candidates'
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_regional_trends(self, candidates_df: pd.DataFrame, 
                               regions_df: pd.DataFrame) -> Dict:
        """Analyze trends across different search regions."""
        logger.info("Analyzing regional trends")
        trends = {}
        
        if 'region_id' not in candidates_df.columns:
            return trends
        
        for region_id in candidates_df['region_id'].unique():
            region_candidates = candidates_df[candidates_df['region_id'] == region_id]
            
            if len(region_candidates) == 0:
                continue
            
            # Calculate regional statistics
            region_area = 64.0  # Assume 8x8 degree regions for now
            detection_rate = len(region_candidates) / region_area
            
            avg_motion = region_candidates['motion_arcsec_year'].mean()
            avg_quality = region_candidates['quality_score'].mean()
            
            # Estimate stellar contamination
            fast_motion = (region_candidates['motion_arcsec_year'] > 2.0).sum()
            stellar_contamination = fast_motion / len(region_candidates)
            
            # Calculate efficiency score
            high_quality_fraction = (region_candidates['quality_score'] > 0.7).sum() / len(region_candidates)
            planet_nine_fraction = (
                (region_candidates['motion_arcsec_year'] >= 0.2) & 
                (region_candidates['motion_arcsec_year'] <= 0.8)
            ).sum() / len(region_candidates)
            
            efficiency_score = (high_quality_fraction + planet_nine_fraction) / 2
            
            # Identify anomaly indicators
            anomaly_indicators = []
            if detection_rate > 50:  # Very high detection rate
                anomaly_indicators.append('high_detection_rate')
            if stellar_contamination > 0.7:
                anomaly_indicators.append('high_stellar_contamination') 
            if avg_motion > 5.0:
                anomaly_indicators.append('unusual_motion_distribution')
            if efficiency_score < 0.1:
                anomaly_indicators.append('low_efficiency')
            
            trend = RegionalTrend(
                region_id=region_id,
                detection_rate=detection_rate,
                avg_motion=avg_motion,
                avg_quality=avg_quality,
                stellar_contamination=stellar_contamination,
                efficiency_score=efficiency_score,
                anomaly_indicators=anomaly_indicators
            )
            
            trends[region_id] = asdict(trend)
        
        return trends
    
    def _analyze_cross_correlations(self, candidates_df: pd.DataFrame,
                                  regions_df: pd.DataFrame) -> Dict:
        """Analyze correlations between different properties."""
        logger.info("Analyzing cross-correlations")
        correlations = {}
        
        if len(candidates_df) < 10:
            return correlations
        
        # Select numerical columns for correlation analysis
        numerical_cols = ['motion_arcsec_year', 'quality_score', 'start_flux', 'ra', 'dec']
        available_cols = [col for col in numerical_cols if col in candidates_df.columns]
        
        if len(available_cols) >= 2:
            corr_data = candidates_df[available_cols].dropna()
            
            if len(corr_data) > 5:
                # Calculate correlation matrix
                corr_matrix = corr_data.corr()
                
                # Find strong correlations (|r| > 0.5)
                strong_correlations = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.5:
                            strong_correlations.append({
                                'variable1': corr_matrix.columns[i],
                                'variable2': corr_matrix.columns[j],
                                'correlation': float(corr_value),
                                'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                            })
                
                correlations['strong_correlations'] = strong_correlations
                correlations['correlation_matrix'] = corr_matrix.round(3).to_dict()
        
        return correlations
    
    def _generate_pattern_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate recommendations based on detected patterns."""
        recommendations = []
        
        # Spatial pattern recommendations
        spatial_patterns = analysis_results.get('spatial_patterns', [])
        if len(spatial_patterns) > 0:
            recommendations.append(
                f"Investigate {len(spatial_patterns)} spatial clusters for systematic effects or genuine groupings"
            )
            
            high_sig_spatial = [p for p in spatial_patterns if p.get('significance', 0) > 0.7]
            if high_sig_spatial:
                recommendations.append(
                    "High-significance spatial clusters detected - prioritize for follow-up observations"
                )
        
        # Motion pattern recommendations
        motion_patterns = analysis_results.get('motion_patterns', [])
        planet_nine_excess = [p for p in motion_patterns if p.get('pattern_id') == 'planet_nine_excess']
        if planet_nine_excess:
            recommendations.append(
                "Significant excess in Planet Nine motion range - investigate for instrumental effects"
            )
        
        # Anomaly recommendations
        anomaly_patterns = analysis_results.get('anomaly_patterns', [])
        ultra_slow = [p for p in anomaly_patterns if p.get('pattern_id') == 'ultra_slow_motion']
        if ultra_slow:
            recommendations.append(
                "Ultra-slow motion candidates detected - high priority for Planet Nine follow-up"
            )
        
        # Regional trend recommendations
        regional_trends = analysis_results.get('regional_trends', {})
        low_efficiency_regions = [
            r_id for r_id, trend in regional_trends.items() 
            if trend.get('efficiency_score', 1.0) < 0.2
        ]
        if low_efficiency_regions:
            recommendations.append(
                f"Low efficiency regions detected: {len(low_efficiency_regions)} regions need algorithm tuning"
            )
        
        # Cross-correlation recommendations
        correlations = analysis_results.get('cross_correlations', {})
        strong_corr = correlations.get('strong_correlations', [])
        motion_quality_corr = [
            c for c in strong_corr 
            if 'motion' in c.get('variable1', '') and 'quality' in c.get('variable2', '')
        ]
        if motion_quality_corr:
            recommendations.append(
                "Strong motion-quality correlation suggests possible detection bias"
            )
        
        # Default recommendations if no patterns found
        if not recommendations:
            recommendations.extend([
                "No significant patterns detected - search appears to be performing normally",
                "Consider expanding search area or adjusting detection thresholds",
                "Monitor for patterns as more data is collected"
            ])
        
        return recommendations
    
    def _save_analysis_results(self, results: Dict):
        """Save pattern analysis results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        results_file = self.results_dir / f'pattern_analysis_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Pattern analysis results saved to {results_file}")
    
    def _create_pattern_visualizations(self, candidates_df: pd.DataFrame,
                                     regions_df: pd.DataFrame, 
                                     analysis_results: Dict):
        """Create visualizations for pattern analysis."""
        logger.info("Creating pattern visualizations")
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Sky distribution with spatial clusters
        ax1 = axes[0, 0]
        if len(candidates_df) > 0:
            scatter = ax1.scatter(candidates_df['ra'], candidates_df['dec'],
                                c=candidates_df['motion_arcsec_year'], 
                                cmap='viridis', alpha=0.6, s=30)
            ax1.set_xlabel('RA (degrees)')
            ax1.set_ylabel('Dec (degrees)')
            ax1.set_title('Sky Distribution of Candidates')
            plt.colorbar(scatter, ax=ax1, label='Motion (arcsec/yr)')
            
            # Overlay spatial clusters
            spatial_patterns = analysis_results.get('spatial_patterns', [])
            for pattern in spatial_patterns:
                if pattern.get('significance', 0) > 0.5:
                    circle = plt.Circle((pattern['center_ra'], pattern['center_dec']),
                                      pattern['spread_deg'], fill=False, 
                                      color='red', linewidth=2)
                    ax1.add_patch(circle)
        
        # 2. Motion distribution
        ax2 = axes[0, 1]
        if 'motion_arcsec_year' in candidates_df.columns:
            motions = candidates_df['motion_arcsec_year'].dropna()
            ax2.hist(motions, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvspan(0.2, 0.8, alpha=0.3, color='red', label='Planet Nine Range')
            ax2.set_xlabel('Proper Motion (arcsec/year)')
            ax2.set_ylabel('Count')
            ax2.set_title('Motion Distribution')
            ax2.set_xscale('log')
            ax2.legend()
        
        # 3. Regional efficiency comparison
        ax3 = axes[0, 2]
        regional_trends = analysis_results.get('regional_trends', {})
        if regional_trends:
            regions = list(regional_trends.keys())
            efficiencies = [regional_trends[r]['efficiency_score'] for r in regions]
            detection_rates = [regional_trends[r]['detection_rate'] for r in regions]
            
            ax3.scatter(detection_rates, efficiencies, alpha=0.7, s=50)
            ax3.set_xlabel('Detection Rate (candidates/sq deg)')
            ax3.set_ylabel('Efficiency Score')
            ax3.set_title('Regional Performance')
            ax3.grid(True, alpha=0.3)
        
        # 4. Pattern significance summary
        ax4 = axes[1, 0]
        pattern_types = ['spatial', 'motion', 'temporal', 'anomaly']
        pattern_counts = []
        
        for ptype in pattern_types:
            patterns = analysis_results.get(f'{ptype}_patterns', [])
            significant_patterns = [p for p in patterns if p.get('significance', 0) > 0.3]
            pattern_counts.append(len(significant_patterns))
        
        ax4.bar(pattern_types, pattern_counts, alpha=0.7)
        ax4.set_ylabel('Number of Significant Patterns')
        ax4.set_title('Pattern Detection Summary')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Quality vs Motion scatter
        ax5 = axes[1, 1]
        if 'quality_score' in candidates_df.columns and 'motion_arcsec_year' in candidates_df.columns:
            ax5.scatter(candidates_df['motion_arcsec_year'], candidates_df['quality_score'],
                       alpha=0.6, s=20)
            ax5.set_xlabel('Proper Motion (arcsec/year)')
            ax5.set_ylabel('Quality Score')
            ax5.set_title('Quality vs Motion')
            ax5.set_xscale('log')
            ax5.grid(True, alpha=0.3)
        
        # 6. Anomaly summary
        ax6 = axes[1, 2]
        anomaly_patterns = analysis_results.get('anomaly_patterns', [])
        if anomaly_patterns:
            anomaly_types = [p.get('pattern_id', 'unknown') for p in anomaly_patterns]
            anomaly_counts = [p.get('candidate_count', 0) for p in anomaly_patterns]
            
            ax6.barh(range(len(anomaly_types)), anomaly_counts, alpha=0.7)
            ax6.set_yticks(range(len(anomaly_types)))
            ax6.set_yticklabels(anomaly_types)
            ax6.set_xlabel('Candidate Count')
            ax6.set_title('Anomaly Detection Results')
        else:
            ax6.text(0.5, 0.5, 'No significant\nanomalies detected', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Anomaly Detection Results')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        vis_file = self.results_dir / f'pattern_visualization_{timestamp}.png'
        plt.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Pattern visualization saved to {vis_file}")


def main():
    """Run pattern detection on search results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect patterns in Planet Nine search results')
    parser.add_argument('--database', type=str,
                       default='/Users/jannikschilling/Documents/coding/planetnine/results/large_scale_search/search_progress.db',
                       help='Path to search results database')
    parser.add_argument('--min-candidates', type=int, default=10,
                       help='Minimum candidates per region for analysis')
    
    args = parser.parse_args()
    
    # Load data from database
    db_path = Path(args.database)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.info("Run large_scale_search.py first to generate data")
        return
    
    logger.info(f"Loading data from {db_path}")
    conn = sqlite3.connect(db_path)
    
    # Load candidates
    candidates_df = pd.read_sql_query("""
        SELECT * FROM candidate_detections
    """, conn)
    
    # Load regions
    regions_df = pd.read_sql_query("""
        SELECT * FROM search_regions
        WHERE total_candidates >= ?
    """, conn, params=(args.min_candidates,))
    
    conn.close()
    
    if len(candidates_df) == 0:
        logger.warning("No candidates found in database")
        return
    
    if len(regions_df) == 0:
        logger.warning(f"No regions with >= {args.min_candidates} candidates")
        return
    
    logger.info(f"Loaded {len(candidates_df)} candidates from {len(regions_df)} regions")
    
    # Initialize pattern detector
    detector = PatternDetectionEngine()
    
    # Run pattern analysis
    results = detector.analyze_search_patterns(candidates_df, regions_df)
    
    # Print summary
    print("\n" + "="*70)
    print("🔍 PATTERN DETECTION SUMMARY")
    print("="*70)
    
    spatial_count = len(results.get('spatial_patterns', []))
    motion_count = len(results.get('motion_patterns', []))
    temporal_count = len(results.get('temporal_patterns', []))
    anomaly_count = len(results.get('anomaly_patterns', []))
    
    print(f"Spatial patterns detected: {spatial_count}")
    print(f"Motion patterns detected: {motion_count}")
    print(f"Temporal patterns detected: {temporal_count}")
    print(f"Anomaly patterns detected: {anomaly_count}")
    
    # Print significant patterns
    all_patterns = (results.get('spatial_patterns', []) + 
                   results.get('motion_patterns', []) + 
                   results.get('anomaly_patterns', []))
    
    significant_patterns = [p for p in all_patterns if p.get('significance', 0) > 0.5]
    
    if significant_patterns:
        print(f"\n🎯 SIGNIFICANT PATTERNS:")
        for pattern in significant_patterns[:5]:  # Top 5
            pattern_id = pattern.get('pattern_id', 'unknown')
            significance = pattern.get('significance', 0)
            description = pattern.get('description', 'No description')
            print(f"• {pattern_id}: {description} (sig: {significance:.2f})")
    
    # Print recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"• {rec}")
    
    print(f"\n📊 ANALYSIS SAVED TO: {detector.results_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Enhanced candidate ranking system for identifying the most promising
Planet Nine candidates from large-scale searches.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from dataclasses import dataclass
from loguru import logger
import sqlite3
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.config import RESULTS_DIR


@dataclass
class RankingCriteria:
    """Define ranking criteria for Planet Nine candidates."""
    motion_weight: float = 0.25  # Prefer slow motion (Planet Nine range)
    quality_weight: float = 0.20  # Detection quality
    novelty_weight: float = 0.20  # How unusual/novel the detection is
    validation_weight: float = 0.15  # Distance from known objects
    consistency_weight: float = 0.10  # Multi-epoch consistency
    theoretical_weight: float = 0.10  # Location matches theoretical predictions


class EnhancedCandidateRanker:
    """Advanced ranking system for Planet Nine candidates."""
    
    def __init__(self, criteria: Optional[RankingCriteria] = None):
        self.criteria = criteria or RankingCriteria()
        self.results_dir = RESULTS_DIR / "enhanced_ranking"
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.scaler = StandardScaler()
        
    def rank_candidates(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Apply enhanced ranking to candidate list."""
        logger.info(f"Ranking {len(candidates_df)} candidates with enhanced criteria")
        
        if len(candidates_df) == 0:
            return candidates_df
        
        # Calculate individual ranking scores
        ranked_df = candidates_df.copy()
        
        # 1. Motion score (prefer Planet Nine range)
        ranked_df['motion_score'] = self._calculate_motion_score(ranked_df)
        
        # 2. Quality score (already exists, normalize)
        ranked_df['quality_score_norm'] = self._normalize_score(ranked_df['quality_score'])
        
        # 3. Novelty score (unusual detections)
        ranked_df['novelty_score'] = self._calculate_novelty_score(ranked_df)
        
        # 4. Validation score (distance from known objects)
        ranked_df['validation_score'] = self._calculate_validation_score(ranked_df)
        
        # 5. Consistency score (multi-epoch reliability)
        ranked_df['consistency_score'] = self._calculate_consistency_score(ranked_df)
        
        # 6. Theoretical score (matches predictions)
        ranked_df['theoretical_score'] = self._calculate_theoretical_score(ranked_df)
        
        # Calculate composite ranking score
        ranked_df['ranking_score'] = (
            self.criteria.motion_weight * ranked_df['motion_score'] +
            self.criteria.quality_weight * ranked_df['quality_score_norm'] +
            self.criteria.novelty_weight * ranked_df['novelty_score'] +
            self.criteria.validation_weight * ranked_df['validation_score'] +
            self.criteria.consistency_weight * ranked_df['consistency_score'] +
            self.criteria.theoretical_weight * ranked_df['theoretical_score']
        )
        
        # Sort by ranking score
        ranked_df = ranked_df.sort_values('ranking_score', ascending=False)
        ranked_df['rank'] = range(1, len(ranked_df) + 1)
        
        # Identify tier classifications
        ranked_df['tier'] = self._classify_tiers(ranked_df)
        
        # Add anomaly detection
        ranked_df['anomaly_score'] = self._detect_anomalies(ranked_df)
        
        logger.success(f"Ranking complete. Top score: {ranked_df['ranking_score'].max():.3f}")
        return ranked_df
    
    def _calculate_motion_score(self, df: pd.DataFrame) -> pd.Series:
        """Score based on proper motion (prefer Planet Nine range)."""
        motion = df['motion_arcsec_year']
        
        # Optimal range: 0.2-0.8 arcsec/year for Planet Nine
        score = np.zeros(len(df))
        
        # Perfect score for Planet Nine range
        planet_nine_mask = (motion >= 0.2) & (motion <= 0.8)
        score[planet_nine_mask] = 1.0
        
        # High score for ultra-slow motion (could be very distant)
        ultra_slow_mask = motion < 0.2
        score[ultra_slow_mask] = 0.9
        
        # Medium score for slow TNO range
        slow_tno_mask = (motion > 0.8) & (motion <= 2.0)
        score[slow_tno_mask] = 0.6
        
        # Lower score for faster objects (more likely asteroids/stars)
        fast_mask = motion > 2.0
        score[fast_mask] = np.maximum(0.1, 1.0 / (motion[fast_mask] / 2.0))
        
        return pd.Series(score, index=df.index)
    
    def _calculate_novelty_score(self, df: pd.DataFrame) -> pd.Series:
        """Score based on how unusual/novel the detection is."""
        scores = []
        
        for _, row in df.iterrows():
            score = 0.0
            
            # Reward faint objects (less likely to be known)
            flux = row.get('start_flux', 1000)
            if flux < 1.0:
                score += 0.3
            elif flux < 10.0:
                score += 0.2
            elif flux < 100.0:
                score += 0.1
            
            # Reward consistent flux (not variable stars)
            flux_ratio = row.get('flux_ratio', 1.0)
            if 0.8 <= flux_ratio <= 1.2:
                score += 0.2
            
            # Reward isolated detections (not in crowded fields)
            local_density = row.get('local_density', 0)
            if local_density == 0:
                score += 0.2
            elif local_density <= 2:
                score += 0.1
            
            # Reward high-priority regions
            priority = row.get('region_priority', 'medium')
            if priority == 'high':
                score += 0.2
            elif priority == 'medium':
                score += 0.1
            
            # Bonus for extreme coordinates (far from ecliptic)
            dec = abs(row.get('dec', 0))
            if dec > 30:
                score += 0.1
            
            scores.append(min(1.0, score))
        
        return pd.Series(scores, index=df.index)
    
    def _calculate_validation_score(self, df: pd.DataFrame) -> pd.Series:
        """Score based on distance from known objects."""
        scores = []
        
        for _, row in df.iterrows():
            # High score for large distances from known objects
            validation_distance = row.get('validation_distance', 999)
            
            if validation_distance >= 60:  # 1 arcminute or more
                score = 1.0
            elif validation_distance >= 30:  # 30 arcseconds
                score = 0.8
            elif validation_distance >= 10:  # 10 arcseconds
                score = 0.6
            elif validation_distance >= 5:   # 5 arcseconds
                score = 0.4
            else:
                score = 0.2
            
            # Penalize high validation confidence (likely known object)
            validation_conf = row.get('validation_confidence', 0)
            if validation_conf > 0.8:
                score *= 0.5
            elif validation_conf > 0.6:
                score *= 0.7
            
            scores.append(score)
        
        return pd.Series(scores, index=df.index)
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> pd.Series:
        """Score based on multi-epoch consistency."""
        scores = []
        
        for _, row in df.iterrows():
            score = row.get('quality_score', 0.5)  # Base on detection quality
            
            # Reward consistent motion direction (if available)
            # This would require multi-epoch data analysis
            
            # For now, use flux consistency as proxy
            flux_ratio = row.get('flux_ratio', 1.0)
            consistency_bonus = 1.0 - abs(1.0 - flux_ratio)
            score = (score + consistency_bonus) / 2.0
            
            scores.append(score)
        
        return pd.Series(scores, index=df.index)
    
    def _calculate_theoretical_score(self, df: pd.DataFrame) -> pd.Series:
        """Score based on match to theoretical predictions."""
        scores = []
        
        # Define high-probability regions based on Batygin & Brown (2016)
        high_prob_regions = [
            {'ra_center': 45, 'dec_center': -20, 'radius': 15},   # Anti-clustering
            {'ra_center': 225, 'dec_center': 15, 'radius': 20},   # Perihelion approach
            {'ra_center': 90, 'dec_center': 45, 'radius': 25},    # High galactic latitude
            {'ra_center': 270, 'dec_center': -45, 'radius': 25},  # High galactic latitude
        ]
        
        for _, row in df.iterrows():
            score = 0.1  # Base score
            
            candidate_coord = SkyCoord(
                ra=row.get('ra', 0) * u.deg,
                dec=row.get('dec', 0) * u.deg
            )
            
            # Check distance to high-probability regions
            for region in high_prob_regions:
                region_coord = SkyCoord(
                    ra=region['ra_center'] * u.deg,
                    dec=region['dec_center'] * u.deg
                )
                
                distance = candidate_coord.separation(region_coord).deg
                
                if distance <= region['radius']:
                    # Score based on proximity to center
                    proximity_score = 1.0 - (distance / region['radius'])
                    score = max(score, proximity_score)
            
            # Bonus for theoretical basis of region
            theoretical_basis = row.get('theoretical_basis', '')
            if 'anti-clustering' in theoretical_basis.lower():
                score += 0.2
            elif 'perihelion' in theoretical_basis.lower():
                score += 0.2
            elif 'galactic' in theoretical_basis.lower():
                score += 0.1
            
            scores.append(min(1.0, score))
        
        return pd.Series(scores, index=df.index)
    
    def _normalize_score(self, series: pd.Series) -> pd.Series:
        """Normalize a score to [0, 1] range."""
        if series.max() == series.min():
            return pd.Series(0.5, index=series.index)
        
        return (series - series.min()) / (series.max() - series.min())
    
    def _classify_tiers(self, ranked_df: pd.DataFrame) -> pd.Series:
        """Classify candidates into priority tiers."""
        scores = ranked_df['ranking_score']
        tiers = []
        
        # Define tier thresholds
        tier1_threshold = scores.quantile(0.95)  # Top 5%
        tier2_threshold = scores.quantile(0.85)  # Top 15%
        tier3_threshold = scores.quantile(0.70)  # Top 30%
        
        for score in scores:
            if score >= tier1_threshold:
                tiers.append('Tier_1_Exceptional')
            elif score >= tier2_threshold:
                tiers.append('Tier_2_High_Priority')
            elif score >= tier3_threshold:
                tiers.append('Tier_3_Moderate')
            else:
                tiers.append('Tier_4_Low_Priority')
        
        return pd.Series(tiers, index=ranked_df.index)
    
    def _detect_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """Detect anomalous candidates using isolation forest."""
        if len(df) < 10:
            return pd.Series(0.0, index=df.index)
        
        # Select features for anomaly detection
        features = ['motion_arcsec_year', 'start_flux', 'quality_score', 
                   'validation_distance', 'ranking_score']
        
        feature_data = df[features].fillna(0)
        
        # Standardize features
        feature_scaled = self.scaler.fit_transform(feature_data)
        
        # Apply isolation forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(feature_scaled)
        anomaly_scores = iso_forest.score_samples(feature_scaled)
        
        # Convert to anomaly scores (higher = more anomalous)
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        
        return pd.Series(anomaly_scores, index=df.index)
    
    def generate_ranking_report(self, ranked_df: pd.DataFrame) -> Dict:
        """Generate comprehensive ranking report."""
        logger.info("Generating ranking report")
        
        report = {
            'ranking_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_candidates': len(ranked_df),
                'tier_1_count': len(ranked_df[ranked_df['tier'] == 'Tier_1_Exceptional']),
                'tier_2_count': len(ranked_df[ranked_df['tier'] == 'Tier_2_High_Priority']),
                'tier_3_count': len(ranked_df[ranked_df['tier'] == 'Tier_3_Moderate']),
                'tier_4_count': len(ranked_df[ranked_df['tier'] == 'Tier_4_Low_Priority']),
                'top_score': float(ranked_df['ranking_score'].max()),
                'median_score': float(ranked_df['ranking_score'].median())
            },
            'top_candidates': [],
            'tier_analysis': {},
            'anomaly_candidates': []
        }
        
        # Top 10 candidates
        top_10 = ranked_df.head(10)
        for _, candidate in top_10.iterrows():
            report['top_candidates'].append({
                'rank': int(candidate['rank']),
                'detection_id': candidate.get('detection_id', 'unknown'),
                'ra': float(candidate.get('ra', 0)),
                'dec': float(candidate.get('dec', 0)),
                'motion_arcsec_year': float(candidate.get('motion_arcsec_year', 0)),
                'ranking_score': float(candidate['ranking_score']),
                'tier': candidate['tier'],
                'motion_score': float(candidate['motion_score']),
                'novelty_score': float(candidate['novelty_score']),
                'validation_score': float(candidate['validation_score']),
                'theoretical_score': float(candidate['theoretical_score'])
            })
        
        # Tier analysis
        for tier in ranked_df['tier'].unique():
            tier_subset = ranked_df[ranked_df['tier'] == tier]
            report['tier_analysis'][tier] = {
                'count': len(tier_subset),
                'avg_ranking_score': float(tier_subset['ranking_score'].mean()),
                'avg_motion': float(tier_subset['motion_arcsec_year'].mean()),
                'avg_quality': float(tier_subset['quality_score'].mean())
            }
        
        # Top anomalies
        top_anomalies = ranked_df.nlargest(5, 'anomaly_score')
        for _, anomaly in top_anomalies.iterrows():
            report['anomaly_candidates'].append({
                'detection_id': anomaly.get('detection_id', 'unknown'),
                'rank': int(anomaly['rank']),
                'anomaly_score': float(anomaly['anomaly_score']),
                'ranking_score': float(anomaly['ranking_score']),
                'tier': anomaly['tier']
            })
        
        return report
    
    def create_ranking_visualizations(self, ranked_df: pd.DataFrame):
        """Create visualizations for ranking analysis."""
        logger.info("Creating ranking visualizations")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Ranking score distribution
        ax1 = axes[0, 0]
        ax1.hist(ranked_df['ranking_score'], bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Ranking Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Ranking Scores')
        ax1.grid(True, alpha=0.3)
        
        # 2. Tier distribution
        ax2 = axes[0, 1]
        tier_counts = ranked_df['tier'].value_counts()
        ax2.pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%')
        ax2.set_title('Candidate Tier Distribution')
        
        # 3. Motion vs Ranking Score
        ax3 = axes[0, 2]
        scatter = ax3.scatter(ranked_df['motion_arcsec_year'], ranked_df['ranking_score'],
                             c=ranked_df['quality_score'], cmap='viridis', alpha=0.6)
        ax3.set_xlabel('Proper Motion (arcsec/year)')
        ax3.set_ylabel('Ranking Score')
        ax3.set_title('Motion vs Ranking Score')
        ax3.set_xscale('log')
        plt.colorbar(scatter, ax=ax3, label='Quality Score')
        
        # 4. Score components analysis
        ax4 = axes[1, 0]
        score_components = ['motion_score', 'novelty_score', 'validation_score', 
                           'theoretical_score', 'consistency_score']
        
        top_candidates = ranked_df.head(20)
        for i, component in enumerate(score_components):
            if component in top_candidates.columns:
                ax4.scatter([i] * len(top_candidates), top_candidates[component], 
                           alpha=0.6, label=f'Top 20')
        
        ax4.set_xticks(range(len(score_components)))
        ax4.set_xticklabels([s.replace('_score', '') for s in score_components], rotation=45)
        ax4.set_ylabel('Score Component Value')
        ax4.set_title('Score Components for Top Candidates')
        ax4.grid(True, alpha=0.3)
        
        # 5. Anomaly detection
        ax5 = axes[1, 1]
        ax5.scatter(ranked_df['ranking_score'], ranked_df['anomaly_score'], alpha=0.6)
        ax5.set_xlabel('Ranking Score')
        ax5.set_ylabel('Anomaly Score')
        ax5.set_title('Ranking vs Anomaly Scores')
        ax5.grid(True, alpha=0.3)
        
        # 6. Sky distribution of top candidates
        ax6 = axes[1, 2]
        top_50 = ranked_df.head(50)
        scatter = ax6.scatter(top_50['ra'], top_50['dec'], 
                             c=top_50['ranking_score'], cmap='plasma', s=50)
        ax6.set_xlabel('RA (degrees)')
        ax6.set_ylabel('Dec (degrees)')
        ax6.set_title('Sky Distribution of Top 50 Candidates')
        plt.colorbar(scatter, ax=ax6, label='Ranking Score')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'ranking_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create detailed tier comparison
        self._create_tier_comparison_plot(ranked_df)
    
    def _create_tier_comparison_plot(self, ranked_df: pd.DataFrame):
        """Create detailed comparison of different tiers."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Motion distribution by tier
        ax1 = axes[0, 0]
        for tier in ranked_df['tier'].unique():
            tier_data = ranked_df[ranked_df['tier'] == tier]
            ax1.hist(tier_data['motion_arcsec_year'], bins=20, alpha=0.6, 
                    label=tier, density=True)
        ax1.set_xlabel('Proper Motion (arcsec/year)')
        ax1.set_ylabel('Normalized Count')
        ax1.set_title('Motion Distribution by Tier')
        ax1.set_xscale('log')
        ax1.legend()
        
        # Quality distribution by tier
        ax2 = axes[0, 1]
        tier_quality = []
        tier_labels = []
        for tier in ranked_df['tier'].unique():
            tier_data = ranked_df[ranked_df['tier'] == tier]
            tier_quality.append(tier_data['quality_score'].values)
            tier_labels.append(tier)
        
        ax2.boxplot(tier_quality, labels=tier_labels)
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Quality Distribution by Tier')
        ax2.tick_params(axis='x', rotation=45)
        
        # Novelty vs Validation score by tier
        ax3 = axes[1, 0]
        colors = ['red', 'orange', 'yellow', 'green']
        for i, tier in enumerate(ranked_df['tier'].unique()):
            tier_data = ranked_df[ranked_df['tier'] == tier]
            ax3.scatter(tier_data['novelty_score'], tier_data['validation_score'],
                       c=colors[i % len(colors)], alpha=0.6, label=tier)
        ax3.set_xlabel('Novelty Score')
        ax3.set_ylabel('Validation Score')
        ax3.set_title('Novelty vs Validation by Tier')
        ax3.legend()
        
        # Ranking score evolution
        ax4 = axes[1, 1]
        ax4.plot(range(1, len(ranked_df) + 1), ranked_df['ranking_score'], 'b-', alpha=0.7)
        ax4.set_xlabel('Candidate Rank')
        ax4.set_ylabel('Ranking Score')
        ax4.set_title('Ranking Score vs Rank')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'tier_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Run enhanced candidate ranking on search results."""
    import argparse
    from large_scale_search import LargeScaleSearchManager
    
    parser = argparse.ArgumentParser(description='Enhanced Planet Nine candidate ranking')
    parser.add_argument('--database', type=str, 
                       default='/Users/jannikschilling/Documents/coding/planetnine/results/large_scale_search/search_progress.db',
                       help='Path to search results database')
    parser.add_argument('--min-quality', type=float, default=0.3,
                       help='Minimum quality score for ranking')
    
    args = parser.parse_args()
    
    # Load candidates from database
    db_path = Path(args.database)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.info("Run large_scale_search.py first to generate candidates")
        return
    
    logger.info(f"Loading candidates from {db_path}")
    conn = sqlite3.connect(db_path)
    
    candidates_df = pd.read_sql_query("""
        SELECT c.*, r.priority as region_priority, r.theoretical_basis
        FROM candidate_detections c
        JOIN search_regions r ON c.region_id = r.region_id
        WHERE c.quality_score >= ?
    """, conn, params=(args.min_quality,))
    
    conn.close()
    
    if len(candidates_df) == 0:
        logger.warning(f"No candidates found with quality >= {args.min_quality}")
        return
    
    logger.info(f"Loaded {len(candidates_df)} candidates for ranking")
    
    # Initialize ranker
    ranker = EnhancedCandidateRanker()
    
    # Rank candidates
    ranked_df = ranker.rank_candidates(candidates_df)
    
    # Generate report
    report = ranker.generate_ranking_report(ranked_df)
    
    # Create visualizations
    ranker.create_ranking_visualizations(ranked_df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save ranked candidates
    ranked_df.to_csv(ranker.results_dir / f'ranked_candidates_{timestamp}.csv', index=False)
    
    # Save report
    with open(ranker.results_dir / f'ranking_report_{timestamp}.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*70)
    print("🏆 ENHANCED CANDIDATE RANKING SUMMARY")
    print("="*70)
    print(f"Total candidates ranked: {len(ranked_df)}")
    print(f"Tier 1 (Exceptional): {report['ranking_summary']['tier_1_count']}")
    print(f"Tier 2 (High Priority): {report['ranking_summary']['tier_2_count']}")
    print(f"Tier 3 (Moderate): {report['ranking_summary']['tier_3_count']}")
    print(f"Top ranking score: {report['ranking_summary']['top_score']:.3f}")
    
    print(f"\n🎯 TOP 5 CANDIDATES:")
    for candidate in report['top_candidates'][:5]:
        print(f"Rank {candidate['rank']}: "
              f"Score {candidate['ranking_score']:.3f}, "
              f"Motion {candidate['motion_arcsec_year']:.3f} arcsec/yr, "
              f"{candidate['tier']}")
    
    print(f"\n📊 ANALYSIS SAVED TO: {ranker.results_dir}")


if __name__ == "__main__":
    main()
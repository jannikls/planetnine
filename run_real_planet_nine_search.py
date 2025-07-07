#!/usr/bin/env python
"""
Run a real Planet Nine search using actual astronomical survey data.
This is the main entry point for conducting a genuine search with
DECaLS images and proper multi-epoch motion detection.
"""

import numpy as np
from pathlib import Path
from loguru import logger
import json
from datetime import datetime
import argparse
from typing import List, Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.processing.real_planet_nine_pipeline import RealPlanetNinePipeline
from src.models.orbit_predictor import PlanetNineOrbitPredictor

class RealPlanetNineSearch:
    """Conduct real Planet Nine search with actual survey data."""
    
    def __init__(self, max_workers: int = 4):
        self.pipeline = RealPlanetNinePipeline()
        self.orbit_predictor = PlanetNineOrbitPredictor()
        self.max_workers = max_workers
        
        self.results_dir = Path("results/real_planet_nine_search")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Define search regions based on theoretical predictions
        self.search_regions = self._define_search_regions()
    
    def _define_search_regions(self) -> List[Dict]:
        """Define high-priority search regions based on Planet Nine predictions."""
        
        # Get theoretical prediction
        prediction = self.orbit_predictor.predict_current_position()
        
        regions = []
        
        # Primary search region around predicted position
        regions.append({
            'name': 'primary_prediction',
            'ra': prediction['ra'],
            'dec': prediction['dec'],
            'size': 1.0,  # 1 degree
            'priority': 'highest',
            'description': 'Primary theoretical prediction region'
        })
        
        # Uncertainty cone around prediction
        uncertainty_radius = prediction['uncertainty_deg']
        
        # Create grid of regions covering uncertainty area
        grid_size = 0.5  # 0.5 degree tiles
        n_tiles = int(2 * uncertainty_radius / grid_size) + 1
        
        for i in range(n_tiles):
            for j in range(n_tiles):
                ra_offset = (i - n_tiles//2) * grid_size
                dec_offset = (j - n_tiles//2) * grid_size
                
                # Skip if outside uncertainty radius
                if np.sqrt(ra_offset**2 + dec_offset**2) > uncertainty_radius:
                    continue
                
                # Skip primary region
                if abs(ra_offset) < grid_size/2 and abs(dec_offset) < grid_size/2:
                    continue
                
                regions.append({
                    'name': f'uncertainty_grid_{i}_{j}',
                    'ra': prediction['ra'] + ra_offset / np.cos(np.radians(prediction['dec'])),
                    'dec': prediction['dec'] + dec_offset,
                    'size': grid_size,
                    'priority': 'high',
                    'description': f'Uncertainty region tile ({i},{j})'
                })
        
        # Anti-clustering regions (opposite to TNO clustering)
        anti_cluster_regions = [
            {'ra': 50.0, 'dec': -15.0},
            {'ra': 70.0, 'dec': -25.0},
            {'ra': 90.0, 'dec': -20.0},
        ]
        
        for i, pos in enumerate(anti_cluster_regions):
            regions.append({
                'name': f'anti_clustering_{i+1}',
                'ra': pos['ra'],
                'dec': pos['dec'],
                'size': 0.75,
                'priority': 'medium',
                'description': 'Anti-clustering region opposite TNO perihelia'
            })
        
        logger.info(f"Defined {len(regions)} search regions")
        logger.info(f"Primary prediction: RA={prediction['ra']:.2f}°, Dec={prediction['dec']:.2f}°")
        logger.info(f"Uncertainty radius: {uncertainty_radius:.2f}°")
        
        return regions
    
    def search_single_region(self, region: Dict) -> Dict:
        """Search a single region for Planet Nine candidates."""
        
        logger.info(f"Searching region: {region['name']} "
                   f"(RA={region['ra']:.2f}°, Dec={region['dec']:.2f}°, "
                   f"size={region['size']}°, priority={region['priority']})")
        
        start_time = datetime.now()
        
        # Run pipeline on this region
        results = self.pipeline.process_region(
            ra_center=region['ra'],
            dec_center=region['dec'],
            size_deg=region['size'],
            band='r'  # Use r-band for primary search
        )
        
        # Add region metadata to results
        results['region'] = region
        results['search_time'] = (datetime.now() - start_time).total_seconds()
        
        # Log results
        if results['status'] == 'completed':
            logger.success(f"Region {region['name']}: {results['candidates_found']} candidates found")
        else:
            logger.warning(f"Region {region['name']}: {results['status']} - {results.get('errors', [])}")
        
        return results
    
    def run_parallel_search(self, regions: List[Dict] = None, 
                          priority_filter: str = None) -> Dict:
        """Run parallel search across multiple regions."""
        
        if regions is None:
            regions = self.search_regions
        
        # Filter by priority if requested
        if priority_filter:
            regions = [r for r in regions if r['priority'] == priority_filter]
        
        logger.info(f"Starting parallel search of {len(regions)} regions with {self.max_workers} workers")
        
        search_results = {
            'start_time': datetime.now().isoformat(),
            'total_regions': len(regions),
            'regions_completed': 0,
            'total_candidates': 0,
            'high_quality_candidates': 0,
            'planet_nine_compatible': 0,
            'region_results': [],
            'all_candidates': []
        }
        
        # Process regions in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all regions
            future_to_region = {
                executor.submit(self.search_single_region, region): region
                for region in regions
            }
            
            # Process completed regions
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                
                try:
                    result = future.result()
                    search_results['region_results'].append(result)
                    search_results['regions_completed'] += 1
                    
                    # Collect candidates
                    if result.get('candidates'):
                        search_results['all_candidates'].extend(result['candidates'])
                        search_results['total_candidates'] += len(result['candidates'])
                        
                        # Count high quality candidates
                        for candidate in result['candidates']:
                            if candidate.get('quality_score', 0) > 5.0:
                                search_results['high_quality_candidates'] += 1
                            if candidate.get('planet_nine_compatible', False):
                                search_results['planet_nine_compatible'] += 1
                    
                    # Progress update
                    logger.info(f"Progress: {search_results['regions_completed']}/{len(regions)} regions completed")
                    
                except Exception as e:
                    logger.error(f"Region {region['name']} failed: {e}")
                    search_results['region_results'].append({
                        'region': region,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        search_results['end_time'] = datetime.now().isoformat()
        search_results['total_time_seconds'] = (
            datetime.fromisoformat(search_results['end_time']) - 
            datetime.fromisoformat(search_results['start_time'])
        ).total_seconds()
        
        # Generate summary statistics
        search_results['summary'] = self._generate_search_summary(search_results)
        
        # Save results
        self._save_search_results(search_results)
        
        return search_results
    
    def _generate_search_summary(self, results: Dict) -> Dict:
        """Generate summary statistics from search results."""
        
        # Coverage statistics
        total_area = sum(r['region']['size']**2 for r in results['region_results'])
        successful_regions = sum(1 for r in results['region_results'] 
                               if r.get('status') == 'completed')
        
        # Candidate statistics
        motion_distribution = []
        quality_distribution = []
        
        for candidate in results['all_candidates']:
            if 'total_motion_arcsec_year' in candidate:
                motion_distribution.append(candidate['total_motion_arcsec_year'])
            if 'quality_score' in candidate:
                quality_distribution.append(candidate['quality_score'])
        
        summary = {
            'coverage': {
                'total_area_sq_deg': total_area,
                'successful_regions': successful_regions,
                'failed_regions': results['total_regions'] - successful_regions,
                'success_rate': successful_regions / results['total_regions'] if results['total_regions'] > 0 else 0
            },
            'candidates': {
                'total': results['total_candidates'],
                'high_quality': results['high_quality_candidates'],
                'planet_nine_compatible': results['planet_nine_compatible'],
                'detection_rate_per_sq_deg': results['total_candidates'] / total_area if total_area > 0 else 0
            },
            'performance': {
                'total_time_seconds': results['total_time_seconds'],
                'time_per_region': results['total_time_seconds'] / results['total_regions'] if results['total_regions'] > 0 else 0,
                'regions_per_hour': 3600 * results['total_regions'] / results['total_time_seconds'] if results['total_time_seconds'] > 0 else 0
            }
        }
        
        if motion_distribution:
            summary['motion_statistics'] = {
                'min': np.min(motion_distribution),
                'max': np.max(motion_distribution),
                'mean': np.mean(motion_distribution),
                'median': np.median(motion_distribution),
                'in_planet_nine_range': sum(1 for m in motion_distribution if 0.2 <= m <= 0.8)
            }
        
        if quality_distribution:
            summary['quality_statistics'] = {
                'min': np.min(quality_distribution),
                'max': np.max(quality_distribution),
                'mean': np.mean(quality_distribution),
                'median': np.median(quality_distribution)
            }
        
        return summary
    
    def _save_search_results(self, results: Dict):
        """Save search results to files."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full results JSON
        json_file = self.results_dir / f"search_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save candidate catalog
        if results['all_candidates']:
            catalog_file = self.results_dir / f"candidate_catalog_{timestamp}.json"
            with open(catalog_file, 'w') as f:
                json.dump(results['all_candidates'], f, indent=2, default=str)
        
        # Generate human-readable report
        report_file = self.results_dir / f"search_report_{timestamp}.md"
        self._write_search_report(results, report_file)
        
        logger.success(f"Search results saved to {json_file}")
        logger.success(f"Report saved to {report_file}")
    
    def _write_search_report(self, results: Dict, report_file: Path):
        """Write human-readable search report."""
        
        with open(report_file, 'w') as f:
            f.write("# Real Planet Nine Search Report\n\n")
            f.write(f"**Search Date**: {results['start_time']}\n")
            f.write(f"**Pipeline Version**: Real astronomical data integration v1.0\n\n")
            
            f.write("## Executive Summary\n\n")
            
            if results['total_candidates'] > 0:
                f.write(f"🎯 **{results['total_candidates']} moving object candidates detected**\n")
                f.write(f"- {results['high_quality_candidates']} high-quality candidates\n")
                f.write(f"- {results['planet_nine_compatible']} compatible with Planet Nine predictions\n\n")
            else:
                f.write("No moving object candidates detected in searched regions.\n\n")
            
            f.write("## Search Coverage\n\n")
            summary = results['summary']
            f.write(f"- **Total area searched**: {summary['coverage']['total_area_sq_deg']:.1f} square degrees\n")
            f.write(f"- **Regions processed**: {summary['coverage']['successful_regions']}/{results['total_regions']}\n")
            f.write(f"- **Success rate**: {summary['coverage']['success_rate']:.1%}\n")
            f.write(f"- **Processing time**: {summary['performance']['total_time_seconds']:.1f} seconds\n")
            f.write(f"- **Processing rate**: {summary['performance']['regions_per_hour']:.1f} regions/hour\n\n")
            
            if 'motion_statistics' in summary:
                f.write("## Candidate Motion Distribution\n\n")
                stats = summary['motion_statistics']
                f.write(f"- **Motion range**: {stats['min']:.2f} - {stats['max']:.2f} arcsec/year\n")
                f.write(f"- **Mean motion**: {stats['mean']:.2f} arcsec/year\n")
                f.write(f"- **In Planet Nine range (0.2-0.8)**: {stats['in_planet_nine_range']} candidates\n\n")
            
            f.write("## Top Candidates\n\n")
            
            # Sort candidates by quality score
            sorted_candidates = sorted(
                results['all_candidates'], 
                key=lambda x: x.get('quality_score', 0), 
                reverse=True
            )[:5]  # Top 5
            
            for i, candidate in enumerate(sorted_candidates, 1):
                f.write(f"### Candidate {i}\n")
                f.write(f"- **Position**: RA={candidate['ra']:.4f}°, Dec={candidate['dec']:.4f}°\n")
                f.write(f"- **Motion**: {candidate.get('total_motion_arcsec_year', 'N/A'):.2f} arcsec/year\n")
                f.write(f"- **Quality score**: {candidate.get('quality_score', 0):.2f}\n")
                f.write(f"- **Detections**: {candidate.get('detections', 'N/A')} epochs\n")
                f.write(f"- **Planet Nine compatible**: {'Yes' if candidate.get('planet_nine_compatible') else 'No'}\n\n")
            
            f.write("## Recommendations\n\n")
            
            if results['planet_nine_compatible'] > 0:
                f.write("1. **Immediate follow-up observations required** for Planet Nine compatible candidates\n")
                f.write("2. Obtain additional epochs to confirm motion and calculate orbits\n")
                f.write("3. Check candidates against latest MPC and survey catalogs\n")
                f.write("4. Perform photometric analysis to estimate distance\n")
            else:
                f.write("1. Continue search in additional high-priority regions\n")
                f.write("2. Consider deeper imaging or different filter bands\n")
                f.write("3. Validate pipeline performance with known TNO recovery\n")
            
            f.write("\n## Data Sources\n\n")
            f.write("- **Primary survey**: Dark Energy Camera Legacy Survey (DECaLS)\n")
            f.write("- **Cross-match catalogs**: WISE, Gaia DR3\n")
            f.write("- **Multi-epoch baseline**: Using different DECaLS data releases\n")

def main():
    """Main entry point for real Planet Nine search."""
    
    parser = argparse.ArgumentParser(description="Run real Planet Nine search with astronomical survey data")
    parser.add_argument('--priority', choices=['highest', 'high', 'medium', 'all'],
                       default='highest', help='Priority level of regions to search')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--test', action='store_true',
                       help='Run test search on small region')
    
    args = parser.parse_args()
    
    print("🌌 REAL PLANET NINE SEARCH")
    print("=" * 60)
    print("Using actual DECaLS survey data for moving object detection")
    print()
    
    # Initialize search
    search = RealPlanetNineSearch(max_workers=args.workers)
    
    if args.test:
        print("Running TEST search on single region...")
        test_region = {
            'name': 'test_region',
            'ra': 50.0,
            'dec': -15.0,
            'size': 0.25,
            'priority': 'test',
            'description': 'Test region for pipeline validation'
        }
        
        results = search.search_single_region(test_region)
        
        print(f"\nTest Results:")
        print(f"Status: {results['status']}")
        print(f"Candidates found: {results.get('candidates_found', 0)}")
        print(f"Processing time: {results.get('search_time', 0):.1f} seconds")
        
    else:
        print(f"Starting {args.priority.upper()} priority region search...")
        print(f"Using {args.workers} parallel workers")
        print()
        
        # Run full search
        priority_filter = None if args.priority == 'all' else args.priority
        results = search.run_parallel_search(priority_filter=priority_filter)
        
        # Print summary
        print(f"\n📊 SEARCH COMPLETED")
        print(f"Regions searched: {results['regions_completed']}/{results['total_regions']}")
        print(f"Total candidates: {results['total_candidates']}")
        print(f"High-quality candidates: {results['high_quality_candidates']}")
        print(f"Planet Nine compatible: {results['planet_nine_compatible']}")
        print(f"Total time: {results['summary']['performance']['total_time_seconds']:.1f} seconds")
        
        if results['total_candidates'] > 0:
            print(f"\n🎯 CANDIDATES FOUND! Check {search.results_dir} for details")
        else:
            print(f"\n📋 No candidates found in searched regions")
    
    print(f"\n✅ Search complete! Results saved to {search.results_dir}")

if __name__ == "__main__":
    main()
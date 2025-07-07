#!/usr/bin/env python
"""
Validate the Planet Nine detection pipeline using known trans-Neptunian objects (TNOs).
This ensures our pipeline can detect real moving objects before searching for Planet Nine.
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.jplhorizons import Horizons
from loguru import logger
from datetime import datetime
import json
from pathlib import Path

from src.processing.real_planet_nine_pipeline import RealPlanetNinePipeline

class TNOValidator:
    """Validate detection pipeline with known TNOs."""
    
    def __init__(self):
        self.pipeline = RealPlanetNinePipeline()
        self.results_dir = Path("results/tno_validation")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Known TNOs for validation (bright enough for detection)
        self.test_tnos = [
            {
                'name': 'Makemake',
                'designation': '136472',
                'magnitude': 17.0,
                'expected_motion': 0.3  # arcsec/year approximate
            },
            {
                'name': 'Quaoar',
                'designation': '50000',
                'magnitude': 18.5,
                'expected_motion': 0.4
            },
            {
                'name': 'Sedna',
                'designation': '90377',
                'magnitude': 21.0,
                'expected_motion': 0.1
            },
            {
                'name': 'Orcus',
                'designation': '90482',
                'magnitude': 19.0,
                'expected_motion': 0.35
            },
            {
                'name': '2015 TG387 (The Goblin)',
                'designation': '2015 TG387',
                'magnitude': 23.5,
                'expected_motion': 0.05
            }
        ]
    
    def get_tno_position(self, designation: str, epoch: str = 'now') -> Optional[Dict]:
        """Get TNO position from JPL Horizons."""
        
        try:
            obj = Horizons(id=designation, location='500@399', epochs=epoch)
            eph = obj.ephemerides()
            
            return {
                'ra': float(eph['RA'][0]),
                'dec': float(eph['DEC'][0]),
                'magnitude': float(eph['V'][0]) if 'V' in eph.colnames else None,
                'ra_rate': float(eph['RA_rate'][0]),  # arcsec/hour
                'dec_rate': float(eph['DEC_rate'][0])  # arcsec/hour
            }
        except Exception as e:
            logger.error(f"Failed to get position for {designation}: {e}")
            return None
    
    def validate_single_tno(self, tno_info: Dict) -> Dict:
        """Validate pipeline can detect a single known TNO."""
        
        logger.info(f"Validating detection of {tno_info['name']} ({tno_info['designation']})")
        
        # Get current position
        position = self.get_tno_position(tno_info['designation'])
        
        if not position:
            return {
                'tno': tno_info['name'],
                'status': 'failed',
                'error': 'Could not get TNO position'
            }
        
        logger.info(f"{tno_info['name']} current position: RA={position['ra']:.3f}°, "
                   f"Dec={position['dec']:.3f}°, V={position.get('magnitude', '?')}")
        
        # Calculate motion in arcsec/year
        motion_ra_year = position['ra_rate'] * 24 * 365.25  # Convert from arcsec/hour
        motion_dec_year = position['dec_rate'] * 24 * 365.25
        total_motion = np.sqrt(motion_ra_year**2 + motion_dec_year**2)
        
        logger.info(f"Expected motion: {total_motion:.2f} arcsec/year")
        
        # Process region around TNO
        results = self.pipeline.process_region(
            ra_center=position['ra'],
            dec_center=position['dec'],
            size_deg=0.1,  # 6 arcmin region
            band='r'
        )
        
        validation_result = {
            'tno': tno_info['name'],
            'designation': tno_info['designation'],
            'position': position,
            'expected_motion_arcsec_year': total_motion,
            'pipeline_results': results,
            'detected': False,
            'matched_candidate': None
        }
        
        # Check if TNO was detected
        if results['status'] == 'completed' and results.get('candidates'):
            for candidate in results['candidates']:
                # Check if candidate position matches TNO
                sep = self._calculate_separation(
                    candidate['ra'], candidate['dec'],
                    position['ra'], position['dec']
                )
                
                # Allow 5 arcsec tolerance
                if sep < 5.0:
                    validation_result['detected'] = True
                    validation_result['matched_candidate'] = candidate
                    
                    # Compare motion
                    motion_error = abs(candidate['total_motion_arcsec_year'] - total_motion)
                    validation_result['motion_error_arcsec_year'] = motion_error
                    
                    logger.success(f"✅ Detected {tno_info['name']}! "
                                 f"Motion: {candidate['total_motion_arcsec_year']:.2f} "
                                 f"(expected: {total_motion:.2f}) arcsec/year")
                    break
        
        if not validation_result['detected']:
            logger.warning(f"❌ Failed to detect {tno_info['name']}")
            
            # Analyze why
            if results['status'] == 'no_coverage':
                validation_result['failure_reason'] = 'No survey coverage'
            elif results['status'] == 'insufficient_epochs':
                validation_result['failure_reason'] = 'Insufficient multi-epoch data'
            elif results['candidates_found'] == 0:
                validation_result['failure_reason'] = 'No moving objects detected'
            else:
                validation_result['failure_reason'] = 'TNO not matched to any candidate'
        
        return validation_result
    
    def validate_all_tnos(self) -> Dict:
        """Validate pipeline with all test TNOs."""
        
        logger.info("Starting TNO validation suite")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'tnos_tested': len(self.test_tnos),
            'tnos_detected': 0,
            'overall_success_rate': 0.0,
            'individual_results': [],
            'summary': {}
        }
        
        for tno in self.test_tnos:
            result = self.validate_single_tno(tno)
            validation_results['individual_results'].append(result)
            
            if result['detected']:
                validation_results['tnos_detected'] += 1
        
        # Calculate statistics
        validation_results['overall_success_rate'] = (
            validation_results['tnos_detected'] / validation_results['tnos_tested']
        )
        
        # Analyze results by magnitude
        bright_tnos = [r for r in validation_results['individual_results'] 
                       if r['position'].get('magnitude', 99) < 20]
        faint_tnos = [r for r in validation_results['individual_results']
                     if r['position'].get('magnitude', 99) >= 20]
        
        validation_results['summary'] = {
            'bright_detection_rate': (
                sum(1 for t in bright_tnos if t['detected']) / len(bright_tnos)
                if bright_tnos else 0
            ),
            'faint_detection_rate': (
                sum(1 for t in faint_tnos if t['detected']) / len(faint_tnos)
                if faint_tnos else 0
            ),
            'motion_accuracy': self._calculate_motion_accuracy(validation_results),
            'recommendations': self._generate_recommendations(validation_results)
        }
        
        # Save results
        self._save_validation_results(validation_results)
        
        return validation_results
    
    def _calculate_separation(self, ra1: float, dec1: float, 
                            ra2: float, dec2: float) -> float:
        """Calculate angular separation in arcseconds."""
        
        c1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
        c2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
        
        return c1.separation(c2).arcsec
    
    def _calculate_motion_accuracy(self, results: Dict) -> Dict:
        """Calculate motion measurement accuracy statistics."""
        
        motion_errors = []
        
        for result in results['individual_results']:
            if result['detected'] and 'motion_error_arcsec_year' in result:
                motion_errors.append(result['motion_error_arcsec_year'])
        
        if motion_errors:
            return {
                'mean_error': np.mean(motion_errors),
                'std_error': np.std(motion_errors),
                'max_error': np.max(motion_errors),
                'samples': len(motion_errors)
            }
        else:
            return {'mean_error': None, 'samples': 0}
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        
        recommendations = []
        
        success_rate = results['overall_success_rate']
        
        if success_rate >= 0.8:
            recommendations.append("Pipeline successfully validates with known TNOs")
            recommendations.append("Ready for Planet Nine search deployment")
        elif success_rate >= 0.5:
            recommendations.append("Pipeline shows moderate success with known TNOs")
            recommendations.append("Consider parameter tuning for improved sensitivity")
        else:
            recommendations.append("Pipeline needs improvement before Planet Nine search")
            recommendations.append("Review detection thresholds and data quality requirements")
        
        # Specific recommendations
        bright_rate = results['summary']['bright_detection_rate']
        faint_rate = results['summary']['faint_detection_rate']
        
        if bright_rate < 0.9:
            recommendations.append("Investigate why bright TNOs are being missed")
        
        if faint_rate < 0.3:
            recommendations.append("Expected: faint TNO detection is challenging")
            recommendations.append("Planet Nine search may require deeper imaging")
        
        motion_acc = results['summary']['motion_accuracy']
        if motion_acc and motion_acc['mean_error'] > 0.1:
            recommendations.append("Motion measurement accuracy needs improvement")
            recommendations.append("Consider longer time baselines or better alignment")
        
        return recommendations
    
    def _save_validation_results(self, results: Dict):
        """Save validation results."""
        
        # Save detailed JSON
        json_file = self.results_dir / f"tno_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary report
        report_file = self.results_dir / "validation_report.md"
        with open(report_file, 'w') as f:
            f.write("# TNO Detection Validation Report\n\n")
            f.write(f"**Date**: {results['timestamp']}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **TNOs tested**: {results['tnos_tested']}\n")
            f.write(f"- **TNOs detected**: {results['tnos_detected']}\n")
            f.write(f"- **Overall success rate**: {results['overall_success_rate']:.1%}\n")
            f.write(f"- **Bright TNO detection rate**: {results['summary']['bright_detection_rate']:.1%}\n")
            f.write(f"- **Faint TNO detection rate**: {results['summary']['faint_detection_rate']:.1%}\n\n")
            
            f.write("## Individual Results\n\n")
            for result in results['individual_results']:
                status = "✅ DETECTED" if result['detected'] else "❌ MISSED"
                f.write(f"### {result['tno']} - {status}\n")
                f.write(f"- Position: RA={result['position']['ra']:.3f}°, "
                       f"Dec={result['position']['dec']:.3f}°\n")
                f.write(f"- Expected motion: {result['expected_motion_arcsec_year']:.2f} arcsec/year\n")
                
                if result['detected']:
                    cand = result['matched_candidate']
                    f.write(f"- Detected motion: {cand['total_motion_arcsec_year']:.2f} arcsec/year\n")
                    f.write(f"- Motion error: {result.get('motion_error_arcsec_year', 'N/A'):.3f} arcsec/year\n")
                    f.write(f"- Quality score: {cand['quality_score']:.2f}\n")
                else:
                    f.write(f"- Failure reason: {result.get('failure_reason', 'Unknown')}\n")
                f.write("\n")
            
            f.write("## Recommendations\n\n")
            for rec in results['summary']['recommendations']:
                f.write(f"- {rec}\n")
        
        logger.success(f"Validation results saved to {json_file}")
        logger.success(f"Report saved to {report_file}")

def main():
    """Run TNO validation suite."""
    
    print("🔬 TRANS-NEPTUNIAN OBJECT DETECTION VALIDATION")
    print("=" * 60)
    print("Validating Planet Nine detection pipeline with known TNOs")
    print()
    
    validator = TNOValidator()
    
    # Run validation
    results = validator.validate_all_tnos()
    
    # Print summary
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"TNOs tested: {results['tnos_tested']}")
    print(f"TNOs detected: {results['tnos_detected']}")
    print(f"Success rate: {results['overall_success_rate']:.1%}")
    
    print(f"\n📈 Detection rates by magnitude:")
    print(f"Bright TNOs (V<20): {results['summary']['bright_detection_rate']:.1%}")
    print(f"Faint TNOs (V≥20): {results['summary']['faint_detection_rate']:.1%}")
    
    if results['summary']['motion_accuracy']['samples'] > 0:
        print(f"\n🎯 Motion measurement accuracy:")
        print(f"Mean error: {results['summary']['motion_accuracy']['mean_error']:.3f} arcsec/year")
        print(f"Std deviation: {results['summary']['motion_accuracy']['std_error']:.3f} arcsec/year")
    
    print(f"\n💡 Recommendations:")
    for rec in results['summary']['recommendations']:
        print(f"  • {rec}")
    
    print(f"\n✅ Validation complete! Check results/ for detailed reports.")

if __name__ == "__main__":
    main()
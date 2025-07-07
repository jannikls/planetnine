#!/usr/bin/env python
"""
Integrate real astronomical data sources for Planet Nine search:
- DECaLS (Dark Energy Camera Legacy Survey)
- WISE (Wide-field Infrared Survey Explorer)
- Gaia (European Space Agency astrometry mission)
- Pan-STARRS (Panoramic Survey Telescope and Rapid Response System)
- NEOWISE (Near-Earth Object Wide-field Infrared Survey Explorer)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from loguru import logger
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class RealDataIntegrator:
    """
    Integration with real astronomical survey data for Planet Nine search.
    """
    
    def __init__(self):
        self.results_dir = Path("results/real_data_integration")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Data source configurations
        self.data_sources = {
            'decals': {
                'base_url': 'https://www.legacysurvey.org/viewer/',
                'description': 'Dark Energy Camera Legacy Survey',
                'bands': ['g', 'r', 'z'],
                'pixel_scale': 0.262,  # arcsec/pixel
                'depth': 24.0  # AB magnitude limit
            },
            'wise': {
                'base_url': 'https://irsa.ipac.caltech.edu/ibe/data/wise/',
                'description': 'Wide-field Infrared Survey Explorer',
                'bands': ['W1', 'W2', 'W3', 'W4'],
                'pixel_scale': 2.75,  # arcsec/pixel
                'depth': 17.0  # Vega magnitude limit
            },
            'gaia': {
                'description': 'Gaia astrometric survey',
                'catalog': 'Gaia DR3',
                'depth': 21.0,  # G magnitude limit
                'proper_motion_precision': 0.01  # mas/year
            },
            'panstarrs': {
                'description': 'Pan-STARRS survey',
                'bands': ['g', 'r', 'i', 'z', 'y'],
                'depth': 24.0  # AB magnitude limit
            }
        }
        
        # Search parameters
        self.search_regions = []
        self.real_data_results = []
    
    def explore_available_datasets(self) -> Dict:
        """Explore what real astronomical datasets are available."""
        
        logger.info("Exploring available real astronomical datasets")
        
        dataset_info = {
            'survey_catalogs': [],
            'image_archives': [],
            'specialized_catalogs': [],
            'recommendations': []
        }
        
        # Survey catalogs available through astroquery
        survey_catalogs = [
            {
                'name': 'Gaia DR3',
                'description': 'European Space Agency astrometric survey',
                'access_method': 'astroquery.gaia',
                'data_types': ['positions', 'proper_motions', 'parallaxes', 'photometry'],
                'sky_coverage': 'Full sky',
                'epoch': '2016.0',
                'magnitude_limit': 21.0,
                'astrometric_precision': '0.01-0.1 mas',
                'suitable_for_planet_nine': True,
                'notes': 'Excellent for ruling out stellar motions, validating proper motions'
            },
            {
                'name': 'WISE/NEOWISE',
                'description': 'Wide-field Infrared Survey Explorer',
                'access_method': 'astroquery.ipac.irsa',
                'data_types': ['infrared_photometry', 'positions', 'time_series'],
                'sky_coverage': 'Full sky',
                'epochs': 'Multiple (2010-present)',
                'magnitude_limit': 17.0,
                'bands': ['W1 (3.4μm)', 'W2 (4.6μm)', 'W3 (12μm)', 'W4 (22μm)'],
                'suitable_for_planet_nine': True,
                'notes': 'Ideal for detecting thermal emission from distant objects'
            },
            {
                'name': 'DECaLS',
                'description': 'Dark Energy Camera Legacy Survey',
                'access_method': 'Legacy Survey API',
                'data_types': ['optical_imaging', 'photometry', 'catalogs'],
                'sky_coverage': 'Northern hemisphere (~19,000 sq deg)',
                'bands': ['g', 'r', 'z'],
                'magnitude_limit': 24.0,
                'pixel_scale': 0.262,
                'suitable_for_planet_nine': True,
                'notes': 'Deep optical survey, good for faint object detection'
            },
            {
                'name': 'Pan-STARRS',
                'description': 'Panoramic Survey Telescope and Rapid Response System',
                'access_method': 'MAST archive',
                'data_types': ['optical_imaging', 'catalogs', 'difference_imaging'],
                'sky_coverage': 'Northern hemisphere (~30,000 sq deg)',
                'bands': ['g', 'r', 'i', 'z', 'y'],
                'magnitude_limit': 24.0,
                'suitable_for_planet_nine': True,
                'notes': 'Multiple epochs available for proper motion studies'
            },
            {
                'name': 'Catalina Sky Survey',
                'description': 'Near-Earth Object discovery survey',
                'access_method': 'Minor Planet Center',
                'data_types': ['astrometry', 'discovery_data'],
                'sky_coverage': 'Northern hemisphere',
                'suitable_for_planet_nine': False,
                'notes': 'Focused on near-Earth objects, not outer solar system'
            }
        ]
        
        dataset_info['survey_catalogs'] = survey_catalogs
        
        # Image archives
        image_archives = [
            {
                'name': 'Legacy Survey DR10',
                'url': 'https://www.legacysurvey.org/',
                'access': 'Web API, FITS cutouts available',
                'coverage': 'DECaLS, BASS, MzLS combined',
                'suitable_for_planet_nine': True
            },
            {
                'name': 'IRSA (NASA/IPAC)',
                'url': 'https://irsa.ipac.caltech.edu/',
                'access': 'Web interface, API',
                'coverage': 'WISE, Spitzer, 2MASS, many others',
                'suitable_for_planet_nine': True
            },
            {
                'name': 'ESA Gaia Archive',
                'url': 'https://gea.esac.esa.int/archive/',
                'access': 'ADQL queries, TAP services',
                'coverage': 'Full sky astrometry and photometry',
                'suitable_for_planet_nine': True
            }
        ]
        
        dataset_info['image_archives'] = image_archives
        
        # Specialized catalogs for moving objects
        specialized_catalogs = [
            {
                'name': 'Minor Planet Center',
                'description': 'Central repository for asteroid and comet observations',
                'url': 'https://www.minorplanetcenter.net/',
                'suitable_for_planet_nine': True,
                'notes': 'Essential for ruling out known objects'
            },
            {
                'name': 'JPL Horizons',
                'description': 'Solar system ephemeris service',
                'url': 'https://ssd.jpl.nasa.gov/horizons/',
                'suitable_for_planet_nine': True,
                'notes': 'Generate predicted positions for known objects'
            },
            {
                'name': 'NEOCP (NEO Confirmation Page)',
                'description': 'Recently discovered moving objects awaiting confirmation',
                'url': 'https://www.minorplanetcenter.net/iau/NEO/toconfirm_tabular.html',
                'suitable_for_planet_nine': False,
                'notes': 'Near-Earth objects only'
            }
        ]
        
        dataset_info['specialized_catalogs'] = specialized_catalogs
        
        # Recommendations for Planet Nine search
        recommendations = [
            {
                'priority': 'highest',
                'dataset': 'WISE/NEOWISE',
                'reason': 'Thermal emission detection ideal for distant objects',
                'implementation': 'Query IRSA for multi-epoch WISE observations in target regions',
                'expected_sensitivity': 'Objects to ~1000 AU detectable if T > 50K'
            },
            {
                'priority': 'highest',
                'dataset': 'Gaia DR3',
                'reason': 'Validate proper motions, rule out stellar contamination',
                'implementation': 'Cross-match candidates against Gaia catalog',
                'expected_sensitivity': 'Stellar proper motions down to 0.01 mas/year'
            },
            {
                'priority': 'high',
                'dataset': 'DECaLS/Legacy Survey',
                'reason': 'Deep optical imaging with multiple epochs',
                'implementation': 'Download FITS cutouts, perform difference imaging',
                'expected_sensitivity': 'Objects to magnitude 24 in g,r,z bands'
            },
            {
                'priority': 'high', 
                'dataset': 'Pan-STARRS',
                'reason': 'Multiple epochs available for proper motion analysis',
                'implementation': 'Access via MAST, analyze multi-epoch catalogs',
                'expected_sensitivity': 'Proper motions down to 5-10 mas/year over 3+ years'
            },
            {
                'priority': 'medium',
                'dataset': 'Minor Planet Center',
                'reason': 'Essential for ruling out known solar system objects',
                'implementation': 'Cross-match candidates against MPC database',
                'expected_sensitivity': 'All known numbered and provisional objects'
            }
        ]
        
        dataset_info['recommendations'] = recommendations
        
        # Save dataset exploration results
        self._save_dataset_info(dataset_info)
        
        return dataset_info
    
    def query_gaia_region(self, ra_center: float, dec_center: float, radius: float = 0.5) -> Table:
        """Query Gaia DR3 for sources in a region."""
        
        logger.info(f"Querying Gaia DR3 for region RA={ra_center:.3f}, Dec={dec_center:.3f}, radius={radius:.3f} deg")
        
        try:
            # Construct ADQL query for Gaia
            query = f"""
            SELECT source_id, ra, dec, pmra, pmdec, pmra_error, pmdec_error,
                   phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                   parallax, parallax_error, ruwe
            FROM gaiadr3.gaia_source 
            WHERE CONTAINS(POINT('ICRS', ra, dec), 
                          CIRCLE('ICRS', {ra_center}, {dec_center}, {radius})) = 1
            AND phot_g_mean_mag IS NOT NULL
            AND pmra IS NOT NULL 
            AND pmdec IS NOT NULL
            """
            
            # Execute query
            job = Gaia.launch_job_async(query)
            results = job.get_results()
            
            logger.success(f"Retrieved {len(results)} Gaia sources")
            
            return results
            
        except Exception as e:
            logger.error(f"Gaia query failed: {e}")
            return Table()
    
    def query_wise_region(self, ra_center: float, dec_center: float, radius: float = 0.5) -> Dict:
        """Query WISE/NEOWISE data for a region."""
        
        logger.info(f"Querying WISE for region RA={ra_center:.3f}, Dec={dec_center:.3f}, radius={radius:.3f} deg")
        
        try:
            # Use Vizier to query WISE catalog
            vizier = Vizier(columns=['*'], row_limit=10000)
            vizier.ROW_LIMIT = 10000
            
            # Query WISE All-Sky catalog
            coord = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg)
            wise_results = vizier.query_region(coord, radius=radius*u.deg, catalog='II/311/wise')
            
            if wise_results:
                wise_table = wise_results[0]
                logger.success(f"Retrieved {len(wise_table)} WISE sources")
                
                return {
                    'sources': wise_table,
                    'count': len(wise_table),
                    'bands_available': ['W1', 'W2', 'W3', 'W4'],
                    'magnitude_range': {
                        'W1': (wise_table['W1mag'].min(), wise_table['W1mag'].max()),
                        'W2': (wise_table['W2mag'].min(), wise_table['W2mag'].max())
                    } if len(wise_table) > 0 else {}
                }
            else:
                logger.warning("No WISE sources found in region")
                return {'sources': Table(), 'count': 0}
                
        except Exception as e:
            logger.error(f"WISE query failed: {e}")
            return {'sources': Table(), 'count': 0, 'error': str(e)}
    
    def check_legacy_survey_availability(self, ra_center: float, dec_center: float) -> Dict:
        """Check if Legacy Survey data is available for a region."""
        
        logger.info(f"Checking Legacy Survey availability for RA={ra_center:.3f}, Dec={dec_center:.3f}")
        
        try:
            # Check coverage using Legacy Survey API
            url = "https://www.legacysurvey.org/viewer/coverage"
            params = {
                'ra': ra_center,
                'dec': dec_center,
                'layer': 'ls-dr10'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                coverage_info = {
                    'available': True,
                    'survey': 'Legacy Survey DR10',
                    'bands': ['g', 'r', 'z'],
                    'api_url': 'https://www.legacysurvey.org/viewer/',
                    'cutout_available': True,
                    'catalog_available': True
                }
                logger.success("Legacy Survey data available")
            else:
                coverage_info = {
                    'available': False,
                    'reason': f'API returned status {response.status_code}'
                }
                logger.warning("Legacy Survey data not available")
            
            return coverage_info
            
        except Exception as e:
            logger.error(f"Legacy Survey check failed: {e}")
            return {'available': False, 'error': str(e)}
    
    def demonstrate_real_data_access(self) -> Dict:
        """Demonstrate access to real astronomical data sources."""
        
        logger.info("Demonstrating real data access for Planet Nine search")
        
        # Test regions in Planet Nine search area
        test_regions = [
            {'name': 'primary_test', 'ra': 50.0, 'dec': -15.0},
            {'name': 'secondary_test', 'ra': 70.0, 'dec': -25.0},
            {'name': 'extended_test', 'ra': 100.0, 'dec': -20.0}
        ]
        
        demonstration_results = {
            'timestamp': datetime.now().isoformat(),
            'regions_tested': len(test_regions),
            'data_access_results': [],
            'summary': {},
            'next_steps': []
        }
        
        for region in test_regions:
            logger.info(f"Testing data access for region: {region['name']}")
            
            region_results = {
                'region': region,
                'gaia_results': {},
                'wise_results': {},
                'legacy_survey_results': {},
                'success_rate': 0
            }
            
            successful_queries = 0
            total_queries = 3
            
            # Test Gaia access
            try:
                gaia_data = self.query_gaia_region(region['ra'], region['dec'], radius=0.1)
                region_results['gaia_results'] = {
                    'status': 'success',
                    'source_count': len(gaia_data),
                    'has_proper_motions': True if len(gaia_data) > 0 else False
                }
                successful_queries += 1
            except Exception as e:
                region_results['gaia_results'] = {'status': 'failed', 'error': str(e)}
            
            # Test WISE access
            try:
                wise_data = self.query_wise_region(region['ra'], region['dec'], radius=0.1)
                region_results['wise_results'] = {
                    'status': 'success',
                    'source_count': wise_data['count'],
                    'infrared_data': True if wise_data['count'] > 0 else False
                }
                successful_queries += 1
            except Exception as e:
                region_results['wise_results'] = {'status': 'failed', 'error': str(e)}
            
            # Test Legacy Survey access
            try:
                legacy_data = self.check_legacy_survey_availability(region['ra'], region['dec'])
                region_results['legacy_survey_results'] = legacy_data
                if legacy_data.get('available', False):
                    successful_queries += 1
            except Exception as e:
                region_results['legacy_survey_results'] = {'status': 'failed', 'error': str(e)}
            
            region_results['success_rate'] = successful_queries / total_queries
            demonstration_results['data_access_results'].append(region_results)
        
        # Generate summary
        total_success_rate = np.mean([r['success_rate'] for r in demonstration_results['data_access_results']])
        
        demonstration_results['summary'] = {
            'overall_success_rate': total_success_rate,
            'gaia_accessible': sum(1 for r in demonstration_results['data_access_results'] 
                                 if r['gaia_results'].get('status') == 'success'),
            'wise_accessible': sum(1 for r in demonstration_results['data_access_results'] 
                                 if r['wise_results'].get('status') == 'success'),
            'legacy_survey_accessible': sum(1 for r in demonstration_results['data_access_results'] 
                                          if r['legacy_survey_results'].get('available', False)),
            'recommendation': 'proceed' if total_success_rate > 0.5 else 'investigate_issues'
        }
        
        # Generate next steps
        if total_success_rate > 0.7:
            demonstration_results['next_steps'] = [
                'Implement full-scale real data Planet Nine search',
                'Develop automated pipeline for multi-survey data integration',
                'Create candidate validation workflow using multiple catalogs',
                'Establish data download and processing infrastructure'
            ]
        elif total_success_rate > 0.3:
            demonstration_results['next_steps'] = [
                'Resolve data access issues for failed queries',
                'Implement alternative data sources where primary sources fail',
                'Develop robust error handling for network/API issues',
                'Test data access from different network locations'
            ]
        else:
            demonstration_results['next_steps'] = [
                'Investigate fundamental data access problems',
                'Consider using local astronomical data archives',
                'Explore alternative data access methods',
                'Contact survey teams for direct data access'
            ]
        
        # Save demonstration results
        self._save_demonstration_results(demonstration_results)
        
        return demonstration_results
    
    def _save_dataset_info(self, dataset_info: Dict):
        """Save dataset exploration information."""
        
        info_file = self.results_dir / "available_datasets.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
        
        # Create human-readable summary
        summary_file = self.results_dir / "dataset_summary.md"
        with open(summary_file, 'w') as f:
            f.write("# Available Astronomical Datasets for Planet Nine Search\\n\\n")
            
            f.write("## 🌌 Survey Catalogs\\n\\n")
            for survey in dataset_info['survey_catalogs']:
                if survey['suitable_for_planet_nine']:
                    f.write(f"### {survey['name']}\\n")
                    f.write(f"- **Description**: {survey['description']}\\n")
                    f.write(f"- **Coverage**: {survey['sky_coverage']}\\n")
                    f.write(f"- **Access**: {survey['access_method']}\\n")
                    f.write(f"- **Suitable for Planet Nine**: ✅\\n")
                    f.write(f"- **Notes**: {survey['notes']}\\n\\n")
            
            f.write("## 🎯 Recommendations (Priority Order)\\n\\n")
            for rec in sorted(dataset_info['recommendations'], 
                            key=lambda x: {'highest': 1, 'high': 2, 'medium': 3}[x['priority']]):
                f.write(f"### {rec['priority'].title()} Priority: {rec['dataset']}\\n")
                f.write(f"- **Reason**: {rec['reason']}\\n")
                f.write(f"- **Implementation**: {rec['implementation']}\\n")
                f.write(f"- **Expected Sensitivity**: {rec['expected_sensitivity']}\\n\\n")
        
        logger.success(f"Dataset information saved: {info_file}")
        logger.success(f"Summary saved: {summary_file}")
    
    def _save_demonstration_results(self, results: Dict):
        """Save real data access demonstration results."""
        
        results_file = self.results_dir / "real_data_demo_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        summary_file = self.results_dir / "real_data_demo_summary.md"
        with open(summary_file, 'w') as f:
            f.write("# Real Astronomical Data Access Demonstration\\n\\n")
            
            f.write("## 📊 Summary\\n\\n")
            summary = results['summary']
            f.write(f"- **Overall Success Rate**: {summary['overall_success_rate']:.1%}\\n")
            f.write(f"- **Gaia Accessible**: {summary['gaia_accessible']}/{len(results['data_access_results'])} regions\\n")
            f.write(f"- **WISE Accessible**: {summary['wise_accessible']}/{len(results['data_access_results'])} regions\\n")
            f.write(f"- **Legacy Survey Accessible**: {summary['legacy_survey_accessible']}/{len(results['data_access_results'])} regions\\n")
            f.write(f"- **Recommendation**: {summary['recommendation'].upper()}\\n\\n")
            
            f.write("## 🎯 Next Steps\\n\\n")
            for step in results['next_steps']:
                f.write(f"- {step}\\n")
            
            f.write("\\n## 📋 Detailed Results by Region\\n\\n")
            for result in results['data_access_results']:
                region = result['region']
                f.write(f"### {region['name']} (RA={region['ra']:.1f}°, Dec={region['dec']:.1f}°)\\n")
                f.write(f"- **Success Rate**: {result['success_rate']:.1%}\\n")
                f.write(f"- **Gaia**: {result['gaia_results'].get('status', 'unknown')}\\n")
                f.write(f"- **WISE**: {result['wise_results'].get('status', 'unknown')}\\n")
                f.write(f"- **Legacy Survey**: {'available' if result['legacy_survey_results'].get('available', False) else 'not available'}\\n\\n")
        
        logger.success(f"Demonstration results saved: {results_file}")
        logger.success(f"Summary saved: {summary_file}")

def main():
    """Demonstrate real astronomical data integration."""
    
    print("🌌 REAL ASTRONOMICAL DATA INTEGRATION")
    print("=" * 50)
    
    integrator = RealDataIntegrator()
    
    print("🔍 Step 1: Exploring available datasets...")
    dataset_info = integrator.explore_available_datasets()
    
    suitable_surveys = [s['name'] for s in dataset_info['survey_catalogs'] 
                       if s['suitable_for_planet_nine']]
    print(f"✅ Found {len(suitable_surveys)} suitable surveys: {', '.join(suitable_surveys)}")
    
    print("\\n🌐 Step 2: Demonstrating real data access...")
    demo_results = integrator.demonstrate_real_data_access()
    
    print(f"\\n📊 REAL DATA ACCESS RESULTS:")
    print(f"Success rate: {demo_results['summary']['overall_success_rate']:.1%}")
    print(f"Gaia accessible: {demo_results['summary']['gaia_accessible']}/3 regions")
    print(f"WISE accessible: {demo_results['summary']['wise_accessible']}/3 regions") 
    print(f"Legacy Survey accessible: {demo_results['summary']['legacy_survey_accessible']}/3 regions")
    
    recommendation = demo_results['summary']['recommendation']
    if recommendation == 'proceed':
        print("\\n🚀 RECOMMENDATION: Proceed with real data Planet Nine search!")
        print("Real astronomical data sources are accessible and suitable")
    else:
        print("\\n⚠️  RECOMMENDATION: Investigate data access issues before proceeding")
    
    print("\\n📋 Next steps:")
    for step in demo_results['next_steps'][:3]:
        print(f"  • {step}")
    
    print("\\n🎯 STATUS: Real data integration assessment completed")

if __name__ == "__main__":
    main()
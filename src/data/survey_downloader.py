import os
import requests
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astroquery.ipac.irsa import Irsa
from astroquery.utils import commons
from loguru import logger
from tqdm import tqdm
import time
from urllib.parse import urlencode

from ..config import RAW_DATA_DIR, config


class SurveyDownloader:
    """Base class for downloading astronomical survey data."""
    
    def __init__(self, survey_name: str):
        self.survey_name = survey_name
        self.survey_config = config['surveys'][survey_name]
        self.output_dir = RAW_DATA_DIR / survey_name
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def download_region(self, ra_center: float, dec_center: float, 
                       width: float, height: float, 
                       epoch: Optional[str] = None) -> List[Path]:
        """Download data for a specific sky region."""
        raise NotImplementedError


class DECaLSDownloader(SurveyDownloader):
    """Download data from the DECam Legacy Survey."""
    
    def __init__(self):
        super().__init__('decals')
        self.base_url = "https://www.legacysurvey.org/viewer/"
        self.cutout_url = self.base_url + "fits-cutout"
        
    def download_region(self, ra_center: float, dec_center: float,
                       width: float, height: float,
                       epoch: Optional[str] = None) -> List[Path]:
        """Download DECaLS cutouts for a region."""
        downloaded_files = []
        
        width_arcsec = width * 3600
        height_arcsec = height * 3600
        
        for band in self.survey_config['bands']:
            logger.info(f"Downloading DECaLS {band}-band for RA={ra_center}, Dec={dec_center}")
            
            params = {
                'ra': ra_center,
                'dec': dec_center,
                'width': int(width_arcsec),
                'height': int(height_arcsec),
                'layer': f'ls-dr9-{band}',
                'pixscale': self.survey_config['pixel_scale'],
                'bands': band
            }
            
            try:
                response = requests.get(self.cutout_url, params=params, timeout=300)
                response.raise_for_status()
                
                filename = f"decals_{ra_center:.2f}_{dec_center:.2f}_{band}.fits"
                if epoch:
                    filename = f"decals_{ra_center:.2f}_{dec_center:.2f}_{band}_{epoch}.fits"
                
                filepath = self.output_dir / filename
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                downloaded_files.append(filepath)
                logger.success(f"Downloaded {filepath}")
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to download DECaLS {band}-band: {e}")
                
        return downloaded_files
    
    def download_catalog(self, ra_center: float, dec_center: float,
                        radius: float) -> Path:
        """Download DECaLS source catalog for a region."""
        logger.info(f"Downloading DECaLS catalog for RA={ra_center}, Dec={dec_center}, radius={radius}")
        
        catalog_url = "https://www.legacysurvey.org/viewer/cat.json"
        params = {
            'ralo': ra_center - radius,
            'rahi': ra_center + radius,
            'declo': dec_center - radius,
            'dechi': dec_center + radius,
        }
        
        try:
            response = requests.get(catalog_url, params=params, timeout=300)
            response.raise_for_status()
            
            filename = f"decals_catalog_{ra_center:.2f}_{dec_center:.2f}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(response.text)
                
            logger.success(f"Downloaded catalog to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to download DECaLS catalog: {e}")
            return None


class WISEDownloader(SurveyDownloader):
    """Download data from WISE/NEOWISE surveys."""
    
    def __init__(self):
        super().__init__('wise')
        Irsa.ROW_LIMIT = 50000
        
    def download_region(self, ra_center: float, dec_center: float,
                       width: float, height: float,
                       epoch: Optional[str] = None) -> List[Path]:
        """Download WISE image data and catalogs for a region."""
        downloaded_files = []
        
        coord = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg)
        radius = np.sqrt((width/2)**2 + (height/2)**2) * u.deg
        
        for catalog_name in self.survey_config['catalogs']:
            logger.info(f"Querying WISE catalog {catalog_name}")
            
            try:
                result_table = Irsa.query_region(
                    coord,
                    catalog=catalog_name,
                    spatial='Cone',
                    radius=radius
                )
                
                filename = f"wise_{catalog_name}_{ra_center:.2f}_{dec_center:.2f}.fits"
                if epoch:
                    filename = f"wise_{catalog_name}_{ra_center:.2f}_{dec_center:.2f}_{epoch}.fits"
                    
                filepath = self.output_dir / filename
                result_table.write(filepath, format='fits', overwrite=True)
                
                downloaded_files.append(filepath)
                logger.success(f"Downloaded {len(result_table)} sources to {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to download WISE catalog {catalog_name}: {e}")
                
        self._download_wise_images(ra_center, dec_center, width, height, downloaded_files)
        
        return downloaded_files
    
    def _download_wise_images(self, ra: float, dec: float, 
                             width: float, height: float,
                             file_list: List[Path]) -> None:
        """Download WISE image cutouts using IRSA cutout service."""
        cutout_url = "https://irsa.ipac.caltech.edu/cgi-bin/Cutouts/wise"
        
        size_arcsec = max(width, height) * 3600
        
        for band in ['1', '2', '3', '4']:
            logger.info(f"Downloading WISE W{band} image")
            
            params = {
                'mission': 'wise',
                'band': band,
                'coadd_type': 'all',
                'ra': ra,
                'dec': dec,
                'size': size_arcsec,
                'file_type': 'fits'
            }
            
            try:
                response = requests.get(cutout_url, params=params, timeout=300)
                if response.status_code == 200:
                    filename = f"wise_W{band}_{ra:.2f}_{dec:.2f}.fits"
                    filepath = self.output_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                        
                    file_list.append(filepath)
                    logger.success(f"Downloaded WISE W{band} image")
                    
            except Exception as e:
                logger.error(f"Failed to download WISE W{band} image: {e}")
            
            time.sleep(2)


class MultiEpochDownloader:
    """Manage downloads across multiple epochs for proper motion detection."""
    
    def __init__(self):
        self.decals_downloader = DECaLSDownloader()
        self.wise_downloader = WISEDownloader()
        
    def download_all_regions(self, regions: Optional[List[str]] = None) -> Dict[str, List[Path]]:
        """Download data for all configured regions."""
        if regions is None:
            regions = list(config['search_regions'].keys())
            
        all_downloads = {}
        
        for region_key in regions:
            region = config['search_regions'][region_key]
            logger.info(f"Downloading data for region: {region['name']}")
            
            downloads = self.download_region(
                region['ra_center'],
                region['dec_center'],
                region['width'],
                region['height']
            )
            
            all_downloads[region_key] = downloads
            
        return all_downloads
    
    def download_region(self, ra: float, dec: float, 
                       width: float, height: float) -> List[Path]:
        """Download data from all surveys for a single region."""
        all_files = []
        
        decals_files = self.decals_downloader.download_region(ra, dec, width, height)
        all_files.extend(decals_files)
        
        catalog_file = self.decals_downloader.download_catalog(ra, dec, max(width, height)/2)
        if catalog_file:
            all_files.append(catalog_file)
        
        wise_files = self.wise_downloader.download_region(ra, dec, width, height)
        all_files.extend(wise_files)
        
        return all_files


def test_download():
    """Test downloading a small region."""
    downloader = MultiEpochDownloader()
    
    test_ra = 45.0
    test_dec = 20.0
    test_size = 0.5
    
    logger.info(f"Testing download for {test_size}x{test_size} degree region")
    files = downloader.download_region(test_ra, test_dec, test_size, test_size)
    
    logger.info(f"Downloaded {len(files)} files:")
    for f in files:
        logger.info(f"  - {f}")
        
    return files


if __name__ == "__main__":
    test_download()
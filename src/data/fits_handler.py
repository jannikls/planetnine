import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
from loguru import logger
import warnings
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)


class FITSHandler:
    """Handle FITS file operations for astronomical images."""
    
    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)
        self.hdulist = None
        self.header = None
        self.data = None
        self.wcs = None
        self._load()
        
    def _load(self):
        """Load FITS file and extract key components."""
        try:
            self.hdulist = fits.open(self.filepath)
            
            if len(self.hdulist) == 0:
                raise ValueError(f"Empty FITS file: {self.filepath}")
                
            self.header = self.hdulist[0].header
            self.data = self.hdulist[0].data
            
            if self.data is None and len(self.hdulist) > 1:
                self.header = self.hdulist[1].header
                self.data = self.hdulist[1].data
                
            self.wcs = WCS(self.header)
            
            logger.debug(f"Loaded FITS: {self.filepath}, shape: {self.data.shape if self.data is not None else 'None'}")
            
        except Exception as e:
            logger.error(f"Failed to load FITS file {self.filepath}: {e}")
            raise
            
    def close(self):
        """Close the FITS file."""
        if self.hdulist is not None:
            self.hdulist.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    @property
    def image_shape(self) -> Tuple[int, int]:
        """Get image dimensions."""
        if self.data is not None:
            return self.data.shape
        return (0, 0)
        
    @property
    def pixel_scale(self) -> float:
        """Get pixel scale in arcseconds/pixel."""
        if 'CD1_1' in self.header:
            cd11 = self.header['CD1_1']
            cd12 = self.header.get('CD1_2', 0)
            cd21 = self.header.get('CD2_1', 0)
            cd22 = self.header.get('CD2_2', cd11)
            pixel_scale = 3600 * np.sqrt(np.abs(cd11*cd22 - cd12*cd21))
        elif 'CDELT1' in self.header:
            pixel_scale = abs(self.header['CDELT1']) * 3600
        else:
            logger.warning("No pixel scale found in header")
            pixel_scale = 1.0
            
        return pixel_scale
        
    @property
    def observation_time(self) -> Optional[Time]:
        """Get observation time from header."""
        for key in ['DATE-OBS', 'DATE', 'MJD-OBS']:
            if key in self.header:
                try:
                    if key == 'MJD-OBS':
                        return Time(self.header[key], format='mjd')
                    else:
                        return Time(self.header[key])
                except:
                    continue
        return None
        
    @property
    def filter_name(self) -> Optional[str]:
        """Get filter/band name from header."""
        for key in ['FILTER', 'BAND', 'FILTNAM']:
            if key in self.header:
                return str(self.header[key])
        return None
        
    def pixel_to_sky(self, x: Union[float, np.ndarray], 
                     y: Union[float, np.ndarray]) -> SkyCoord:
        """Convert pixel coordinates to sky coordinates."""
        return self.wcs.pixel_to_world(x, y)
        
    def sky_to_pixel(self, ra: Union[float, np.ndarray], 
                     dec: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert sky coordinates to pixel coordinates."""
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        x, y = self.wcs.world_to_pixel(coord)
        return x, y
        
    def get_subimage(self, ra_center: float, dec_center: float, 
                     size: float) -> Tuple[np.ndarray, WCS]:
        """Extract a subimage centered on given coordinates.
        
        Args:
            ra_center: Right ascension in degrees
            dec_center: Declination in degrees
            size: Size of cutout in arcseconds
            
        Returns:
            Tuple of (image_data, wcs)
        """
        x_center, y_center = self.sky_to_pixel(ra_center, dec_center)
        
        size_pixels = int(size / self.pixel_scale)
        
        x_min = int(x_center - size_pixels/2)
        x_max = int(x_center + size_pixels/2)
        y_min = int(y_center - size_pixels/2)
        y_max = int(y_center + size_pixels/2)
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.data.shape[1], x_max)
        y_max = min(self.data.shape[0], y_max)
        
        subimage = self.data[y_min:y_max, x_min:x_max]
        
        sub_wcs = self.wcs.deepcopy()
        sub_wcs.wcs.crpix[0] -= x_min
        sub_wcs.wcs.crpix[1] -= y_min
        
        return subimage, sub_wcs
        
    def calculate_background_stats(self, sigma: float = 3.0) -> Dict[str, float]:
        """Calculate background statistics using sigma clipping."""
        mean, median, std = sigma_clipped_stats(self.data, sigma=sigma)
        
        return {
            'mean': float(mean),
            'median': float(median),
            'std': float(std),
            'mad': float(np.median(np.abs(self.data - median)))
        }


class MultiEpochFITS:
    """Handle multiple FITS files from different epochs."""
    
    def __init__(self, fits_files: List[Path]):
        self.fits_files = sorted(fits_files)
        self.handlers = []
        self.epochs = []
        self._load_all()
        
    def _load_all(self):
        """Load all FITS files and extract epochs."""
        for filepath in self.fits_files:
            try:
                handler = FITSHandler(filepath)
                self.handlers.append(handler)
                
                obs_time = handler.observation_time
                if obs_time:
                    self.epochs.append(obs_time)
                else:
                    logger.warning(f"No observation time found for {filepath}")
                    self.epochs.append(None)
                    
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
                
    def close_all(self):
        """Close all FITS files."""
        for handler in self.handlers:
            handler.close()
            
    def get_aligned_images(self, reference_idx: int = 0) -> List[np.ndarray]:
        """Get all images aligned to a reference image."""
        if not self.handlers:
            return []
            
        reference = self.handlers[reference_idx]
        aligned_images = [reference.data]
        
        for i, handler in enumerate(self.handlers):
            if i == reference_idx:
                continue
                
            aligned_img = self._align_images(reference, handler)
            aligned_images.append(aligned_img)
            
        return aligned_images
        
    def _align_images(self, ref_handler: FITSHandler, 
                      target_handler: FITSHandler) -> np.ndarray:
        """Align target image to reference using WCS."""
        logger.warning("Image alignment not fully implemented yet")
        return target_handler.data
        
    def get_time_baseline(self) -> float:
        """Get time baseline in years between first and last epoch."""
        valid_epochs = [e for e in self.epochs if e is not None]
        if len(valid_epochs) < 2:
            return 0.0
            
        time_diff = (valid_epochs[-1] - valid_epochs[0]).to(u.year).value
        return time_diff


def test_fits_handler():
    """Test FITS handling functionality."""
    test_files = list(Path("data/raw/decals").glob("*.fits"))[:2]
    
    if not test_files:
        logger.warning("No test FITS files found")
        return
        
    logger.info(f"Testing with {len(test_files)} files")
    
    handler = FITSHandler(test_files[0])
    logger.info(f"Image shape: {handler.image_shape}")
    logger.info(f"Pixel scale: {handler.pixel_scale:.3f} arcsec/pixel")
    logger.info(f"Observation time: {handler.observation_time}")
    logger.info(f"Filter: {handler.filter_name}")
    
    stats = handler.calculate_background_stats()
    logger.info(f"Background stats: {stats}")
    
    test_coord = handler.pixel_to_sky(100, 100)
    logger.info(f"Pixel (100,100) -> Sky: {test_coord}")
    
    handler.close()


if __name__ == "__main__":
    test_fits_handler()
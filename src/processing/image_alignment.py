import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.nddata import CCDData
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.optimize import minimize
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from loguru import logger

from ..data.fits_handler import FITSHandler
from ..config import PROCESSED_DATA_DIR


class ImageAligner:
    """Align multi-epoch astronomical images for proper motion detection."""
    
    def __init__(self, pixel_tolerance: float = 0.1):
        """
        Initialize the image aligner.
        
        Args:
            pixel_tolerance: Maximum acceptable alignment error in pixels
        """
        self.pixel_tolerance = pixel_tolerance
        self.aligned_dir = PROCESSED_DATA_DIR / "aligned"
        self.aligned_dir.mkdir(exist_ok=True, parents=True)
        
    def align_image_stack(self, image_files: List[Path], 
                         reference_idx: int = 0) -> List[Path]:
        """
        Align a stack of images to a reference image.
        
        Args:
            image_files: List of FITS image files
            reference_idx: Index of reference image in the list
            
        Returns:
            List of aligned image file paths
        """
        if len(image_files) < 2:
            logger.warning("Need at least 2 images for alignment")
            return image_files
            
        logger.info(f"Aligning {len(image_files)} images to reference {reference_idx}")
        
        # Load reference image
        ref_handler = FITSHandler(image_files[reference_idx])
        ref_sources = self._extract_sources(ref_handler.data, ref_handler.header)
        ref_coords = self._sources_to_coords(ref_sources, ref_handler.wcs)
        
        aligned_files = []
        
        for i, img_file in enumerate(image_files):
            if i == reference_idx:
                # Reference image doesn't need alignment
                aligned_path = self.aligned_dir / f"aligned_{img_file.name}"
                self._copy_with_header_update(img_file, aligned_path, "REFERENCE")
                aligned_files.append(aligned_path)
                continue
                
            logger.info(f"Aligning {img_file.name} to reference")
            
            try:
                # Load target image
                target_handler = FITSHandler(img_file)
                target_sources = self._extract_sources(target_handler.data, target_handler.header)
                target_coords = self._sources_to_coords(target_sources, target_handler.wcs)
                
                # Find transformation
                shift_x, shift_y, quality = self._find_alignment_transform(
                    ref_coords, target_coords
                )
                
                # Apply transformation
                aligned_data = self._apply_shift(target_handler.data, shift_x, shift_y)
                
                # Save aligned image
                aligned_path = self.aligned_dir / f"aligned_{img_file.name}"
                self._save_aligned_image(
                    aligned_data, target_handler.header, aligned_path,
                    shift_x, shift_y, quality
                )
                
                aligned_files.append(aligned_path)
                logger.success(f"Aligned {img_file.name}: shift=({shift_x:.2f}, {shift_y:.2f}), quality={quality:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to align {img_file.name}: {e}")
                # Copy original file as fallback
                aligned_path = self.aligned_dir / f"failed_align_{img_file.name}"
                self._copy_with_header_update(img_file, aligned_path, "FAILED")
                aligned_files.append(aligned_path)
                
        ref_handler.close()
        return aligned_files
    
    def _extract_sources(self, data: np.ndarray, header: fits.Header,
                        detection_threshold: float = 5.0) -> np.ndarray:
        """
        Extract point sources from an image.
        
        Args:
            data: Image data array
            header: FITS header
            detection_threshold: Detection threshold in sigma
            
        Returns:
            Array of source positions (x, y, flux)
        """
        # Calculate background statistics
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
        # Simple peak detection (could be improved with proper source extraction)
        threshold = median + detection_threshold * std
        
        # Find local maxima above threshold
        from scipy.ndimage import maximum_filter
        
        # Create a mask for potential sources
        mask = data > threshold
        
        # Apply maximum filter to find local peaks
        local_maxima = maximum_filter(data, size=5) == data
        source_mask = mask & local_maxima
        
        # Get source positions
        y_coords, x_coords = np.where(source_mask)
        fluxes = data[source_mask]
        
        # Filter out sources too close to edges
        margin = 10
        h, w = data.shape
        valid = ((x_coords > margin) & (x_coords < w - margin) & 
                (y_coords > margin) & (y_coords < h - margin))
        
        # Sort by flux and keep brightest sources
        indices = np.argsort(fluxes[valid])[::-1][:100]  # Top 100 sources
        
        sources = np.column_stack([
            x_coords[valid][indices],
            y_coords[valid][indices], 
            fluxes[valid][indices]
        ])
        
        logger.debug(f"Extracted {len(sources)} sources")
        return sources
    
    def _sources_to_coords(self, sources: np.ndarray, wcs: WCS) -> SkyCoord:
        """Convert pixel coordinates to sky coordinates."""
        if len(sources) == 0:
            return SkyCoord([], [], unit="deg")
            
        x_coords = sources[:, 0]
        y_coords = sources[:, 1]
        
        sky_coords = wcs.pixel_to_world(x_coords, y_coords)
        return sky_coords
    
    def _find_alignment_transform(self, ref_coords: SkyCoord, 
                                target_coords: SkyCoord,
                                match_radius: float = 2.0) -> Tuple[float, float, float]:
        """
        Find the transformation between reference and target coordinates.
        
        Args:
            ref_coords: Reference sky coordinates
            target_coords: Target sky coordinates
            match_radius: Matching radius in arcseconds
            
        Returns:
            Tuple of (shift_x, shift_y, quality)
        """
        if len(ref_coords) == 0 or len(target_coords) == 0:
            raise ValueError("Need sources in both images for alignment")
            
        # Cross-match sources
        idx, d2d, _ = ref_coords.match_to_catalog_sky(target_coords)
        matches = d2d < match_radius * u.arcsec
        
        if np.sum(matches) < 3:
            raise ValueError(f"Only {np.sum(matches)} matched sources, need at least 3")
            
        # Get matched coordinates
        ref_matched = ref_coords[matches]
        target_matched = target_coords[idx[matches]]
        
        # Calculate offsets in arcseconds
        dra = (target_matched.ra - ref_matched.ra) * np.cos(ref_matched.dec)
        ddec = target_matched.dec - ref_matched.dec
        
        # Convert to pixels (assuming 0.262 arcsec/pixel)
        pixel_scale = 0.262  # arcsec/pixel
        dx_pixels = dra.to(u.arcsec).value / pixel_scale
        dy_pixels = ddec.to(u.arcsec).value / pixel_scale
        
        # Robust estimate of shift (median)
        shift_x = np.median(dx_pixels)
        shift_y = np.median(dy_pixels)
        
        # Calculate quality metric (RMS of residuals)
        residuals_x = dx_pixels - shift_x
        residuals_y = dy_pixels - shift_y
        rms = np.sqrt(np.mean(residuals_x**2 + residuals_y**2))
        quality = 1.0 / (1.0 + rms)  # Quality metric [0, 1]
        
        logger.debug(f"Matched {np.sum(matches)} sources, RMS={rms:.3f} pixels")
        
        return shift_x, shift_y, quality
    
    def _apply_shift(self, data: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
        """Apply sub-pixel shift to image data."""
        # Use scipy.ndimage.shift for sub-pixel accuracy
        shifted_data = shift(data, (shift_y, shift_x), mode='constant', cval=0.0)
        return shifted_data
    
    def _save_aligned_image(self, data: np.ndarray, header: fits.Header,
                          output_path: Path, shift_x: float, shift_y: float,
                          quality: float):
        """Save aligned image with alignment metadata."""
        # Update header with alignment information
        header['ALGSHIFT'] = True
        header['ALGSHFTX'] = (shift_x, 'Alignment shift in X (pixels)')
        header['ALGSHFTY'] = (shift_y, 'Alignment shift in Y (pixels)')
        header['ALGQUAL'] = (quality, 'Alignment quality [0,1]')
        header['ALGDATE'] = (str(Time.now().iso), 'Alignment processing date')
        
        # Save FITS file
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_path, overwrite=True)
    
    def _copy_with_header_update(self, input_path: Path, output_path: Path, 
                                status: str):
        """Copy FITS file with status update."""
        with fits.open(input_path) as hdul:
            header = hdul[0].header.copy()
            header['ALGSTAT'] = (status, 'Alignment status')
            header['ALGDATE'] = (str(Time.now().iso), 'Alignment processing date')
            
            hdu = fits.PrimaryHDU(data=hdul[0].data, header=header)
            hdu.writeto(output_path, overwrite=True)


class StackCreator:
    """Create stacked images from aligned multi-epoch data."""
    
    def __init__(self):
        self.stack_dir = PROCESSED_DATA_DIR / "stacks"
        self.stack_dir.mkdir(exist_ok=True, parents=True)
    
    def create_median_stack(self, aligned_files: List[Path], 
                          output_name: str) -> Path:
        """
        Create median-stacked image from aligned files.
        
        Args:
            aligned_files: List of aligned FITS files
            output_name: Name for output stacked image
            
        Returns:
            Path to stacked image
        """
        logger.info(f"Creating median stack from {len(aligned_files)} images")
        
        # Load all images
        image_data = []
        headers = []
        
        for img_file in aligned_files:
            with fits.open(img_file) as hdul:
                image_data.append(hdul[0].data)
                headers.append(hdul[0].header)
        
        # Create median stack
        image_stack = np.array(image_data)
        median_image = np.median(image_stack, axis=0)
        
        # Create combined header
        combined_header = headers[0].copy()
        combined_header['STACKTYP'] = ('MEDIAN', 'Type of stack')
        combined_header['STACKNUM'] = (len(aligned_files), 'Number of stacked images')
        combined_header['STACKDT'] = (str(Time.now().iso), 'Stack creation date')
        
        # Add information about input files
        for i, img_file in enumerate(aligned_files):
            combined_header[f'STACK{i:02d}'] = (img_file.name, f'Input file {i}')
        
        # Save stacked image
        output_path = self.stack_dir / f"{output_name}_median_stack.fits"
        hdu = fits.PrimaryHDU(data=median_image, header=combined_header)
        hdu.writeto(output_path, overwrite=True)
        
        logger.success(f"Created median stack: {output_path}")
        return output_path
    
    def create_difference_images(self, aligned_files: List[Path], 
                               reference_idx: int = 0) -> List[Path]:
        """
        Create difference images: target - reference.
        
        Args:
            aligned_files: List of aligned FITS files
            reference_idx: Index of reference image
            
        Returns:
            List of difference image paths
        """
        logger.info(f"Creating difference images with reference {reference_idx}")
        
        diff_dir = PROCESSED_DATA_DIR / "difference"
        diff_dir.mkdir(exist_ok=True, parents=True)
        
        # Load reference image
        with fits.open(aligned_files[reference_idx]) as ref_hdul:
            ref_data = ref_hdul[0].data
            ref_header = ref_hdul[0].header
        
        difference_files = []
        
        for i, img_file in enumerate(aligned_files):
            if i == reference_idx:
                continue
                
            with fits.open(img_file) as hdul:
                target_data = hdul[0].data
                target_header = hdul[0].header
                
                # Create difference image
                diff_data = target_data - ref_data
                
                # Create difference header
                diff_header = target_header.copy()
                diff_header['DIFFTYPE'] = ('TARGET-REF', 'Type of difference')
                diff_header['DIFFREF'] = (aligned_files[reference_idx].name, 'Reference image')
                diff_header['DIFFTGT'] = (img_file.name, 'Target image')
                diff_header['DIFFDATE'] = (str(Time.now().iso), 'Difference creation date')
                
                # Save difference image
                output_name = f"diff_{img_file.stem}_minus_{aligned_files[reference_idx].stem}.fits"
                output_path = diff_dir / output_name
                
                hdu = fits.PrimaryHDU(data=diff_data, header=diff_header)
                hdu.writeto(output_path, overwrite=True)
                
                difference_files.append(output_path)
                logger.success(f"Created difference image: {output_path}")
        
        return difference_files


def test_image_processing():
    """Test image processing pipeline with available data."""
    from ..config import RAW_DATA_DIR
    
    # Find FITS files
    fits_files = list(RAW_DATA_DIR.glob("**/*.fits"))
    
    if len(fits_files) < 2:
        logger.warning("Need at least 2 FITS files for processing test")
        return
    
    logger.info(f"Testing image processing with {len(fits_files)} files")
    
    # Test alignment
    aligner = ImageAligner()
    try:
        aligned_files = aligner.align_image_stack(fits_files)
        logger.success(f"Aligned {len(aligned_files)} images")
        
        # Test stacking
        stacker = StackCreator()
        stack_path = stacker.create_median_stack(aligned_files, "test_stack")
        
        # Test difference images
        diff_files = stacker.create_difference_images(aligned_files)
        
        logger.success(f"Image processing test completed")
        logger.info(f"Stack: {stack_path}")
        logger.info(f"Differences: {len(diff_files)} files")
        
    except Exception as e:
        logger.error(f"Image processing test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    test_image_processing()
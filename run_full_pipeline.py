#!/usr/bin/env python
"""
Full Planet Nine detection pipeline for a 1 square degree region.
This script runs the complete pipeline from data download to candidate validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from loguru import logger
import pandas as pd
from typing import List, Dict, Tuple
import json

from src.config import config, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from src.data.survey_downloader import DECaLSDownloader
from src.processing.image_alignment import ImageAligner, StackCreator
from test_moving_object_detection import (
    detect_moving_objects_in_difference, 
    match_motion_pairs,
    visualize_detections
)


class PlanetNinePipeline:
    """Complete Planet Nine detection pipeline."""
    
    def __init__(self, target_region: Dict):
        """
        Initialize the pipeline for a target sky region.
        
        Args:
            target_region: Dict with 'ra', 'dec', 'width', 'height' in degrees
        """
        self.target_region = target_region
        self.pipeline_dir = RESULTS_DIR / "pipeline_run"
        self.pipeline_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.downloader = DECaLSDownloader()
        self.aligner = ImageAligner()
        self.stacker = StackCreator()
        
        # Results storage
        self.downloaded_files = []
        self.aligned_files = []
        self.difference_files = []
        self.candidates = []
        
    def step1_download_data(self, bands: List[str] = ['g', 'r', 'z']) -> List[Path]:
        """Use existing data or download multi-epoch data for the target region."""
        logger.info("STEP 1: Finding existing survey data")
        
        # First, check for existing data
        existing_files = []
        for data_dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR / "aligned"]:
            if data_dir.exists():
                fits_files = list(data_dir.glob("**/*.fits"))
                existing_files.extend(fits_files)
        
        if existing_files:
            logger.info(f"Found {len(existing_files)} existing FITS files")
            # Use existing files for the pipeline
            self.downloaded_files = existing_files[:9]  # Use up to 9 files to simulate multi-epoch
            logger.success(f"Using {len(self.downloaded_files)} existing files")
            return self.downloaded_files
        
        # If no existing data, try to download new data
        logger.info("No existing data found, attempting download")
        ra_center = self.target_region['ra']
        dec_center = self.target_region['dec']
        width = self.target_region['width']
        height = self.target_region['height']
        
        logger.info(f"Target region: RA={ra_center}°, Dec={dec_center}°, Size={width}°×{height}°")
        
        downloaded_files = []
        
        # Try to download new data
        try:
            file_paths = self.downloader.download_region(
                ra_center=ra_center, dec_center=dec_center,
                width=width, height=height
            )
            
            if file_paths:
                downloaded_files.extend(file_paths)
                logger.success(f"Downloaded {len(file_paths)} new files")
            else:
                logger.warning("Download failed, creating synthetic test data")
                downloaded_files = self._create_synthetic_test_data()
                
        except Exception as e:
            logger.error(f"Download error: {e}")
            logger.info("Creating synthetic test data instead")
            downloaded_files = self._create_synthetic_test_data()
        
        self.downloaded_files = downloaded_files
        logger.info(f"Using {len(downloaded_files)} files for pipeline")
        return downloaded_files
    
    def _create_synthetic_test_data(self) -> List[Path]:
        """Create synthetic test data for pipeline demonstration."""
        logger.info("Creating synthetic test data")
        
        synthetic_dir = RAW_DATA_DIR / "synthetic"
        synthetic_dir.mkdir(exist_ok=True, parents=True)
        
        synthetic_files = []
        
        # Create 6 synthetic FITS files to simulate multi-epoch observations
        image_size = 512
        
        for i in range(6):
            # Create synthetic image with noise and a few point sources
            image_data = np.random.normal(100, 10, (image_size, image_size))
            
            # Add some point sources
            for j in range(5):
                x = np.random.randint(50, image_size - 50)
                y = np.random.randint(50, image_size - 50)
                brightness = np.random.uniform(500, 2000)
                
                # Add Gaussian PSF
                y_coords, x_coords = np.ogrid[:image_size, :image_size]
                psf = brightness * np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * 3**2))
                image_data += psf
            
            # Add a moving object (same position with small offset between epochs)
            moving_x = 200 + i * 0.5  # Small motion between epochs
            moving_y = 300 + i * 0.2
            moving_brightness = 1000
            
            y_coords, x_coords = np.ogrid[:image_size, :image_size]
            moving_psf = moving_brightness * np.exp(-((x_coords - moving_x)**2 + (y_coords - moving_y)**2) / (2 * 2**2))
            image_data += moving_psf
            
            # Create FITS file
            from astropy.io import fits
            from astropy.wcs import WCS
            
            # Create basic WCS
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [image_size/2, image_size/2]
            wcs.wcs.crval = [180.0, 0.0]  # RA, Dec center
            wcs.wcs.cdelt = [-0.262/3600, 0.262/3600]  # DECaLS pixel scale
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            
            header = wcs.to_header()
            header['OBJECT'] = 'Synthetic Test'
            header['FILTER'] = ['g', 'r', 'z'][i % 3]
            header['EPOCH'] = f'epoch_{i//3 + 1}'
            
            hdu = fits.PrimaryHDU(data=image_data.astype(np.float32), header=header)
            
            filename = f"synthetic_test_{i:02d}.fits"
            filepath = synthetic_dir / filename
            
            hdu.writeto(filepath, overwrite=True)
            synthetic_files.append(filepath)
            
            logger.debug(f"Created synthetic file: {filename}")
        
        logger.success(f"Created {len(synthetic_files)} synthetic test files")
        return synthetic_files
    
    def step2_process_images(self) -> List[Path]:
        """Process images through alignment and differencing pipeline."""
        logger.info("STEP 2: Processing images through alignment pipeline")
        
        if len(self.downloaded_files) < 2:
            logger.error("Need at least 2 images for processing")
            return []
        
        # Group files by band for alignment
        band_groups = {}
        for file_path in self.downloaded_files:
            # Extract band from filename
            parts = file_path.stem.split('_')
            if len(parts) >= 4:
                band = parts[3]  # Assuming format: decals_ra_dec_band_epoch
                if band not in band_groups:
                    band_groups[band] = []
                band_groups[band].append(file_path)
        
        all_aligned_files = []
        all_difference_files = []
        
        # Process each band separately
        for band, files in band_groups.items():
            if len(files) < 2:
                logger.warning(f"Skipping {band}-band: only {len(files)} images")
                continue
                
            logger.info(f"Processing {len(files)} images in {band}-band")
            
            try:
                # Align images
                aligned_files = self.aligner.align_image_stack(files, reference_idx=0)
                all_aligned_files.extend(aligned_files)
                
                # Create difference images
                diff_files = self.stacker.create_difference_images(aligned_files, reference_idx=0)
                all_difference_files.extend(diff_files)
                
                logger.success(f"Processed {band}-band: {len(aligned_files)} aligned, {len(diff_files)} differences")
                
            except Exception as e:
                logger.error(f"Processing error for {band}-band: {e}")
        
        self.aligned_files = all_aligned_files
        self.difference_files = all_difference_files
        
        logger.info(f"Total processed: {len(all_aligned_files)} aligned, {len(all_difference_files)} differences")
        return all_difference_files
    
    def step3_detect_candidates(self, detection_threshold: float = 3.0) -> List[Dict]:
        """Run moving object detection on difference images."""
        logger.info("STEP 3: Detecting moving object candidates")
        
        if not self.difference_files:
            logger.error("No difference images available for detection")
            return []
        
        all_candidates = []
        detection_stats = []
        
        for i, diff_file in enumerate(self.difference_files):
            logger.info(f"Processing difference image {i+1}/{len(self.difference_files)}: {diff_file.name}")
            
            try:
                from astropy.io import fits
                with fits.open(diff_file) as hdul:
                    diff_data = hdul[0].data
                    header = hdul[0].header
                
                # Extract observation info from header
                obs_info = {
                    'file': diff_file.name,
                    'target_file': header.get('DIFFTGT', 'unknown'),
                    'reference_file': header.get('DIFFREF', 'unknown'),
                }
                
                # Detect moving objects
                detections = detect_moving_objects_in_difference(diff_data, detection_threshold)
                
                # Separate appeared and disappeared
                appeared = [d for d in detections if d['type'] == 'appeared']
                disappeared = [d for d in detections if d['type'] == 'disappeared']
                
                # Match motion pairs
                motion_candidates = match_motion_pairs(appeared, disappeared)
                
                # Add metadata to candidates
                for candidate in motion_candidates:
                    candidate.update(obs_info)
                    candidate['detection_id'] = f"det_{i:03d}_{len(all_candidates):03d}"
                
                all_candidates.extend(motion_candidates)
                
                detection_stats.append({
                    'file': diff_file.name,
                    'detections': len(detections),
                    'appeared': len(appeared),
                    'disappeared': len(disappeared),
                    'motion_candidates': len(motion_candidates)
                })
                
                logger.info(f"  Found {len(motion_candidates)} motion candidates")
                
            except Exception as e:
                logger.error(f"Detection error for {diff_file.name}: {e}")
        
        self.candidates = all_candidates
        
        # Save detection statistics
        stats_df = pd.DataFrame(detection_stats)
        stats_path = self.pipeline_dir / "detection_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        
        logger.success(f"Detection complete: {len(all_candidates)} total candidates")
        return all_candidates
    
    def step4_validate_candidates(self) -> pd.DataFrame:
        """Implement statistical validation for candidates."""
        logger.info("STEP 4: Validating candidates with statistical analysis")
        
        if not self.candidates:
            logger.warning("No candidates to validate")
            return pd.DataFrame()
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.candidates)
        
        # Add validation metrics
        validation_metrics = []
        
        for i, candidate in enumerate(self.candidates):
            metrics = self._calculate_validation_metrics(candidate)
            metrics['candidate_id'] = candidate['detection_id']
            validation_metrics.append(metrics)
        
        validation_df = pd.DataFrame(validation_metrics)
        
        # Merge with candidate data
        full_df = df.merge(validation_df, left_on='detection_id', right_on='candidate_id', how='left')
        
        # Apply validation filters
        full_df['is_valid_motion'] = (
            (full_df['motion_arcsec_year'] >= 0.1) & 
            (full_df['motion_arcsec_year'] <= 10.0)
        )
        
        full_df['is_planet_nine_candidate'] = (
            (full_df['motion_arcsec_year'] >= 0.2) & 
            (full_df['motion_arcsec_year'] <= 0.8)
        )
        
        full_df['quality_score'] = (
            full_df['match_score'] * 
            full_df['flux_consistency'] * 
            full_df['motion_quality']
        )
        
        # Save results
        results_path = self.pipeline_dir / "candidate_analysis.csv"
        full_df.to_csv(results_path, index=False)
        
        # Summary statistics
        total_candidates = len(full_df)
        valid_motion = full_df['is_valid_motion'].sum()
        planet_nine_candidates = full_df['is_planet_nine_candidate'].sum()
        high_quality = (full_df['quality_score'] > 0.5).sum()
        
        summary = {
            'total_candidates': int(total_candidates),
            'valid_motion': int(valid_motion),
            'planet_nine_candidates': int(planet_nine_candidates),
            'high_quality': int(high_quality),
            'detection_rate': f"{total_candidates/len(self.difference_files):.1f} per image"
        }
        
        # Save summary
        summary_path = self.pipeline_dir / "validation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.success(f"Validation complete: {planet_nine_candidates} Planet Nine candidates from {total_candidates} total")
        
        return full_df
    
    def _calculate_validation_metrics(self, candidate: Dict) -> Dict:
        """Calculate validation metrics for a single candidate."""
        
        # Flux consistency (how similar are start and end fluxes)
        flux_ratio = candidate['flux_ratio']
        flux_consistency = 1.0 / (1.0 + abs(flux_ratio - 1.0))
        
        # Motion quality (prefer moderate motion, not too fast/slow)
        motion = candidate['motion_arcsec_year']
        if motion < 0.1:
            motion_quality = motion / 0.1  # Penalize very slow motion
        elif motion > 2.0:
            motion_quality = 2.0 / motion  # Penalize very fast motion  
        else:
            motion_quality = 1.0
        
        # Distance from image center (prefer central detections)
        # Assuming 512x512 pixel images for now
        center_x, center_y = 256, 256
        start_dist = np.sqrt((candidate['start_x'] - center_x)**2 + (candidate['start_y'] - center_y)**2)
        edge_penalty = max(0, 1.0 - start_dist / 256)
        
        return {
            'flux_consistency': flux_consistency,
            'motion_quality': motion_quality,
            'edge_penalty': edge_penalty,
            'total_motion_pixels': candidate['motion_pixels']
        }
    
    def step5_create_visualizations(self, candidate_df: pd.DataFrame):
        """Create comprehensive visualizations of detection results."""
        logger.info("STEP 5: Creating detection visualizations")
        
        plots_dir = self.pipeline_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Candidate summary plot
        self._plot_candidate_summary(candidate_df, plots_dir)
        
        # 2. Motion distribution plots
        self._plot_motion_distributions(candidate_df, plots_dir)
        
        # 3. Sky position map
        self._plot_sky_positions(candidate_df, plots_dir)
        
        # 4. Quality assessment plots
        self._plot_quality_assessment(candidate_df, plots_dir)
        
        # 5. Individual candidate stamps (top candidates)
        self._create_candidate_stamps(candidate_df, plots_dir)
        
        logger.success(f"Visualizations saved to {plots_dir}")
    
    def _plot_candidate_summary(self, df: pd.DataFrame, plots_dir: Path):
        """Create summary overview plot."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Motion histogram
        ax1 = axes[0, 0]
        ax1.hist(df['motion_arcsec_year'], bins=20, alpha=0.7, edgecolor='black')
        ax1.axvspan(0.2, 0.8, alpha=0.3, color='red', label='Planet Nine range')
        ax1.set_xlabel('Proper Motion (arcsec/year)')
        ax1.set_ylabel('Count')
        ax1.set_title('Motion Distribution')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Quality scores
        ax2 = axes[0, 1]
        ax2.hist(df['quality_score'], bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Quality Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Quality Score Distribution')
        
        # Flux ratio vs motion
        ax3 = axes[1, 0]
        scatter = ax3.scatter(df['motion_arcsec_year'], df['flux_ratio'], 
                            c=df['quality_score'], cmap='viridis', alpha=0.7)
        ax3.set_xlabel('Proper Motion (arcsec/year)')
        ax3.set_ylabel('Flux Ratio (end/start)')
        ax3.set_title('Motion vs Flux Consistency')
        ax3.set_xscale('log')
        plt.colorbar(scatter, ax=ax3, label='Quality Score')
        
        # Detection statistics
        ax4 = axes[1, 1]
        categories = ['Total', 'Valid Motion', 'Planet Nine', 'High Quality']
        counts = [
            len(df),
            df['is_valid_motion'].sum(),
            df['is_planet_nine_candidate'].sum(),
            (df['quality_score'] > 0.5).sum()
        ]
        bars = ax4.bar(categories, counts, color=['blue', 'green', 'red', 'orange'])
        ax4.set_ylabel('Count')
        ax4.set_title('Detection Categories')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "candidate_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_motion_distributions(self, df: pd.DataFrame, plots_dir: Path):
        """Plot detailed motion analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Linear motion histogram
        ax1 = axes[0]
        ax1.hist(df['motion_arcsec_year'], bins=30, alpha=0.7, edgecolor='black')
        ax1.axvspan(0.2, 0.8, alpha=0.3, color='red', label='Planet Nine range')
        ax1.set_xlabel('Proper Motion (arcsec/year)')
        ax1.set_ylabel('Count')
        ax1.set_title('Linear Scale Motion Distribution')
        ax1.legend()
        
        # Log motion histogram
        ax2 = axes[1]
        ax2.hist(df['motion_arcsec_year'], bins=np.logspace(-1, 1, 30), alpha=0.7, edgecolor='black')
        ax2.axvspan(0.2, 0.8, alpha=0.3, color='red', label='Planet Nine range')
        ax2.set_xlabel('Proper Motion (arcsec/year)')
        ax2.set_ylabel('Count')
        ax2.set_title('Log Scale Motion Distribution')
        ax2.set_xscale('log')
        ax2.legend()
        
        # Motion vs match score
        ax3 = axes[2]
        ax3.scatter(df['motion_arcsec_year'], df['match_score'], alpha=0.6)
        ax3.set_xlabel('Proper Motion (arcsec/year)')
        ax3.set_ylabel('Match Score')
        ax3.set_title('Motion vs Detection Quality')
        ax3.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "motion_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_sky_positions(self, df: pd.DataFrame, plots_dir: Path):
        """Plot candidate positions on sky."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Start positions
        ax1 = axes[0]
        scatter1 = ax1.scatter(df['start_x'], df['start_y'], 
                             c=df['motion_arcsec_year'], cmap='plasma', alpha=0.7)
        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Y Position (pixels)')
        ax1.set_title('Candidate Start Positions')
        plt.colorbar(scatter1, ax=ax1, label='Motion (arcsec/yr)')
        
        # Motion vectors
        ax2 = axes[1]
        for _, row in df.iterrows():
            if row['is_planet_nine_candidate']:
                color = 'red'
                alpha = 1.0
                zorder = 10
            else:
                color = 'blue'
                alpha = 0.5
                zorder = 1
                
            ax2.arrow(row['start_x'], row['start_y'],
                     row['end_x'] - row['start_x'],
                     row['end_y'] - row['start_y'],
                     head_width=5, head_length=5, 
                     fc=color, ec=color, alpha=alpha, zorder=zorder)
        
        ax2.set_xlabel('X Position (pixels)')
        ax2.set_ylabel('Y Position (pixels)')
        ax2.set_title('Motion Vectors (Red = Planet Nine candidates)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "sky_positions.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_assessment(self, df: pd.DataFrame, plots_dir: Path):
        """Plot quality assessment metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Quality components
        ax1 = axes[0, 0]
        ax1.scatter(df['flux_consistency'], df['motion_quality'], 
                   c=df['quality_score'], cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Flux Consistency')
        ax1.set_ylabel('Motion Quality')
        ax1.set_title('Quality Components')
        
        # Match score vs quality
        ax2 = axes[0, 1]
        ax2.scatter(df['match_score'], df['quality_score'], alpha=0.7)
        ax2.set_xlabel('Match Score')
        ax2.set_ylabel('Overall Quality Score')
        ax2.set_title('Match Quality vs Overall Quality')
        
        # Motion vs quality for Planet Nine candidates
        ax3 = axes[1, 0]
        p9_candidates = df[df['is_planet_nine_candidate']]
        others = df[~df['is_planet_nine_candidate']]
        
        ax3.scatter(others['motion_arcsec_year'], others['quality_score'], 
                   alpha=0.5, c='blue', label='Other candidates')
        ax3.scatter(p9_candidates['motion_arcsec_year'], p9_candidates['quality_score'], 
                   alpha=0.8, c='red', label='Planet Nine candidates')
        ax3.set_xlabel('Proper Motion (arcsec/year)')
        ax3.set_ylabel('Quality Score')
        ax3.set_title('Quality vs Motion')
        ax3.legend()
        
        # False positive analysis
        ax4 = axes[1, 1]
        motion_bins = np.logspace(-1, 1, 20)
        ax4.hist(df['motion_arcsec_year'], bins=motion_bins, alpha=0.5, 
                label='All candidates', density=True)
        ax4.hist(df[df['quality_score'] > 0.5]['motion_arcsec_year'], 
                bins=motion_bins, alpha=0.7, label='High quality', density=True)
        ax4.set_xlabel('Proper Motion (arcsec/year)')
        ax4.set_ylabel('Normalized Count')
        ax4.set_title('Quality Filtering Effect')
        ax4.set_xscale('log')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "quality_assessment.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_candidate_stamps(self, df: pd.DataFrame, plots_dir: Path):
        """Create postage stamps for top candidates."""
        # Select top candidates
        top_candidates = df.nlargest(6, 'quality_score')
        
        if len(top_candidates) == 0:
            logger.warning("No candidates found for stamp creation")
            return
        
        stamps_dir = plots_dir / "candidate_stamps"
        stamps_dir.mkdir(exist_ok=True)
        
        for i, (_, candidate) in enumerate(top_candidates.iterrows()):
            self._create_single_stamp(candidate, stamps_dir, i)
        
        logger.info(f"Created stamps for {len(top_candidates)} top candidates")
    
    def _create_single_stamp(self, candidate: Dict, stamps_dir: Path, index: int):
        """Create a single candidate stamp."""
        try:
            # Create a simple visualization since we don't have access to the original images
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            
            # Draw motion vector
            ax.arrow(candidate['start_x'], candidate['start_y'],
                    candidate['end_x'] - candidate['start_x'],
                    candidate['end_y'] - candidate['start_y'],
                    head_width=2, head_length=2, fc='red', ec='red')
            
            # Mark positions
            ax.plot(candidate['start_x'], candidate['start_y'], 'bo', markersize=8, label='Start')
            ax.plot(candidate['end_x'], candidate['end_y'], 'ro', markersize=8, label='End')
            
            # Add candidate info
            info_text = f"ID: {candidate['detection_id']}\n"
            info_text += f"Motion: {candidate['motion_arcsec_year']:.3f} arcsec/yr\n"
            info_text += f"Quality: {candidate['quality_score']:.3f}\n"
            info_text += f"Flux ratio: {candidate['flux_ratio']:.2f}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('X Position (pixels)')
            ax.set_ylabel('Y Position (pixels)')
            ax.set_title(f"Candidate {index+1}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig(stamps_dir / f"candidate_{index+1:02d}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create stamp for candidate {index}: {e}")
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """Run the complete detection pipeline."""
        logger.info("🚀 STARTING FULL PLANET NINE DETECTION PIPELINE")
        logger.info("=" * 60)
        
        start_time = Time.now()
        
        try:
            # Step 1: Download data
            self.step1_download_data()
            
            # Step 2: Process images  
            self.step2_process_images()
            
            # Step 3: Detect candidates
            self.step3_detect_candidates()
            
            # Step 4: Validate candidates
            candidate_df = self.step4_validate_candidates()
            
            # Step 5: Create visualizations
            if not candidate_df.empty:
                self.step5_create_visualizations(candidate_df)
            
            # Final summary
            end_time = Time.now()
            duration = (end_time - start_time).to(u.minute).value
            
            logger.success("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Processing time: {duration:.1f} minutes")
            logger.info(f"Results saved to: {self.pipeline_dir}")
            
            return candidate_df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Run the full pipeline on a test region."""
    
    # Define target region (1 square degree)
    target_region = {
        'ra': 180.0,    # degrees
        'dec': 0.0,     # degrees  
        'width': 1.0,   # degrees
        'height': 1.0   # degrees
    }
    
    logger.info(f"Running Planet Nine pipeline on region: {target_region}")
    
    # Initialize and run pipeline
    pipeline = PlanetNinePipeline(target_region)
    results_df = pipeline.run_full_pipeline()
    
    # Print final summary
    if not results_df.empty:
        total = len(results_df)
        planet_nine = results_df['is_planet_nine_candidate'].sum()
        high_quality = (results_df['quality_score'] > 0.5).sum()
        
        print("\n" + "="*50)
        print("🎯 FINAL DETECTION SUMMARY")
        print("="*50)
        print(f"Total candidates found: {total}")
        print(f"Planet Nine candidates: {planet_nine}")
        print(f"High quality candidates: {high_quality}")
        
        if planet_nine > 0:
            print(f"\n🎉 Found {planet_nine} Planet Nine candidates!")
            print("Top candidates:")
            top_p9 = results_df[results_df['is_planet_nine_candidate']].nlargest(3, 'quality_score')
            for i, (_, row) in enumerate(top_p9.iterrows()):
                print(f"  {i+1}. {row['detection_id']}: {row['motion_arcsec_year']:.3f} arcsec/yr (quality: {row['quality_score']:.3f})")
        else:
            print("No Planet Nine candidates found in this region.")
    else:
        print("No candidates detected.")


if __name__ == "__main__":
    main()
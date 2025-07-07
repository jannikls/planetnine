#!/usr/bin/env python
"""
Test moving object detection on difference images
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.ndimage import label, center_of_mass
from loguru import logger

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.data.fits_handler import FITSHandler


def create_synthetic_moving_object(image_shape, motion_pixels, brightness=1000):
    """Create a synthetic moving object for testing."""
    h, w = image_shape
    
    # Start position (random)
    start_x = np.random.randint(50, w-50)
    start_y = np.random.randint(50, h-50)
    
    # End position after motion
    end_x = start_x + motion_pixels[0]
    end_y = start_y + motion_pixels[1]
    
    # Create two images with the object at different positions
    image1 = np.random.normal(0, 10, image_shape)  # Background noise
    image2 = np.random.normal(0, 10, image_shape)
    
    # Add point source (Gaussian PSF)
    y_coords, x_coords = np.ogrid[:h, :w]
    
    # Object in first image
    psf1 = brightness * np.exp(-((x_coords - start_x)**2 + (y_coords - start_y)**2) / (2 * 2**2))
    image1 += psf1
    
    # Object in second image (moved)
    psf2 = brightness * np.exp(-((x_coords - end_x)**2 + (y_coords - end_y)**2) / (2 * 2**2))
    image2 += psf2
    
    return image1, image2, (start_x, start_y), (end_x, end_y)


def detect_moving_objects_in_difference(diff_image, detection_threshold=3.0):
    """
    Detect moving objects in difference images.
    
    Args:
        diff_image: Difference image (target - reference)
        detection_threshold: Detection threshold in sigma
        
    Returns:
        List of detected object positions and properties
    """
    logger.info("Detecting moving objects in difference image")
    
    # Use robust statistics for background estimation
    from astropy.stats import sigma_clipped_stats
    mean, median, std = sigma_clipped_stats(diff_image, sigma=3.0, maxiters=5)
    
    logger.info(f"Image stats: mean={mean:.2f}, median={median:.2f}, std={std:.2f}")
    
    # Create detection masks for positive and negative sources
    positive_mask = diff_image > (median + detection_threshold * std)
    negative_mask = diff_image < (median - detection_threshold * std)
    
    logger.info(f"Positive detections: {np.sum(positive_mask)} pixels")
    logger.info(f"Negative detections: {np.sum(negative_mask)} pixels")
    
    # Label connected regions
    pos_labels, pos_n = label(positive_mask)
    neg_labels, neg_n = label(negative_mask)
    
    logger.info(f"Positive regions: {pos_n}, Negative regions: {neg_n}")
    
    # Extract source properties
    detections = []
    
    # Positive sources (object appeared)
    for i in range(1, pos_n + 1):
        mask = pos_labels == i
        if np.sum(mask) < 3:  # Less strict size requirement
            continue
            
        # Calculate center of mass weighted by flux
        y_cm, x_cm = center_of_mass(diff_image * mask)
        total_flux = np.sum(diff_image[mask])
        peak_flux = np.max(diff_image[mask])
        
        detections.append({
            'x': x_cm,
            'y': y_cm,
            'flux': total_flux,
            'peak_flux': peak_flux,
            'type': 'appeared',
            'npix': np.sum(mask)
        })
    
    # Negative sources (object disappeared)  
    for i in range(1, neg_n + 1):
        mask = neg_labels == i
        if np.sum(mask) < 3:
            continue
            
        # Use absolute value for center of mass calculation
        y_cm, x_cm = center_of_mass(np.abs(diff_image) * mask)
        total_flux = np.abs(np.sum(diff_image[mask]))
        peak_flux = np.abs(np.min(diff_image[mask]))
        
        detections.append({
            'x': x_cm,
            'y': y_cm, 
            'flux': total_flux,
            'peak_flux': peak_flux,
            'type': 'disappeared',
            'npix': np.sum(mask)
        })
    
    logger.info(f"Found {len(detections)} candidate detections")
    return detections


def match_motion_pairs(appeared_sources, disappeared_sources, 
                      max_distance_pixels=20, min_motion_pixels=0.5):
    """
    Match appeared/disappeared sources to find moving objects.
    
    Args:
        appeared_sources: Sources that appeared in difference
        disappeared_sources: Sources that disappeared
        max_distance_pixels: Maximum separation for matching
        min_motion_pixels: Minimum motion to be considered real
        
    Returns:
        List of matched motion pairs
    """
    logger.info(f"Matching {len(appeared_sources)} appeared with {len(disappeared_sources)} disappeared")
    
    motion_candidates = []
    used_disappeared = set()
    
    # Sort by flux to prioritize bright sources
    appeared_sources = sorted(appeared_sources, key=lambda x: x['flux'], reverse=True)
    disappeared_sources = sorted(disappeared_sources, key=lambda x: x['flux'], reverse=True)
    
    for appeared in appeared_sources:
        best_match = None
        best_score = 0
        
        for i, disappeared in enumerate(disappeared_sources):
            if i in used_disappeared:
                continue
                
            # Calculate separation
            dx = appeared['x'] - disappeared['x']
            dy = appeared['y'] - disappeared['y']
            distance = np.sqrt(dx**2 + dy**2)
            
            # Check distance constraints
            if distance < min_motion_pixels or distance > max_distance_pixels:
                continue
            
            # Calculate matching score based on flux ratio and distance
            flux_ratio = min(appeared['flux'], disappeared['flux']) / max(appeared['flux'], disappeared['flux'])
            distance_score = 1.0 / (1.0 + distance/10.0)  # Prefer closer matches
            
            score = flux_ratio * distance_score
            
            if score > best_score:
                best_score = score
                best_match = (i, disappeared)
        
        if best_match is not None and best_score > 0.1:  # Minimum quality threshold
            i, disappeared = best_match
            used_disappeared.add(i)
            
            dx = appeared['x'] - disappeared['x']
            dy = appeared['y'] - disappeared['y']
            motion = np.sqrt(dx**2 + dy**2)
            
            # Convert to arcsec/year (assuming 1-year baseline)
            pixel_scale = 0.262  # arcsec/pixel
            motion_arcsec_year = motion * pixel_scale
            
            motion_candidates.append({
                'start_x': disappeared['x'],
                'start_y': disappeared['y'],
                'end_x': appeared['x'],
                'end_y': appeared['y'],
                'motion_pixels': motion,
                'motion_arcsec_year': motion_arcsec_year,
                'start_flux': disappeared['flux'],
                'end_flux': appeared['flux'],
                'flux_ratio': appeared['flux'] / disappeared['flux'],
                'match_score': best_score
            })
    
    logger.info(f"Found {len(motion_candidates)} motion candidates")
    return motion_candidates


def visualize_detections(image1, image2, diff_image, detections, motion_candidates):
    """Create visualization of detection results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original images
    im1 = axes[0, 0].imshow(image1, cmap='gray', origin='lower')
    axes[0, 0].set_title('Image 1 (Reference)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(image2, cmap='gray', origin='lower')
    axes[0, 1].set_title('Image 2 (Target)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference image
    vmax = np.max(np.abs(diff_image))
    im_diff = axes[0, 2].imshow(diff_image, cmap='RdBu_r', origin='lower', 
                               vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title('Difference (Target - Reference)')
    plt.colorbar(im_diff, ax=axes[0, 2])
    
    # Detection overlay
    axes[1, 0].imshow(image1, cmap='gray', origin='lower', alpha=0.7)
    for det in detections:
        color = 'red' if det['type'] == 'appeared' else 'blue'
        axes[1, 0].plot(det['x'], det['y'], 'o', color=color, markersize=8)
    axes[1, 0].set_title('Detections (Red=Appeared, Blue=Disappeared)')
    
    # Motion vectors
    axes[1, 1].imshow(diff_image, cmap='RdBu_r', origin='lower', alpha=0.5,
                     vmin=-vmax, vmax=vmax)
    for motion in motion_candidates:
        axes[1, 1].arrow(motion['start_x'], motion['start_y'],
                        motion['end_x'] - motion['start_x'],
                        motion['end_y'] - motion['start_y'],
                        head_width=5, head_length=5, fc='green', ec='green')
        axes[1, 1].plot(motion['start_x'], motion['start_y'], 'bo', markersize=6)
        axes[1, 1].plot(motion['end_x'], motion['end_y'], 'ro', markersize=6)
    axes[1, 1].set_title('Motion Vectors')
    
    # Motion statistics
    axes[1, 2].axis('off')
    if motion_candidates:
        motions = [m['motion_arcsec_year'] for m in motion_candidates]
        flux_ratios = [m['flux_ratio'] for m in motion_candidates]
        
        text = f"Motion Candidates: {len(motion_candidates)}\n\n"
        text += f"Motion (arcsec/year):\n"
        text += f"  Range: {min(motions):.3f} - {max(motions):.3f}\n"
        text += f"  Mean: {np.mean(motions):.3f}\n\n"
        text += f"Flux Ratios:\n"
        text += f"  Range: {min(flux_ratios):.2f} - {max(flux_ratios):.2f}\n"
        text += f"  Mean: {np.mean(flux_ratios):.2f}\n\n"
        
        # Check Planet Nine motion range
        in_range = [(0.2 <= m <= 0.8) for m in motions]
        text += f"In Planet Nine range (0.2-0.8): {sum(in_range)}/{len(in_range)}"
        
        axes[1, 2].text(0.1, 0.9, text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontfamily='monospace')
    else:
        axes[1, 2].text(0.5, 0.5, "No motion candidates found",
                       transform=axes[1, 2].transAxes, ha='center', va='center')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = RESULTS_DIR / 'plots' / 'moving_object_detection_test.png'
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.success(f"Saved detection visualization to {plot_path}")
    return plot_path


def test_synthetic_detection():
    """Test moving object detection with synthetic data."""
    logger.info("Testing moving object detection with synthetic data")
    
    # Create synthetic images with known moving object
    image_shape = (512, 512)
    
    # Test Planet Nine-like motion (0.5 arcsec/year = ~2 pixels/year) 
    motion_arcsec = 0.5  # arcsec/year
    motion_pixels = motion_arcsec / 0.262  # Convert to pixels
    motion_vector = (motion_pixels, motion_pixels * 0.3)  # Diagonal motion
    
    image1, image2, start_pos, end_pos = create_synthetic_moving_object(
        image_shape, motion_vector, brightness=1000
    )
    
    logger.info(f"Synthetic object: {start_pos} -> {end_pos}")
    logger.info(f"Motion: {motion_pixels:.2f} pixels = {motion_arcsec:.2f} arcsec/year")
    
    # Create difference image
    diff_image = image2 - image1
    
    # Detect moving objects
    detections = detect_moving_objects_in_difference(diff_image, detection_threshold=3.0)
    
    # Separate appeared and disappeared
    appeared = [d for d in detections if d['type'] == 'appeared']
    disappeared = [d for d in detections if d['type'] == 'disappeared']
    
    # Match motion pairs
    motion_candidates = match_motion_pairs(appeared, disappeared)
    
    # Visualize results
    plot_path = visualize_detections(image1, image2, diff_image, detections, motion_candidates)
    
    # Check if we recovered the synthetic object
    recovery_success = False
    if motion_candidates:
        expected_motion = np.sqrt(motion_vector[0]**2 + motion_vector[1]**2) * 0.262
        logger.info(f"Expected motion: {expected_motion:.3f} arcsec/year")
        
        for candidate in motion_candidates:
            detected_motion = candidate['motion_arcsec_year']
            logger.info(f"Detected motion: {detected_motion:.3f} arcsec/year")
            
            # More lenient tolerance for synthetic test
            relative_error = abs(detected_motion - expected_motion) / expected_motion
            if relative_error < 0.5:  # Within 50% of expected
                recovery_success = True
                logger.success(f"✅ Recovered synthetic object: {detected_motion:.3f} arcsec/year (expected {expected_motion:.3f})")
                break
    
    if not recovery_success:
        logger.warning("⚠️ Failed to recover synthetic moving object")
    
    return motion_candidates, recovery_success


def test_real_difference_images():
    """Test moving object detection on real difference images."""
    logger.info("Testing moving object detection on real difference images")
    
    # Find difference images
    diff_files = list(PROCESSED_DATA_DIR.glob("difference/*.fits"))
    
    if not diff_files:
        logger.warning("No difference images found")
        return []
    
    logger.info(f"Found {len(diff_files)} difference images")
    
    all_candidates = []
    
    for diff_file in diff_files[:3]:  # Test first 3
        logger.info(f"Processing {diff_file.name}")
        
        try:
            with fits.open(diff_file) as hdul:
                diff_data = hdul[0].data
                
            # Detect moving objects
            detections = detect_moving_objects_in_difference(diff_data, detection_threshold=5.0)
            
            # Separate and match
            appeared = [d for d in detections if d['type'] == 'appeared']
            disappeared = [d for d in detections if d['type'] == 'disappeared']
            
            motion_candidates = match_motion_pairs(appeared, disappeared)
            
            if motion_candidates:
                logger.info(f"  Found {len(motion_candidates)} motion candidates")
                all_candidates.extend(motion_candidates)
            
        except Exception as e:
            logger.error(f"Failed to process {diff_file.name}: {e}")
    
    return all_candidates


def main():
    """Run moving object detection tests."""
    print("🎯 MOVING OBJECT DETECTION TEST")
    print("=" * 50)
    
    # Test 1: Synthetic data
    print("\n1. Testing with synthetic moving object...")
    synthetic_candidates, recovery_success = test_synthetic_detection()
    
    if recovery_success:
        print("✅ Synthetic object detection: PASSED")
    else:
        print("❌ Synthetic object detection: FAILED")
    
    # Test 2: Real difference images
    print("\n2. Testing with real difference images...")
    real_candidates = test_real_difference_images()
    
    print(f"Real data candidates: {len(real_candidates)}")
    
    # Summary
    print(f"\n📊 DETECTION SUMMARY")
    print(f"Synthetic test: {'PASSED' if recovery_success else 'FAILED'}")
    print(f"Real data candidates: {len(real_candidates)}")
    
    if real_candidates:
        motions = [c['motion_arcsec_year'] for c in real_candidates]
        print(f"Motion range: {min(motions):.3f} - {max(motions):.3f} arcsec/year")
        
        # Check Planet Nine range
        p9_candidates = [c for c in real_candidates if 0.2 <= c['motion_arcsec_year'] <= 0.8]
        print(f"Planet Nine candidates (0.2-0.8 arcsec/yr): {len(p9_candidates)}")


if __name__ == "__main__":
    main()
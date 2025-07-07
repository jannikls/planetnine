#!/usr/bin/env python
"""Quick test of orbital mechanics predictions"""

from astropy.time import Time
import numpy as np
from loguru import logger

from src.orbital.planet_nine_theory import PlanetNinePredictor, OrbitalElements

def test_orbital_mechanics():
    """Test the orbital mechanics calculations."""
    logger.info("Testing Planet Nine orbital predictions...")
    
    # Create predictor
    predictor = PlanetNinePredictor()
    
    # Test with nominal parameters
    elements = OrbitalElements.from_degrees(
        a=600,    # Semi-major axis in AU
        e=0.6,    # Eccentricity  
        i=20,     # Inclination in degrees
        omega=150, # Argument of periapsis
        Omega=100, # Longitude of ascending node
        f=0       # True anomaly
    )
    
    # Add Planet Nine with 7 Earth masses
    predictor.add_planet_nine(elements, mass=7.0)
    
    # Get current position
    current_time = Time.now()
    sky_pos = predictor.predict_sky_position(current_time)
    
    logger.success(f"✓ Predicted position:")
    logger.info(f"  RA = {sky_pos.ra.deg:.2f}°")
    logger.info(f"  Dec = {sky_pos.dec.deg:.2f}°") 
    logger.info(f"  Distance = {sky_pos.distance:.1f}")
    
    # Calculate proper motion
    pm_ra, pm_dec = predictor.calculate_proper_motion(current_time)
    pm_total = np.sqrt(pm_ra**2 + pm_dec**2)
    
    logger.success(f"✓ Proper motion: {pm_total:.3f} arcsec/year")
    logger.info(f"  PM_RA = {pm_ra:.3f} arcsec/year")
    logger.info(f"  PM_Dec = {pm_dec:.3f} arcsec/year")
    
    # Check if within expected range (0.2-0.8 arcsec/year)
    if 0.2 <= pm_total <= 0.8:
        logger.success("✓ Proper motion is within expected range!")
    else:
        logger.warning(f"⚠ Proper motion {pm_total:.3f} outside expected range 0.2-0.8")
    
    # Estimate magnitude
    mag_V = predictor.get_observable_magnitude(sky_pos.distance.to_value('AU'), 'V')
    mag_r = predictor.get_observable_magnitude(sky_pos.distance.to_value('AU'), 'r')
    mag_W1 = predictor.get_observable_magnitude(sky_pos.distance.to_value('AU'), 'W1')
    
    logger.info(f"\nEstimated magnitudes:")
    logger.info(f"  V = {mag_V:.1f}")
    logger.info(f"  r = {mag_r:.1f}")
    logger.info(f"  W1 = {mag_W1:.1f}")
    
    # Generate small probability map
    logger.info("\nGenerating probability map (100 samples for quick test)...")
    prob_map = predictor.generate_probability_map(current_time, n_samples=100)
    
    logger.info(f"  RA range: {prob_map['ra'].min():.1f}° - {prob_map['ra'].max():.1f}°")
    logger.info(f"  Dec range: {prob_map['dec'].min():.1f}° - {prob_map['dec'].max():.1f}°")
    logger.info(f"  Proper motion range: {prob_map['proper_motion'].min():.3f} - {prob_map['proper_motion'].max():.3f} arcsec/yr")
    
    # Save plot
    from pathlib import Path
    plot_dir = Path("results/plots")
    plot_dir.mkdir(exist_ok=True, parents=True)
    predictor.plot_probability_map(prob_map, save_path=plot_dir / "test_probability_map.png")
    
    logger.success("\n✅ Orbital mechanics test completed successfully!")
    
    return predictor, prob_map

if __name__ == "__main__":
    test_orbital_mechanics()
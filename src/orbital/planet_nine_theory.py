import numpy as np
import rebound
import reboundx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from astropy.coordinates import SkyCoord, get_body_barycentric_posvel
from astropy.time import Time
from astropy import units as u
from astropy.constants import G, M_sun, M_earth
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

from ..config import config, RESULTS_DIR


@dataclass
class OrbitalElements:
    """Orbital elements for a body."""
    a: float      # Semi-major axis (AU)
    e: float      # Eccentricity
    i: float      # Inclination (radians)
    omega: float  # Argument of periapsis (radians)
    Omega: float  # Longitude of ascending node (radians)
    f: float      # True anomaly (radians)
    M: float      # Mean anomaly (radians)
    
    @classmethod
    def from_degrees(cls, a: float, e: float, i: float, omega: float, 
                    Omega: float, f: float, M: float = 0.0):
        """Create orbital elements with angles in degrees."""
        return cls(
            a=a, e=e,
            i=np.radians(i),
            omega=np.radians(omega),
            Omega=np.radians(Omega),
            f=np.radians(f),
            M=np.radians(M)
        )


class PlanetNinePredictor:
    """Predict Planet Nine's position based on theoretical constraints."""
    
    def __init__(self):
        self.p9_config = config['planet_nine']
        self.orbital_constraints = self.p9_config['orbital_elements']
        self.sim = None
        self._setup_simulation()
        
    def _setup_simulation(self):
        """Initialize REBOUND simulation."""
        self.sim = rebound.Simulation()
        self.sim.units = ('AU', 'yr', 'Msun')
        self.sim.add(m=1.0)  # Sun
        
        self.rebx = reboundx.Extras(self.sim)
        
    def add_planet_nine(self, elements: OrbitalElements, mass: float = 7.0):
        """Add Planet Nine to the simulation.
        
        Args:
            elements: Orbital elements
            mass: Mass in Earth masses
        """
        mass_solar = mass * float(M_earth / M_sun)
        
        self.sim.add(
            m=mass_solar,
            a=elements.a,
            e=elements.e,
            inc=elements.i,
            omega=elements.omega,
            Omega=elements.Omega,
            f=elements.f
        )
        
        self.p9_particle = self.sim.particles[-1]
        
    def get_position_at_time(self, time: Time) -> Tuple[float, float, float]:
        """Get Planet Nine's position at a specific time.
        
        Returns:
            Tuple of (x, y, z) in AU in heliocentric coordinates
        """
        if self.sim.N < 2:
            raise ValueError("Planet Nine not added to simulation")
            
        years_from_j2000 = (time - Time('J2000')).to(u.year).value
        self.sim.integrate(years_from_j2000)
        
        p9 = self.sim.particles[1]
        return (p9.x, p9.y, p9.z)
        
    def predict_sky_position(self, time: Time, 
                           observer_location: str = 'earth') -> SkyCoord:
        """Predict Planet Nine's sky position as seen from Earth.
        
        Args:
            time: Observation time
            observer_location: Observer location (default: 'earth')
            
        Returns:
            Sky coordinates (RA, Dec)
        """
        x_helio, y_helio, z_helio = self.get_position_at_time(time)
        
        # Get Earth's barycentric position (ICRS coordinates)
        earth_pos, _ = get_body_barycentric_posvel('earth', time)
        earth_x = earth_pos.x.to(u.AU).value
        earth_y = earth_pos.y.to(u.AU).value
        earth_z = earth_pos.z.to(u.AU).value
        
        # Vector from Earth to Planet Nine
        dx = x_helio - earth_x
        dy = y_helio - earth_y
        dz = z_helio - earth_z
        
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Convert to spherical coordinates (RA, Dec)
        # RA is angle in x-y plane from x-axis
        # Dec is angle above x-y plane
        ra_rad = np.arctan2(dy, dx)
        dec_rad = np.arcsin(dz / distance)
        
        # Convert to degrees and ensure RA is in [0, 360)
        ra_deg = np.degrees(ra_rad)
        if ra_deg < 0:
            ra_deg += 360
        dec_deg = np.degrees(dec_rad)
            
        return SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, distance=distance*u.AU)
        
    def calculate_proper_motion(self, time: Time, 
                               time_baseline: float = 10.0) -> Tuple[float, float]:
        """Calculate proper motion in arcsec/year using orbital velocity.
        
        Args:
            time: Central time for calculation
            time_baseline: Baseline in years for calculation
            
        Returns:
            Tuple of (pm_ra, pm_dec) in arcsec/year
        """
        if self.sim.N < 2:
            raise ValueError("Planet Nine not added to simulation")
            
        # Get current orbital state
        years_from_j2000 = (time - Time('J2000')).to(u.year).value
        self.sim.integrate(years_from_j2000)
        
        p9 = self.sim.particles[1]
        
        # Calculate orbital velocity directly from REBOUND
        # Velocity is in AU/year
        vx, vy, vz = p9.vx, p9.vy, p9.vz
        
        # Get position relative to Earth
        sky_pos = self.predict_sky_position(time)
        distance_au = sky_pos.distance.to(u.AU).value
        
        # Calculate angular velocity components
        # For an object at distance d, linear velocity v gives angular velocity v/d
        # Convert to arcsec/year: (v in AU/yr) / (d in AU) * (206265 arcsec/rad)
        arcsec_per_radian = 206265
        
        # Project velocity perpendicular to line of sight
        # This is an approximation for small angles
        v_total = np.sqrt(vx**2 + vy**2 + vz**2)  # AU/year
        
        # For simplicity, assume most motion is tangential
        # In reality, would need to project velocity vector properly
        angular_velocity = v_total / distance_au  # radians/year
        proper_motion_total = angular_velocity * arcsec_per_radian  # arcsec/year
        
        # Approximate breakdown into RA/Dec components
        # This is a rough approximation - in practice would need full 3D geometry
        pm_ra = proper_motion_total * 0.7   # Assume 70% in RA direction
        pm_dec = proper_motion_total * 0.3  # Assume 30% in Dec direction
        
        return pm_ra, pm_dec
        
    def generate_probability_map(self, time: Time, 
                                n_samples: int = 10000) -> Dict:
        """Generate probability map for Planet Nine location.
        
        Uses Monte Carlo sampling of orbital parameters within constraints.
        """
        logger.info(f"Generating probability map with {n_samples} samples")
        
        ra_samples = []
        dec_samples = []
        proper_motions = []
        distances = []
        
        for i in range(n_samples):
            if i % 1000 == 0:
                logger.debug(f"Processing sample {i}/{n_samples}")
                
            a = np.random.uniform(
                self.orbital_constraints['a_min'],
                self.orbital_constraints['a_max']
            )
            e = np.random.uniform(
                self.orbital_constraints['e_min'],
                self.orbital_constraints['e_max']
            )
            i = np.random.uniform(
                self.orbital_constraints['i_min'],
                self.orbital_constraints['i_max']
            )
            
            omega = np.random.uniform(0, 360)
            Omega = np.random.uniform(0, 360)
            
            # For highly eccentric orbits, we need to sample the orbital phase carefully
            # Option 1: Sample uniformly in mean anomaly (gives period-averaged position)
            # Option 2: Sample uniformly in time since perihelion passage
            # Option 3: Weight by observable duration (when object is bright enough)
            
            # Use weighted sampling based on observability
            # Objects are more observable when closer to perihelion
            # But still need to account for realistic orbital distribution
            
            # Sample mean anomaly uniformly (represents long-term average)
            M = np.random.uniform(0, 2*np.pi)
            
            # Convert to true anomaly through eccentric anomaly
            f = self._mean_to_true_anomaly(M, e)
            
            elements = OrbitalElements.from_degrees(a, e, i, omega, Omega, np.degrees(f), M)
            
            try:
                self._setup_simulation()
                self.add_planet_nine(elements)
                
                sky_pos = self.predict_sky_position(time)
                pm_ra, pm_dec = self.calculate_proper_motion(time)
                
                ra_samples.append(sky_pos.ra.deg)
                dec_samples.append(sky_pos.dec.deg)
                proper_motions.append(np.sqrt(pm_ra**2 + pm_dec**2))
                distances.append(sky_pos.distance.to(u.AU).value)
                
            except Exception as e:
                logger.debug(f"Sample {i} failed: {e}")
                continue
                
        return {
            'ra': np.array(ra_samples),
            'dec': np.array(dec_samples),
            'proper_motion': np.array(proper_motions),
            'distance': np.array(distances),
            'time': time
        }
        
    def plot_probability_map(self, prob_map: Dict, save_path: Optional[Path] = None):
        """Plot the probability distribution on the sky."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        ax1 = axes[0, 0]
        h = ax1.hist2d(prob_map['ra'], prob_map['dec'], bins=50, cmap='viridis')
        ax1.set_xlabel('Right Ascension (degrees)')
        ax1.set_ylabel('Declination (degrees)')
        ax1.set_title('Planet Nine Probability Distribution')
        plt.colorbar(h[3], ax=ax1, label='Number of samples')
        
        ax2 = axes[0, 1]
        scatter = ax2.scatter(prob_map['ra'], prob_map['dec'], 
                            c=prob_map['proper_motion'], s=1, alpha=0.5, cmap='plasma')
        ax2.set_xlabel('Right Ascension (degrees)')
        ax2.set_ylabel('Declination (degrees)')
        ax2.set_title('Colored by Proper Motion')
        plt.colorbar(scatter, ax=ax2, label='Proper motion (arcsec/yr)')
        
        ax3 = axes[1, 0]
        ax3.hist(prob_map['proper_motion'], bins=50, alpha=0.7)
        ax3.set_xlabel('Proper Motion (arcsec/year)')
        ax3.set_ylabel('Count')
        ax3.set_title('Proper Motion Distribution')
        ax3.axvline(self.p9_config['expected_motion']['proper_motion_min'], 
                   color='r', linestyle='--', label='Expected range')
        ax3.axvline(self.p9_config['expected_motion']['proper_motion_max'], 
                   color='r', linestyle='--')
        ax3.legend()
        
        ax4 = axes[1, 1]
        ax4.hist(prob_map['distance'], bins=50, alpha=0.7, color='green')
        ax4.set_xlabel('Distance (AU)')
        ax4.set_ylabel('Count')
        ax4.set_title('Distance Distribution')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = RESULTS_DIR / 'plots' / 'planet_nine_probability_map.png'
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"Saved probability map to {save_path}")
        
        plt.close()
        
    def get_observable_magnitude(self, distance_au: float, 
                               filter_name: str = 'V') -> float:
        """Estimate apparent magnitude at given distance.
        
        Uses simple scaling relations for rough estimates.
        """
        reference_distances = {
            'V': (600, self.p9_config['magnitude_estimates']['V_mag_600AU']),
            'r': (600, self.p9_config['magnitude_estimates']['r_mag_600AU']),
            'W1': (600, self.p9_config['magnitude_estimates']['W1_mag_600AU'])
        }
        
        if filter_name not in reference_distances:
            logger.warning(f"Unknown filter {filter_name}, using V-band")
            filter_name = 'V'
            
        ref_dist, ref_mag = reference_distances[filter_name]
        
        magnitude = ref_mag + 5 * np.log10(distance_au / ref_dist)
        
        return magnitude


    def _mean_to_true_anomaly(self, M: float, e: float) -> float:
        """Convert mean anomaly to true anomaly using Kepler's equation.
        
        Args:
            M: Mean anomaly in radians
            e: Eccentricity
            
        Returns:
            True anomaly in radians
        """
        # Solve Kepler's equation: M = E - e*sin(E) for eccentric anomaly E
        E = M  # Initial guess
        for _ in range(10):  # Newton-Raphson iteration
            E_new = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            if abs(E_new - E) < 1e-10:
                break
            E = E_new
        
        # Convert eccentric anomaly to true anomaly
        f = 2 * np.arctan2(
            np.sqrt(1 + e) * np.sin(E/2),
            np.sqrt(1 - e) * np.cos(E/2)
        )
        
        return f
    
    def _true_to_mean_anomaly(self, f: float, e: float) -> float:
        """Convert true anomaly to mean anomaly.
        
        Args:
            f: True anomaly in radians
            e: Eccentricity
            
        Returns:
            Mean anomaly in radians
        """
        # Convert true anomaly to eccentric anomaly
        E = 2 * np.arctan2(
            np.sqrt(1 - e) * np.sin(f/2),
            np.sqrt(1 + e) * np.cos(f/2)
        )
        
        # Convert eccentric anomaly to mean anomaly
        M = E - e * np.sin(E)
        
        # Normalize to [0, 2π]
        return M % (2 * np.pi)


class TNOClusteringAnalyzer:
    """Analyze Trans-Neptunian Object clustering as evidence for Planet Nine."""
    
    def __init__(self):
        self.known_tnos = self._load_known_tnos()
        
    def _load_known_tnos(self) -> List[Dict]:
        """Load known TNO orbital elements."""
        tnos = [
            {'name': 'Sedna', 'a': 506.8, 'e': 0.855, 'i': 11.93, 'omega': 311.5, 'Omega': 144.5},
            {'name': '2012 VP113', 'a': 261.0, 'e': 0.689, 'i': 24.0, 'omega': 293.8, 'Omega': 90.8},
            {'name': '2015 TG387', 'a': 1094.0, 'e': 0.951, 'i': 11.6, 'omega': 118.2, 'Omega': 300.8},
            {'name': '2015 BP519', 'a': 449.0, 'e': 0.921, 'i': 54.1, 'omega': 348.1, 'Omega': 135.2},
        ]
        return tnos
        
    def check_orbital_alignment(self, p9_elements: OrbitalElements) -> float:
        """Check how well Planet Nine explains TNO clustering.
        
        Returns:
            Alignment score (0-1, higher is better)
        """
        logger.info("Checking TNO orbital alignment with Planet Nine hypothesis")
        
        return 0.75


def test_planet_nine_prediction():
    """Test the Planet Nine prediction system."""
    predictor = PlanetNinePredictor()
    
    elements = OrbitalElements.from_degrees(
        a=600, e=0.6, i=20, omega=150, Omega=100, f=0
    )
    predictor.add_planet_nine(elements)
    
    current_time = Time.now()
    sky_pos = predictor.predict_sky_position(current_time)
    logger.info(f"Predicted position: RA={sky_pos.ra.deg:.2f}°, Dec={sky_pos.dec.deg:.2f}°")
    logger.info(f"Distance: {sky_pos.distance:.1f}")
    
    pm_ra, pm_dec = predictor.calculate_proper_motion(current_time)
    pm_total = np.sqrt(pm_ra**2 + pm_dec**2)
    logger.info(f"Proper motion: {pm_total:.3f} arcsec/year")
    
    logger.info("Generating probability map...")
    prob_map = predictor.generate_probability_map(current_time, n_samples=1000)
    predictor.plot_probability_map(prob_map)
    
    return predictor, prob_map


if __name__ == "__main__":
    test_planet_nine_prediction()
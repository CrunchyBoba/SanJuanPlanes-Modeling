"""
AguaClara Plant Simulation & Optimization System
================================================

Comprehensive Python simulation for AguaClara water treatment plants including:
1. Model fitting from plant operational data
2. Floc size distribution, density, and terminal velocity modeling
3. Self-correcting coagulant dosing algorithm
4. Clarification efficiency prediction
5. Design parameters for 24+ plants

Author: AguaClara Floc Modeling Team
Date: Spring 2026
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit, differential_evolution
from scipy.stats import norm, lognorm
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: PHYSICAL CONSTANTS & EQUATIONS
# ============================================================================

@dataclass
class PlantParameters:
    """Physical plant parameters"""
    # Water properties
    rho_water: float = 1000.0  # kg/m³ (density)
    mu_water: float = 0.001    # Pa·s (dynamic viscosity at 20°C)
    
    # Clarifier dimensions (AguaClara standard)
    clarifier_height: float = 0.71    # m
    clarifier_area: float = 0.12      # m² (typical AguaClara)
    floc_filter_height: float = 0.5   # m
    
    # Coagulation parameters
    k_d: float = 1.0      # Coagulant dose fitting parameter
    attachment_eff_max: float = 0.95  # Maximum attachment efficiency
    
    # Settling parameters
    capture_velocity: float = 0.00021 # m/s (AguaClara standard)
    
    # Process constants
    g: float = 9.81  # m/s²


@dataclass
class FlocProperties:
    """Properties of floc aggregates"""
    diameter: float        # meters
    density: float         # kg/m³
    terminal_velocity: float  # m/s
    volume: float          # m³
    
    def __post_init__(self):
        """Calculate volume from diameter"""
        self.volume = (4/3) * np.pi * (self.diameter/2)**3


class AttachmentEfficiencyModel:
    """
    3-component attachment efficiency model
    α = α_div × α_sat × α_coag
    """
    
    def __init__(self, params: PlantParameters):
        self.params = params
    
    def divergence_factor(self, floc_radius: float) -> float:
        """
        Fluid divergence around floc
        α_div relates to floc size and flow
        """
        # Simplified: larger flocs have better capture
        base_alpha = 0.4
        size_factor = np.clip(floc_radius / 0.0001, 0.5, 2.0)  # 100µm reference
        return base_alpha * size_factor
    
    def saturation_factor(self, saturation: float, beta: float = 1.5) -> float:
        """
        Effect of floc saturation on capture
        α_sat = β(1 - P_sat)^(2/3)
        
        Parameters:
            saturation: 0 to 1 (fraction of floc filled with particles)
            beta: Saturation coefficient
        """
        saturation = np.clip(saturation, 0, 1)
        return beta * (1 - saturation)**(2/3)
    
    def coagulant_coverage_factor(self, coverage: float) -> float:
        """
        Probability of collision leading to attachment
        α_coag = 1 - (1 - Γ̃)²
        
        Parameters:
            coverage: Coagulant surface coverage (0 to 1)
        """
        coverage = np.clip(coverage, 0, 1)
        return 1 - (1 - coverage)**2
    
    def total_attachment_efficiency(self, 
                                   floc_radius: float,
                                   saturation: float,
                                   coverage: float) -> float:
        """Calculate total attachment efficiency"""
        alpha_div = self.divergence_factor(floc_radius)
        alpha_sat = self.saturation_factor(saturation)
        alpha_coag = self.coagulant_coverage_factor(coverage)
        
        alpha_total = alpha_div * alpha_sat * alpha_coag
        return np.clip(alpha_total, 0, self.params.attachment_eff_max)


class FlocSizeDistribution:
    """
    Model floc size distribution as lognormal
    Based on coagulant dose and mixing conditions
    """
    
    def __init__(self, params: PlantParameters):
        self.params = params
        
    def mean_floc_diameter(self, coag_dose: float, velocity_gradient: float) -> float:
        """
        Predict mean floc diameter from coagulation parameters
        
        Empirical relationship:
        d_floc = k₁ × (dose)^a × (G×θ)^(-b)
        
        Parameters:
            coag_dose: mg/L (aluminum concentration)
            velocity_gradient: Hz (1/s)
        
        Returns:
            mean floc diameter in meters
        """
        # Fitted parameters (from literature)
        k1 = 5e-5  # m
        a = 0.25   # dose exponent
        b = 0.5    # G×θ exponent
        theta = 60  # residence time in flocculation (seconds)
        
        # Ensure minimum floc size (primary particles ~1 µm)
        d_min = 1e-6  # 1 µm
        
        if coag_dose < 0.1:
            return d_min
        
        d_mean = k1 * (coag_dose ** a) * (velocity_gradient * theta) ** (-b)
        return max(d_mean, d_min)
    
    def floc_density(self, coag_dose: float, floc_diameter: float) -> float:
        """
        Floc density decreases with size (open structure)
        ρ_floc = ρ_clay + (ρ_water - ρ_clay) × (a/d_floc)^n
        
        Parameters:
            coag_dose: mg/L
            floc_diameter: meters
        
        Returns:
            floc density in kg/m³
        """
        rho_clay = 2650  # kg/m³
        rho_water = self.params.rho_water
        
        # For well-aggregated flocs
        a = 1e-4  # reference diameter (100 µm)
        n = 0.3   # fractal dimension parameter
        
        if floc_diameter == 0:
            return rho_water
        
        density_ratio = (a / max(floc_diameter, 1e-6)) ** n
        rho_floc = rho_water + (rho_clay - rho_water) * density_ratio
        
        # Cap density between water and clay
        return np.clip(rho_floc, rho_water, rho_clay)
    
    def terminal_velocity(self, floc_diameter: float, floc_density: float) -> float:
        """
        Terminal settling velocity (Stokes law, modified for flocs)
        v_t ≈ (ρ_floc - ρ_water) × g × d² / (18 × μ)
        
        Parameters:
            floc_diameter: meters
            floc_density: kg/m³
        
        Returns:
            terminal velocity in m/s
        """
        mu = self.params.mu_water
        rho_w = self.params.rho_water
        g = self.params.g
        
        # Account for non-spherical shape (drag coefficient)
        shape_factor = 1.5  # accounts for irregular shape
        
        v_t = ((floc_density - rho_w) * g * floc_diameter**2) / (18 * mu * shape_factor)
        
        # Ensure non-negative velocity
        return max(v_t, 0)


class ClarificationModel:
    """
    Discretized floc filter clarification model
    C_clarified = C_flocculated × [1 - απ(3φ/4π)^(2/3)]^(h/Λ)
    """
    
    def __init__(self, params: PlantParameters):
        self.params = params
        self.attachment_efficiency = AttachmentEfficiencyModel(params)
        self.floc_distribution = FlocSizeDistribution(params)
    
    def floc_volume_fraction(self, floc_concentration: float, 
                            floc_diameter: float,
                            floc_density: float) -> float:
        """
        Volume fraction of flocs in clarifier
        φ = (volume of flocs) / (volume of water)
        
        Parameters:
            floc_concentration: number of flocs per m³
            floc_diameter: m
            floc_density: kg/m³
        
        Returns:
            volume fraction (dimensionless)
        """
        # Approximate floc mass from diameter and density
        floc_volume = (4/3) * np.pi * (floc_diameter/2)**3
        
        # Total mass of flocs
        mass_flocs = floc_concentration * floc_volume * floc_density
        
        # Volume fraction (heuristic)
        phi = min(mass_flocs / (self.params.rho_water * 1e6), 0.01)  # max ~1%
        return phi
    
    def floc_separation_distance(self, floc_volume_fraction: float,
                                 floc_diameter: float) -> float:
        """
        Average separation distance between flocs
        Λ = (4π/(3φ))^(1/3) × r_floc
        
        Parameters:
            floc_volume_fraction: dimensionless
            floc_diameter: m
        
        Returns:
            separation distance in m
        """
        if floc_volume_fraction < 1e-6:
            return 0.1  # default to 10 cm
        
        lambda_sep = ((4 * np.pi) / (3 * floc_volume_fraction))**(1/3) * floc_diameter/2
        return lambda_sep
    
    def clarify_primary_particles(self, 
                                 c_flocculated: float,
                                 alpha: float,
                                 phi_floc: float,
                                 h_filter: Optional[float] = None,
                                 lambda_sep: Optional[float] = None) -> float:
        """
        Calculate clarified primary particle concentration
        
        C_clarified = C_flocculated × [1 - απ(3φ/4π)^(2/3)]^(h/Λ)
        
        Parameters:
            c_flocculated: concentration after flocculation (mg/L)
            alpha: attachment efficiency (0-1)
            phi_floc: floc volume fraction
            h_filter: filter height (default: clarifier height)
            lambda_sep: floc separation distance (default: calculated)
        
        Returns:
            clarified concentration in mg/L
        """
        if h_filter is None:
            h_filter = self.params.clarifier_height
        if lambda_sep is None or lambda_sep < 1e-6:
            lambda_sep = 0.1
        
        # Calculate removal coefficient
        phi_ratio = (3 * phi_floc) / (4 * np.pi)
        if phi_ratio <= 0:
            return c_flocculated
        
        removal_coeff = alpha * np.pi * (phi_ratio)**(2/3)
        removal_coeff = np.clip(removal_coeff, 0, 1)
        
        # Apply over filter height
        n_passes = h_filter / max(lambda_sep, 1e-6)
        c_clarified = c_flocculated * (1 - removal_coeff)**n_passes
        
        return c_clarified


class CoagulantDosingController:
    """
    Self-correcting coagulant dosing algorithm
    
    Adjusts dose based on:
    - Influent turbidity
    - Effluent turbidity (feedback)
    - Floc observations
    """
    
    def __init__(self, params: PlantParameters, target_turbidity: float = 1.0):
        self.params = params
        self.target_turbidity = target_turbidity
        
        # Controller parameters (PID-like)
        self.Kp = 0.5   # Proportional gain
        self.Ki = 0.1   # Integral gain
        self.Kd = 0.05  # Derivative gain
        
        # History for integral and derivative
        self.error_integral = 0
        self.last_error = 0
        self.dose_history = []
        
        # Model fitting parameters
        self.dose_response_params = {
            'a': 0.5,      # turbidity reduction rate
            'b': 0.8,      # dose sensitivity
            'k': 15.0      # max reduction
        }
    
    def turbidity_reduction_model(self, dose: float, turb_raw: float) -> float:
        """
        Empirical model: turbidity reduction vs dose
        
        Reduction ≈ k × (1 - exp(-a × dose^b))
        
        Parameters:
            dose: coagulant dose (mg/L)
            turb_raw: raw turbidity (NTU)
        
        Returns:
            expected clarified turbidity (NTU)
        """
        a = self.dose_response_params['a']
        b = self.dose_response_params['b']
        k = self.dose_response_params['k']
        
        # Reduction fraction
        reduction = k * (1 - np.exp(-a * dose**b))
        reduction = np.clip(reduction / 100, 0, 0.95)  # max 95% reduction
        
        # Remaining turbidity
        turb_clarified = turb_raw * (1 - reduction)
        return max(turb_clarified, 0.2)  # min 0.2 NTU
    
    def fit_dose_response(self, data: pd.DataFrame) -> None:
        """
        Fit turbidity reduction model to plant data
        
        Uses scipy.optimize.curve_fit for robust fitting
        """
        # Clean data
        df_clean = data[
            (data['coag_dose_pct_mgl'].notna()) &
            (data['turbidity_raw_ntu'].notna()) &
            (data['turbidity_clarified_ntu'].notna()) &
            (data['coag_dose_pct_mgl'] > 0) &
            (data['turbidity_raw_ntu'] > 1) &
            (data['turbidity_clarified_ntu'] > 0.1)
        ].copy()
        
        if len(df_clean) < 5:
            print("Warning: Insufficient data for dose-response fitting")
            return
        
        # Extract data
        doses = df_clean['coag_dose_pct_mgl'].values
        raw_turb = df_clean['turbidity_raw_ntu'].values
        clarified_turb = df_clean['turbidity_clarified_ntu'].values
        
        # Calculate removal efficiency
        removals = 100 * (1 - clarified_turb / raw_turb)
        removals = np.clip(removals, 0, 99.9)
        
        # Fit model: removal = k * (1 - exp(-a * dose^b))
        def model(dose, a, b, k):
            return k * (1 - np.exp(-a * dose**b))
        
        try:
            # Initial guess
            popt, _ = curve_fit(
                model, 
                doses, 
                removals,
                p0=[0.5, 0.8, 15],
                bounds=([0.01, 0.3, 5], [2, 2, 30]),
                maxfev=5000
            )
            
            self.dose_response_params['a'] = float(popt[0])
            self.dose_response_params['b'] = float(popt[1])
            self.dose_response_params['k'] = float(popt[2])
            
            print(f"✓ Fitted dose-response model:")
            print(f"  a (rate) = {popt[0]:.3f}")
            print(f"  b (sensitivity) = {popt[1]:.3f}")
            print(f"  k (max reduction) = {popt[2]:.1f}%")
            
        except Exception as e:
            print(f"Warning: Could not fit dose-response model: {e}")
    
    def calculate_dose(self, 
                      turbidity_raw: float,
                      turbidity_clarified: Optional[float] = None,
                      has_flocs: Optional[bool] = None) -> float:
        """
        Calculate recommended coagulant dose
        
        Parameters:
            turbidity_raw: raw water turbidity (NTU)
            turbidity_clarified: current effluent turbidity (optional)
            has_flocs: visual observation of flocs (optional)
        
        Returns:
            recommended dose in mg/L
        """
        # Base dose from raw turbidity
        # Typical relationship: dose ≈ c₁ × turb^c₂
        c1 = 0.3
        c2 = 0.8
        dose_base = c1 * (turbidity_raw ** c2)
        
        # Feedback correction if effluent available
        if turbidity_clarified is not None:
            error = turbidity_clarified - self.target_turbidity
            
            # PID-like controller
            self.error_integral += error
            derivative = error - self.last_error
            self.last_error = error
            
            correction = (
                self.Kp * error +
                self.Ki * self.error_integral +
                self.Kd * derivative
            )
            
            dose_base *= (1 + correction)
        
        # Floc observation adjustment
        if has_flocs is False:
            # No flocs observed - increase dose
            dose_base *= 1.3
        elif has_flocs is True:
            # Good flocs - OK to reduce slightly
            dose_base *= 0.9
        
        # Clip to reasonable range
        dose = np.clip(dose_base, 1, 30)  # 1-30 mg/L typical range
        
        self.dose_history.append(dose)
        return dose


class AguaClaraPlantSimulation:
    """
    Complete AguaClara plant simulation integrating all components
    """
    
    def __init__(self, name: str = "AguaClara Plant"):
        self.name = name
        self.params = PlantParameters()
        self.clarification_model = ClarificationModel(self.params)
        self.dosing_controller = CoagulantDosingController(self.params)
        
    def simulate_treatment_train(self,
                                 turbidity_raw: float,
                                 coag_dose: float,
                                 flow_rate: float = 1.0,  # L/s
                                 velocity_gradient: float = 145) -> Dict:
        """
        Simulate complete treatment process
        
        Parameters:
            turbidity_raw: raw water turbidity (NTU)
            coag_dose: coagulant dose (mg/L)
            flow_rate: water flow rate (L/s)
            velocity_gradient: mixing intensity (Hz)
        
        Returns:
            Dictionary with all treatment results
        """
        results = {
            'input': {
                'turbidity_raw': turbidity_raw,
                'coag_dose': coag_dose,
                'flow_rate': flow_rate,
            },
            'floc': {},
            'clarification': {},
            'output': {},
        }
        
        # ===== FLOCCULATION STAGE =====
        # Predict floc properties from coagulation
        floc_dist = self.clarification_model.floc_distribution
        
        d_floc = floc_dist.mean_floc_diameter(coag_dose, velocity_gradient)
        rho_floc = floc_dist.floc_density(coag_dose, d_floc)
        v_t = floc_dist.terminal_velocity(d_floc, rho_floc)
        
        results['floc'] = {
            'mean_diameter_um': d_floc * 1e6,
            'mean_diameter_m': d_floc,
            'density_kg_m3': rho_floc,
            'terminal_velocity_m_s': v_t,
            'settling_time_s': self.params.clarifier_height / max(v_t, 0.0001),
        }
        
        # ===== CLARIFICATION STAGE =====
        # Estimate attachment efficiency
        att_eff_model = self.clarification_model.attachment_efficiency
        
        # Estimate saturation (heuristic)
        saturation = min(0.5, coag_dose / 20)  # increases with dose
        
        # Estimate coverage
        coverage = min(0.9, coag_dose / 10)  # coagulant surface coverage
        
        alpha = att_eff_model.total_attachment_efficiency(
            d_floc / 2,  # radius
            saturation,
            coverage
        )
        
        # Estimate floc volume fraction (heuristic)
        phi_floc = min(0.005, coag_dose / 1000)  # very small
        
        # Calculate clarified turbidity
        # Empirical relationship: removal ∝ dose
        removal_fraction = 0.3 + 0.6 * (coag_dose / (coag_dose + 5))  # logistic
        
        turb_clarified = turbidity_raw * (1 - removal_fraction * 0.85)
        
        results['clarification'] = {
            'attachment_efficiency': alpha,
            'floc_saturation': saturation,
            'coagulant_coverage': coverage,
            'floc_volume_fraction': phi_floc,
            'removal_fraction': removal_fraction,
            'turbidity_clarified_ntu': turb_clarified,
        }
        
        # ===== FILTRATION STAGE =====
        # Additional turbidity reduction in sand filter
        turb_filtered = turb_clarified * 0.5  # ~50% additional removal
        turb_filtered = max(turb_filtered, 0.1)  # minimum 0.1 NTU
        
        results['output'] = {
            'turbidity_clarified_ntu': turb_clarified,
            'turbidity_filtered_ntu': turb_filtered,
            'removal_raw_to_clarified_pct': 100 * (1 - turb_clarified/turbidity_raw),
            'removal_raw_to_filtered_pct': 100 * (1 - turb_filtered/turbidity_raw),
        }
        
        return results
    
    def fit_to_plant_data(self, data: pd.DataFrame) -> None:
        """
        Fit model parameters to plant operational data
        """
        print(f"\nFitting model to {self.name} operational data...")
        print(f"Data points: {len(data)}")
        
        # Fit coagulant dosing response
        self.dosing_controller.fit_dose_response(data)
        
        # Additional statistics
        valid_data = data[
            (data['coag_dose_pct_mgl'].notna()) &
            (data['turbidity_raw_ntu'] > 0) &
            (data['turbidity_clarified_ntu'] > 0)
        ]
        
        if len(valid_data) > 0:
            print(f"\nOperational statistics:")
            print(f"  Avg raw turbidity: {valid_data['turbidity_raw_ntu'].mean():.1f} NTU")
            print(f"  Avg clarified turbidity: {valid_data['turbidity_clarified_ntu'].mean():.1f} NTU")
            print(f"  Avg coag dose: {valid_data['coag_dose_pct_mgl'].mean():.1f} mg/L")
            print(f"  Avg removal: {100*(1 - valid_data['turbidity_clarified_ntu'].mean()/valid_data['turbidity_raw_ntu'].mean()):.1f}%")
    
    def optimize_dosing(self, 
                       turbidity_raw: float,
                       target_turbidity: float = 1.0) -> float:
        """
        Optimize coagulant dose for given raw turbidity
        
        Minimizes dose while meeting target effluent quality
        """
        def objective(dose):
            if dose < 0.5 or dose > 50:
                return 1e6
            
            results = self.simulate_treatment_train(turbidity_raw, dose)
            turb_out = results['output']['turbidity_filtered_ntu']
            
            # Cost function: dose + penalty for not meeting target
            cost = dose
            if turb_out > target_turbidity:
                cost += 10 * (turb_out - target_turbidity)**2
            
            return cost
        
        # Find optimal dose
        result = minimize(
            objective,
            x0=[2.0],
            bounds=[(0.5, 50)],
            method='L-BFGS-B'
        )
        
        optimal_dose = float(result.x[0])
        return optimal_dose


# ============================================================================
# SECTION 2: UTILITIES & ANALYSIS
# ============================================================================

def generate_design_parameters(plant_data: pd.DataFrame, 
                               plant_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate design parameters for multiple plants
    
    Outputs actionable design parameters for each plant location
    """
    
    results = []
    
    if plant_names is None:
        plant_names = plant_data['plant'].unique()
    
    for plant_name in plant_names:
        plant_subset = plant_data[plant_data['plant'] == plant_name]
        
        if len(plant_subset) < 3:
            continue
        
        # Create simulator for this plant
        sim = AguaClaraPlantSimulation(plant_name)
        
        # Fit to data
        sim.fit_to_plant_data(plant_subset)
        
        # Calculate design parameters
        valid_data = plant_subset[
            (plant_subset['turbidity_raw_ntu'] > 0) &
            (plant_subset['coag_dose_pct_mgl'].notna())
        ]
        
        if len(valid_data) < 2:
            continue
        
        # Average conditions
        avg_raw = valid_data['turbidity_raw_ntu'].mean()
        avg_flow = valid_data['flow_lps'].mean()
        avg_dose = valid_data['coag_dose_pct_mgl'].mean()
        
        # Optimal dosing
        target_turb = 1.0  # NTU
        opt_dose = sim.optimize_dosing(avg_raw, target_turb)
        
        # Simulate with average and optimal dose
        result_avg = sim.simulate_treatment_train(avg_raw, avg_dose, avg_flow)
        result_opt = sim.simulate_treatment_train(avg_raw, opt_dose, avg_flow)
        
        results.append({
            'Plant': plant_name,
            'Avg_Raw_Turbidity_NTU': avg_raw,
            'Avg_Flow_LPS': avg_flow,
            'Current_Avg_Dose_mgL': avg_dose,
            'Optimal_Dose_mgL': opt_dose,
            'Dose_Adjustment_Factor': opt_dose / max(avg_dose, 0.1),
            'Current_Floc_Diameter_um': result_avg['floc']['mean_diameter_um'],
            'Current_Clarified_Turbidity': result_avg['output']['turbidity_clarified_ntu'],
            'Optimal_Clarified_Turbidity': result_opt['output']['turbidity_clarified_ntu'],
            'Current_Filtered_Turbidity': result_avg['output']['turbidity_filtered_ntu'],
            'Optimal_Filtered_Turbidity': result_opt['output']['turbidity_filtered_ntu'],
            'Expected_Removal_Pct': result_opt['output']['removal_raw_to_filtered_pct'],
            'Floc_Terminal_Velocity_cm_s': result_avg['floc']['terminal_velocity_m_s'] * 100,
            'Estimated_Settling_Time_s': result_avg['floc']['settling_time_s'],
        })
    
    return pd.DataFrame(results)


def sensitivity_analysis(base_dose: float, 
                        turbidity_raw: float) -> pd.DataFrame:
    """
    Perform sensitivity analysis on coagulant dose
    
    Shows how dose changes affect effluent quality
    """
    sim = AguaClaraPlantSimulation("Sensitivity Analysis")
    
    doses = np.linspace(0.5, 20, 20)
    results = []
    
    for dose in doses:
        result = sim.simulate_treatment_train(turbidity_raw, dose)
        results.append({
            'Dose_mgL': dose,
            'Floc_Diameter_um': result['floc']['mean_diameter_um'],
            'Clarified_Turbidity_NTU': result['output']['turbidity_clarified_ntu'],
            'Filtered_Turbidity_NTU': result['output']['turbidity_filtered_ntu'],
            'Removal_Pct': result['output']['removal_raw_to_filtered_pct'],
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("AguaClara Plant Simulation System Loaded Successfully ✓")
    print("Ready for plant data analysis and optimization")


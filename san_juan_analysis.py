"""
San Juan Planes Analysis & Optimization
========================================

Analyze operational data from San Juan Planes AguaClara plant
and generate optimized design parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from aguaclara_simulation import (
    AguaClaraPlantSimulation,
    generate_design_parameters,
    sensitivity_analysis
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD AND EXPLORE DATA
# ============================================================================

print("="*80)
print("SAN JUAN PLANES AGUACLARA PLANT - SIMULATION & OPTIMIZATION")
print("="*80)

# Load data
excel_file = '/mnt/user-data/uploads/San_Juan_Planes.xlsx'
df = pd.read_excel(excel_file, sheet_name='San Juan Planes')

print(f"\nData loaded: {len(df)} operational records")
print(f"Date range: {df['record_date'].min()} to {df['record_date'].max()}")

# ============================================================================
# DATA CLEANING & PREPARATION
# ============================================================================

print("\n" + "="*80)
print("DATA QUALITY ASSESSMENT")
print("="*80)

# Check missing values
print("\nMissing values:")
missing_cols = [
    'turbidity_raw_ntu', 'turbidity_clarified_ntu', 'turbidity_filtered_ntu',
    'coag_dose_pct_mgl', 'flow_lps'
]
for col in missing_cols:
    missing_count = df[col].isna().sum()
    pct = 100 * missing_count / len(df)
    print(f"  {col}: {missing_count} ({pct:.1f}%)")

# Create clean dataset
df_clean = df[
    (df['turbidity_raw_ntu'] > 0) &
    (df['turbidity_clarified_ntu'] > 0) &
    (df['turbidity_filtered_ntu'] > 0) &
    (df['coag_dose_pct_mgl'] > 0) &
    (df['flow_lps'] > 0)
].copy()

print(f"\nAfter cleaning: {len(df_clean)} valid records ({100*len(df_clean)/len(df):.1f}%)")

# ============================================================================
# BASIC STATISTICS
# ============================================================================

print("\n" + "="*80)
print("OPERATIONAL STATISTICS - SAN JUAN PLANES")
print("="*80)

stats_data = {
    'Raw Turbidity (NTU)': df_clean['turbidity_raw_ntu'],
    'Clarified Turbidity (NTU)': df_clean['turbidity_clarified_ntu'],
    'Filtered Turbidity (NTU)': df_clean['turbidity_filtered_ntu'],
    'Coagulant Dose (mg/L)': df_clean['coag_dose_pct_mgl'],
    'Flow Rate (L/s)': df_clean['flow_lps'],
}

for name, series in stats_data.items():
    print(f"\n{name}:")
    print(f"  Mean:   {series.mean():.2f}")
    print(f"  Median: {series.median():.2f}")
    print(f"  Std:    {series.std():.2f}")
    print(f"  Min:    {series.min():.2f}")
    print(f"  Max:    {series.max():.2f}")

# Calculate treatment efficiency
df_clean['removal_raw_to_clarified_pct'] = 100 * (1 - df_clean['turbidity_clarified_ntu'] / df_clean['turbidity_raw_ntu'])
df_clean['removal_raw_to_filtered_pct'] = 100 * (1 - df_clean['turbidity_filtered_ntu'] / df_clean['turbidity_raw_ntu'])

print(f"\nTreatment Efficiency:")
print(f"  Raw → Clarified: {df_clean['removal_raw_to_clarified_pct'].mean():.1f}% ± {df_clean['removal_raw_to_clarified_pct'].std():.1f}%")
print(f"  Raw → Filtered:  {df_clean['removal_raw_to_filtered_pct'].mean():.1f}% ± {df_clean['removal_raw_to_filtered_pct'].std():.1f}%")

# ============================================================================
# MODEL FITTING
# ============================================================================

print("\n" + "="*80)
print("MODEL FITTING & CALIBRATION")
print("="*80)

sim = AguaClaraPlantSimulation("San Juan Planes")
sim.fit_to_plant_data(df_clean)

# ============================================================================
# DOSE-RESPONSE RELATIONSHIP
# ============================================================================

print("\n" + "="*80)
print("DOSE-RESPONSE ANALYSIS")
print("="*80)

# Analyze relationship between dose and turbidity reduction
print("\nAnalyzing dose-response relationship...")

# Group by dose ranges
dose_groups = pd.cut(df_clean['coag_dose_pct_mgl'], bins=5)
dose_analysis = df_clean.groupby(dose_groups).agg({
    'coag_dose_pct_mgl': 'mean',
    'turbidity_raw_ntu': 'mean',
    'turbidity_clarified_ntu': 'mean',
    'turbidity_filtered_ntu': 'mean',
    'removal_raw_to_filtered_pct': 'mean',
}).dropna()

print("\nDose Response (grouped by coagulant dose):")
print(dose_analysis.to_string())

# ============================================================================
# OPTIMAL DOSING
# ============================================================================

print("\n" + "="*80)
print("OPTIMAL DOSING STRATEGY")
print("="*80)

# Calculate optimal doses for various raw turbidity levels
turbidity_scenarios = [10, 20, 50, 100, 200, 500]
target_turbidity = 1.0  # NTU

print(f"\nOptimal coagulant doses (target: {target_turbidity} NTU filtered):")
print(f"{'Raw Turbidity (NTU)':<20} {'Current Dose':<20} {'Optimal Dose':<20} {'Change':<15}")
print("-" * 75)

optimal_doses = {}
for turb in turbidity_scenarios:
    opt_dose = sim.optimize_dosing(turb, target_turbidity)
    
    # Estimate what current dose would be
    current_dose = 0.3 * (turb ** 0.8)  # heuristic
    
    change = ((opt_dose - current_dose) / current_dose) * 100
    
    optimal_doses[turb] = opt_dose
    print(f"{turb:<20} {current_dose:<20.2f} {opt_dose:<20.2f} {change:+.1f}%")

# ============================================================================
# FLOC CHARACTERIZATION
# ============================================================================

print("\n" + "="*80)
print("FLOC CHARACTERIZATION AT VARIOUS DOSES")
print("="*80)

print(f"\n{'Dose (mg/L)':<15} {'Floc Dia (μm)':<20} {'Density (kg/m³)':<20} {'Terminal Vel (mm/s)':<20}")
print("-" * 75)

for dose in [5, 10, 15, 20]:
    result = sim.simulate_treatment_train(100, dose)  # 100 NTU reference
    d_um = result['floc']['mean_diameter_um']
    rho = result['floc']['density_kg_m3']
    v_mm = result['floc']['terminal_velocity_m_s'] * 1000
    print(f"{dose:<15} {d_um:<20.1f} {rho:<20.0f} {v_mm:<20.3f}")

# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SENSITIVITY ANALYSIS")
print("="*80)

# Analyze sensitivity at typical raw turbidity
typical_raw = df_clean['turbidity_raw_ntu'].median()
print(f"\nSensitivity analysis at typical raw turbidity ({typical_raw:.1f} NTU):")

sensitivity_df = sensitivity_analysis(10, typical_raw)
print(sensitivity_df.head(10).to_string())

# ============================================================================
# SIMULATION RESULTS
# ============================================================================

print("\n" + "="*80)
print("SIMULATION RESULTS - SCENARIOS")
print("="*80)

scenarios = [
    {'name': 'Low Turbidity (20 NTU)', 'raw': 20, 'dose': 3},
    {'name': 'Medium Turbidity (100 NTU)', 'raw': 100, 'dose': 8},
    {'name': 'High Turbidity (300 NTU)', 'raw': 300, 'dose': 15},
    {'name': 'Very High (500 NTU)', 'raw': 500, 'dose': 20},
]

for scenario in scenarios:
    print(f"\n{scenario['name']}:")
    print(f"  Raw turbidity: {scenario['raw']} NTU, Dose: {scenario['dose']} mg/L")
    
    result = sim.simulate_treatment_train(scenario['raw'], scenario['dose'])
    
    print(f"  Floc properties:")
    print(f"    • Diameter: {result['floc']['mean_diameter_um']:.1f} μm")
    print(f"    • Terminal velocity: {result['floc']['terminal_velocity_m_s']*1000:.2f} mm/s")
    print(f"    • Settling time: {result['floc']['settling_time_s']:.1f} sec")
    
    print(f"  Treatment results:")
    print(f"    • Clarified turbidity: {result['output']['turbidity_clarified_ntu']:.2f} NTU")
    print(f"    • Filtered turbidity: {result['output']['turbidity_filtered_ntu']:.2f} NTU")
    print(f"    • Removal efficiency: {result['output']['removal_raw_to_filtered_pct']:.1f}%")

# ============================================================================
# GENERATE DESIGN PARAMETERS
# ============================================================================

print("\n" + "="*80)
print("DESIGN PARAMETERS FOR PLANTS")
print("="*80)

design_params = generate_design_parameters(df, plant_names=['San Juan Planes'])

print("\nGenerated design parameters:")
print(design_params.to_string(index=False))

# ============================================================================
# VALIDATION WITH ACTUAL DATA
# ============================================================================

print("\n" + "="*80)
print("MODEL VALIDATION")
print("="*80)

print("\nComparing model predictions vs actual observations:")

# Select sample of actual data
sample_df = df_clean.sample(min(20, len(df_clean)), random_state=42)

print(f"\n{'#':<3} {'Raw':<8} {'Dose':<8} {'Actual':<10} {'Predicted':<12} {'Error':<8}")
print(f"{'':3} {'NTU':<8} {'mg/L':<8} {'Filt NTU':<10} {'Filt NTU':<12} {'NTU':<8}")
print("-" * 60)

errors = []
for idx, row in sample_df.iterrows():
    raw = row['turbidity_raw_ntu']
    dose = row['coag_dose_pct_mgl']
    actual = row['turbidity_filtered_ntu']
    
    pred = sim.simulate_treatment_train(raw, dose)
    predicted = pred['output']['turbidity_filtered_ntu']
    
    error = abs(predicted - actual)
    errors.append(error)
    
    if idx < 20:  # Print first 20
        print(f"{idx:<3} {raw:<8.1f} {dose:<8.1f} {actual:<10.2f} {predicted:<12.2f} {error:<8.2f}")

print(f"\nValidation statistics:")
print(f"  Mean Absolute Error: {np.mean(errors):.3f} NTU")
print(f"  RMSE: {np.sqrt(np.mean(np.array(errors)**2)):.3f} NTU")
print(f"  Max Error: {np.max(errors):.3f} NTU")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save design parameters
output_file = '/mnt/user-data/outputs/san_juan_design_parameters.csv'
design_params.to_csv(output_file, index=False)
print(f"✓ Design parameters saved: {output_file}")

# Save sensitivity analysis
output_file = '/mnt/user-data/outputs/sensitivity_analysis.csv'
sensitivity_df.to_csv(output_file, index=False)
print(f"✓ Sensitivity analysis saved: {output_file}")

# Save dose response
dose_analysis.to_csv('/mnt/user-data/outputs/dose_response_analysis.csv')
print(f"✓ Dose response analysis saved")

print("\n" + "="*80)
print("ANALYSIS COMPLETE ✓")
print("="*80)

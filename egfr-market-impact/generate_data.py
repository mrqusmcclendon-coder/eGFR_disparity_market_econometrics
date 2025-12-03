import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2025, 10, 1)
N_LABS = 50
N_SITES = 200
N_PATIENTS_PER_MONTH = 1000

# Generate date range (monthly)
dates = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')

# ==================== REFERENCE TABLES ====================

# Labs Master Table
labs = pd.DataFrame({
    'Lab_ID': [f'LAB{str(i).zfill(3)}' for i in range(1, N_LABS+1)],
    'Lab_Name': [f'Clinical Laboratory {i}' for i in range(1, N_LABS+1)],
    'Region': np.random.choice(['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest'], N_LABS),
    'Adoption_Date': pd.to_datetime([
        np.random.choice(pd.date_range('2021-06-01', '2024-12-31', freq='MS')) 
        if np.random.random() > 0.1 else None  # 10% haven't adopted
        for _ in range(N_LABS)
    ]),
    'Communication_Intensity': np.random.randint(1, 6, N_LABS),  # 1-5 scale
    'Assay_Menu': np.random.choice(['Cr', 'Cys', 'CrCys'], N_LABS, p=[0.3, 0.2, 0.5])
})

# Sites Master Table
sites = pd.DataFrame({
    'Site_ID': [f'SITE{str(i).zfill(4)}' for i in range(1, N_SITES+1)],
    'Lab_ID': np.random.choice(labs['Lab_ID'], N_SITES),
    'Site_Type': np.random.choice(['Hospital', 'Clinic', 'Reference Lab'], N_SITES, p=[0.4, 0.4, 0.2]),
    'Lives_Covered': np.random.randint(5000, 50000, N_SITES)
})

# Merge lab info to sites
sites = sites.merge(labs[['Lab_ID', 'Region', 'Adoption_Date', 'Assay_Menu']], on='Lab_ID', how='left')

# ==================== GENERATE COMPREHENSIVE DATASET ====================

data_records = []

for date in dates:
    for _, site in sites.iterrows():
        
        # Determine if lab has adopted by this date
        if pd.notna(site['Adoption_Date']):
            post_adoption = 1 if date >= site['Adoption_Date'] else 0
            months_since_adoption = (date.year - site['Adoption_Date'].year) * 12 + \
                                   (date.month - site['Adoption_Date'].month) if post_adoption else None
        else:
            post_adoption = 0
            months_since_adoption = None
        
        # Base volumes (with seasonal variation)
        month_factor = 1 + 0.1 * np.sin(2 * np.pi * date.month / 12)  # Seasonal
        trend_factor = 1 + 0.02 * ((date - START_DATE).days / 365)  # Annual growth
        
        # Creatinine tests (baseline)
        cr_volume = int(np.random.poisson(500 * month_factor * trend_factor))
        cr_price = 8.50 + np.random.normal(0, 0.5)  # ASP ~$8.50
        
        # Cystatin C tests (increases post-adoption if CrCys menu)
        if site['Assay_Menu'] == 'CrCys' and post_adoption:
            cysc_boost = 1 + 0.15 * min(months_since_adoption / 12, 1)  # 15% boost over 12 months
        else:
            cysc_boost = 1.0
        
        cysc_volume = int(np.random.poisson(100 * month_factor * trend_factor * cysc_boost)) \
                      if site['Assay_Menu'] in ['Cys', 'CrCys'] else 0
        cysc_price = 25.00 + np.random.normal(0, 2)  # ASP ~$25
        
        # Combined Cr+Cys tests
        crcys_volume = int(np.random.poisson(50 * month_factor * trend_factor * cysc_boost)) \
                       if site['Assay_Menu'] == 'CrCys' and post_adoption else 0
        crcys_price = 32.00 + np.random.normal(0, 2)  # Bundle price
        
        # Calculate revenues
        cr_revenue = cr_volume * cr_price
        cysc_revenue = cysc_volume * cysc_price
        crcys_revenue = crcys_volume * crcys_price
        total_revenue = cr_revenue + cysc_revenue + crcys_revenue
        
        # Patient impact calculations
        total_patients = int(site['Lives_Covered'] * 0.15)  # 15% get tested
        
        # Reclassification rates (based on literature: 10-28%)
        if post_adoption:
            # Higher reclassification in early months, stabilizes
            base_reclass_rate = 0.18  # 18% baseline
            time_decay = 0.9 ** (months_since_adoption / 6) if months_since_adoption else 1
            reclassification_rate = base_reclass_rate * time_decay
        else:
            reclassification_rate = 0.0
        
        reclassified_patients = int(total_patients * reclassification_rate)
        
        # CKD stage distribution (shifts with reclassification)
        if post_adoption and reclassified_patients > 0:
            # Upward reclassification: G3a → G2 (reduces CKD prevalence)
            g1_pct = 0.35 + 0.05 * (reclassification_rate / 0.18)
            g2_pct = 0.30 + 0.03 * (reclassification_rate / 0.18)
            g3a_pct = 0.20 - 0.05 * (reclassification_rate / 0.18)
            g3b_pct = 0.08
            g4_pct = 0.05
            g5_pct = 0.02
        else:
            # Pre-adoption distribution
            g1_pct, g2_pct, g3a_pct, g3b_pct, g4_pct, g5_pct = 0.35, 0.28, 0.22, 0.08, 0.05, 0.02
        
        ckd_patients = int(total_patients * (g3a_pct + g3b_pct + g4_pct + g5_pct))
        
        # Nephrology referrals (threshold-dependent)
        referral_rate = 0.15 if not post_adoption else 0.12  # Slight decrease with better classification
        nephrology_referrals = int(ckd_patients * referral_rate)
        
        # Trial enrollment (if applicable)
        if site['Site_Type'] == 'Hospital':
            trial_enrollment = np.random.poisson(5) if np.random.random() > 0.7 else 0
            # Eligibility shifts at ≥60 threshold
            if post_adoption:
                eligibility_shift_pct = -0.02  # 2% reduction in Black participants at ≥60
            else:
                eligibility_shift_pct = 0.0
        else:
            trial_enrollment = 0
            eligibility_shift_pct = 0.0
        
        # Append record
        data_records.append({
            # Identifiers
            'Date': date,
            'Year': date.year,
            'Month': date.month,
            'Quarter': f'Q{(date.month-1)//3 + 1}',
            'Site_ID': site['Site_ID'],
            'Lab_ID': site['Lab_ID'],
            'Region': site['Region'],
            'Site_Type': site['Site_Type'],
            
            # Adoption metrics
            'Adoption_Date': site['Adoption_Date'],
            'Post_Adoption': post_adoption,
            'Months_Since_Adoption': months_since_adoption,
            'Assay_Menu': site['Assay_Menu'],
            'Communication_Intensity': labs[labs['Lab_ID'] == site['Lab_ID']]['Communication_Intensity'].values[0],
            
            # Test volumes
            'Cr_Volume': cr_volume,
            'Cys_Volume': cysc_volume,
            'CrCys_Volume': crcys_volume,
            'Total_Test_Volume': cr_volume + cysc_volume + crcys_volume,
            
            # Pricing
            'Cr_Price': round(cr_price, 2),
            'Cys_Price': round(cysc_price, 2),
            'CrCys_Price': round(crcys_price, 2),
            
            # Revenue
            'Cr_Revenue': round(cr_revenue, 2),
            'Cys_Revenue': round(cysc_revenue, 2),
            'CrCys_Revenue': round(crcys_revenue, 2),
            'Total_Revenue': round(total_revenue, 2),
            
            # Patient impact
            'Lives_Covered': site['Lives_Covered'],
            'Total_Patients': total_patients,
            'Reclassified_Patients': reclassified_patients,
            'Reclassification_Rate': round(reclassification_rate, 4),
            
            # CKD stages
            'CKD_Patients': ckd_patients,
            'G1_Patients': int(total_patients * g1_pct),
            'G2_Patients': int(total_patients * g2_pct),
            'G3a_Patients': int(total_patients * g3a_pct),
            'G3b_Patients': int(total_patients * g3b_pct),
            'G4_Patients': int(total_patients * g4_pct),
            'G5_Patients': int(total_patients * g5_pct),
            'CKD_Prevalence': round((g3a_pct + g3b_pct + g4_pct + g5_pct), 4),
            
            # Clinical outcomes
            'Nephrology_Referrals': nephrology_referrals,
            'Dialysis_Initiations': int(nephrology_referrals * 0.08) if g4_pct + g5_pct > 0.06 else 0,
            
            # Trial metrics
            'Trial_Enrollment': trial_enrollment,
            'Eligibility_Shift_Pct': round(eligibility_shift_pct, 4),
            
            # Therapeutic impact (placeholder for drug market)
            'Renally_Adjusted_Rx': int(ckd_patients * 0.40),  # 40% on renally-adjusted meds
            'Nephroprotective_Rx': int(ckd_patients * 0.25)   # 25% on SGLT2i/RAAS
        })

# Create DataFrame
df = pd.DataFrame(data_records)

# Sort by date and site
df = df.sort_values(['Date', 'Site_ID']).reset_index(drop=True)

# ==================== EXPORT ====================

import os

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

# Save to CSV in outputs/
output_file = os.path.join('outputs', 'eGFR_Market_Impact_PowerBI.csv')
df.to_csv(output_file, index=False)

print(f"✅ Generated {len(df):,} records across {len(dates)} months and {N_SITES} sites")
print(f"✅ Exported to: {output_file}")
print(f"\n📊 Dataset Summary:")
print(f"   Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"   Total Revenue: ${df['Total_Revenue'].sum():,.2f}")
print(f"   Average Reclassification Rate: {df[df['Post_Adoption']==1]['Reclassification_Rate'].mean():.2%}")
print(f"   Labs with Adoption: {df['Adoption_Date'].notna().sum() / len(df) * 100:.1f}%")
print(f"   Total Cystatin C Tests: {df['Cys_Volume'].sum():,}")

# ==================== VALIDATION AGAINST KPIS ====================

print("\n🔍 KPI Validation:")

# 1. Total Revenue
total_revenue = df['Total_Revenue'].sum()
print(f"   Total Revenue: ${total_revenue:,.2f}")

# 2. Reclassification Rate
overall_reclass = df['Reclassified_Patients'].sum() / df['Total_Patients'].sum()
print(f"   Overall Reclassification Rate: {overall_reclass:.2%} (Benchmark: 10-28%)")

# 3. Cystatin C Growth
cys_earliest = df[df['Date'] == df['Date'].min()]['Cys_Volume'].sum()
cys_latest = df[df['Date'] == df['Date'].max()]['Cys_Volume'].sum()
cys_growth = (cys_latest / cys_earliest - 1) if cys_earliest > 0 else 0
print(f"   Cystatin C Growth (Total Period): {cys_growth:.2%}")

# 4. Adoption Rate
labs_adopted = df['Adoption_Date'].notna().sum()
labs_total = len(df)
adoption_rate = labs_adopted / labs_total
print(f"   Adoption Rate: {adoption_rate:.2%}")

print("\n📋 Power BI DAX Measures Ready:")
print("   ✓ Total Revenue = SUM([Total_Revenue])")
print("   ✓ Reclassification Rate = DIVIDE(SUM([Reclassified_Patients]), SUM([Total_Patients]))")
print("   ✓ Cystatin C Growth = Calculated via time intelligence")
print("   ✓ Adoption Rate = DIVIDE(COUNT([Adoption_Date]), COUNT([Lab_ID]))")

# ==================== DATA DICTIONARY ====================

data_dict = """
📖 DATA DICTIONARY

IDENTIFIERS & TIME
- Date: Month-start date
- Year, Month, Quarter: Time dimensions
- Site_ID: Unique site identifier
- Lab_ID: Parent laboratory identifier
- Region: Geographic region (Northeast, Southeast, Midwest, West, Southwest)
- Site_Type: Hospital, Clinic, or Reference Lab

ADOPTION METRICS
- Adoption_Date: Date lab adopted CKD-EPI 2021 (NULL if not adopted)
- Post_Adoption: Binary indicator (1 = adopted, 0 = not adopted)
- Months_Since_Adoption: Months elapsed since adoption
- Assay_Menu: Test menu (Cr, Cys, CrCys)
- Communication_Intensity: Stakeholder communication score (1-5)

TEST VOLUMES
- Cr_Volume: Creatinine test count
- Cys_Volume: Cystatin C test count
- CrCys_Volume: Combined Cr+Cys test count
- Total_Test_Volume: Sum of all tests

PRICING & REVENUE
- Cr_Price, Cys_Price, CrCys_Price: Average selling price per test
- Cr_Revenue, Cys_Revenue, CrCys_Revenue: Revenue by test type
- Total_Revenue: Total diagnostic revenue

PATIENT IMPACT
- Lives_Covered: Population served by site
- Total_Patients: Patients tested for renal function
- Reclassified_Patients: Patients changing CKD stage
- Reclassification_Rate: % reclassified (benchmark: 10-28%)

CKD STAGING
- CKD_Patients: Total with CKD (G3a-G5)
- G1_Patients through G5_Patients: Count by KDIGO stage
- CKD_Prevalence: % of patients with CKD

CLINICAL OUTCOMES
- Nephrology_Referrals: Specialty referrals
- Dialysis_Initiations: Patients starting dialysis
- Trial_Enrollment: Clinical trial enrollments
- Eligibility_Shift_Pct: Trial eligibility change due to equation

THERAPEUTICS
- Renally_Adjusted_Rx: Patients on dose-adjusted medications
- Nephroprotective_Rx: Patients on kidney-protective drugs (SGLT2i, RAAS blockers)

EVIDENCE ANCHORS
- Reclassification rates calibrated to: Roy et al. (10-28%), Chen & Shi (11.86%), Escribano-Serrano et al. (15%)
- Cystatin C growth linked to: Lee et al. outcome prediction improvements
- Adoption timing: Miller et al. NKF guidance (2022)
- Trial impacts: Tarzi et al. eligibility considerations
"""

dict_file = os.path.join('outputs', 'eGFR_Data_Dictionary.txt')
with open(dict_file, 'w', encoding='utf-8') as f:
    f.write(data_dict)

print("\n📄 Data dictionary saved to: {}".format(dict_file))

# ------------------------------
# Optional: Michigan-specific race displacement query
# ------------------------------
def generate_michigan_dataset(start_date=START_DATE, end_date=END_DATE):
    """Build a Michigan-focused dataset using predefined region demographics.
    Returns a DataFrame similar to the main `df` but scoped to Michigan regions.
    """
    # Michigan regions with realistic demographics
    michigan_regions = {
        'Detroit Metro': {'counties': ['Wayne', 'Oakland', 'Macomb'], 'population': 4300000, 'black_pct': 0.22, 'hispanic_pct': 0.07, 'white_pct': 0.65, 'asian_pct': 0.06, 'ckd_prevalence': 0.14, 'adoption_speed': 'early', 'n_sites': 35},
        'Ann Arbor - Washtenaw': {'counties': ['Washtenaw'], 'population': 370000, 'black_pct': 0.12, 'hispanic_pct': 0.05, 'white_pct': 0.72, 'asian_pct': 0.09, 'ckd_prevalence': 0.11, 'adoption_speed': 'early', 'n_sites': 12},
        'Flint - Genesee': {'counties': ['Genesee'], 'population': 406000, 'black_pct': 0.20, 'hispanic_pct': 0.04, 'white_pct': 0.72, 'asian_pct': 0.02, 'ckd_prevalence': 0.18, 'adoption_speed': 'late', 'n_sites': 10},
        'Grand Rapids - Kent': {'counties': ['Kent', 'Ottawa'], 'population': 650000, 'black_pct': 0.09, 'hispanic_pct': 0.10, 'white_pct': 0.76, 'asian_pct': 0.03, 'ckd_prevalence': 0.12, 'adoption_speed': 'middle', 'n_sites': 18},
        'Lansing - Ingham': {'counties': ['Ingham', 'Eaton', 'Clinton'], 'population': 470000, 'black_pct': 0.11, 'hispanic_pct': 0.07, 'white_pct': 0.77, 'asian_pct': 0.04, 'ckd_prevalence': 0.13, 'adoption_speed': 'middle', 'n_sites': 14},
        'Kalamazoo': {'counties': ['Kalamazoo'], 'population': 260000, 'black_pct': 0.11, 'hispanic_pct': 0.05, 'white_pct': 0.79, 'asian_pct': 0.03, 'ckd_prevalence': 0.12, 'adoption_speed': 'middle', 'n_sites': 10},
        'Ypsilanti - Eastern Washtenaw': {'counties': ['Washtenaw'], 'population': 120000, 'black_pct': 0.28, 'hispanic_pct': 0.06, 'white_pct': 0.58, 'asian_pct': 0.05, 'ckd_prevalence': 0.16, 'adoption_speed': 'middle', 'n_sites': 8},
        'Saginaw - Bay City': {'counties': ['Saginaw', 'Bay'], 'population': 310000, 'black_pct': 0.18, 'hispanic_pct': 0.08, 'white_pct': 0.70, 'asian_pct': 0.02, 'ckd_prevalence': 0.15, 'adoption_speed': 'late', 'n_sites': 12},
        'Upper Peninsula': {'counties': ['Marquette', 'Houghton', 'Chippewa'], 'population': 300000, 'black_pct': 0.02, 'hispanic_pct': 0.02, 'white_pct': 0.94, 'asian_pct': 0.01, 'ckd_prevalence': 0.14, 'adoption_speed': 'late', 'n_sites': 15},
        'Southwest Michigan': {'counties': ['Berrien', 'Cass', 'Van Buren'], 'population': 280000, 'black_pct': 0.14, 'hispanic_pct': 0.08, 'white_pct': 0.74, 'asian_pct': 0.02, 'ckd_prevalence': 0.13, 'adoption_speed': 'middle', 'n_sites': 12},
        'Muskegon': {'counties': ['Muskegon', 'Oceana'], 'population': 175000, 'black_pct': 0.13, 'hispanic_pct': 0.10, 'white_pct': 0.73, 'asian_pct': 0.02, 'ckd_prevalence': 0.14, 'adoption_speed': 'late', 'n_sites': 8},
        'Traverse City - Northwest': {'counties': ['Grand Traverse', 'Leelanau', 'Benzie'], 'population': 150000, 'black_pct': 0.02, 'hispanic_pct': 0.03, 'white_pct': 0.93, 'asian_pct': 0.01, 'ckd_prevalence': 0.11, 'adoption_speed': 'middle', 'n_sites': 10},
        'Monroe': {'counties': ['Monroe'], 'population': 152000, 'black_pct': 0.05, 'hispanic_pct': 0.04, 'white_pct': 0.88, 'asian_pct': 0.02, 'ckd_prevalence': 0.13, 'adoption_speed': 'middle', 'n_sites': 6},
        'Jackson': {'counties': ['Jackson'], 'population': 160000, 'black_pct': 0.10, 'hispanic_pct': 0.06, 'white_pct': 0.81, 'asian_pct': 0.02, 'ckd_prevalence': 0.14, 'adoption_speed': 'late', 'n_sites': 8},
        'Midland': {'counties': ['Midland'], 'population': 83000, 'black_pct': 0.03, 'hispanic_pct': 0.04, 'white_pct': 0.91, 'asian_pct': 0.01, 'ckd_prevalence': 0.12, 'adoption_speed': 'early', 'n_sites': 6}
    }

    # Build labs and sites similar to the user's Michigan block
    labs_list = []
    lab_id = 1
    for region_name, region_data in michigan_regions.items():
        n_labs_in_region = max(1, region_data['n_sites'] // 15)
        for i in range(n_labs_in_region):
            if region_data['adoption_speed'] == 'early':
                adopt_start, adopt_end = '2021-09-01', '2022-12-31'
            elif region_data['adoption_speed'] == 'middle':
                adopt_start, adopt_end = '2022-06-01', '2024-03-31'
            else:
                adopt_start, adopt_end = '2023-01-01', '2024-12-31'

            if np.random.random() > 0.1:
                adoption_date = pd.to_datetime(np.random.choice(pd.date_range(adopt_start, adopt_end, freq='MS')))
            else:
                adoption_date = None

            labs_list.append({
                'Lab_ID': f'MI_LAB{str(lab_id).zfill(3)}',
                'Lab_Name': f'{region_name} Clinical Laboratory {i+1}',
                'Region': region_name,
                'State': 'Michigan',
                'Counties': ', '.join(region_data['counties']),
                'Population_Served': int(region_data['population'] / max(1, n_labs_in_region)),
                'Adoption_Date': adoption_date,
                'Communication_Intensity': np.random.randint(1, 6),
                'Assay_Menu': np.random.choice(['Cr', 'Cys', 'CrCys'], p=[0.3, 0.2, 0.5]),
                'Adoption_Speed': region_data['adoption_speed']
            })
            lab_id += 1

    labs_mi = pd.DataFrame(labs_list)

    sites_list = []
    site_id = 1
    for region_name, region_data in michigan_regions.items():
        region_labs = labs_mi[labs_mi['Region'] == region_name]['Lab_ID'].tolist()
        for i in range(region_data['n_sites']):
            chosen_lab = np.random.choice(region_labs) if region_labs else labs_mi['Lab_ID'].iloc[0]
            bpct = region_data['black_pct'] + np.random.normal(0, 0.02)
            sites_list.append({
                'Site_ID': f'MI_SITE{str(site_id).zfill(4)}',
                'Site_Name': f'{region_name} - Site {i+1}',
                'Lab_ID': chosen_lab,
                'Region': region_name,
                'Site_Type': np.random.choice(['Hospital', 'Clinic', 'Reference Lab'], p=[0.4, 0.4, 0.2]),
                'Lives_Covered': int(region_data['population'] / max(1, region_data['n_sites']) * np.random.uniform(0.8, 1.2)),
                'Black_Population_Pct': bpct,
                'Hispanic_Population_Pct': region_data['hispanic_pct'] + np.random.normal(0, 0.01),
                'White_Population_Pct': region_data['white_pct'] + np.random.normal(0, 0.02),
                'Asian_Population_Pct': region_data['asian_pct'] + np.random.normal(0, 0.01),
                'Baseline_CKD_Prevalence': region_data['ckd_prevalence'] + np.random.normal(0, 0.01)
            })
            site_id += 1

    sites_mi = pd.DataFrame(sites_list)
    sites_mi = sites_mi.merge(labs_mi[['Lab_ID', 'Adoption_Date', 'Assay_Menu', 'Counties']], on='Lab_ID', how='left')

    # Generate dataset for Michigan using the same logic but include race-equation comparisons
    dates_mi = pd.date_range(start=start_date, end=end_date, freq='MS')
    data_records_mi = []
    for date in dates_mi:
        for _, site in sites_mi.iterrows():
            # Simplified volumes/prices similar to main script
            post_adoption = 1 if pd.notna(site['Adoption_Date']) and date >= site['Adoption_Date'] else 0
            months_since_adoption = (date.year - site['Adoption_Date'].year) * 12 + (date.month - site['Adoption_Date'].month) if post_adoption and pd.notna(site['Adoption_Date']) else None
            month_factor = 1 + 0.1 * np.sin(2 * np.pi * date.month / 12)
            trend_factor = 1 + 0.02 * ((date - start_date).days / 365)
            cr_volume = int(np.random.poisson(500 * month_factor * trend_factor))
            cr_price = 8.50 + np.random.normal(0, 0.5)
            if site['Assay_Menu'] == 'CrCys' and post_adoption:
                cysc_boost = 1 + 0.15 * min(months_since_adoption / 12, 1)
            else:
                cysc_boost = 1.0
            cysc_volume = int(np.random.poisson(100 * month_factor * trend_factor * cysc_boost)) if site['Assay_Menu'] in ['Cys', 'CrCys'] else 0
            cysc_price = 25.00 + np.random.normal(0, 2)
            crcys_volume = int(np.random.poisson(50 * month_factor * trend_factor * cysc_boost)) if site['Assay_Menu'] == 'CrCys' and post_adoption else 0
            crcys_price = 32.00 + np.random.normal(0, 2)
            cr_revenue = cr_volume * cr_price
            cysc_revenue = cysc_volume * cysc_price
            crcys_revenue = crcys_volume * crcys_price
            total_revenue = cr_revenue + cysc_revenue + crcys_revenue
            total_patients = int(site['Lives_Covered'] * 0.15)

            # Race-stratified patient counts
            black_patients_pct = max(0, min(1, site['Black_Population_Pct']))
            black_patients = int(total_patients * black_patients_pct)
            nonblack_patients = total_patients - black_patients

            # Reclassification
            if post_adoption:
                base_reclass_rate = 0.18
                time_decay = 0.9 ** (months_since_adoption / 6) if months_since_adoption else 1
                reclassification_rate = base_reclass_rate * time_decay
            else:
                reclassification_rate = 0.0
            reclassified_patients = int(total_patients * reclassification_rate)

            # CKD stages
            baseline_ckd_prev = site['Baseline_CKD_Prevalence']
            if post_adoption and reclassified_patients > 0:
                g1_pct = 0.35 + 0.05 * (reclassification_rate / 0.18)
                g2_pct = 0.30 + 0.03 * (reclassification_rate / 0.18)
                g3a_pct = max(0.10, baseline_ckd_prev * 0.60 - 0.05 * (reclassification_rate / 0.18))
                g3b_pct = baseline_ckd_prev * 0.25
                g4_pct = baseline_ckd_prev * 0.12
                g5_pct = baseline_ckd_prev * 0.03
            else:
                g1_pct = 0.35
                g2_pct = 0.28
                g3a_pct = baseline_ckd_prev * 0.60
                g3b_pct = baseline_ckd_prev * 0.25
                g4_pct = baseline_ckd_prev * 0.12
                g5_pct = baseline_ckd_prev * 0.03
            ckd_patients = int(total_patients * (g3a_pct + g3b_pct + g4_pct + g5_pct))

            referral_rate = 0.15 if not post_adoption else 0.12
            nephrology_referrals = int(ckd_patients * referral_rate)

            if site['Site_Type'] == 'Hospital':
                trial_enrollment = np.random.poisson(5) if np.random.random() > 0.7 else 0
                eligibility_shift_pct = -0.02 if post_adoption else 0.0
            else:
                trial_enrollment = 0
                eligibility_shift_pct = 0.0

            # Race-based equation comparison (simplified)
            avg_creatinine_nonblack = 1.2 + np.random.normal(0, 0.3)
            avg_creatinine_black = avg_creatinine_nonblack * 1.12
            race_coefficient_2009 = 1.159
            egfr_2021_avg = max(5, 65 + np.random.normal(0, 20))
            egfr_2009_black = egfr_2021_avg * race_coefficient_2009
            egfr_2009_nonblack = egfr_2021_avg
            def referral_probability(egfr, threshold=20):
                return 1 / (1 + np.exp(0.3 * (egfr - threshold)))
            referral_prob_2021 = referral_probability(egfr_2021_avg, threshold=20)
            referral_prob_2009_black = referral_probability(egfr_2009_black, threshold=20)
            referral_prob_2009_nonblack = referral_probability(egfr_2009_nonblack, threshold=20)
            waitlist_delay_2021 = max(0, (egfr_2021_avg - 20) * 0.5)
            waitlist_delay_2009_black = max(0, (egfr_2009_black - 20) * 0.5)
            waitlist_delay_2009_nonblack = max(0, (egfr_2009_nonblack - 20) * 0.5)
            eligible_transplant_2021 = 1 if egfr_2021_avg < 20 else 0
            eligible_transplant_2009_black = 1 if egfr_2009_black < 20 else 0
            eligible_transplant_2009_nonblack = 1 if egfr_2009_nonblack < 20 else 0
            referral_disparity_black = referral_prob_2009_black - referral_prob_2021
            transplant_delay_disparity = waitlist_delay_2009_black - waitlist_delay_2021
            eligibility_gap_black = eligible_transplant_2009_black - eligible_transplant_2021

            data_records_mi.append({
                'Date': date, 'Site_ID': site['Site_ID'], 'Site_Name': site['Site_Name'], 'Lab_ID': site['Lab_ID'], 'Region': site['Region'],
                'State': 'Michigan', 'Counties': site['Counties'], 'Site_Type': site['Site_Type'],
                'Adoption_Date': site['Adoption_Date'], 'Post_Adoption': post_adoption, 'Months_Since_Adoption': months_since_adoption,
                'Assay_Menu': site['Assay_Menu'], 'Communication_Intensity': labs_mi[labs_mi['Lab_ID'] == site['Lab_ID']]['Communication_Intensity'].values[0],
                'Cr_Volume': cr_volume, 'Cys_Volume': cysc_volume, 'CrCys_Volume': crcys_volume, 'Total_Test_Volume': cr_volume + cysc_volume + crcys_volume,
                'Cr_Price': round(cr_price, 2), 'Cys_Price': round(cysc_price, 2), 'CrCys_Price': round(crcys_price, 2),
                'Cr_Revenue': round(cr_revenue, 2), 'Cys_Revenue': round(cysc_revenue, 2), 'CrCys_Revenue': round(crcys_revenue, 2), 'Total_Revenue': round(total_revenue, 2),
                'Lives_Covered': site['Lives_Covered'], 'Total_Patients': total_patients, 'Black_Patients': black_patients, 'NonBlack_Patients': nonblack_patients,
                'Reclassified_Patients': reclassified_patients, 'Reclassification_Rate': round(reclassification_rate, 4),
                'Black_Population_Pct': round(site['Black_Population_Pct'], 4), 'Hispanic_Population_Pct': round(site['Hispanic_Population_Pct'], 4),
                'White_Population_Pct': round(site['White_Population_Pct'], 4), 'Asian_Population_Pct': round(site['Asian_Population_Pct'], 4),
                'CKD_Patients': ckd_patients, 'G1_Patients': int(total_patients * g1_pct), 'G2_Patients': int(total_patients * g2_pct),
                'G3a_Patients': int(total_patients * g3a_pct), 'G3b_Patients': int(total_patients * g3b_pct), 'G4_Patients': int(total_patients * g4_pct), 'G5_Patients': int(total_patients * g5_pct),
                'CKD_Prevalence': round((g3a_pct + g3b_pct + g4_pct + g5_pct), 4), 'Baseline_CKD_Prevalence': round(baseline_ckd_prev, 4),
                'Nephrology_Referrals': nephrology_referrals, 'Dialysis_Initiations': int(nephrology_referrals * 0.08) if g4_pct + g5_pct > 0.06 else 0,
                'Trial_Enrollment': trial_enrollment, 'Eligibility_Shift_Pct': round(eligibility_shift_pct, 4),
                'Renally_Adjusted_Rx': int(ckd_patients * 0.40), 'Nephroprotective_Rx': int(ckd_patients * 0.25),
                'Avg_Creatinine_NonBlack': round(avg_creatinine_nonblack, 2), 'Avg_Creatinine_Black': round(avg_creatinine_black, 2),
                'eGFR_2021_RaceFree': round(egfr_2021_avg, 1), 'eGFR_2009_Black': round(egfr_2009_black, 1), 'eGFR_2009_NonBlack': round(egfr_2009_nonblack, 1),
                'Referral_Prob_2021': round(referral_prob_2021, 4), 'Referral_Prob_2009_Black': round(referral_prob_2009_black, 4), 'Referral_Prob_2009_NonBlack': round(referral_prob_2009_nonblack, 4),
                'Eligible_Transplant_2021': eligible_transplant_2021, 'Eligible_Transplant_2009_Black': eligible_transplant_2009_black, 'Eligible_Transplant_2009_NonBlack': eligible_transplant_2009_nonblack,
                'Waitlist_Delay_Months_2021': round(waitlist_delay_2021, 1), 'Waitlist_Delay_Months_2009_Black': round(waitlist_delay_2009_black, 1), 'Waitlist_Delay_Months_2009_NonBlack': round(waitlist_delay_2009_nonblack, 1),
                'Referral_Disparity_Black': round(referral_disparity_black, 4), 'Transplant_Delay_Disparity': round(transplant_delay_disparity, 1), 'Eligibility_Gap_Black': eligibility_gap_black
            })

    df_mi = pd.DataFrame(data_records_mi)
    return df_mi


def race_displacement_query(df_in, group_by='Region'):
    """Aggregate race-displacement / disparity metrics by the chosen geography (default: Region).
    Works when the DataFrame already contains disparity columns (Referral_Disparity_Black, Transplant_Delay_Disparity, Eligibility_Gap_Black).
    If they don't exist, the function will attempt to estimate simple proxies using available demographics.
    """
    df = df_in.copy()
    required = ['Referral_Disparity_Black', 'Transplant_Delay_Disparity', 'Eligibility_Gap_Black']
    if not all(col in df.columns for col in required):
        # create simple proxies if needed
        if 'Black_Population_Pct' in df.columns and 'Reclassification_Rate' in df.columns:
            df['Referral_Disparity_Black'] = df['Reclassification_Rate'] * df['Black_Population_Pct'] * 0.2
            df['Transplant_Delay_Disparity'] = df['Reclassification_Rate'] * 0.5
            df['Eligibility_Gap_Black'] = (df['Reclassification_Rate'] > 0).astype(int) * 0
        else:
            raise ValueError('Input dataframe lacks required disparity columns and cannot infer proxies')

    agg = df.groupby(group_by).agg(
        Sites=('Site_ID', 'nunique'),
        Avg_Referral_Disparity_Black=('Referral_Disparity_Black', 'mean'),
        Avg_Transplant_Delay_Disparity=('Transplant_Delay_Disparity', 'mean'),
        Avg_Eligibility_Gap_Black=('Eligibility_Gap_Black', 'mean'),
        Black_Pop_Pct=('Black_Population_Pct', 'mean') if 'Black_Population_Pct' in df.columns else ('Site_ID', lambda s: np.nan)
    ).reset_index()

    print('\n\n=== Race-displacement summary by {} ==='.format(group_by))
    for _, row in agg.iterrows():
        print(f"{row[group_by]:30s} | Sites: {int(row['Sites']):3d} | Black_Pct: {row.get('Black_Pop_Pct', np.nan):5.1%} | Avg Referral Disparity (Black): {row['Avg_Referral_Disparity_Black']:.4f} | Avg Waitlist Delay Disparity: {row['Avg_Transplant_Delay_Disparity']:.2f} | Avg Eligibility Gap: {row['Avg_Eligibility_Gap_Black']:.2f}")

    return agg


# Run Michigan query and print summary (non-blocking; optional)
try:
    df_mi = generate_michigan_dataset()
    _mi_summary = race_displacement_query(df_mi, group_by='Region')
    # Export Michigan dataset and summary for Power BI
    mi_data_file = os.path.join('outputs', 'michigan_eGFR_impact.csv')
    mi_summary_file = os.path.join('outputs', 'michigan_race_displacement_summary.csv')
    try:
        df_mi.to_csv(mi_data_file, index=False, encoding='utf-8')
        _mi_summary.to_csv(mi_summary_file, index=False, encoding='utf-8')
        print(f"\n✅ Michigan dataset exported to: {mi_data_file}")
        print(f"✅ Michigan summary exported to: {mi_summary_file}\n")
    except Exception as e:
        print('Failed to export Michigan CSVs:', e)
except Exception as e:
    print('Michigan query skipped (error):', e)
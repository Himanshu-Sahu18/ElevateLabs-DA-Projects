import pandas as pd
import numpy as np

# Load the data
input_path = 'Financial KPI Analysis for a Startup/startup_growth_investment_data.csv'
df = pd.read_csv(input_path)

# 1. Years Active
current_year = 2025
df['Years_Active'] = current_year - df['Year Founded']
df['Years_Active'] = df['Years_Active'].replace(0, np.nan)  # Avoid division by zero

# 2. Burn Rate (Yearly)
df['Burn_Rate_Yearly'] = df['Investment Amount (USD)'] / df['Years_Active']

# 3. Estimated Customers (100K per funding round)
df['Estimated_Customers'] = df['Funding Rounds'] * 100_000

# 4. CAC (Customer Acquisition Cost)
df['CAC'] = df['Investment Amount (USD)'] / df['Estimated_Customers']

# 5. LTV (Valuation per Customer)
df['LTV'] = df['Valuation (USD)'] / df['Estimated_Customers']

# 6. LTV:CAC Ratio
df['LTV_CAC_Ratio'] = df['LTV'] / df['CAC']

# 7. Runway (Years) - assuming $5M cash reserve
df['Runway_Years'] = 5_000_000 / df['Burn_Rate_Yearly']

# 8. Cohort Analysis by Year Founded
cohort_df = df.groupby('Year Founded')[['Investment Amount (USD)', 'Valuation (USD)']].mean().reset_index()
cohort_df.rename(columns={
    'Investment Amount (USD)': 'Avg_Investment_Amount',
    'Valuation (USD)': 'Avg_Valuation'
}, inplace=True)

# Export enriched data for Tableau/Excel
df.to_csv('Financial KPI Analysis for a Startup/startup_growth_investment_data_enriched.csv', index=False)
cohort_df.to_csv('Financial KPI Analysis for a Startup/startup_cohort_analysis.csv', index=False)

print('KPI calculation complete. Enriched data exported for Tableau/Excel.') 
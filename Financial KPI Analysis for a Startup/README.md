# Financial KPI Analysis for a Startup

## Objective

Analyze monthly revenue, burn rate, CAC, LTV, and run rate for an early-stage startup.

## Dashboard


## Tools Used

- **Python (Pandas)**: For data processing and KPI calculations
- **Excel**: For modeling and template creation
- **Tableau**: For dashboard and data visualization

## Mini Guide

1. **Collect Financials**: Gather data on expenses, revenue, and customer base.
2. **Compute LTV:CAC Ratio**: Calculate Lifetime Value (LTV) and Customer Acquisition Cost (CAC) and their ratio.
3. **Build Dashboard**: Use Tableau to create a dashboard with trend indicators for key metrics.
4. **Perform Cohort Analysis**: Analyze monthly customer groups to understand trends and retention.

## KPIs Calculated

- **Monthly Revenue**
- **Burn Rate**
- **CAC (Customer Acquisition Cost)**
- **LTV (Lifetime Value)**
- **LTV:CAC Ratio**
- **Run Rate**
- **Cohort Analysis** (by monthly customer groups)

## Data Sources

- **startup_growth_investment_data.csv**: Raw data containing startup funding, valuation, and growth information.

## Output Files & Deliverables

- **Tableau Dashboard**: Interactive dashboard visualizing trends and KPIs.
- **LTV:CAC Report (PDF)**: Summary report of LTV:CAC analysis.
- **Excel Model Template**: For financial modeling and scenario analysis.
- **startup_growth_investment_data_enriched.csv**: Enriched dataset with all calculated KPIs.
- **startup_cohort_analysis.csv**: Cohort analysis by year founded.

## Usage

1. Ensure you have Python 3.x installed.
2. Install dependencies:
   ```bash
   pip install pandas numpy
   ```
3. Place `startup_growth_investment_data.csv` in the project directory.
4. Run the analysis script:
   ```bash
   python financial_kpi_analysis.py
   ```
5. The enriched data and cohort analysis will be exported as CSV files for use in Tableau or Excel.
6. Use Tableau to build dashboards and Excel for further modeling as needed.

## Dependencies

- pandas
- numpy

## Visualization

The output CSV files can be imported into Tableau or Excel for further analysis and visualization.

## License

This project is for educational and analytical purposes.

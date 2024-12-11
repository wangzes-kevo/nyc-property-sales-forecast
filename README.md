# nyc-property-sales-forecast

This is the repo for the project: Model Comparison for NYC Property Sales Forecasting. The performance of SARIMA, LSTM, and Propeht are compared using their best rolling validation performance.

## Data
The data source is NYC's annualized sales update released by the NYC Department of Finance. We gathered data for Manhattan, Bronx, Brooklyn, Queens, and Staten Island from 2003 to 2023 and then stacked them into one table. Then, the empty string in the table was marked as missing value and we removed rows where the sale price is 0 (transfer) or all entries are missing.

Then, we did three different aggregations for separate experiments: 
- Directly aggregate **all the daily sales data** info into monthly sales no matter which building\_class\_at\_time\_of\_sale it belongs to;
- Select the entries where building\_class\_at\_time\_of\_sale is **One or Two Family Dwellings** (starts with label 'A' or 'B') and then aggregate it into monthly sales info;
- Select the entries where building\_class\_at\_time\_of\_sale is **Walk up or Elevator Apartments** (starts with label 'C' or 'D') and then aggregate it into monthly sales info;

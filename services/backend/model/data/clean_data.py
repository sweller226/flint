import pandas as pd

data_file = "data.csv"
df = pd.read_csv(data_file)

print(df.head())

# 1. Load with memory optimization
df = pd.read_csv('data.csv', low_memory=False)

# 2. Parse ts_event with strict formatting (Regex-style pattern matching)
# We use ISO8601 because your data contains the 'T' and 'Z' markers.
# errors='coerce' ensures that the malformed row 450120 is turned into a Null (NaT)
df['ts_event'] = pd.to_datetime(df['ts_event'], format='ISO8601', errors='coerce')

# 3. Remove NaN (Crucial: This removes the corrupted row 450120)
df = df.dropna(subset=['ts_event', 'close'])

# 4. Remove Interval Tickers (Calendar Spreads)
df = df[~df['symbol'].str.contains('-', na=False)]

# 5. Sort the data by time (Required for forecasting)
# This ensures that your H, M, U, Z blocks are chronologically ordered.
df = df.sort_values('ts_event')

# 6. Categorize by Contract Month (H, M, U, Z)
df['contract_month'] = df['symbol'].str[2]

# 7. Drop unneeded columns
cols_to_drop = ['rtype', 'publisher_id', 'instrument_id', 'symbol']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 8. Split the Data into sterile copies
df_h = df[df['contract_month'] == 'H'].copy()
df_m = df[df['contract_month'] == 'M'].copy()
df_u = df[df['contract_month'] == 'U'].copy()
df_z = df[df['contract_month'] == 'Z'].copy()

df_h.to_csv('df_h.csv', index=False)
df_m.to_csv('df_m.csv', index=False)
df_u.to_csv('df_u.csv', index=False)
df_z.to_csv('df_z.csv', index=False)

print('DataFrames successfully exported to CSV files: df_h.csv, df_m.csv, df_u.csv, df_z.csv')
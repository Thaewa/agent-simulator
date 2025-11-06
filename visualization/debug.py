import pandas as pd

df = pd.read_csv("output_logs/aggregate_combined.csv")
print(df.head())
print(df.dtypes)
import pandas as pd

df = pd.read_csv("20240121_shopee_sample_data (1).csv")

# Menampilkan 5 baris pertama
print(df.head())

# Menampilkan struktur data
print(df.info())

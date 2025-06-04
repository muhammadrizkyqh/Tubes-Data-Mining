import pandas as pd
import numpy as np

def convert_k(x):
    try:
        x = str(x).lower().replace("favorite", "").replace("(", "").replace(")", "").strip()
        if 'k' in x:
            return float(x.replace('k', '')) * 1000
        return float(x)
    except:
        return np.nan

# Load the data
df = pd.read_csv("20240121_shopee_sample_data (1).csv")

# Apply the conversion function to relevant columns
df['total_sold'] = df['total_sold'].apply(convert_k)
df['total_rating'] = df['total_rating'].apply(convert_k)
df['favorite'] = df['favorite'].apply(convert_k)
df['item_rating'] = pd.to_numeric(df['item_rating'], errors='coerce')

# Display the cleaned data
print("Data after conversion:")
print(df[['total_sold', 'total_rating', 'favorite', 'item_rating']].head())

# Create a subset with selected columns and drop missing values
selected_df = df[['total_sold', 'total_rating', 'favorite', 'item_rating']].copy()
selected_df = selected_df.dropna()

# Scale the data for machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_df)

# Convert back to DataFrame with original column names
scaled_df = pd.DataFrame(scaled_data, columns=selected_df.columns)

# Inisialisasi model K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Simpan hasil cluster ke DataFrame
selected_df['cluster'] = clusters

print(f"\nOriginal data shape: {df.shape}")
print(f"Selected data shape after dropping NaN: {selected_df.shape}")
print("\nCleaned selected data:")
print(selected_df.head())
print("\nScaled data:")
print(scaled_df.head())
print(f"\nCluster distribution:")
print(selected_df['cluster'].value_counts().sort_index())

# Save the cleaned and scaled data
df.to_csv("cleaned_shopee_data.csv", index=False)
selected_df.to_csv("selected_clean_data.csv", index=False)
scaled_df.to_csv("scaled_data.csv", index=False)
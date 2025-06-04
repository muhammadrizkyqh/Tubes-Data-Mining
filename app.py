# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Tugas Besar ML - Shopee Analysis", layout="wide")

# ==================== Judul & Pendahuluan ====================
st.title("📘 Tugas Besar Machine Learning")
st.markdown("**Kelompok: Nama anggota 1, 2, 3, 4**")

st.header("1. Business Understanding")
st.write("""
Shopee adalah salah satu platform e-commerce terbesar di Asia Tenggara. Penjual menghadapi tantangan dalam menentukan jenis produk yang paling berpotensi untuk menghasilkan penjualan tinggi. 
Tujuan analisis ini adalah:
- Mengelompokkan produk berdasarkan karakteristik (Clustering)
- Memprediksi potensi penjualan produk baru (Regression)
""")

# ==================== Upload Data ====================
st.header("2. Data Understanding")
uploaded_file = st.file_uploader("Unggah dataset Shopee (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Mentah")
    st.write(df.head())

    st.markdown("**Statistik Ringkasan**")
    st.write(df.describe())

    # ==================== Preprocessing ====================
    st.header("3. Data Preparation")

    def convert_k(x):
        try:
            x = str(x).lower().replace("k", "").replace("(", "").replace(")", "").replace("favorite", "").strip()
            return float(x) * 1000 if 'k' in str(x) else float(x)
        except:
            return np.nan

    df['total_sold'] = df['total_sold'].apply(convert_k)
    df['total_rating'] = df['total_rating'].apply(convert_k)
    df['favorite'] = df['favorite'].apply(convert_k)
    df['item_rating'] = pd.to_numeric(df['item_rating'], errors='coerce')

    selected = df[['price_ori', 'price_actual', 'item_rating', 'total_rating', 'favorite', 'total_sold']].dropna()

    st.success("Data berhasil dibersihkan dan dipilih 6 fitur utama.")
    st.write(selected.head())

    # ==================== Clustering ====================
    st.header("4.1 Clustering (Unsupervised Learning)")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(selected.drop(columns='total_sold'))

    kmeans = KMeans(n_clusters=3, random_state=42)
    selected['cluster'] = kmeans.fit_predict(scaled)

    st.write("Hasil Clustering:")
    st.dataframe(selected[['price_actual', 'item_rating', 'total_sold', 'favorite', 'cluster']].head())

    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=scaled[:, 0], y=scaled[:, 1], hue=selected['cluster'], palette='Set2', ax=ax1)
    plt.title("Visualisasi Cluster (fitur terstandardisasi)")
    st.pyplot(fig1)

    # ==================== Regression ====================
    st.header("4.2 Regression (Supervised Learning)")

    X = selected.drop(columns=['total_sold', 'cluster'])
    y = selected['total_sold']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown(f"""
    - **R² Score:** {r2_score(y_test, y_pred):.4f}
    - **MSE:** {mean_squared_error(y_test, y_pred):,.2f}
    """)

    # ==================== Prediksi Produk Baru ====================
    st.header("5. Deployment: Prediksi Produk Baru")

    with st.form("form_prediksi"):
        st.markdown("Masukkan fitur produk baru:")
        col1, col2 = st.columns(2)
        with col1:
            price_ori = st.number_input("Harga Awal", 0, 1000000, value=100000)
            price_actual = st.number_input("Harga Setelah Diskon", 0, 1000000, value=85000)
            item_rating = st.slider("Rating Produk", 0.0, 5.0, 4.5)
        with col2:
            total_rating = st.number_input("Jumlah Rating", 0, 100000, value=1500)
            favorite = st.number_input("Jumlah Favorit", 0, 100000, value=300)
        submitted = st.form_submit_button("Prediksi")

        if submitted:
            new_data = pd.DataFrame([{
                'price_ori': price_ori,
                'price_actual': price_actual,
                'item_rating': item_rating,
                'total_rating': total_rating,
                'favorite': favorite
            }])
            prediksi = model.predict(new_data)[0]
            st.success(f"📦 Prediksi jumlah terjual: {int(prediksi)} unit")

    # ==================== Export Dataset (optional) ====================
    st.download_button("⬇️ Unduh Data Hasil Clustering", data=selected.to_csv(index=False), file_name="shopee_clustered.csv", mime='text/csv')

else:
    st.info("Silakan unggah dataset untuk melanjutkan.")

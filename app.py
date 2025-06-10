import streamlit as st
import pandas as pd
import os
import urllib.parse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Prediksi Harga Motor", layout="wide")

# =========================
# Sidebar Upload Section
# =========================
st.sidebar.markdown("## 🔧 Manajemen Data")

# Upload CSV Motor
uploaded_csv = st.sidebar.file_uploader("📄 Upload file CSV motor", type=["csv"])
if uploaded_csv is not None:
    try:
        new_df = pd.read_csv(uploaded_csv)
        new_df.to_csv("motor.csv", index=False)
        st.sidebar.success("✅ Dataset motor berhasil diunggah!")
        st.rerun()  # force reload page & cache
    except Exception as e:
        st.sidebar.error(f"❌ Gagal upload: {e}")

# Upload Gambar
uploaded_image = st.sidebar.file_uploader("🖼️ Upload gambar motor", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    file_name = uploaded_image.name.lower().replace(" ", "")
    file_path = os.path.join("images", file_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_image.read())
    st.sidebar.success(f"✅ Gambar disimpan sebagai: {file_name}")

# =========================
# Load & Encode Dataset (No Cache)
# =========================
def load_data():
    return pd.read_csv("motor.csv")

df = load_data()

df_encoded = df.copy()
encoders = {}
for col in ['brand', 'model', 'condition']:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df_encoded.drop('price', axis=1)
y = df_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# =========================
# Sidebar Input User
# =========================
st.sidebar.header("🛠️ Input Data Motor")
brand = st.sidebar.selectbox("Merek", df['brand'].unique())
model_motor = st.sidebar.selectbox("Model", df[df['brand'] == brand]['model'].unique())
year = st.sidebar.slider("Tahun", 2010, 2024, 2020)
km = st.sidebar.slider("Kilometer (ribu)", 1, 100, 20)
cc = st.sidebar.slider("Kapasitas Mesin (cc)", 100, 200, 125)
condition = st.sidebar.selectbox("Kondisi", df['condition'].unique())

input_df = pd.DataFrame([{
    "brand": brand,
    "model": model_motor,
    "year": year,
    "km": km * 1000,
    "cc": cc,
    "condition": condition
}])

input_encoded = input_df.copy()
for col in ['brand', 'model', 'condition']:
    input_encoded[col] = encoders[col].transform(input_encoded[col])

# =========================
# Gambar Otomatis
# =========================
model_slug = model_motor.lower().replace(" ", "")
img_paths = [
    f"images/{model_slug}.jpg",
    f"images/{model_slug}.jpeg",
    f"images/{model_slug}.png"
]

img_path = next((path for path in img_paths if os.path.exists(path)), "images/default.jpg")

st.sidebar.image(img_path, caption=f"Gambar: {model_motor}", use_container_width=True)

# =========================
# Tabs Layout
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data", "📈 Evaluasi", "📉 Visualisasi", "🔮 Prediksi", "🗂️ Riwayat"
])

with tab1:
    st.subheader("📋 Dataset Motor Bekas")
    st.dataframe(df)

with tab2:
    st.subheader("📈 Evaluasi Model")
    st.metric("Mean Absolute Error", f"Rp {mae:,.0f}")
    st.write("Model: Random Forest Regressor")

with tab3:
    st.subheader("📉 Distribusi Harga")
    fig, ax = plt.subplots()
    sns.histplot(df['price'], bins=10, kde=True, ax=ax)
    st.pyplot(fig)

with tab4:
    st.subheader("🔮 Prediksi Harga Motor")
    st.image(img_path, caption=f"Gambar: {model_motor}", use_container_width=True)

    if st.button("Prediksi Sekarang"):
        harga = model.predict(input_encoded)[0]
        st.success(f"💰 Perkiraan Harga: Rp {harga:,.0f}")

        hasil_log = input_df.copy()
        hasil_log["predicted_price"] = harga
        try:
            existing = pd.read_csv("riwayat_prediksi.csv")
            hasil_log = pd.concat([existing, hasil_log], ignore_index=True)
        except FileNotFoundError:
            pass
        hasil_log.to_csv("riwayat_prediksi.csv", index=False)
        st.info("✅ Data prediksi telah disimpan ke riwayat_prediksi.csv")

        # WhatsApp
        pesan = f"""Halo Admin, saya ingin menanyakan harga motor bekas:

📌 Merek: {brand}
📌 Model: {model_motor}
📅 Tahun: {year}
🛣️ Jarak Tempuh: {km * 1000} km
💨 CC: {cc}
✅ Kondisi: {condition}

💰 Estimasi Harga: Rp {harga:,.0f}

Dikirim dari aplikasi Prediksi Harga Motor Beb.
"""
        encoded = urllib.parse.quote(pesan)
        wa_link = f"https://wa.me/6281234567890?text={encoded}"

        st.markdown(f"""
        <a href="{wa_link}" target="_blank">
            <button style='background-color:#25D366;color:white;padding:10px;border:none;border-radius:5px;font-size:16px'>
                📲 Kirim ke WhatsApp
            </button>
        </a>
        """, unsafe_allow_html=True)

with tab5:
    st.subheader("🗂️ Riwayat Prediksi Harga Motor")
    try:
        riwayat_df = pd.read_csv("riwayat_prediksi.csv")
        st.dataframe(riwayat_df, use_container_width=True)
    except FileNotFoundError:
        st.warning("Belum ada riwayat prediksi.")

st.markdown("---")
st.caption("🚀 Dibuat dengan ❤️ oleh Bos Beb")

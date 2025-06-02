import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Prediksi Harga Motor", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("motor.csv")

df = load_data()

# Encode
df_encoded = df.copy()
le = LabelEncoder()
for col in ['brand', 'model', 'condition']:
    df_encoded[col] = le.fit_transform(df[col])

X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
mae = mean_absolute_error(y_test, predicted)

# Sidebar input
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
    input_encoded[col] = le.transform(input_encoded[col])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "📈 Evaluasi", "📉 Visualisasi", "🔮 Prediksi"])

with tab1:
    st.subheader("📋 Data Motor")
    st.dataframe(df)

with tab2:
    st.subheader("📈 Evaluasi Model")
    st.metric("Mean Absolute Error", f"Rp {mae:,.0f}")
    st.write("Model: Random Forest Regressor")

with tab3:
    st.subheader("📉 Distribusi Harga Motor")
    fig, ax = plt.subplots()
    sns.histplot(df['price'], bins=10, kde=True, ax=ax)
    st.pyplot(fig)

with tab4:
    st.subheader("🔮 Prediksi Harga Motor Bekas")
    if st.button("Prediksi Sekarang"):
        harga = model.predict(input_encoded)[0]
        st.success(f"💰 Perkiraan Harga: Rp {harga:,.0f}")

st.markdown("---")
st.caption("Dibuat oleh Beb 🔧 dengan Streamlit 🚀")

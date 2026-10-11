import json, os
import pandas as pd
import streamlit as st
import joblib
from features import add_features, RAW_COLUMNS   # pickle ko ye import chahiye

st.set_page_config(page_title="Seattle House Price Predictor", page_icon="🏠", layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load("model.pkl")

bundle = load_bundle()
model, low, high = bundle["model"], bundle["low"], bundle["high"]

st.title("🏠 Seattle House Price Predictor")
st.caption(f"Model: {bundle['best_name']}  |  Predicts price with a likely range")

tab1, tab2 = st.tabs(["Predict", "About the model"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        bedrooms = st.number_input("Bedrooms", 1, 10, 3)
        bathrooms = st.number_input("Bathrooms", 0.5, 8.0, 2.0, step=0.25)
        sqft_living = st.number_input("Living area (sqft)", 300, 12000, 2000)
        sqft_lot = st.number_input("Lot size (sqft)", 500, 100000, 7000)
        floors = st.number_input("Floors", 1.0, 3.5, 1.0, step=0.5)
    with c2:
        waterfront = st.selectbox("Waterfront", [0, 1])
        view = st.slider("View (0-4)", 0, 4, 0)
        condition = st.slider("Condition (1-5)", 1, 5, 3)
        grade = st.slider("Grade (1-13)", 1, 13, 7)
        sqft_above = st.number_input("Sqft above ground", 300, 10000, 1500)
    with c3:
        sqft_basement = st.number_input("Sqft basement", 0, 5000, 0)
        yr_built = st.number_input("Year built", 1900, 2015, 1990)
        yr_renovated = st.number_input("Year renovated (0 = never)", 0, 2015, 0)
        zipcode = st.number_input("Zipcode", 98001, 98199, 98103)
        lat = st.number_input("Latitude", 47.15, 47.78, 47.56, format="%.4f")
        long = st.number_input("Longitude", -122.52, -121.31, -122.21, format="%.4f")

    if st.button("Predict Price", type="primary"):
        row = {
            "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living,
            "sqft_lot": sqft_lot, "floors": floors, "waterfront": waterfront,
            "view": view, "condition": condition, "grade": grade,
            "sqft_above": sqft_above, "sqft_basement": sqft_basement,
            "yr_built": yr_built, "yr_renovated": yr_renovated, "zipcode": zipcode,
            "lat": lat, "long": long,
            "sqft_living15": sqft_living, "sqft_lot15": sqft_lot,   # padosi ghar ka avg: simple default
            "sale_year": 2015, "sale_month": 6,
        }
        X_new = pd.DataFrame([row])[RAW_COLUMNS]
        price = float(model.predict(X_new)[0])
        p_low = min(float(low.predict(X_new)[0]), price)
        p_high = max(float(high.predict(X_new)[0]), price)

        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated price", f"${price:,.0f}")
        m2.metric("Likely low", f"${p_low:,.0f}")
        m3.metric("Likely high", f"${p_high:,.0f}")
        st.map(pd.DataFrame({"lat": [lat], "lon": [long]}))

with tab2:
    if os.path.exists("metrics.json"):
        m = json.load(open("metrics.json"))
        a, b, c = st.columns(3)
        a.metric("R² (test)", m["r2"])
        b.metric("Avg. error (MAE)", f"${m['mae']:,}")
        c.metric("CV R² (mean)", m["cv_r2_mean"])
    if os.path.exists("results.csv"):
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv("results.csv"), hide_index=True)
    for img, cap in [("images/feature_importance.png", "What drives the price"),
                     ("images/actual_vs_predicted.png", "Actual vs predicted")]:
        if os.path.exists(img):
            st.image(img, caption=cap)
    st.caption("Data: King County (Seattle) house sales 2014-15. Prices in USD.")
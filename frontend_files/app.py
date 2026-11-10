# Streamlit frontend for the SuperKart sales forecasting API
import io
import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")

st.set_page_config(page_title="SuperKart Sales Forecaster", page_icon="🛒", layout="centered")
st.title("🛒 SuperKart Sales Forecaster")
st.caption("Predict the total revenue of a product in a given store.")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction (CSV)"])

with tab_single:
    st.subheader("Product")
    c1, c2 = st.columns(2)
    product_id_char = c1.selectbox("Product family (ID prefix)", ["FD", "NC", "DR"],
                                   help="FD = food, NC = non-consumable, DR = drinks")
    product_category = c2.selectbox("Product category", ["Perishables", "Non Perishables"])
    sugar = c1.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
    weight = c2.number_input("Product weight", min_value=0.1, max_value=50.0, value=12.5, step=0.1)
    area = c1.number_input("Allocated display area (ratio)", min_value=0.0, max_value=1.0, value=0.06, step=0.001, format="%.3f")
    mrp = c2.number_input("Product MRP", min_value=1.0, max_value=1000.0, value=147.0, step=1.0)

    st.subheader("Store")
    c3, c4 = st.columns(2)
    store_type = c3.selectbox("Store type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
    store_size = c4.selectbox("Store size", ["Small", "Medium", "High"])
    city = c3.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
    age = c4.number_input("Store age (years)", min_value=0, max_value=100, value=16, step=1)

    if st.button("Predict sales", type="primary"):
        payload = {
            "Product_Weight": weight,
            "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": area,
            "Product_MRP": mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": int(age),
            "Product_Type_Category": product_category,
        }
        try:
            resp = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            if resp.ok:
                st.metric("Predicted sales total", f"{resp.json()['Predicted_Sales']:,.2f}")
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")

with tab_batch:
    st.write("Upload a CSV with the columns: Product_Weight, Product_Sugar_Content, Product_Allocated_Area, Product_MRP, "
             "Store_Size, Store_Location_City_Type, Store_Type, Product_Id_char, Store_Age_Years, Product_Type_Category.")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None and st.button("Predict for all rows"):
        try:
            resp = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": (file.name, file.getvalue(), "text/csv")}, timeout=120)
            if resp.ok:
                preds = pd.Series(resp.json()).rename("Predicted_Sales")
                preds.index = preds.index.astype(int)
                result = pd.read_csv(io.BytesIO(file.getvalue())).join(preds.sort_index())
                st.dataframe(result, use_container_width=True)
                st.metric("Total predicted sales", f"{result['Predicted_Sales'].sum():,.2f}")
                st.download_button("Download predictions", result.to_csv(index=False).encode("utf-8"),
                                   "superkart_predictions.csv", "text/csv")
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")

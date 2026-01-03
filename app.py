import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

st.set_page_config(page_title="Visualisasi Orde Reaksi", layout="centered")

st.title("Visualisasi Penentuan Orde Reaksi")
st.write("Upload data waktu dan absorbansi (CSV)")

uploaded_file = st.file_uploader(
    "Upload file CSV (kolom: waktu, absorbansi)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Input")
    st.dataframe(df)

    t = df.iloc[:, 0].values
    A = df.iloc[:, 1].values

    results = []

    # ORDE 0
    slope0, intercept0, r0, _, _ = linregress(t, A)
    results.append(("Orde 0", r0, r0**2, abs(slope0)))

    # ORDE 1
    lnA = np.log(A)
    slope1, intercept1, r1, _, _ = linregress(t, lnA)
    results.append(("Orde 1", r1, r1**2, abs(slope1)))

    # ORDE 2
    invA = 1 / A
    slope2, intercept2, r2, _, _ = linregress(t, invA)
    results.append(("Orde 2", r2, r2**2, abs(slope2)))

    result_df = pd.DataFrame(
        results,
        columns=["Orde Reaksi", "r", "R²", "k"]
    )

    st.subheader("Hasil Analisis Kinetika")
    st.dataframe(result_df)

    best_order = result_df.loc[result_df["R²"].idxmax()]
    st.success(f"Reaksi mengikuti kinetika **{best_order['Orde Reaksi']}**")

    st.write(
        f"Model dengan R² tertinggi ({best_order['R²']:.3f}) "
        "menunjukkan linearitas terbaik terhadap data eksperimen."
    )


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Visualisasi Orde Reaksi",
    layout="centered"
)

st.title("Visualisasi Penentuan Orde Reaksi")
st.write(
    """
    Aplikasi ini menentukan orde reaksi berdasarkan data waktu dan absorbansi.
    Analisis dilakukan dengan membandingkan linearitas model orde 0, 1, dan 2
    menggunakan nilai koefisien korelasi (r) dan determinasi (R²).
    """
)

st.info(
    "Asumsi: absorbansi berbanding lurus dengan konsentrasi (Hukum Lambert–Beer)."
)

# =========================
# UPLOAD DATA
# =========================
uploaded_file = st.file_uploader(
    "Upload file CSV (kolom 1 = waktu, kolom 2 = absorbansi)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Eksperimen")
    st.dataframe(df)

    t = df.iloc[:, 0].values
    A = df.iloc[:, 1].values

    # Validasi data
    if np.any(A <= 0):
        st.error("Absorbansi harus bernilai positif untuk analisis orde 1 dan 2.")
        st.stop()

    results = []

    # =========================
    # ORDE 0
    # =========================
    coef0 = np.polyfit(t, A, 1)
    pred0 = np.polyval(coef0, t)
    r0 = np.corrcoef(A, pred0)[0, 1]
    k0 = abs(coef0[0])
    results.append(["Orde 0", r0, r0**2, k0])

    # =========================
    # ORDE 1
    # =========================
    lnA = np.log(A)
    coef1 = np.polyfit(t, lnA, 1)
    pred1 = np.polyval(coef1, t)
    r1 = np.corrcoef(lnA, pred1)[0, 1]
    k1 = abs(coef1[0])
    results.append(["Orde 1", r1, r1**2, k1])

    # =========================
    # ORDE 2
    # =========================
    invA = 1 / A
    coef2 = np.polyfit(t, invA, 1)
    pred2 = np.polyval(coef2, t)
    r2 = np.corrcoef(invA, pred2)[0, 1]
    k2 = abs(coef2[0])
    results.append(["Orde 2", r2, r2**2, k2])

    # =========================
    # TABEL HASIL
    # =========================
    result_df = pd.DataFrame(
        results,
        columns=["Orde Reaksi", "r", "R²", "k"]
    )

    st.subheader("Hasil Analisis Kinetika")
    st.dataframe(result_df.style.format({
        "r": "{:.3f}",
        "R²": "{:.3f}",
        "k": "{:.5f}"
    }))

    # =========================
    # PENENTUAN ORDE TERBAIK
    # =========================
    best = result_df.loc[result_df["R²"].idxmax()]

    st.success(
        f"Orde reaksi yang paling sesuai adalah **{best['Orde Reaksi']}** "
        f"(R² = {best['R²']:.3f})."
    )

    # =========================
    # INTERPRETASI OTOMATIS
    # =========================
    if best["Orde Reaksi"] == "Orde 0":
        interpretation = (
            "Reaksi mengikuti kinetika orde nol, yang menunjukkan bahwa laju reaksi "
            "tidak bergantung pada konsentrasi reaktan. Kondisi ini dapat terjadi "
            "ketika salah satu reaktan berada dalam jumlah berlebih."
        )
    elif best["Orde Reaksi"] == "Orde 1":
        interpretation = (
            "Reaksi mengikuti kinetika orde satu, yang menunjukkan bahwa laju reaksi "
            "berbanding lurus dengan konsentrasi reaktan. Model ini umum ditemukan "
            "pada reaksi degradasi senyawa pangan dan oksidasi."
        )
    else:
        interpretation = (
            "Reaksi mengikuti kinetika orde dua, yang menunjukkan bahwa laju reaksi "
            "bergantung pada kuadrat konsentrasi reaktan atau interaksi dua spesies "
            "reaktan."
        )

    st.write("### Interpretasi")
    st.write(interpretation)

    # =========================
    # VISUALISASI GRAFIK
    # =========================
    st.subheader("Visualisasi Linearitas Tiap Orde")

    fig, axs = plt.subplots(3, 1, figsize=(6, 12))

    # Orde 0 plot
    axs[0].scatter(t, A)
    axs[0].plot(t, pred0)
    axs[0].set_title("Orde 0: Absorbansi vs Waktu")
    axs[0].set_xlabel("Waktu")
    axs[0].set_ylabel("Absorbansi")

    # Orde 1 plot
    axs[1].scatter(t, lnA)
    axs[1].plot(t, pred1)
    axs[1].set_title("Orde 1: ln(Absorbansi) vs Waktu")
    axs[1].set_xlabel("Waktu")
    axs[1].set_ylabel("ln(Absorbansi)")

    # Orde 2 plot
    axs[2].scatter(t, invA)
    axs[2].plot(t, pred2)
    axs[2].set_title("Orde 2: 1/Absorbansi vs Waktu")
    axs[2].set_xlabel("Waktu")
    axs[2].set_ylabel("1/Absorbansi")

    st.pyplot(fig)

else:
    st.warning("Silakan upload file CSV untuk memulai analisis.")

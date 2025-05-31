import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Konfigurasi halaman
st.set_page_config(page_title="🎵 Rekomendasi Musik", layout="wide")

# Fungsi load data
@st.cache_data
def load_data(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File '{csv_path}' tidak ditemukan di folder {os.getcwd()}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    df.columns = df.columns.str.strip().str.lower()
    return df

# Load dataset
data_path = "musik.csv"
try:
    musik_df = load_data(data_path)
except Exception as e:
    st.error(f"❌ Terjadi kesalahan saat memuat data: {e}")
    st.stop()

# Bersihkan data
musik_df.dropna(subset=["judul_musik", "artist", "genre", "tempo", "energy", "danceability"], inplace=True)

# Label encoding untuk genre
label_encoder = LabelEncoder()
musik_df["genre_label"] = label_encoder.fit_transform(musik_df["genre"])

# Split data
fitur = musik_df[["tempo", "energy", "danceability"]]
target = musik_df["genre_label"]
X_train, X_test, y_train, y_test = train_test_split(fitur, target, test_size=0.2, random_state=42)

# Model Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Session state untuk riwayat
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar navigasi
st.sidebar.markdown("<h2 style='color:#6C63FF;'>🔍 Navigasi</h2>", unsafe_allow_html=True)
halaman = st.sidebar.radio("Pilih halaman:", ["Beranda", "Distribusi Musik", "Rekomendasi Musik"])

# ------------------------ HALAMAN BERANDA ------------------------
if halaman == "Beranda":
    st.markdown("<h1 style='color:#6C63FF;'>🎵 Beranda</h1>", unsafe_allow_html=True)

    # Top 10 Musik Populer
    st.markdown("### 🔥 10 Musik Terpopuler")
    if "popularity" in musik_df.columns:
        top10 = musik_df.sort_values(by="popularity", ascending=False).drop_duplicates("judul_musik").head(10)
        st.table(top10[["judul_musik", "artist", "popularity"]])
    else:
        st.warning("Kolom 'popularity' tidak ditemukan.")

    st.divider()

    # Riwayat pencarian
    st.markdown("### 🕘 Riwayat Pencarian")
    if st.session_state.history:
        riwayat_display = []
        for entry in st.session_state.history:
            riwayat_display.append({
                "Judul Musik": entry["judul_input"],
                "Genre Prediksi": entry["genre_prediksi"],
                "Lagu Rekomendasi": ", ".join(entry["rekomendasi"][:5])  # tampilkan maksimal 5 lagu
            })
        df_history = pd.DataFrame(riwayat_display)
        st.table(df_history)
        
        # Tombol hapus riwayat
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.history.clear()
            st.experimental_rerun()
    else:
        st.info("Belum ada pencarian.")

# ------------------------ HALAMAN DISTRIBUSI ------------------------
elif halaman == "Distribusi Musik":
    st.markdown("<h1 style='color:#6C63FF;'>📊 Distribusi Musik</h1>", unsafe_allow_html=True)

    st.markdown("### 🎤 10 Artis Terpopuler")
    top_artists = musik_df["artist"].value_counts().head(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top_artists.values, y=top_artists.index, ax=ax1, palette="coolwarm")
    ax1.set_xlabel("Jumlah Lagu")
    ax1.set_ylabel("Artis")
    st.pyplot(fig1)

    st.markdown("### 🎼 10 Genre Terpopuler")
    top_genres = musik_df["genre"].value_counts().head(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax2, palette="viridis")
    ax2.set_xlabel("Jumlah Lagu")
    ax2.set_ylabel("Genre")
    st.pyplot(fig2)

# ------------------------ HALAMAN REKOMENDASI ------------------------
elif halaman == "Rekomendasi Musik":
    st.markdown("<h1 style='color:#6C63FF;'>🎧 Rekomendasi Musik</h1>", unsafe_allow_html=True)
    st.markdown("Masukkan judul musik untuk menemukan lagu serupa berdasarkan fitur audio dan genre.")

    with st.form(key="form_rekomendasi"):
        judul_input = st.text_input("🎵 Masukkan judul musik:")
        submit = st.form_submit_button("🔍 Cari rekomendasi")

    if submit:
        if not judul_input.strip():
            st.warning("⚠️ Masukkan judul musik yang valid.")
        else:
            hasil = musik_df[musik_df["judul_musik"].str.contains(judul_input, case=False, na=False)]
            if hasil.empty:
                st.error("❌ Judul musik tidak ditemukan.")
            else:
                sampel = hasil.iloc[0]
                fitur_input = [[sampel["tempo"], sampel["energy"], sampel["danceability"]]]
                pred_label = rf.predict(fitur_input)[0]
                pred_genre = label_encoder.inverse_transform([pred_label])[0]

                # Filter lagu lain dari genre yang sama, kecuali lagu yang dicari
                rekomendasi = musik_df[
                    (musik_df["genre"] == pred_genre) &
                    (musik_df["judul_musik"] != sampel["judul_musik"])
                ]
                rekomendasi_sample = rekomendasi[["judul_musik", "artist"]].drop_duplicates().sample(
                    n=min(5, len(rekomendasi)), random_state=42
                )

                st.success(f"✅ Lagu ditemukan! Genre: **{pred_genre}**")
                st.markdown("### 🎯 Rekomendasi Lagu Serupa")
                st.table(rekomendasi_sample)

                # Simpan ke riwayat pencarian dalam session state
                rekom_list = rekomendasi_sample["judul_musik"].tolist()
                st.session_state.history.append({
                    "judul_input": judul_input,
                    "genre_prediksi": pred_genre,
                    "rekomendasi": rekom_list
                })
                if len(st.session_state.history) > 10:
                    st.session_state.history.pop(0)

# ------------------------ FOOTER ------------------------
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 Dibuat dengan ❤️ menggunakan Streamlit</p>",
    unsafe_allow_html=True
)

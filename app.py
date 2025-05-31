import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Rekomendasi Musik", layout="wide")

# Fungsi load data CSV
@st.cache_data
def load_data(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File '{csv_path}' tidak ditemukan di folder {os.getcwd()}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    df.columns = df.columns.str.strip().str.lower()
    return df

data_path = "musik.csv"  # Pastikan file musik.csv ada di folder project

# Load data
try:
    musik_df = load_data(data_path)
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")
    st.stop()

# Bersihkan data (hapus baris dengan NA di kolom penting)
musik_df.dropna(subset=["judul_musik", "artist", "genre", "tempo", "energy", "danceability"], inplace=True)

# Encode genre menjadi label numerik
label_encoder = LabelEncoder()
musik_df["genre_label"] = label_encoder.fit_transform(musik_df["genre"])

# Split data fitur dan target
fitur = musik_df[["tempo", "energy", "danceability"]]
target = musik_df["genre_label"]
X_train, X_test, y_train, y_test = train_test_split(fitur, target, test_size=0.2, random_state=42)

# Latih model Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Inisialisasi riwayat pencarian pada session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar navigasi
st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih halaman:", ["Beranda", "Distribusi Musik", "Rekomendasi Musik"])

if halaman == "Beranda":
    st.title("Beranda")
    st.subheader("10 Musik Terpopuler")
    if "popularity" in musik_df.columns:
        top10 = musik_df.sort_values(by="popularity", ascending=False).drop_duplicates("judul_musik").head(10)
        st.table(top10[["judul_musik", "artist", "popularity"]])
    else:
        st.warning("Kolom 'popularity' tidak ditemukan dalam data.")

    st.subheader("Riwayat Pencarian")
    if st.session_state.history:
        df_history = pd.DataFrame(reversed(st.session_state.history), columns=["Judul Musik", "Genre Rekomendasi"])
        st.table(df_history)

        # Tambahkan hasil pencarian terakhir
        st.subheader("Hasil Pencarian Terakhir")
        judul_terakhir, genre_terakhir = st.session_state.history[-1]
        st.markdown(f"**Judul:** {judul_terakhir}  \n**Genre Rekomendasi:** {genre_terakhir}")

        rekom_terakhir = musik_df[
            (musik_df["genre"] == genre_terakhir) &
            (~musik_df["judul_musik"].str.contains(judul_terakhir, case=False, na=False))
        ].drop_duplicates(subset=["judul_musik", "artist"])

        rekom_sample = rekom_terakhir[["judul_musik", "artist"]].sample(
            n=min(5, len(rekom_terakhir)), random_state=42
        )

        st.markdown("**Rekomendasi Musik Serupa:**")
        st.table(rekom_sample)
    else:
        st.write("Belum ada pencarian.")

elif halaman == "Distribusi Musik":
    st.title("Distribusi Musik")

    st.subheader("10 Artis Terpopuler")
    if "artist" in musik_df.columns:
        top_artists = musik_df["artist"].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_artists.values, y=top_artists.index, ax=ax)
        ax.set_xlabel("Jumlah Lagu")
        ax.set_ylabel("Artis")
        st.pyplot(fig)

    st.subheader("10 Genre Terpopuler")
    if "genre" in musik_df.columns:
        top_genres = musik_df["genre"].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax)
        ax.set_xlabel("Jumlah Lagu")
        ax.set_ylabel("Genre")
        st.pyplot(fig)

elif halaman == "Rekomendasi Musik":
    st.title("Rekomendasi Musik dengan Random Forest")

    with st.form(key="form_rekomendasi"):
        judul_input = st.text_input("Masukkan judul musik:")
        submit = st.form_submit_button("Cari rekomendasi")

    if submit:
        if not judul_input.strip():
            st.warning("Masukkan judul musik yang valid.")
        else:
            hasil = musik_df[musik_df["judul_musik"].str.contains(judul_input, case=False, na=False)]
            if hasil.empty:
                st.error("Judul musik tidak ditemukan.")
            else:
                sampel = hasil.iloc[0]

                # Validasi fitur tidak kosong
                if pd.isnull(sampel[["tempo", "energy", "danceability"]]).any():
                    st.error("Data musik tidak lengkap untuk rekomendasi.")
                else:
                    fitur_input = [[sampel["tempo"], sampel["energy"], sampel["danceability"]]]
                    pred_label = rf.predict(fitur_input)[0]
                    pred_genre = label_encoder.inverse_transform([pred_label])[0]

                    rekomendasi = musik_df[
                        (musik_df["genre"] == pred_genre) &
                        (musik_df["judul_musik"] != sampel["judul_musik"])
                    ]
                    rekomendasi_sample = rekomendasi[["judul_musik", "artist"]].drop_duplicates().sample(
                        n=min(5, len(rekomendasi)), random_state=42
                    )

                    st.write(f"Rekomendasi berdasarkan genre **{pred_genre}**:")
                    st.table(rekomendasi_sample)

                    # Simpan ke riwayat
                    st.session_state.history.append([judul_input, pred_genre])
                    if len(st.session_state.history) > 10:
                        st.session_state.history.pop(0)

"""
Aplikasi Klasifikasi Kategori Judul Berita Detik
UAS Machine Learning 2025
"""
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
MODEL_PATH = "model_klasifikasi_berita.joblib"

# Isi dengan hasil evaluasi test dari notebook 02 (biarkan None jika belum).
AKURASI_TEST = None      # contoh: 0.87
F1_MACRO_TEST = None     # contoh: 0.84

NAMA_KATEGORI = {
    "finance": "Keuangan & Ekonomi",
    "food": "Kuliner",
    "health": "Kesehatan",
    "hot": "Hiburan & Selebritas",
    "inet": "Teknologi & Internet",
    "news": "Berita Umum",
    "oto": "Otomotif",
    "sport": "Olahraga",
    "travel": "Wisata & Perjalanan",
}

CONTOH_JUDUL = [
    "Harga Emas Antam Naik Rp 5.000 per Gram Hari Ini",
    "Timnas Indonesia Menang Telak di Laga Persahabatan",
    "Resep Rendang Daging Empuk yang Mudah Dibuat di Rumah",
    "Samsung Rilis Galaxy Terbaru dengan Kamera 108 MP",
]

# Harus SAMA PERSIS dengan preprocessing di notebook 01.
NORMALISASI = {
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "tdk": "tidak",
    "nggak": "tidak", "gak": "tidak", "ga": "tidak", "krn": "karena",
    "dlm": "dalam", "dg": "dengan", "sbg": "sebagai", "tak": "tidak",
}
SKALA_SKOR = 2.0   # dipakai hanya untuk model tanpa predict_proba (mis. Linear SVM)


# ----------------------------------------------------------------------------
# Pemuatan model dan preprocessing
# ----------------------------------------------------------------------------
@st.cache_resource
def muat_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def muat_nlp():
    stopwords = set(StopWordRemoverFactory().get_stop_words())
    stemmer = StemmerFactory().create_stemmer()
    return stopwords, stemmer


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(judul, stopwords, stemmer, rinci=False):
    """Mengembalikan teks bersih (dan tahapan-tahapannya jika rinci=True)."""
    bersih = clean_text(judul)
    token = [NORMALISASI.get(t, t) for t in bersih.split()]
    token = [t for t in token if t not in stopwords and len(t) > 1]
    hasil = " ".join(stemmer.stem(t) for t in token)
    if rinci:
        return hasil, {"Case folding & cleaning": bersih,
                       "Setelah stopword removal": " ".join(token),
                       "Setelah stemming": hasil}
    return hasil


def hitung_skor(model, X):
    """Skor per kelas. predict_proba jika ada, jika tidak softmax dari decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    d = model.decision_function(X) * SKALA_SKOR
    d = d - d.max(axis=1, keepdims=True)
    e = np.exp(d)
    return e / e.sum(axis=1, keepdims=True)


def prediksi(judul_list, bundle, stopwords, stemmer):
    bersih = [preprocess(j, stopwords, stemmer) for j in judul_list]
    X = bundle["vectorizer"].transform(bersih)
    skor = hitung_skor(bundle["model"], X)
    kelas = bundle["model"].classes_
    return bersih, kelas, skor


# ----------------------------------------------------------------------------
# Tampilan
# ----------------------------------------------------------------------------
def set_contoh(teks):
    st.session_state["judul"] = teks


def tab_prediksi(bundle, stopwords, stemmer):
    st.subheader("Prediksi satu judul")
    st.text_input("Masukkan judul berita:", key="judul",
                  placeholder="Contoh: Harga Emas Antam Naik Rp 5.000 per Gram")

    st.caption("Atau coba salah satu contoh:")
    kolom = st.columns(len(CONTOH_JUDUL))
    for k, teks in zip(kolom, CONTOH_JUDUL):
        k.button(teks[:32] + "…", on_click=set_contoh, args=(teks,), use_container_width=True)

    if st.button("Prediksi kategori", type="primary"):
        judul = st.session_state.get("judul", "").strip()
        if not judul:
            st.warning("Judul masih kosong. Tulis dulu judul beritanya.")
            return
        bersih, rinci = preprocess(judul, *muat_nlp(), rinci=True)
        if not bersih:
            st.warning("Judul tidak memiliki kata bermakna setelah preprocessing. Coba judul yang lebih panjang.")
            return

        _, kelas, skor = prediksi([judul], bundle, stopwords, stemmer)
        skor = skor[0]
        urut = np.argsort(skor)[::-1]
        top = kelas[urut[0]]

        st.success(f"Kategori prediksi: **{NAMA_KATEGORI.get(top, top)}** (`{top}`)")
        c1, c2 = st.columns(2)
        c1.metric("Skor kelas teratas", f"{skor[urut[0]] * 100:.1f}%")
        c2.metric("Selisih dengan kelas kedua",
                  f"{(skor[urut[0]] - skor[urut[1]]) * 100:.1f} poin")

        tabel = pd.DataFrame({
            "Kategori": [NAMA_KATEGORI.get(k, k) for k in kelas[urut]],
            "Skor (%)": np.round(skor[urut] * 100, 1),
        }).set_index("Kategori")
        st.bar_chart(tabel)

        with st.expander("Lihat tahapan preprocessing"):
            for nama, isi in rinci.items():
                st.markdown(f"**{nama}**")
                st.code(isi if isi else "(kosong)")


def tab_batch(bundle, stopwords, stemmer):
    st.subheader("Prediksi banyak judul sekaligus")
    st.write("Unggah file CSV yang memiliki kolom bernama **title**.")
    file = st.file_uploader("File CSV", type="csv")
    if file is None:
        return
    df = pd.read_csv(file)
    if "title" not in df.columns:
        st.error("Kolom `title` tidak ditemukan di file CSV.")
        return
    if len(df) > 2000:
        st.info("File berisi lebih dari 2.000 baris, hanya 2.000 baris pertama yang diproses.")
        df = df.head(2000)

    with st.spinner("Memproses..."):
        _, kelas, skor = prediksi(df["title"].astype(str).tolist(), bundle, stopwords, stemmer)
    df["kategori_prediksi"] = kelas[skor.argmax(axis=1)]
    df["skor_teratas_%"] = (skor.max(axis=1) * 100).round(1)

    st.dataframe(df, use_container_width=True)
    st.download_button("Unduh hasil (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "hasil_prediksi.csv", "text/csv")


def tab_tentang(bundle):
    st.subheader("Tentang aplikasi")
    st.markdown(f"""
Aplikasi ini mengklasifikasikan **judul berita berbahasa Indonesia** ke dalam {len(bundle['labels'])} kategori Detik.

**Model:** {bundle.get('model_name', '-')}, dengan fitur TF-IDF (unigram + bigram)
**Dataset:** judul berita Detik, Januari sampai Juni 2020
**Preprocessing:** case folding, cleaning, normalisasi, stopword removal, stemming (Sastrawi)
""")
    if AKURASI_TEST is not None and F1_MACRO_TEST is not None:
        a, b = st.columns(2)
        a.metric("Accuracy (data test)", f"{AKURASI_TEST:.2%}")
        b.metric("F1 macro (data test)", f"{F1_MACRO_TEST:.2%}")

    st.markdown("**Daftar kategori**")
    st.table(pd.DataFrame({"Kode": bundle["labels"],
                           "Nama": [NAMA_KATEGORI.get(k, k) for k in bundle["labels"]]}).set_index("Kode"))
    st.caption("Skor ditampilkan sebagai ukuran keyakinan relatif model, bukan probabilitas terkalibrasi. "
               "Data dari tahun 2020 (banyak bertema COVID-19), sehingga akurasi bisa menurun untuk berita di luar periode itu.")


def main():
    st.set_page_config(page_title="Klasifikasi Berita Detik", page_icon="📰", layout="centered")
    st.title("📰 Klasifikasi Kategori Judul Berita")
    st.write("Tempel judul berita, dan model akan menebak kategorinya.")

    try:
        bundle = muat_model()
    except FileNotFoundError:
        st.error(f"File model `{MODEL_PATH}` tidak ditemukan. Letakkan di folder yang sama dengan app.py.")
        st.stop()
    stopwords, stemmer = muat_nlp()

    t1, t2, t3 = st.tabs(["Prediksi", "Prediksi Batch (CSV)", "Tentang Model"])
    with t1:
        tab_prediksi(bundle, stopwords, stemmer)
    with t2:
        tab_batch(bundle, stopwords, stemmer)
    with t3:
        tab_tentang(bundle)


if __name__ == "__main__":
    main()

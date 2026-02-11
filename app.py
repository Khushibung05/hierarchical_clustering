import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="News Topic Discovery Dashboard",
    layout="wide"
)

# -----------------------------
# STYLING (YOUR REFERENCE STYLE)
# -----------------------------
st.markdown(
    """
<style>

/* ===== GLOBAL BACKGROUND ===== */
.stApp {
    background: linear-gradient(
        135deg,
        #f0f4ff,
        #e6f7f1,
        #fff7e6
    );
    background-attachment: fixed;
    padding: 100px;
}

/* ===== TITLE SPACING ===== */
h1 {
    margin-top: 2.5rem !important;
}

/* ===== SIDEBAR FIX ===== */
section[data-testid="stSidebar"] {
    height: 100vh;
    overflow-y: auto !important;
    padding-bottom: 2rem;
}

/* ===== MAIN CONTENT CARD ===== */
.block-container {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

/* ===== SIDEBAR GRADIENT ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1f3c88,
        #2a5298,
        #1e3c72
    );
    color: white;
}

/* ===== SIDEBAR LABELS ===== */
section[data-testid="stSidebar"] label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* ===== SLIDER VALUES ===== */
section[data-testid="stSidebar"] .stSlider span {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #ffdddd !important;
}

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
}

/* ===== ALERTS ===== */
div.stAlert-success {
    background: linear-gradient(90deg, #e0f8e9, #c6f6d5);
    border-radius: 10px;
}

div.stAlert-error {
    background: linear-gradient(90deg, #ffe0e0, #ffbdbd);
    border-radius: 10px;
}

/* ===== FOOTER ===== */
footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# APP TITLE & DESCRIPTION
# -----------------------------
st.title("🟣 News Topic Discovery Dashboard")
st.write(
    "This system uses **Hierarchical Clustering** to automatically group similar news articles "
    "based on textual similarity."
)
st.markdown("👉 *Discover hidden themes without defining categories upfront.*")

# -----------------------------
# HELPERS
# -----------------------------

def detect_text_column(df: pd.DataFrame):
    """Try to detect a text column automatically."""
    preferred = [
        "news headline",
        "headline",
        "news",
        "text",
        "sentence",
        "article",
        "content",
    ]

    cols = list(df.columns)
    cols_lower = [str(c).strip().lower() for c in cols]

    # 1) exact preferred matches
    for p in preferred:
        if p in cols_lower:
            return cols[cols_lower.index(p)]

    # 2) first object column
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        return obj_cols[0]

    return cols[0] if cols else None


def safe_series_text(s: pd.Series):
    s = s.fillna("").astype(str)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def top_keywords_per_cluster(X_tfidf, labels, feature_names, top_n=10):
    clusters = np.unique(labels)
    rows = []

    for c in clusters:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue

        # mean tf-idf vector for this cluster
        cluster_mean = X_tfidf[idx].mean(axis=0)

        # convert to 1D numpy array safely
        mean_vec = np.asarray(cluster_mean).ravel()

        top_idx = mean_vec.argsort()[::-1][:top_n]

        keywords = [feature_names[i] for i in top_idx if mean_vec[i] > 0]

        rows.append((int(c), int(len(idx)), ", ".join(keywords)))

    return pd.DataFrame(rows, columns=["Cluster ID", "# Articles", "Top Keywords"])



# -----------------------------
# SIDEBAR: INPUT SECTION
# -----------------------------
st.sidebar.header("🔧 Controls")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload News Dataset (CSV)", type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Text Vectorization")

max_features = st.sidebar.slider(
    "Maximum TF-IDF Features", 100, 2000, 1000, step=50
)

use_stopwords = st.sidebar.checkbox(
    "Use English Stopwords", value=True
)

ngram_choice = st.sidebar.selectbox(
    "N-gram Range (Bonus)",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"],
    index=0,
)

if ngram_choice == "Unigrams":
    ngram_range = (1, 1)
elif ngram_choice == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("🌳 Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"],
    index=0,
)

# For this dashboard, keep metric fixed as requested
metric = st.sidebar.selectbox(
    "Distance Metric",
    ["euclidean"],
    index=0,
)

subset_for_dendrogram = st.sidebar.slider(
    "Number of Articles for Dendrogram",
    20, 200, 120, step=10
)

st.sidebar.markdown("---")

# IMPORTANT: two-stage workflow
btn_dendro = st.sidebar.button("🟦 Generate Dendrogram")

st.sidebar.markdown("---")
st.sidebar.subheader("🟩 Apply Clustering")

k_clusters = st.sidebar.slider(
    "Number of Clusters (K)",
    2, 12, 4, step=1
)

btn_cluster = st.sidebar.button("🟩 Apply Clustering")

# -----------------------------
# DATA LOAD
# -----------------------------

def load_default_kaggle_like():
    """Optional helper if user does not upload. (Kept simple: requires local file path.)"""
    return None


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except:
        df = pd.read_csv(uploaded_file, encoding="latin1")

else:
    df = None

# -----------------------------
# MAIN PANEL
# -----------------------------
if df is None:
    st.warning("Please upload a CSV dataset to begin clustering.")
    st.stop()

st.subheader("📌 Dataset Preview")
st.write("Detected columns:", list(df.columns))
st.dataframe(df.head(8), use_container_width=True)

# -----------------------------
# Detect Text Column
# -----------------------------
text_col_guess = detect_text_column(df)

text_col = st.selectbox(
    "✅ Select the column containing the news text/headlines",
    options=list(df.columns),
    index=list(df.columns).index(text_col_guess) if text_col_guess in df.columns else 0,
)

texts = safe_series_text(df[text_col])

if texts.str.len().mean() < 5:
    st.error(
        "The selected text column seems empty or too short. Please choose the correct column."
    )
    st.stop()

# -----------------------------
# TF-IDF
# -----------------------------
stop_words = "english" if use_stopwords else None

vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    max_features=max_features,
    ngram_range=ngram_range,
)

X = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

st.success(f"TF-IDF created successfully. Shape: {X.shape}")

# -----------------------------
# DENDROGRAM SECTION
# -----------------------------
st.markdown("---")
st.subheader("🌳 Dendrogram (Core)")

st.write(
    "A dendrogram is the **hierarchical clustering tree**. "
    "Look for **large vertical jumps** — they indicate strong topic separation."
)

# cut line slider (optional)
cut_height = st.slider(
    "Optional: Cut line height (visual only)",
    0.0, 25.0, 8.0, 0.5
)

if btn_dendro:
    # subset for dendrogram
    n = min(subset_for_dendrogram, X.shape[0])
    X_sub = X[:n].toarray()

    # SciPy linkage
    # ward expects euclidean
    Z = linkage(X_sub, method=linkage_method)

    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, ax=ax, no_labels=True)

    ax.axhline(y=cut_height, linewidth=3)
    ax.set_title(f"Dendrogram ({linkage_method.title()} Linkage)")
    ax.set_xlabel("Article Index")
    ax.set_ylabel("Distance")

    st.pyplot(fig, use_container_width=True)
else:
    st.info("Click **🟦 Generate Dendrogram** to build and view the hierarchical tree.")

# -----------------------------
# APPLY CLUSTERING
# -----------------------------
st.markdown("---")
st.subheader("🧩 Clustering Results")

if btn_cluster:
    # NOTE:
    # - sklearn AgglomerativeClustering supports metric='euclidean'
    # - ward linkage requires euclidean

    if linkage_method == "ward" and metric != "euclidean":
        st.error("Ward linkage only works with Euclidean distance.")
        st.stop()

    model = AgglomerativeClustering(
        n_clusters=k_clusters,
        linkage=linkage_method,
        metric=metric,
    )

    labels = model.fit_predict(X.toarray())
    df_out = df.copy()
    df_out["Cluster"] = labels

    # -----------------------------
    # VALIDATION SECTION
    # -----------------------------
    st.subheader("📊 Validation (Unsupervised)")

    try:
        sil = silhouette_score(X, labels)
        st.metric("Silhouette Score", f"{sil:.4f}")

        st.write(
            "**How to read it:**  \n"
            "• Close to **1** → clusters are well-separated  \n"
            "• Close to **0** → clusters overlap  \n"
            "• **Negative** → poor clustering"
        )
    except Exception:
        st.warning("Silhouette score could not be computed for this configuration.")

    # -----------------------------
    # PCA VISUALIZATION
    # -----------------------------
    st.markdown("---")
    st.subheader("📌 Cluster Visualization (PCA 2D)")

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X.toarray())

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)

    ax2.set_title("2D Projection of Headlines (PCA)")
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")

    st.pyplot(fig2, use_container_width=False)

    # -----------------------------
    # CLUSTER SUMMARY (BUSINESS VIEW)
    # -----------------------------
    st.markdown("---")
    st.subheader("📋 Cluster Summary (Business View)")

    summary = (
        df_out.groupby("Cluster")
        .size()
        .reset_index(name="# Articles")
        .sort_values("Cluster")
    )

    keywords_df = top_keywords_per_cluster(
        X, labels, feature_names, top_n=10
    )

    summary_full = summary.merge(
        keywords_df,
        left_on="Cluster",
        right_on="Cluster ID",
        how="left",
    ).drop(columns=["Cluster ID"])

    st.dataframe(summary_full, use_container_width=True)

    # -----------------------------
    # REPRESENTATIVE SNIPPET
    # -----------------------------
    st.subheader("🧾 Representative Headlines")

    for c in sorted(df_out["Cluster"].unique()):
        sample = (
            df_out[df_out["Cluster"] == c][text_col]
            .astype(str)
            .head(3)
            .tolist()
        )

        st.write(f"🟣 **Cluster {int(c)}**")
        for s in sample:
            st.write("•", s[:160] + ("..." if len(s) > 160 else ""))

    # -----------------------------
    # BUSINESS INTERPRETATION (HUMAN LANGUAGE)
    # -----------------------------
    st.markdown("---")
    st.subheader("💡 Business Interpretation (Human Language)")

    st.write(
        "Below is a **plain-language** interpretation. "
        "These are *probable themes* based on top keywords and sample headlines."
    )

    # simple heuristic: show keywords as theme
    for _, row in summary_full.iterrows():
        cid = int(row["Cluster"])
        kw = str(row.get("Top Keywords", ""))

        if kw.strip() == "":
            msg = "A mixed set of headlines with no strong repeating terms."
        else:
            msg = f"Articles that frequently mention: {kw.split(',')[0:4]}"

        st.write(f"🟣 **Cluster {cid}:** {msg}")

    # -----------------------------
    # USER GUIDANCE
    # -----------------------------
    st.info(
        "Articles grouped in the same cluster share similar vocabulary and themes. "
        "These clusters can be used for automatic tagging, recommendations, and content organization."
    )

    # download results
    st.download_button(
        "⬇️ Download Clustered CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="news_clusters.csv",
        mime="text/csv",
    )

else:
    st.info("Set your parameters and click **🟩 Apply Clustering** to generate clusters.")

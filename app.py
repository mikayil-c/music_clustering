from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

from clustering import (
    clear_measurement_caches,
    measure_algorithm,
    safe_metric,
)
from config import (
    ARTIST_CANDIDATES,
    FEATURE_LABELS,
    FEATURE_LABELS_TR,
    GENRE_CANDIDATES,
    TITLE_CANDIDATES,
)
from metadata import first_existing_column, readable_song_name

# -----------------------------------------------------------------------------
# App configuration and project metadata
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Music Cluster Lab",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "language" not in st.session_state:
    st.session_state.language = "en"


def t(english: str, turkish: str) -> str:
    return turkish if st.session_state.language == "tr" else english


title_col, language_col = st.columns([8, 1])
with title_col:
    st.title("🎵 Music Cluster Lab")
with language_col:
    if st.button(
        "Türkçe" if st.session_state.language == "en" else "English",
        key="language_toggle",
        use_container_width=True,
    ):
        st.session_state.language = "tr" if st.session_state.language == "en" else "en"
        st.rerun()

feature_labels = (
    FEATURE_LABELS_TR if st.session_state.language == "tr" else FEATURE_LABELS
)
algorithm_options = {
    "kmeans": "K-Means",
    "kmedoids": "K-Medoids",
    "dbscan": "DBSCAN",
    "gmm": "Gaussian Mixture Model (GMM)",
}
scaler_options = {
    "standard": ("StandardScaler", StandardScaler),
    "minmax": ("MinMaxScaler", MinMaxScaler),
    "maxabs": ("MaxAbsScaler", MaxAbsScaler),
}
st.markdown(
    t(
        "An unsupervised learning application that clusters songs by numerical musical features "
        "and helps discover similar tracks.",
        "Şarkıları sayısal müzikal özelliklerine göre kümeleyen ve benzer parçaları keşfetmeye yardımcı olan "
        "denetimsiz öğrenme uygulaması.",
    )
)

# -----------------------------------------------------------------------------
# File upload is in the main page on purpose, so it is easy to find.
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    t("Upload a song dataset CSV to get started", "Başlamak için şarkı veri setinin CSV dosyasını yükle"),
    type=["csv"],
    help=t(
        "Use a CSV containing Spotify audio features. The file must contain at least two numerical musical features.",
        "Spotify ses özelliklerini içeren bir CSV kullan. Dosyada en az iki sayısal müzikal özellik bulunmalı.",
    ),
)

if uploaded_file is None:
    st.info(
        t(
            "**No dataset has been uploaded yet.** Select a CSV file here. "
            "Recommended columns: danceability, energy, tempo, acousticness, "
            "instrumentalness, valence, loudness, speechiness, and liveness.",
            "**Henüz veri seti yüklenmedi.** CSV dosyanı bu alandan seç. "
            "Önerilen sütunlar: danceability, energy, tempo, acousticness, "
            "instrumentalness, valence, loudness, speechiness ve liveness.",
        )
    )
    with st.expander(t("Example expected data structure", "Beklenen veri yapısı örneği")):
        st.dataframe(
            pd.DataFrame(
                {
                    "track_name": ["Example Song A", "Example Song B"],
                    "artists": ["Artist A", "Artist B"],
                    "danceability": [0.72, 0.31],
                    "energy": [0.88, 0.22],
                    "tempo": [128.0, 76.0],
                    "acousticness": [0.05, 0.91],
                    "valence": [0.68, 0.24],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file, low_memory=False)
except Exception as exc:
    st.error(t(f"Could not read the CSV file: {exc}", f"CSV dosyası okunamadı: {exc}"))
    st.stop()

if raw_df.empty:
    st.error(t("The file contains no data rows.", "Dosyada veri satırı bulunamadı."))
    st.stop()

available_feature_names = [feature for feature in FEATURE_LABELS if feature in raw_df.columns]

if len(available_feature_names) < 2:
    st.error(
        t(
            "At least two musical features could not be found in the CSV file. Check the expected column names.",
            "CSV dosyasında en az iki müzikal özellik bulunamadı. Beklenen sütun adlarını kontrol et.",
        )
    )
    st.write(t("Detected columns:", "Algılanan sütunlar:"), raw_df.columns.tolist())
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header(t("1. Musical features", "1. Müzikal özellikler"))
selected_features = st.sidebar.multiselect(
    t("Features used for clustering", "Kümelemede kullanılacak özellikler"),
    options=available_feature_names,
    default=available_feature_names,
    format_func=lambda x: feature_labels.get(x, x),
    help=t(
        "Label and metadata fields such as song name, artist, genre, and popularity are not passed to the model.",
        "Şarkı adı, sanatçı, tür ve popülerlik gibi etiket/metaveri alanları modele verilmez.",
    ),
)

if len(selected_features) < 2:
    st.warning(t("Select at least two musical features for clustering.", "Kümeleme için en az iki müzikal özellik seçmelisin."))
    st.stop()

sample = raw_df.reset_index(drop=True)
if len(sample) < 3:
    st.error(t("At least three songs are required for clustering.", "Kümeleme için en az üç şarkı gerekli."))
    st.stop()

sample_size = len(sample)
st.info(
    t(
        f"All records in the uploaded cleaned dataset will be used: **{sample_size:,} songs**.",
        f"Yüklenen temiz veri setindeki tüm kayıtlar kullanılacak: **{sample_size:,} şarkı**.",
    )
)

st.sidebar.header(t("2. Scaling and clustering", "2. Ölçeklendirme ve kümeleme"))
algorithm = st.sidebar.selectbox(
    t("Clustering algorithm", "Kümeleme algoritması"),
    options=list(algorithm_options),
    format_func=algorithm_options.get,
    help=t(
        "DBSCAN finds density-based clusters and can separate noise. GMM models each cluster with a Gaussian distribution.",
        "DBSCAN yoğunluk tabanlı kümeler bulur ve gürültüyü ayırabilir. GMM, her kümeyi Gauss dağılımıyla modeller.",
    ),
)
algorithm_label = algorithm_options[algorithm]

n_clusters_max = min(10, sample_size)
if algorithm == "dbscan":
    eps = st.sidebar.slider(
        "DBSCAN eps",
        min_value=0.05,
        max_value=5.0,
        value=0.5,
        step=0.05,
        help=t(
            "Neighborhood radius. Higher values can produce larger clusters and fewer noise points.",
            "Komşuluk yarıçapı. Daha yüksek değerler daha büyük ve daha az sayıda küme oluşturabilir.",
        ),
    )
    min_samples = st.sidebar.slider(
        t("DBSCAN minimum samples", "DBSCAN minimum örnek sayısı"),
        min_value=2,
        max_value=min(20, sample_size),
        value=min(5, sample_size),
        help=t(
            "Number of neighbors required for a point to be considered a core sample.",
            "Bir noktanın çekirdek örnek sayılması için gereken komşu sayısı.",
        ),
    )
    n_clusters = min(2, sample_size)
else:
    eps = 0.5
    min_samples = 5
    n_clusters = st.sidebar.slider(
        t(
            f"{algorithm_label} number of clusters (k)",
            f"{algorithm_label} küme sayısı (k)",
        ),
        min_value=1,
        max_value=max(1, n_clusters_max),
        value=min(4, max(1, n_clusters_max)),
        help=t(
            "The k value can be 1, but Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics cannot be calculated for a single cluster.",
            "K değeri 1 olabilir; ancak tek kümede Silhouette, Davies-Bouldin ve Calinski-Harabasz metrikleri hesaplanamaz.",
        ),
    )

model_features = selected_features
X_features = sample[model_features].to_numpy(dtype=float)
scaler_key = st.sidebar.selectbox(
    t("Scaling method", "Ölçeklendirme yöntemi"),
    options=list(scaler_options),
    format_func=lambda key: scaler_options[key][0],
    help=t(
        "Scaling balances the effect of features with different units on the model.",
        "Ölçeklendirme, farklı ölçü birimlerine sahip özelliklerin model üzerindeki etkisini dengeler.",
    ),
)
scaler_label, scaler_class = scaler_options[scaler_key]
X_scaled = scaler_class().fit_transform(X_features)

if st.sidebar.button(
    t("Measure again", "Yeniden ölç"),
    help=t(
        "Clear cached results for the selected algorithm and run the measurement again.",
        "Seçili algoritmanın cache'lenmiş sonuçlarını siler ve ölçümü tekrar çalıştırır.",
    ),
):
    clear_measurement_caches()

# Metadata labels are kept outside the model and used only for external validation.
genre_col = first_existing_column(list(sample.columns), GENRE_CANDIDATES)
reference_labels = (
    sample[genre_col].astype("string").to_numpy() if genre_col is not None else None
)

with st.spinner(t(f"Measuring {algorithm_label}...", f"{algorithm_label} ölçülüyor...")):
    (
        labels,
        elapsed,
        metrics,
        external_metrics,
        cluster_names,
        k_evaluation_df,
    ) = measure_algorithm(
        X_scaled,
        model_features,
        int(n_clusters),
        reference_labels,
        algorithm,
        n_clusters_max,
        eps=eps,
        min_samples=min_samples,
    )

# Build the enriched result frame.
result_df = sample.copy()
result_df["cluster_id"] = labels
result_df["cluster_name"] = [cluster_names[int(x)] for x in labels]

# Two-dimensional projection is for exploration, not for clustering itself.
pca = PCA(n_components=2, random_state=42)
coordinates = pca.fit_transform(X_scaled)
result_df["PCA 1"] = coordinates[:, 0]
result_df["PCA 2"] = coordinates[:, 1]

# Detect metadata columns, after standardized header names.
title_col = first_existing_column(list(result_df.columns), TITLE_CANDIDATES)
artist_col = first_existing_column(list(result_df.columns), ARTIST_CANDIDATES)

# -----------------------------------------------------------------------------
# High-level summary
# -----------------------------------------------------------------------------
metric_cols = st.columns(7)
metric_cols[0].metric(t("Songs analyzed", "Analiz edilen şarkı"), f"{len(result_df):,}")
metric_cols[1].metric(t("Musical features", "Müzikal özellik"), len(model_features))
metric_cols[2].metric(t("Clusters found", "Bulunan küme"), metrics["cluster_count"])
metric_cols[3].metric("Silhouette Score", safe_metric(metrics["silhouette"]))
metric_cols[4].metric("Davies–Bouldin", safe_metric(metrics["davies_bouldin"]))
metric_cols[5].metric("Calinski–Harabasz", safe_metric(metrics["calinski_harabasz"]))
metric_cols[6].metric(t("Model runtime", "Model süresi"), f"{elapsed:.3f} {'sec' if st.session_state.language == 'en' else 'sn'}")
if algorithm == "dbscan" and 0 in labels:
    st.caption(t(
        f"DBSCAN marked {int(np.sum(labels == 0)):,} songs as noise.",
        f"DBSCAN {int(np.sum(labels == 0)):,} şarkıyı gürültü olarak işaretledi.",
    ))

st.caption(
    t(
        "Note: Cluster numbers are not genre labels. Automatically generated profile descriptions "
        "summarize the features that stand out in each cluster for the selected sample.",
        "Not: Küme numaraları tür etiketleri değildir. Otomatik profil ifadeleri, "
        "her kümenin seçilen veri örnekleminde öne çıkan özelliklerini özetler.",
    )
)

song_count_label = t("Song count", "Şarkı sayısı")
feature_label = t("Feature", "Özellik")
value_label = t("Value", "Değer")
genre_label = t("Genre", "Tür")
duration_label = t("Duration (sec)", "Süre (sn)")
k_value_label = t("K value", "K değeri")

# Tabs for a research-style workflow.
tab_overview, tab_dataset, tab_map, tab_profiles, tab_discovery = st.tabs(
    [
        t("Overview", "Genel bakış"),
        t("Dataset", "Veri seti"),
        t("Cluster map", "Küme haritası"),
        t("Cluster profiles", "Küme profilleri"),
        t("Song discovery", "Şarkı keşfi"),
    ]
)

with tab_overview:
    st.subheader(t(f"{algorithm_label} results summary", f"{algorithm_label} sonuç özeti"))
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(t("**Cluster sizes**", "**Kümelerin büyüklükleri**"))
        cluster_sizes = (
            result_df.groupby(["cluster_id", "cluster_name"], dropna=False)
            .size()
            .reset_index(name=song_count_label)
            .sort_values(song_count_label, ascending=False)
        )
        if not cluster_sizes.empty:
            fig_sizes = px.bar(
                cluster_sizes,
                x="cluster_name",
                y=song_count_label,
                color="cluster_name",
                title=t("Song count by cluster", "Kümelerdeki şarkı sayısı"),
                labels={"cluster_name": t("Cluster profile", "Küme profili")},
            )
            fig_sizes.update_layout(showlegend=False, xaxis_tickangle=-25)
            st.plotly_chart(fig_sizes, use_container_width=True)
        st.dataframe(cluster_sizes, use_container_width=True, hide_index=True)
    with right:
        st.markdown(t("**How to read these results**", "**Bu sonuçlar nasıl okunmalı?**"))
        st.write(
            t(
                f"{algorithm_label} assigns each song to a cluster according to the selected k value. Evaluate cluster profiles and actual song examples together with the metrics.",
                f"{algorithm_label}, her şarkıyı seçilen k değerine göre bir kümeye atar. Küme profillerini ve gerçek şarkı örneklerini metriklerle birlikte değerlendir.",
            )
        )
        st.markdown(t("**Selected model metrics**", "**Seçilen modelin metrikleri**"))
        st.write(f"- Silhouette Score: **{safe_metric(metrics['silhouette'])}**")
        st.write(t(
            f"- Davies–Bouldin Index: **{safe_metric(metrics['davies_bouldin'])}** (lower is generally better)",
            f"- Davies–Bouldin Index: **{safe_metric(metrics['davies_bouldin'])}** (düşük olması genellikle iyidir)",
        ))
        st.write(t(
            f"- Calinski–Harabasz Score: **{safe_metric(metrics['calinski_harabasz'])}** (higher is generally better)",
            f"- Calinski–Harabasz Score: **{safe_metric(metrics['calinski_harabasz'])}** (yüksek olması genellikle iyidir)",
        ))
        st.write(t(f"- Measured model runtime: **{elapsed:.3f} sec**", f"- Modelin ölçülen çalışma süresi: **{elapsed:.3f} sn**"))
        if genre_col is not None:
            st.markdown(t("**External validation (using genre labels)**", "**Dış doğrulama (tür etiketleriyle)**"))
            st.write(
                f"- Adjusted Rand Index: **{safe_metric(external_metrics['adjusted_rand'])}**"
            )
            st.write(
                f"- Normalized Mutual Information: **{safe_metric(external_metrics['normalized_mutual_info'])}**"
            )
            st.write(
                f"- Homogeneity: **{safe_metric(external_metrics['homogeneity'])}**"
            )
            st.write(
                f"- Completeness: **{safe_metric(external_metrics['completeness'])}**"
            )
            st.write(f"- V-measure: **{safe_metric(external_metrics['v_measure'])}**")
            st.caption(
                t(
                    "These metrics measure how closely cluster labels match genre labels. Because genre information is not provided to the model, this is an independent check.",
                    "Bu metrikler, küme etiketlerinin tür etiketleriyle ne kadar örtüştüğünü ölçer. Tür bilgisi modele verilmediği için bağımsız bir kontrol sağlar.",
                )
            )
        st.info(
            t(
                "These metrics evaluate cluster geometry and do not by themselves prove the most musically meaningful clustering. Also inspect the cluster profiles and actual song examples.",
                "Bu ölçütler küme geometrisini değerlendirir; müzikal açıdan en anlamlı kümelemeyi tek başlarına kanıtlamaz. Küme profillerini ve gerçek şarkı örneklerini de incele.",
            )
        )

with tab_dataset:
    st.subheader(t("Dataset details", "Veri seti özellikleri"))
    info_cols = st.columns(3)
    info_cols[0].metric(t("Row count", "Satır sayısı"), f"{len(sample):,}")
    info_cols[1].metric(t("Total columns", "Toplam sütun"), len(sample.columns))
    info_cols[2].metric(t("Musical features used", "Kullanılan müzikal özellik"), len(model_features))

    st.markdown(t("**Outlier overview**", "**Aykırı değer görünümü**"))
    outlier_df = sample[model_features].melt(
        var_name=feature_label,
        value_name=value_label,
    )
    outlier_fig = px.box(
        outlier_df,
        x=feature_label,
        y=value_label,
        points="outliers",
        title=t("Musical feature distributions and outliers", "Müzikal özelliklerin dağılımı ve aykırı değerleri"),
        labels={value_label: value_label},
    )
    outlier_fig.update_xaxes(tickangle=-35)
    st.plotly_chart(outlier_fig, use_container_width=True)
    st.caption(
        t(
            "Points outside a feature's box range are statistical outliers. These records are not removed automatically.",
            "Box-plot üzerindeki noktalar, ilgili özelliğin kutu aralığı dışındaki istatistiksel aykırı değerlerdir. Bu kayıtlar otomatik olarak silinmez.",
        )
    )

    if genre_col is None:
        st.info(t("No genre information was found in the CSV file.", "CSV dosyasında tür bilgisi bulunamadı."))
    else:
        genre_counts = (
            sample[genre_col]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis(genre_label)
            .reset_index(name=song_count_label)
        )
        st.markdown(t(f"**Genre distribution (`{genre_col}`)**", f"**Tür dağılımı (`{genre_col}`)**"))
        st.metric(t("Genre count", "Tür sayısı"), len(genre_counts))
        genre_fig = px.bar(
            genre_counts,
            x=genre_label,
            y=song_count_label,
            title=t("Song count by genre", "Türlere göre şarkı sayıları"),
        )
        genre_fig.update_xaxes(tickangle=-35)
        st.plotly_chart(genre_fig, use_container_width=True)
        st.dataframe(genre_counts, use_container_width=True, hide_index=True)

    if algorithm == "dbscan":
        st.info(t(
            "K-value comparisons are not performed for DBSCAN; results depend on eps and minimum samples.",
            "DBSCAN için k değerleri karşılaştırması yapılmaz; sonuçlar eps ve minimum örnek sayısına bağlıdır.",
        ))
    else:
        st.markdown(t(f"**{algorithm_label} evaluation by k value**", f"**{algorithm_label} için k değerlerine göre değerlendirme**"))
        st.write(
            t(
                f"The charts below compare {algorithm_label} results from k=1 to k using musical features transformed with {scaler_label}.",
                f"Aşağıdaki grafikler, {scaler_label} uygulanmış müzikal özelliklerle k=1 ile k arasındaki {algorithm_label} sonuçlarını karşılaştırır.",
            )
        )
    chart_data = [
        ("Objective", t("Clustering objective", "Kümeleme amacı"), t("Total distance", "Toplam uzaklık")),
        ("Silhouette Score", "Silhouette Score", "Silhouette Score"),
        ("Davies–Bouldin Index", "Davies–Bouldin Index", "Davies–Bouldin Index"),
        ("Calinski–Harabasz Score", "Calinski–Harabasz Score", "Calinski–Harabasz Score"),
    ]
    if genre_col is not None:
        chart_data.extend(
            [
                ("Adjusted Rand Index", "Adjusted Rand Index", "ARI"),
                (
                    "Normalized Mutual Information",
                    "Normalized Mutual Information",
                    "NMI",
                ),
                ("V-measure", "V-measure", "V-measure"),
            ]
        )
    evaluation_display_df = k_evaluation_df.rename(
        columns={"K value": k_value_label, "Duration (sec)": duration_label}
    )
    if algorithm != "dbscan":
        for metric_name, title, y_label in chart_data:
            metric_df = evaluation_display_df.dropna(subset=[metric_name])
            st.plotly_chart(
                px.line(
                    metric_df,
                    x=k_value_label,
                    y=metric_name,
                    markers=True,
                    title=title,
                    labels={metric_name: y_label},
                ),
                use_container_width=True,
            )
        st.dataframe(
            evaluation_display_df.style.format(
                {
                    "Objective": "{:.1f}",
                    "Silhouette Score": "{:.3f}",
                    "Davies–Bouldin Index": "{:.3f}",
                    "Calinski–Harabasz Score": "{:.1f}",
                    duration_label: "{:.3f}",
                    "Adjusted Rand Index": "{:.3f}",
                    "Normalized Mutual Information": "{:.3f}",
                    "Homogeneity": "{:.3f}",
                    "Completeness": "{:.3f}",
                    "V-measure": "{:.3f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab_map:
    st.subheader(t("Two-dimensional cluster map with PCA", "PCA ile iki boyutlu küme haritası"))
    hover_cols = [c for c in [title_col, artist_col, "cluster_name"] if c is not None]
    fig = px.scatter(
        result_df,
        x="PCA 1",
        y="PCA 2",
        color="cluster_name",
        hover_data=hover_cols,
        opacity=0.78,
        title=t(f"{algorithm_label} results — PCA projection", f"{algorithm_label} sonuçları — PCA izdüşümü"),
        labels={"PCA 1": t("PCA component 1", "PCA bileşeni 1"), "PCA 2": t("PCA component 2", "PCA bileşeni 2")},
    )
    fig.update_traces(marker={"size": 7})
    st.plotly_chart(fig, use_container_width=True)
    explained = float(np.sum(pca.explained_variance_ratio_))
    st.caption(
        t(
            f"The first two PCA components explain {explained * 100:.1f}% of the variance in the scaled features. Clusters were created from all selected scaled features, not from the PCA plot. Reducing the data to two dimensions may hide some distance relationships.",
            f"İlk iki PCA bileşeni, ölçeklenmiş özelliklerdeki varyansın %{explained * 100:.1f}'ini açıklıyor. Kümeler PCA grafiği üzerinde değil, tüm seçili ölçeklenmiş özellikler üzerinde oluşturuldu. İki boyuta indirgeme bazı uzaklık ilişkilerini gizleyebilir.",
        )
    )

with tab_profiles:
    st.subheader(t("Musical feature profiles by cluster", "Kümelerin müzikal özellik profilleri"))
    profile_df = pd.DataFrame(X_features, columns=model_features)
    profile_df["cluster_id"] = labels
    profile_raw = profile_df.groupby("cluster_id")[model_features].mean()
    profile_raw.index = [cluster_names[int(i)] for i in profile_raw.index]
    profile_raw = profile_raw.rename(columns=feature_labels)

    profile_z_df = pd.DataFrame(X_scaled, columns=model_features)
    profile_z_df["cluster_id"] = labels
    profile_z = profile_z_df.groupby("cluster_id")[model_features].mean()
    profile_z.index = [cluster_names[int(i)] for i in profile_z.index]
    profile_z = profile_z.rename(columns=feature_labels)

    st.markdown(t("**Average features per cluster (original scale)**", "**Küme başına ortalama özellikler (orijinal ölçek)**"))
    st.dataframe(profile_raw.round(3), use_container_width=True)
    st.markdown(t("**Relative profile map**", "**Göreli profil haritası**"))
    fig_heat = px.imshow(
        profile_z,
        color_continuous_scale="RdBu_r",
        zmin=-2,
        zmax=2,
        aspect="auto",
        text_auto=".2f",
        labels={
            "x": t("Musical feature", "Müzikal özellik"),
            "y": t("Cluster profile", "Küme profili"),
            "color": t("Deviation from mean", "Ortalamadan sapma"),
        },
        title=t("Standardized feature means by cluster relative to the dataset", "Kümelerin veri geneline göre standardize edilmiş özellik ortalamaları"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        t(
            "A positive value means that the cluster is above the dataset average for that feature; a negative value means it is below. The colors show relative profiles within the sample, not the song's absolute energy or valence.",
            "Pozitif değer, o kümenin ilgili özellikte veri genelinin üzerinde; negatif değer ise altında olduğunu gösterir. Bu renkler şarkının mutlak enerji veya duygu düzeyi değil, örneklem içindeki göreli profildir.",
        )
    )

with tab_discovery:
    st.subheader(t("Select a song and find similar tracks", "Bir şarkı seç ve benzer parçaları bul"))
    if len(result_df) < 2:
        st.info(t("At least two rows are required to recommend similar songs.", "Benzer şarkı önermek için en az iki satır gerekli."))
    else:
        result_df["_display_name"] = [
            readable_song_name(row, title_col, artist_col)
            for _, row in result_df.iterrows()
        ]
        display_names = result_df["_display_name"].to_dict()

        @st.fragment
        def render_discovery() -> None:
            options = result_df.index.tolist()
            selected_idx = st.selectbox(
                t("Song", "Şarkı"),
                options=options,
                format_func=lambda index: display_names[index],
            )
            same_cluster_only = st.checkbox(
                t("Search the same cluster first", "Önce aynı kümedeki şarkıları ara"), value=True
            )
            n_recommendations = st.slider(t("Similar songs to show", "Gösterilecek benzer şarkı"), 3, 15, 8)

            selected_cluster = int(result_df.loc[selected_idx, "cluster_id"])
            eligible = np.ones(len(result_df), dtype=bool)
            eligible[selected_idx] = False
            if same_cluster_only:
                eligible &= labels == selected_cluster
            candidate_idx = np.where(eligible)[0]

            if len(candidate_idx) == 0:
                st.warning(
                    t(
                        "No other songs were found with these settings. Try turning off the same-cluster filter.",
                        "Bu koşullarda başka şarkı bulunamadı. Aynı küme filtresini kapatmayı dene.",
                    )
                )
            else:
                distances = np.linalg.norm(
                    X_scaled[candidate_idx] - X_scaled[selected_idx], axis=1
                )
                order = np.argsort(distances)[:n_recommendations]
                top_idx = candidate_idx[order]
                recommendations = result_df.iloc[top_idx].copy()
                distance_label = t("Distance in feature space", "Özellik uzayında uzaklık")
                cluster_profile_label = t("Cluster profile", "Küme profili")
                recommendations[distance_label] = distances[order]
                recommendations[cluster_profile_label] = recommendations["cluster_id"].map(
                    cluster_names
                )
                rename_map: dict[str, str] = {}
                show_cols: list[str] = []
                if title_col is not None and title_col in recommendations.columns:
                    song_label = t("Song", "Şarkı")
                    rename_map[title_col] = song_label
                    show_cols.append(song_label)
                else:
                    song_label = t("Song", "Şarkı")
                    recommendations[song_label] = recommendations.index.map(
                        lambda i: t(f"CSV row {i + 1}", f"CSV satırı {i + 1}")
                    )
                    show_cols.append(song_label)

                if artist_col is not None and artist_col in recommendations.columns:
                    artist_label = t("Artist", "Sanatçı")
                    rename_map[artist_col] = artist_label
                    show_cols.append(artist_label)

                recommendations = recommendations.rename(columns=rename_map)
                show_cols.extend([cluster_profile_label, distance_label])
                show_cols = [c for c in show_cols if c in recommendations.columns]

                st.write(
                    t(
                        "Recommendations are ranked by Euclidean distance after scaling the selected musical features. They are not based on listening behavior or personal taste data.",
                        "Öneriler, seçili müzikal özellikler ölçeklendikten sonra Öklid uzaklığıyla sıralanır. Bu, dinleme davranışı veya kişisel zevk verisine dayalı bir öneri değildir.",
                    )
                )
                st.dataframe(
                    recommendations.loc[:, show_cols].round(3),
                    use_container_width=True,
                    hide_index=True,
                )

        render_discovery()

# Download active-model assignments including cluster names and PCA coordinates.
export_df = result_df.drop(columns=["_display_name"], errors="ignore").copy()
st.divider()
st.subheader(t("Export results", "Sonuçları dışa aktar"))
st.write(
    t(
        f"Download the {algorithm_label} cluster labels, cluster profiles, and PCA coordinates as a CSV file.",
        f"{algorithm_label} küme etiketlerini, küme profilini ve PCA koordinatlarını CSV olarak indirebilirsin.",
    )
)
st.download_button(
    t("Download clustered songs as CSV", "Kümelenmiş şarkıları CSV indir"),
    data=export_df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"clustered_songs_{algorithm}.csv",
    mime="text/csv",
)

with st.expander(t("Methodological notes", "Metodolojik notlar")):
    st.markdown(
        t(
            "- Only selected numerical musical features are used for clustering; song name, artist, and genre labels are not passed to the model.\n"
            "- There is no in-app sampling or data preprocessing; all rows in the uploaded cleaned CSV file are used.\n"
            "- PCA is used only for visualization; models are not trained on PCA coordinates.\n"
            "- Internal metrics do not definitively measure musical quality. Interpret clusters together with profile tables and song examples.\n"
            "- If a genre column exists, ARI, NMI, homogeneity, completeness, and V-measure are calculated as external validation without providing genre labels to the model.\n"
            "- Cluster profile names are based on each cluster's two most distinctive standardized features relative to the dataset rather than fixed thresholds.\n"
            f"- {algorithm_label} results are created in feature space after applying {scaler_label}. K-Medoids centers are actual songs in the dataset; DBSCAN marks noise points separately.",
            "- Kümeleme için yalnızca seçilmiş sayısal müzikal özellikler kullanılır; şarkı adı, sanatçı ve tür etiketleri modele verilmez.\n"
            "- Uygulama içi örneklem veya veri ön işleme yoktur; yüklenen temiz CSV dosyasındaki tüm satırlar kullanılır.\n"
            "- PCA yalnızca görselleştirme içindir; modeller PCA koordinatları üzerinde eğitilmez.\n"
            "- İçsel metrikler müzikal kaliteyi kesin olarak ölçmez. Kümeler, profil tabloları ve şarkı örnekleriyle birlikte yorumlanmalıdır.\n"
            "- Tür sütunu varsa ARI, NMI, homogeneity, completeness ve V-measure, tür etiketleri modele verilmeden dış doğrulama olarak hesaplanır.\n"
            "- Küme profil adları, sabit eşikler yerine her kümenin veri genelinden standardize edilmiş en ayırt edici iki özelliğine göre oluşturulur.\n"
            f"- {algorithm_label} sonuçları {scaler_label} sonrası özellik uzayında oluşur. K-Medoids'te merkezler veri setindeki gerçek şarkılardır; DBSCAN gürültü noktalarını ayrı işaretler.",
        )
    )

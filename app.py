from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

from clustering import (
    calculate_metrics,
    calculate_external_metrics,
    evaluate_k_range,
    fit_labels,
    make_cluster_names,
    safe_metric,
)
from config import (
    ARTIST_CANDIDATES,
    FEATURE_LABELS,
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

st.title("🎵 Music Cluster Lab")
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
    "Şarkıları sayısal müzikal özelliklerine göre farklı kümeleme algoritmalarıyla kümeleyen ve "
    "benzer parçaları keşfetmeye yardımcı olan denetimsiz öğrenme uygulaması."
)

# -----------------------------------------------------------------------------
# File upload is in the main page on purpose, so it is easy to find.
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Başlamak için şarkı veri setinin CSV dosyasını yükle",
    type=["csv"],
    help="Spotify ses özelliklerini içeren bir CSV kullan. Dosyada en az iki sayısal müzikal özellik bulunmalı.",
)

if uploaded_file is None:
    st.info(
        "**Henüz veri seti yüklenmedi.** CSV dosyanı bu alandan seç. "
        "Önerilen sütunlar: danceability, energy, tempo, acousticness, "
        "instrumentalness, valence, loudness, speechiness ve liveness."
    )
    with st.expander("Beklenen veri yapısı örneği"):
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
    st.error(f"CSV dosyası okunamadı: {exc}")
    st.stop()

if raw_df.empty:
    st.error("Dosyada veri satırı bulunamadı.")
    st.stop()

available_feature_names = [feature for feature in FEATURE_LABELS if feature in raw_df.columns]

if len(available_feature_names) < 2:
    st.error(
        "CSV dosyasında en az iki müzikal özellik bulunamadı. "
        "Beklenen sütun adlarını kontrol et."
    )
    st.write("Algılanan sütunlar:", raw_df.columns.tolist())
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("1. Müzikal özellikler")
selected_features = st.sidebar.multiselect(
    "Kümelemede kullanılacak özellikler",
    options=available_feature_names,
    default=available_feature_names,
    format_func=lambda x: FEATURE_LABELS.get(x, x),
    help="Şarkı adı, sanatçı, tür ve popülerlik gibi etiket/metaveri alanları modele verilmez.",
)

if len(selected_features) < 2:
    st.warning("Kümeleme için en az iki müzikal özellik seçmelisin.")
    st.stop()

sample = raw_df.reset_index(drop=True)
if len(sample) < 3:
    st.error("Kümeleme için en az üç şarkı gerekli.")
    st.stop()

sample_size = len(sample)
st.info(
    f"Yüklenen temiz veri setindeki tüm kayıtlar kullanılacak: **{sample_size:,} şarkı**."
)

st.sidebar.header("2. Ölçeklendirme ve kümeleme")
algorithm = st.sidebar.selectbox(
    "Kümeleme algoritması",
    options=list(algorithm_options),
    format_func=algorithm_options.get,
    help="DBSCAN yoğunluk tabanlı kümeler bulur ve gürültüyü ayırabilir. GMM, her kümeyi Gauss dağılımıyla modeller.",
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
        help="Komşuluk yarıçapı. Daha yüksek değerler daha büyük ve daha az sayıda küme oluşturabilir.",
    )
    min_samples = st.sidebar.slider(
        "DBSCAN minimum örnek sayısı",
        min_value=2,
        max_value=min(20, sample_size),
        value=min(5, sample_size),
        help="Bir noktanın çekirdek örnek sayılması için gereken komşu sayısı.",
    )
    n_clusters = min(2, sample_size)
else:
    eps = 0.5
    min_samples = 5
    n_clusters = st.sidebar.slider(
        f"{algorithm_label} küme sayısı (k)",
        min_value=1,
        max_value=max(1, n_clusters_max),
        value=min(4, max(1, n_clusters_max)),
        help="K değeri 1 olabilir; ancak tek kümede Silhouette, Davies–Bouldin ve Calinski–Harabasz metrikleri hesaplanamaz.",
    )

model_features = selected_features
X_features = sample[model_features].to_numpy(dtype=float)
scaler_key = st.sidebar.selectbox(
    "Ölçeklendirme yöntemi",
    options=list(scaler_options),
    format_func=lambda key: scaler_options[key][0],
    help="Ölçeklendirme, farklı ölçü birimlerine sahip özelliklerin model üzerindeki etkisini dengeler.",
)
scaler_label, scaler_class = scaler_options[scaler_key]
X_scaled = scaler_class().fit_transform(X_features)

# Metadata labels are kept outside the model and used only for external validation.
genre_col = first_existing_column(list(sample.columns), GENRE_CANDIDATES)
reference_labels = (
    sample[genre_col].astype("string").to_numpy() if genre_col is not None else None
)

with st.spinner(f"{algorithm_label} modeli eğitiliyor..."):
    labels, elapsed, _ = fit_labels(
        X_scaled,
        int(n_clusters),
        algorithm,
        eps=eps,
        min_samples=min_samples,
    )
metrics = calculate_metrics(X_scaled, labels)
external_metrics = calculate_external_metrics(labels, reference_labels)
cluster_names = make_cluster_names(X_scaled, model_features, labels)
k_evaluation_df = (
    evaluate_k_range(
        X_scaled,
        range(1, n_clusters_max + 1),
        reference_labels,
        algorithm,
        eps=eps,
        min_samples=min_samples,
    )
    if algorithm != "dbscan"
    else pd.DataFrame()
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
metric_cols[0].metric("Analiz edilen şarkı", f"{len(result_df):,}")
metric_cols[1].metric("Müzikal özellik", len(model_features))
metric_cols[2].metric("Bulunan küme", metrics["cluster_count"])
metric_cols[3].metric("Silhouette Score", safe_metric(metrics["silhouette"]))
metric_cols[4].metric("Davies–Bouldin", safe_metric(metrics["davies_bouldin"]))
metric_cols[5].metric("Calinski–Harabasz", safe_metric(metrics["calinski_harabasz"]))
metric_cols[6].metric("Model süresi", f"{elapsed:.3f} sn")
if algorithm == "dbscan" and 0 in labels:
    st.caption(f"DBSCAN gürültü olarak {int(np.sum(labels == 0)):,} şarkı işaretledi.")

st.caption(
    "Not: Küme numaraları tür etiketleri değildir. Otomatik profil ifadeleri, "
    "her kümenin seçilen veri örneklemine göre hangi özelliklerde öne çıktığını özetler."
)

# Tabs for a research-style workflow.
tab_overview, tab_dataset, tab_map, tab_profiles, tab_discovery = st.tabs(
    [
        "Genel bakış",
        "Veri seti",
        "Küme haritası",
        "Küme profilleri",
        "Şarkı keşfi",
    ]
)

with tab_overview:
    st.subheader(f"{algorithm_label} sonuç özeti")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("**Kümelerin büyüklükleri**")
        cluster_sizes = (
            result_df.groupby(["cluster_id", "cluster_name"], dropna=False)
            .size()
            .reset_index(name="Şarkı sayısı")
            .sort_values("Şarkı sayısı", ascending=False)
        )
        if not cluster_sizes.empty:
            fig_sizes = px.bar(
                cluster_sizes,
                x="cluster_name",
                y="Şarkı sayısı",
                color="cluster_name",
                title="Kümelerdeki şarkı sayısı",
                labels={"cluster_name": "Küme profili"},
            )
            fig_sizes.update_layout(showlegend=False, xaxis_tickangle=-25)
            st.plotly_chart(fig_sizes, use_container_width=True)
        st.dataframe(cluster_sizes, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Bu sonuç nasıl okunmalı?**")
        st.write(
            f"{algorithm_label}, her şarkıyı seçilen k değeri doğrultusunda bir kümeye atar. "
            "Küme profillerini ve gerçek şarkı örneklerini metriklerle birlikte değerlendir."
        )
        st.markdown("**Seçilen modelin metrikleri**")
        st.write(f"- Silhouette Score: **{safe_metric(metrics['silhouette'])}**")
        st.write(f"- Davies–Bouldin Index: **{safe_metric(metrics['davies_bouldin'])}** (düşük olması genellikle iyidir)")
        st.write(f"- Calinski–Harabasz Score: **{safe_metric(metrics['calinski_harabasz'])}** (yüksek olması genellikle iyidir)")
        st.write(f"- Modelin ölçülen çalışma süresi: **{elapsed:.3f} sn**")
        if genre_col is not None:
            st.markdown("**Dış doğrulama (tür etiketi ile)**")
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
                "Bu metrikler, küme etiketlerinin tür etiketleriyle ne kadar örtüştüğünü "
                "ölçer; tür bilgisi modele verilmediği için bağımsız bir kontrol sağlar."
            )
        st.info(
            "Bu ölçütler küme geometrisini değerlendirir; müzikal açıdan en anlamlı "
            "kümelemeyi tek başlarına kanıtlamaz. Küme profillerini ve gerçek şarkı "
            "örneklerini de incele."
        )

with tab_dataset:
    st.subheader("Veri seti özellikleri")
    info_cols = st.columns(3)
    info_cols[0].metric("Satır sayısı", f"{len(sample):,}")
    info_cols[1].metric("Toplam sütun", len(sample.columns))
    info_cols[2].metric("Kullanılan müzikal özellik", len(model_features))

    st.markdown("**Aykırı değer görünümü**")
    outlier_df = sample[model_features].melt(
        var_name="Özellik",
        value_name="Değer",
    )
    outlier_fig = px.box(
        outlier_df,
        x="Özellik",
        y="Değer",
        points="outliers",
        title="Müzikal özelliklerin dağılımı ve aykırı değerleri",
        labels={"Değer": "Değer"},
    )
    outlier_fig.update_xaxes(tickangle=-35)
    st.plotly_chart(outlier_fig, use_container_width=True)
    st.caption(
        "Box-plot üzerindeki noktalar, ilgili özelliğin kutu aralığı dışındaki "
        "istatistiksel aykırı değerlerdir. Bu kayıtlar otomatik olarak silinmez."
    )

    if genre_col is None:
        st.info("CSV dosyasında tür bilgisi bulunamadı.")
    else:
        genre_counts = (
            sample[genre_col]
            .fillna("Bilinmeyen")
            .astype(str)
            .value_counts()
            .rename_axis("Tür")
            .reset_index(name="Şarkı sayısı")
        )
        st.markdown(f"**Tür dağılımı (`{genre_col}`)**")
        st.metric("Tür sayısı", len(genre_counts))
        genre_fig = px.bar(
            genre_counts,
            x="Tür",
            y="Şarkı sayısı",
            title="Türlere göre şarkı sayıları",
        )
        genre_fig.update_xaxes(tickangle=-35)
        st.plotly_chart(genre_fig, use_container_width=True)
        st.dataframe(genre_counts, use_container_width=True, hide_index=True)

    if algorithm == "dbscan":
        st.info("DBSCAN için k değerleri karşılaştırması yapılmaz; sonuçlar eps ve minimum örnek sayısına bağlıdır.")
    else:
        st.markdown(f"**K değerlerine göre {algorithm_label} değerlendirmesi**")
        st.write(
            f"Aşağıdaki grafikler, {scaler_label} uygulanmış müzikal özelliklerle "
            f"k=1 ile k arasındaki {algorithm_label} sonuçlarını karşılaştırır."
        )
    chart_data = [
        ("Objective", "Kümeleme amacı", "Toplam uzaklık"),
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
    if algorithm != "dbscan":
        for metric_name, title, y_label in chart_data:
            metric_df = k_evaluation_df.dropna(subset=[metric_name])
            st.plotly_chart(
                px.line(
                    metric_df,
                    x="K değeri",
                    y=metric_name,
                    markers=True,
                    title=title,
                    labels={metric_name: y_label},
                ),
                use_container_width=True,
            )
        st.dataframe(
            k_evaluation_df.style.format(
                {
                    "Objective": "{:.1f}",
                    "Silhouette Score": "{:.3f}",
                    "Davies–Bouldin Index": "{:.3f}",
                    "Calinski–Harabasz Score": "{:.1f}",
                    "Süre (sn)": "{:.3f}",
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
    st.subheader("PCA ile iki boyutlu küme haritası")
    hover_cols = [c for c in [title_col, artist_col, "cluster_name"] if c is not None]
    fig = px.scatter(
        result_df,
        x="PCA 1",
        y="PCA 2",
        color="cluster_name",
        hover_data=hover_cols,
        opacity=0.78,
        title=f"{algorithm_label} sonuçları — PCA izdüşümü",
        labels={"PCA 1": "PCA bileşeni 1", "PCA 2": "PCA bileşeni 2"},
    )
    fig.update_traces(marker={"size": 7})
    st.plotly_chart(fig, use_container_width=True)
    explained = float(np.sum(pca.explained_variance_ratio_))
    st.caption(
        f"İlk iki PCA bileşeni, ölçeklenmiş özelliklerdeki varyansın %{explained * 100:.1f}'ini açıklıyor. "
        "Kümeler PCA grafiği üzerinde değil, tüm seçili ölçeklenmiş özellikler üzerinde oluşturuldu. "
        "İki boyuta indirgeme bazı uzaklık ilişkilerini gizleyebilir."
    )

with tab_profiles:
    st.subheader("Kümelerin müzikal özellik profilleri")
    profile_df = pd.DataFrame(X_features, columns=model_features)
    profile_df["cluster_id"] = labels
    profile_raw = profile_df.groupby("cluster_id")[model_features].mean()
    profile_raw.index = [cluster_names[int(i)] for i in profile_raw.index]
    profile_raw = profile_raw.rename(columns=FEATURE_LABELS)

    profile_z_df = pd.DataFrame(X_scaled, columns=model_features)
    profile_z_df["cluster_id"] = labels
    profile_z = profile_z_df.groupby("cluster_id")[model_features].mean()
    profile_z.index = [cluster_names[int(i)] for i in profile_z.index]
    profile_z = profile_z.rename(columns=FEATURE_LABELS)

    st.markdown("**Küme başına ortalama özellikler (orijinal ölçek)**")
    st.dataframe(profile_raw.round(3), use_container_width=True)
    st.markdown("**Göreli profil haritası**")
    fig_heat = px.imshow(
        profile_z,
        color_continuous_scale="RdBu_r",
        zmin=-2,
        zmax=2,
        aspect="auto",
        text_auto=".2f",
        labels={"x": "Müzikal özellik", "y": "Küme profili", "color": "Ortalamadan sapma"},
        title="Kümelerin veri geneline göre standardize edilmiş özellik ortalamaları",
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        "Pozitif değer, o kümenin ilgili özellikte veri genelinin üzerinde; negatif değer ise altında olduğunu gösterir. "
        "Bu renkler şarkının mutlak enerji veya duygu düzeyi değil, örneklem içindeki göreli profildir."
    )

with tab_discovery:
    st.subheader("Bir şarkı seç ve benzer parçaları bul")
    if len(result_df) < 2:
        st.info("Benzer şarkı önermek için en az iki satır gerekli.")
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
                "Şarkı",
                options=options,
                format_func=lambda index: display_names[index],
            )
            same_cluster_only = st.checkbox(
                "Önce aynı kümedeki şarkıları ara", value=True
            )
            n_recommendations = st.slider("Gösterilecek benzer şarkı", 3, 15, 8)

            selected_cluster = int(result_df.loc[selected_idx, "cluster_id"])
            eligible = np.ones(len(result_df), dtype=bool)
            eligible[selected_idx] = False
            if same_cluster_only:
                eligible &= labels == selected_cluster
            candidate_idx = np.where(eligible)[0]

            if len(candidate_idx) == 0:
                st.warning(
                    "Bu koşullarda başka şarkı bulunamadı. "
                    "Aynı küme filtresini kapatmayı dene."
                )
            else:
                distances = np.linalg.norm(
                    X_scaled[candidate_idx] - X_scaled[selected_idx], axis=1
                )
                order = np.argsort(distances)[:n_recommendations]
                top_idx = candidate_idx[order]
                recommendations = result_df.iloc[top_idx].copy()
                recommendations["Özellik uzayında uzaklık"] = distances[order]
                recommendations["Küme profili"] = recommendations["cluster_id"].map(
                    cluster_names
                )
                rename_map: dict[str, str] = {}
                show_cols: list[str] = []
                if title_col is not None and title_col in recommendations.columns:
                    rename_map[title_col] = "Şarkı"
                    show_cols.append("Şarkı")
                else:
                    recommendations["Şarkı"] = recommendations.index.map(
                        lambda i: f"CSV satırı {i + 1}"
                    )
                    show_cols.append("Şarkı")

                if artist_col is not None and artist_col in recommendations.columns:
                    rename_map[artist_col] = "Sanatçı"
                    show_cols.append("Sanatçı")

                recommendations = recommendations.rename(columns=rename_map)
                show_cols.extend(["Küme profili", "Özellik uzayında uzaklık"])
                show_cols = [c for c in show_cols if c in recommendations.columns]

                st.write(
                    "Öneriler, seçili müzikal özellikler ölçeklendikten sonra Öklid uzaklığıyla sıralanır. "
                    "Bu, dinleme davranışı veya kişisel zevk verisine dayalı bir öneri değildir."
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
st.subheader("Sonuçları dışa aktar")
st.write(
    f"{algorithm_label} küme etiketlerini, küme profilini ve PCA koordinatlarını "
    "CSV olarak indirebilirsin."
)
st.download_button(
    "Kümelenmiş şarkıları CSV indir",
    data=export_df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"clustered_songs_{algorithm}.csv",
    mime="text/csv",
)

with st.expander("Metodolojik notlar"):
    st.markdown(
        "- Kümeleme için yalnızca seçilmiş sayısal müzikal özellikler kullanılır; şarkı adı, sanatçı ve tür etiketleri modele verilmez.\n"
        "- Uygulama içi örneklem veya veri ön işleme yoktur; yüklenen temiz CSV dosyasındaki tüm satırlar kullanılır.\n"
        "- PCA yalnızca görselleştirme içindir; modeller PCA koordinatları üzerinde eğitilmez.\n"
        "- İçsel metrikler müzikal kaliteyi kesin olarak ölçmez. Kümeler, profil tabloları ve şarkı örnekleriyle birlikte yorumlanmalıdır.\n"
        "- Tür sütunu varsa ARI, NMI, homogeneity, completeness ve V-measure, tür etiketlerini modele vermeden dış doğrulama olarak hesaplanır.\n"
        "- Küme profil adları, sabit eşikler yerine her kümenin veri genelinden standardize edilmiş en ayırt edici iki özelliğine göre oluşturulur.\n"
        f"- {algorithm_label} sonuçları {scaler_label} sonrası özellik uzayında oluşur. K-Medoids'te merkezler veri setindeki gerçek şarkılardır; DBSCAN gürültü noktalarını ayrı işaretler."
    )

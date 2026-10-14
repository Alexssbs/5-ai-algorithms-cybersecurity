import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Agrupamiento de Incidentes - K-Means", page_icon="🔵", layout="wide")

st.title("🔵 Agrupamiento de Incidentes con K-Means")
st.markdown("""
Modelo **no supervisado** que agrupa incidentes de un SOC según su comportamiento.
Usa **K-Means** para encontrar patrones y priorizar la atención del analista.
""")

# --- Cargar datos ---
@st.cache_data
def cargar_datos():
    return pd.read_csv("incidentes_agrupados.csv")

st.sidebar.header("📂 Datos de entrada")
archivo_subido = st.sidebar.file_uploader("Sube tu CSV (o usa 'incidentes_agrupados.csv')", type=["csv"])

if archivo_subido is not None:
    df = pd.read_csv(archivo_subido)
    st.sidebar.success("Archivo cargado.")
else:
    try:
        df = cargar_datos()
        st.sidebar.info("Usando dataset: incidentes_agrupados.csv")
    except FileNotFoundError:
        st.sidebar.error("No se encontró 'incidentes_agrupados.csv'.")

# --- Seleccionar características ---
st.sidebar.header("⚙️ Características para el clustering")
columnas_disponibles = [c for c in df.columns if c != "number"]

# Por defecto, usar todas
columnas_seleccionadas = st.sidebar.multiselect(
    "Selecciona las características:",
    options=columnas_disponibles,
    default=columnas_disponibles
)

if len(columnas_seleccionadas) < 2:
    st.warning("⚠️ Selecciona al menos 2 características para el clustering.")
    st.stop()

X = df[columnas_seleccionadas].copy()
X = X.fillna(0)

# --- Escalado ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Configuración del modelo ---
st.sidebar.header("⚙️ Configuración de K-Means")
k = st.sidebar.slider("Número de clusters (k)", 2, 10, 3, 1)
init_method = st.sidebar.selectbox("Método de inicialización", ["k-means++", "random"])
max_iter = st.sidebar.slider("Máximo de iteraciones", 100, 500, 300, 50)

# --- Entrenamiento ---
with st.spinner(f"Entrenando K-Means con k={k}..."):
    modelo = KMeans(
        n_clusters=k,
        init=init_method,
        max_iter=max_iter,
        random_state=42,
        n_init=10
    )
    df["cluster"] = modelo.fit_predict(X_scaled)

# --- Métricas ---
st.subheader("📈 Métricas del clustering")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Clusters", k)
col2.metric("Inercia", f"{modelo.inertia_:.2f}")
col3.metric("Silhouette Score", f"{silhouette_score(X_scaled, df['cluster']):.3f}")
col4.metric("Incidentes", len(df))

# --- Distribución de clusters ---
st.subheader("📊 Distribución de incidentes por cluster")
dist = df["cluster"].value_counts().sort_index().reset_index()
dist.columns = ["Cluster", "Cantidad"]
dist["Porcentaje"] = (dist["Cantidad"] / len(df) * 100).round(2)

col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(dist, use_container_width=True)
with col2:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=dist, x="Cluster", y="Cantidad", palette="viridis", ax=ax)
    ax.set_title("Distribución de incidentes por cluster")
    st.pyplot(fig)

# --- Perfil de cada cluster (promedio de características) ---
st.subheader("🔍 Perfil de cada cluster")
st.markdown("Promedio de cada característica por cluster (para interpretar qué tipo de incidentes agrupa cada uno):")

perfil = df.groupby("cluster")[columnas_seleccionadas].mean().round(2)
st.dataframe(perfil, use_container_width=True)

# Heatmap del perfil
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(perfil.T, annot=True, fmt=".1f", cmap="coolwarm", ax=ax,
            cbar_kws={"label": "Promedio"})
ax.set_title("Perfil promedio por cluster")
ax.set_xlabel("Cluster")
st.pyplot(fig)

# --- Visualización con PCA ---
st.subheader("🗺️ Visualización de clusters con PCA (2D)")

with st.spinner("Reduciendo dimensionalidad con PCA..."):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Cluster"] = df["cluster"].values

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Cluster", palette="viridis", alpha=0.5, s=15, ax=ax)
ax.set_title(f"Visualización de {k} clusters (PCA)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)")
st.pyplot(fig)

# --- Método del codo ---
with st.expander("📉 Ver método del codo (para elegir k óptimo)"):
    inercias = []
    siluetas = []
    rango_k = range(2, 11)
    with st.spinner("Calculando inercias y siluetas..."):
        for k_test in rango_k:
            km = KMeans(n_clusters=k_test, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inercias.append(km.inertia_)
            siluetas.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(rango_k, inercias, marker="o", color="blue")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inercia")
    axes[0].set_title("Método del codo")
    axes[0].grid(True)

    axes[1].plot(rango_k, siluetas, marker="o", color="green")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score por k")
    axes[1].grid(True)

    st.pyplot(fig)

# --- Explorar incidentes por cluster ---
st.subheader("🔎 Explorar incidentes por cluster")
cluster_seleccionado = st.selectbox("Selecciona un cluster:", sorted(df["cluster"].unique()))
muestra = df[df["cluster"] == cluster_seleccionado].head(20)
st.dataframe(muestra, use_container_width=True)
st.caption(f"Mostrando 20 de {len(df[df['cluster'] == cluster_seleccionado])} incidentes en el cluster {cluster_seleccionado}.")

# --- Predicción de cluster para un incidente nuevo ---
st.subheader("🔍 Predecir el cluster de un incidente nuevo")
st.markdown("Ajusta las características del incidente y el modelo dirá a qué cluster pertenece:")

cols = st.columns(3)
entrada_dict = {}
for i, col in enumerate(columnas_seleccionadas):
    with cols[i % 3]:
        valor_default = float(df[col].median())
        entrada_dict[col] = st.number_input(
            col, value=valor_default, step=1.0, key=f"input_{col}"
        )

if st.button("Predecir cluster"):
    entrada = pd.DataFrame([entrada_dict])[columnas_seleccionadas]
    entrada_scaled = scaler.transform(entrada)
    cluster_pred = modelo.predict(entrada_scaled)[0]
    st.success(f"🎯 El incidente pertenece al **Cluster {cluster_pred}**")

st.markdown("---")
st.caption("Ejemplo didáctico - Tópicos de Ciberseguridad - K-Means para agrupamiento de incidentes en un SOC.")

# https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log


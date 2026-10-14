import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
st.set_page_config(page_title="Detección de Anomalías - Isolation Forest", page_icon="🛡️", layout="wide")

st.title("🛡️ Detección de Anomalías en Logs con Isolation Forest")
st.markdown("""
Modelo **no supervisado** que detecta comportamientos anómalos en logs de red **sin usar etiquetas**.
Usa **Isolation Forest** para identificar registros que se salen de lo normal.
""")

# --- Cargar datos ---
@st.cache_data
def cargar_datos():
    df = pd.read_csv("nsg_dataset.csv")
    return df

st.sidebar.header("📂 Datos de entrada")
archivo_subido = st.sidebar.file_uploader("Sube tu CSV (o usa 'nsg_dataset.csv')", type=["csv"])

if archivo_subido is not None:
    df = pd.read_csv(archivo_subido)
    st.sidebar.success("Archivo cargado.")
else:
    try:
        df = cargar_datos()
        st.sidebar.info("Usando dataset: nsg_dataset.csv")
    except FileNotFoundError:
        st.sidebar.error("No se encontró 'nsg_dataset.csv'.")

# --- Preprocesamiento ---
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hora_del_dia"] = df["timestamp"].dt.hour

# Codificación de variables categóricas
map_protocol = {"ICMP": 0, "UDP": 1, "TCP": 2}
map_action = {"Allow": 0, "Deny": 1}
map_threat_level = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
map_threat_type = {"Malware": 0, "Brute Force": 1, "DDoS": 2, "Port Scan": 3, "SQL Injection": 4}
map_response = {"Monitored": 0, "Escalated": 1, "Blocked": 2, "Alerted": 3}

df["protocol_encoded"] = df["protocol"].map(map_protocol)
df["action_encoded"] = df["action"].map(map_action)
df["threat_level_encoded"] = df["threat_level"].map(map_threat_level)
df["threat_type_encoded"] = df["threat_type"].map(map_threat_type)
df["response_action_encoded"] = df["response_action"].map(map_response)

# Características para el modelo
features = [
    "port", "hora_del_dia", "protocol_encoded", "action_encoded",
    "threat_level_encoded", "threat_type_encoded", "response_action_encoded"
]

X = df[features].copy()

# --- Configuración del modelo desde sidebar ---
st.sidebar.header("⚙️ Configuración del modelo")
contamination = st.sidebar.slider(
    "Contaminación esperada (proporción de anomalías)",
    min_value=0.01, max_value=0.30, value=0.10, step=0.01,
    help="Qué porcentaje de los datos esperas que sean anómalos."
)
n_estimators = st.sidebar.slider("Número de árboles", 50, 300, 100, 50)

# --- Entrenamiento ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

modelo = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    random_state=42
)
modelo.fit(X_scaled)

# Predicciones: -1 = anomalía, 1 = normal
df["anomalia"] = modelo.predict(X_scaled)
df["score_anomalia"] = modelo.decision_function(X_scaled)  # Más negativo = más anómalo
df["es_anomalia"] = df["anomalia"].apply(lambda x: "Sí" if x == -1 else "No")

# --- Métricas ---
total = len(df)
num_anomalias = (df["anomalia"] == -1).sum()

st.subheader("📊 Resumen de la detección")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de registros", total)
col2.metric("Anomalías detectadas", num_anomalias)
col3.metric("Porcentaje", f"{num_anomalias/total*100:.1f}%")
col4.metric("Árboles usados", n_estimators)

# --- Tabla de anomalías ---
st.subheader("🚨 Registros marcados como ANÓMALOS")
columnas_mostrar = [
    "timestamp", "source_ip", "destination_ip", "protocol", "port",
    "action", "threat_level", "threat_type", "response_action", "score_anomalia"
]
anomalias = df[df["anomalia"] == -1][columnas_mostrar].sort_values("score_anomalia")
st.dataframe(anomalias, use_container_width=True)
st.caption(f"Mostrando {len(anomalias)} registros anómalos ordenados por score (más negativo = más anómalo).")

# --- Comparación con etiquetas reales ---
st.subheader("🔍 Comparación con las etiquetas del dataset")
st.markdown("Aunque el modelo **no usó** `threat_level` ni `action` para entrenar, podemos ver si las anomalías coinciden con eventos críticos.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Distribución de threat_level en anomalías:**")
    st.dataframe(df[df["anomalia"] == -1]["threat_level"].value_counts())
with col2:
    st.markdown("**Distribución de action en anomalías:**")
    st.dataframe(df[df["anomalia"] == -1]["action"].value_counts())

# --- Visualización ---
st.subheader("📈 Visualización de las anomalías")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Puerto vs Hora del día
colores = df["anomalia"].map({1: "blue", -1: "red"})
axes[0].scatter(df["hora_del_dia"], df["port"], c=colores, alpha=0.5, s=15)
axes[0].set_xlabel("Hora del día")
axes[0].set_ylabel("Puerto")
axes[0].set_title("Puerto vs Hora (rojo = anomalía)")
axes[0].legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Normal'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Anomalía')
])

# Gráfico 2: Distribución del score
sns.histplot(df["score_anomalia"], bins=40, kde=True, ax=axes[1], color="purple")
axes[1].axvline(0, color="red", linestyle="--", label="Umbral de anomalía")
axes[1].set_xlabel("Score de anomalía (más negativo = más anómalo)")
axes[1].set_title("Distribución del score")
axes[1].legend()

st.pyplot(fig)

# --- Predicción interactiva ---
st.subheader("🔍 Probar un registro nuevo")
st.markdown("Ingresa las características de un log para ver si el modelo lo considera anómalo:")

col1, col2, col3 = st.columns(3)
with col1:
    puerto = st.number_input("Puerto", min_value=1, max_value=65535, value=443)
    hora = st.slider("Hora del día", 0, 23, 12)
with col2:
    protocolo = st.selectbox("Protocolo", ["ICMP", "UDP", "TCP"])
    accion = st.selectbox("Acción", ["Allow", "Deny"])
with col3:
    nivel = st.selectbox("Nivel de amenaza", ["Low", "Medium", "High", "Critical"])
    tipo = st.selectbox("Tipo de amenaza", ["Malware", "Brute Force", "DDoS", "Port Scan", "SQL Injection"])
    respuesta = st.selectbox("Acción de respuesta", ["Monitored", "Escalated", "Blocked", "Alerted"])

if st.button("Analizar registro"):
    entrada = pd.DataFrame({
        "port": [puerto],
        "hora_del_dia": [hora],
        "protocol_encoded": [map_protocol[protocolo]],
        "action_encoded": [map_action[accion]],
        "threat_level_encoded": [map_threat_level[nivel]],
        "threat_type_encoded": [map_threat_type[tipo]],
        "response_action_encoded": [map_response[respuesta]]
    })
    entrada_scaled = scaler.transform(entrada)
    pred = modelo.predict(entrada_scaled)[0]
    score = modelo.decision_function(entrada_scaled)[0]

    if pred == -1:
        st.error(f"🚨 **ANOMALÍA DETECTADA** (Score: {score:.4f})")
    else:
        st.success(f"✅ **Registro NORMAL** (Score: {score:.4f})")

st.markdown("---")
st.caption("Ejemplo didáctico - Tópicos de Ciberseguridad - Isolation Forest para detección de anomalías en logs.")
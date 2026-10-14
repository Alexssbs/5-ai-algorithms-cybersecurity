import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Análisis de Tráfico de Red - Random Forest", page_icon="🌐", layout="wide")

st.title("🌐 Análisis de Tráfico de Red con Random Forest")
st.markdown("""
Modelo de **Random Forest** entrenado con el dataset **CICIDS2017**.
Clasifica flujos de red como **Benigno** o **Ataque** (DDoS, PortScan, etc.).
""")

# --- Cargar datos ---
@st.cache_data
def cargar_datos():
    return pd.read_csv("cicids_muestra.csv")

st.sidebar.header("📂 Datos de entrada")
archivo_subido = st.sidebar.file_uploader("Sube tu CSV (o usa 'cicids_muestra.csv')", type=["csv"])

if archivo_subido is not None:
    df = pd.read_csv(archivo_subido)
    st.sidebar.success("Archivo cargado.")
else:
    try:
        df = cargar_datos()
        st.sidebar.info("Usando dataset: cicids_muestra.csv")
    except FileNotFoundError:
        st.sidebar.error("No se encontró 'cicids_muestra.csv'. Ejecuta primero 'preparar_cicids.py'.")

st.subheader("📊 Vista previa del dataset")
st.dataframe(df.head(10))
st.caption(f"Total de muestras: {len(df)} | Etiquetas únicas: {df['Label'].nunique()}")

# --- Preprocesamiento ---
# Codificar la etiqueta
le = LabelEncoder()
df["Label_encoded"] = le.fit_transform(df["Label"])

# Mostrar mapeo de etiquetas
st.sidebar.header("🏷️ Etiquetas del dataset")
mapeo = dict(zip(le.classes_, le.transform(le.classes_)))
st.sidebar.write(mapeo)

# Separar X e y
X = df.drop(columns=["Label", "Label_encoded"])
y = df["Label_encoded"]

# Convertir a numérico y reemplazar infinitos
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# --- Configuración del modelo desde sidebar ---
st.sidebar.header("⚙️ Configuración del Random Forest")
n_estimators = st.sidebar.slider("Número de árboles", 10, 300, 100, 10)
max_depth = st.sidebar.slider("Profundidad máxima", 2, 30, 10, 1)
test_size = st.sidebar.slider("Tamaño de prueba", 0.1, 0.4, 0.3, 0.05)

# --- División ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# --- Entrenamiento ---
with st.spinner("Entrenando Random Forest..."):
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

# --- Métricas ---
st.subheader("📈 Rendimiento del modelo")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Precisión (Accuracy)", f"{accuracy_score(y_test, y_pred):.2%}")
col2.metric("Muestras de prueba", len(y_test))
col3.metric("Características", X.shape[1])
col4.metric("Árboles", n_estimators)

# Matriz de confusión
st.markdown("**Matriz de confusión**")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
st.pyplot(fig)

# Reporte de clasificación
st.markdown("**Reporte de clasificación**")
st.text(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Importancia de características ---
st.subheader("🔍 Importancia de características")
importancias = pd.DataFrame({
    "Característica": X.columns,
    "Importancia": modelo.feature_importances_
}).sort_values("Importancia", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importancias, x="Importancia", y="Característica", ax=ax, palette="viridis")
ax.set_title("Top 15 características más importantes")
st.pyplot(fig)

# --- Predicción interactiva ---
st.subheader("🔍 Probar un flujo de red nuevo")
st.markdown("Ajusta las características principales y el modelo predecirá el tipo de tráfico:")

col1, col2, col3 = st.columns(3)
with col1:
    dest_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=80)
    flow_duration = st.number_input("Flow Duration", min_value=0, value=1000000, step=1000)
    total_fwd = st.number_input("Total Fwd Packets", min_value=0, value=10)
with col2:
    total_bwd = st.number_input("Total Backward Packets", min_value=0, value=10)
    flow_bytes = st.number_input("Flow Bytes/s", min_value=0.0, value=1000.0, step=100.0)
    flow_packets = st.number_input("Flow Packets/s", min_value=0.0, value=100.0, step=10.0)
with col3:
    packet_len_mean = st.number_input("Packet Length Mean", min_value=0.0, value=100.0, step=10.0)
    syn_flag = st.number_input("SYN Flag Count", min_value=0, value=1)
    ack_flag = st.number_input("ACK Flag Count", min_value=0, value=1)

if st.button("Analizar flujo"):
    # Construir un DataFrame con todas las columnas requeridas
    entrada_dict = {col: 0 for col in X.columns}
    entrada_dict.update({
        "Destination Port": dest_port,
        "Flow Duration": flow_duration,
        "Total Fwd Packets": total_fwd,
        "Total Backward Packets": total_bwd,
        "Flow Bytes/s": flow_bytes,
        "Flow Packets/s": flow_packets,
        "Packet Length Mean": packet_len_mean,
        "SYN Flag Count": syn_flag,
        "ACK Flag Count": ack_flag,
    })
    entrada = pd.DataFrame([entrada_dict])[X.columns]
    entrada = entrada.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)
    pred = modelo.predict(entrada)[0]
    proba = modelo.predict_proba(entrada)[0]
    etiqueta = le.inverse_transform([pred])[0]

    if etiqueta == "BENIGN":
        st.success(f"✅ **TRÁFICO BENIGNO** (Confianza: {max(proba):.2%})")
    else:
        st.error(f"🚨 **ATAQUE DETECTADO: {etiqueta}** (Confianza: {max(proba):.2%})")

st.markdown("---")
st.caption("Ejemplo didáctico - Tópicos de Ciberseguridad - Random Forest para detección de intrusiones con CICIDS2017.")
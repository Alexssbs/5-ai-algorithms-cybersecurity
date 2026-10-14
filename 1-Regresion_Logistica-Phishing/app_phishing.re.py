import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Detección de Phishing - Regresión Logística", page_icon="🎣", layout="centered")

st.title("🎣 Detección de Phishing con Regresión Logística")
st.markdown("""
Modelo de **Machine Learning** entrenado con un dataset real de correos (200 muestras).
Usa **Regresión Logística** para clasificar correos como phishing o legítimos.
""")

# --- Cargar datos ---
@st.cache_data
def cargar_datos():
    df = pd.read_csv("phishing_features.csv")
    return df

st.sidebar.header("📂 Datos de entrada")
archivo_subido = st.sidebar.file_uploader("Sube tu CSV (o usa 'phishing_features.csv')", type=["csv"])

if archivo_subido is not None:
    df = pd.read_csv(archivo_subido)
    st.sidebar.success("Archivo cargado correctamente.")
else:
    try:
        df = cargar_datos()
        st.sidebar.info("Usando dataset real: phishing_features.csv")
    except FileNotFoundError:
        st.sidebar.error("No se encontró 'phishing_features.csv'. Ejecuta primero el script de conversión.")

st.subheader("📊 Vista previa del dataset")
st.dataframe(df.head(10))
st.caption(f"Total de muestras: {len(df)} | Phishing: {df['es_phishing'].sum()} | Legítimos: {len(df) - df['es_phishing'].sum()}")

# --- Entrenamiento del modelo ---
X = df.drop("es_phishing", axis=1)
y = df["es_phishing"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Escalar características (importante para Regresión Logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train_scaled, y_train)

y_pred = modelo.predict(X_test_scaled)

# --- Métricas ---
st.subheader("📈 Rendimiento del modelo")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Precisión (Accuracy)", f"{accuracy_score(y_test, y_pred):.2%}")
with col2:
    st.metric("Muestras de prueba", len(y_test))
with col3:
    st.metric("Características usadas", X.shape[1])

with st.expander("Ver matriz de confusión y reporte detallado"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legítimo", "Phishing"],
                yticklabels=["Legítimo", "Phishing"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    st.pyplot(fig)
    st.text(classification_report(y_test, y_pred, target_names=["Legítimo", "Phishing"]))

# --- Coeficientes del modelo (interpretabilidad) ---
with st.expander("🔍 Ver pesos (coeficientes) del modelo"):
    coefs = pd.DataFrame({
        "Característica": X.columns,
        "Coeficiente": modelo.coef_[0]
    }).sort_values("Coeficiente", ascending=False)
    st.dataframe(coefs)
    st.caption("Valores positivos → empujan hacia 'Phishing'. Valores negativos → empujan hacia 'Legítimo'.")

# --- Predicción interactiva ---
st.subheader("🔍 Probar un correo nuevo")
st.markdown("Ajusta las características extraídas del correo:")

col1, col2 = st.columns(2)
with col1:
    tiene_url = st.checkbox("Tiene URL en el cuerpo")
    tiene_adjunto = st.checkbox("Tiene adjunto")
    num_urgentes = st.slider("Palabras urgentes", 0, 10, 1)
    longitud_asunto = st.slider("Longitud del asunto", 0, 200, 30)
with col2:
    longitud_cuerpo = st.slider("Longitud del cuerpo", 0, 2000, 100)
    tiene_numero = st.checkbox("Tiene números en el asunto")
    remitente_sospechoso = st.checkbox("Remitente sospechoso")

if st.button("Predecir"):
    entrada = pd.DataFrame({
        "tiene_url": [int(tiene_url)],
        "tiene_adjunto": [int(tiene_adjunto)],
        "num_palabras_urgentes": [num_urgentes],
        "longitud_asunto": [longitud_asunto],
        "longitud_cuerpo": [longitud_cuerpo],
        "tiene_numero_asunto": [int(tiene_numero)],
        "remitente_sospechoso": [int(remitente_sospechoso)]
    })
    entrada_scaled = scaler.transform(entrada)
    prediccion = modelo.predict(entrada_scaled)[0]
    probabilidad = modelo.predict_proba(entrada_scaled)[0][1]

    if prediccion == 1:
        st.error(f"⚠️ **PHISHING DETECTADO** (Probabilidad: {probabilidad:.2%})")
    else:
        st.success(f"✅ **CORREO LEGÍTIMO** (Probabilidad de phishing: {probabilidad:.2%})")

st.markdown("---")
st.caption("Ejemplo didáctico - Tópicos de Ciberseguridad - Regresión Logística aplicada a detección de phishing.")
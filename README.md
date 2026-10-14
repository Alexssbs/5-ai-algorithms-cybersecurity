<h1 align="center">🛡️ 5 Algoritmos de IA Aplicados a la Ciberseguridad</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-Académica-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <b>Curso:</b> Tópicos de Ciberseguridad &nbsp;|&nbsp;
  <b>Entrega:</b> 17 de septiembre de 2026 &nbsp;|&nbsp;
  <b>Universidad:</b> UNA Puno
</p>

---

## 📖 Descripción

Este repositorio contiene **5 ejemplos prácticos de algoritmos de Inteligencia Artificial aplicados a la Ciberseguridad**, implementados en Python con **scikit-learn** y desplegados mediante interfaces interactivas en **Streamlit**.

Cada ejemplo cubre un paradigma distinto del aprendizaje automático y resuelve un problema real de seguridad:

| # | Algoritmo | Tipo | Caso de uso | Dataset |
|---|---|---|---|---|
| 1 | **Regresión Logística** | Supervisado (clasificación) | Detección de phishing | Phishing & Benign Email (200 correos) |
| 2 | **Isolation Forest** | No supervisado (anomalías) | Anomalías en logs de red | NSG Dataset (500 logs) |
| 3 | **SVM** | Supervisado (clasificación) | Clasificación de malware | Malware & Benign PE (1,156 archivos) |
| 4 | **Random Forest** | Supervisado (ensemble) | Análisis de tráfico de red | CICIDS2017 (5,000 flujos) |
| 5 | **K-Means** | No supervisado (clustering) | Agrupamiento de incidentes SOC | ServiceNow Event Log (24,918 incidentes) |

---

## 🗂️ Estructura del Repositorio

```
📦 5-Algoritmos-IA-Ciberseguridad
│
├── 📁 1-Regresion_Logistica-Phishing
│   ├── app_phishing.re.py
│   └── phishing_features.csv
│
├── 📁 2-Isolation_Forest-Anomalias_Logs
│   ├── app_anomalias.py
│   └── nsg_dataset.csv
│
├── 📁 3-SVM-Clasif_Malware
│   ├── app_malware_svm.py
│   └── Malware_and_benign_recognition.csv
│
├── 📁 4-Random_Forest-Trafico_de_red
│   ├── app_trafico_rf.py
│   └── cicids_muestra.csv
│
└── 📁 5-KNN-Agrupamientos-incidentes
    ├── app_incidentes_kmeans.py
    └── incidentes_agrupados.csv
```

---

## ⚙️ Requisitos Previos

Asegúrate de tener **Python 3.10 o superior** instalado. Luego instala las dependencias:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

---

## 🚀 Cómo Ejecutar Cada Ejemplo

Cada carpeta contiene su propio script de Streamlit y su dataset. Abre una terminal, navega a la carpeta correspondiente y ejecuta el comando indicado.

### 1️⃣ Regresión Logística — Detección de Phishing

```bash
cd 1-Regresion_Logistica-Phishing
streamlit run app_phishing.re.py
```

> **Resultado esperado:** Precisión ≈ 98.33% con 200 correos (100 phishing, 100 legítimos).

---

### 2️⃣ Isolation Forest — Anomalías en Logs

```bash
cd 2-Isolation_Forest-Anomalias_Logs
streamlit run app_anomalias.py
```

> **Resultado esperado:** Detección del 10% de anomalías (50 de 500 registros) con 100 árboles.

---

### 3️⃣ SVM — Clasificación de Malware

```bash
cd 3-SVM-Clasif_Malware
streamlit run app_malware_svm.py
```

> **Resultado esperado:** Precisión ≈ 96.83% con kernel RBF sobre 1,156 archivos PE.

---

### 4️⃣ Random Forest — Tráfico de Red

```bash
cd 4-Random_Forest-Trafico_de_red
streamlit run app_trafico_rf.py
```

> **Resultado esperado:** Precisión ≈ 99.93% sobre 5,000 flujos de CICIDS2017.

---

### 5️⃣ K-Means — Agrupamiento de Incidentes

```bash
cd 5-KNN-Agrupamientos-incidentes
streamlit run app_incidentes_kmeans.py
```

> **Resultado esperado:** 3 clusters con silhouette score = 0.617 sobre 24,918 incidentes.

---

## 📊 Métricas Globales

<div align="center">

| Algoritmo | Precisión | Muestras | Características |
|:---:|:---:|:---:|:---:|
| Regresión Logística | **98.33%** | 200 | 7 |
| Isolation Forest | **90%** (anomalías) | 500 | 7 |
| SVM (RBF) | **96.83%** | 1,156 | 25 |
| Random Forest | **99.93%** | 5,000 | 78 |
| K-Means | **0.617** (silhouette) | 24,918 | 13 |

</div>

---

## 🧠 Conceptos Clave

- **Aprendizaje supervisado:** Regresión Logística, SVM y Random Forest usan etiquetas para entrenar.
- **Aprendizaje no supervisado:** Isolation Forest y K-Means descubren patrones sin etiquetas.
- **Métricas:** Precisión, recall, F1, matriz de confusión, AUC-ROC (clasificación); silhouette score e inercia (clustering).
- **Preprocesamiento:** Extracción de características, codificación de variables categóricas y estandarización.

---

## 🔗 Datasets Utilizados

| Dataset | Fuente | Uso |
|---|---|---|
| Phishing & Benign Email | Hugging Face | Detección de phishing |
| NSG Dataset | Hugging Face | Anomalías en logs |
| Malware & Benign Recognition | Kaggle | Clasificación de malware |
| CICIDS2017 | Kaggle / UNB | Tráfico de red |
| Incident Management Event Log | UCI ML Repository | Agrupamiento de incidentes |

---

## 👨‍💻 Autor

**Alex Chura**
---

## 📜 Licencia

Este repositorio es de uso **académico y educativo**. Los datasets utilizados pertenecen a sus respectivos autores y se emplean únicamente con fines de investigación y aprendizaje.

---

<p align="center">
  <i>"La IA no reemplaza al analista, lo hace más efectivo."</i>
</p>

<p align="center">
  ⭐ Si te resultó útil, no olvides darle una estrella al repositorio.
</p>

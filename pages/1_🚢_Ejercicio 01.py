import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from os import environ

st.header("Ejercicio 01: Análisis del Dataset Titanic")
st.caption("Objetivo: Preparar los datos para un modelo que prediga la supervivencia de los pasajeros.")

@st.cache_data(show_spinner=False)
def cargar_datos():
    try:
        rutaArchivo = Path(environ.get("data_dir") + "/titanic.csv")
        return pd.read_csv(rutaArchivo)
    except:
        raise FileNotFoundError("No se encontró 'titanic.csv' en la carpeta de datos definida en tus variables de entorno. Asegúrate de que el archivo exista.")

try:
    df = cargar_datos()
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()
    
st.subheader("1) Carga de datos")
st.write("Primeras filas (original):")
st.dataframe(df.head())

st.subheader("2) Remover columnas: Name, Ticket y Cabin")
df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)
st.dataframe(df.head())

st.subheader("3) Verificar nulos y reemplazarlos con la moda o media")
for col in df.columns:
    if df[col].dtype in ["object", "int64"]:
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

st.write("Nulos restantes por columna:")
st.dataframe(df.isna().sum().to_frame("nulos"))

st.subheader("4) Codificación de variables categóricas")
arrCodificar = ["Sex", "Embarked"]
encoders = {}
for col in arrCodificar:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
st.write(f"Columnas codificadas: {', '.join(arrCodificar) if arrCodificar else 'ninguna'}")

st.subheader("5) Estandarización de variables numéricas")
scale_cols = [c for c in ["Age", "Fare"] if c in df.columns]
if scale_cols:
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
st.write(f"Columnas estandarizadas: {', '.join(scale_cols) if scale_cols else 'ninguna'}")

st.subheader("6) División Train/Test (70/30)")
y = df["Survived"].astype(int)
X = df.drop(columns=["Survived"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

st.write("Datos de Entrenamiento")
st.dataframe(pd.concat([X_train, y_train], axis=1).head())

st.write("Datos de Test")
st.dataframe(pd.concat([X_test, y_test], axis=1).head())

st.write(f"Shape X_train: {X_train.shape}")
st.write(f"Shape X_test: {X_test.shape}")
st.write(f"Shape y_train: {y_train.shape}")
st.write(f"Shape y_test: {y_test.shape}")

st.subheader("Entrenamiento del Modelo: Regresión Logística")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.subheader("Predicho vs Real (muestra)")
_map = {0: "No", 1: "Sí"}
viz_numeric = pd.DataFrame({
    "Real": y_test.reset_index(drop=True),
    "Predicho": pd.Series(y_pred)
})
viz_df = pd.DataFrame({
    "Real": viz_numeric["Real"].map(_map),
    "Predicho": viz_numeric["Predicho"].map(_map)
})
st.dataframe(viz_df.head(10))
st.scatter_chart(viz_numeric.rename(columns={"Real": "y", "Predicho": "x"}))

st.subheader("Primeros 5 registros procesados")
df_processed_preview = pd.concat([X, y], axis=1).head(5)
st.dataframe(df_processed_preview)

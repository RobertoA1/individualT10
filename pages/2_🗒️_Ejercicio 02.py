import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from os import environ

st.header("Ejercicio 02: Student Performance (Regresión de G3)")
st.caption("Objetivo: Preparar los datos y entrenar un modelo de regresión para predecir la nota final G3.")

@st.cache_data(show_spinner=False)
def cargar_datos():
    data_dir = environ.get("data_dir", "./datos")
    ruta = Path(data_dir) / "student-mat.csv"
    if not ruta.exists():
        raise FileNotFoundError("No se encontró 'student-mat.csv' en la carpeta de datos definida en tus variables de entorno. Asegúrate de que el archivo exista.")
    return pd.read_csv(ruta, delimiter=";")

try:
    df = cargar_datos()
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

st.subheader("1) Vista inicial del dataset")
st.dataframe(df.head())
st.subheader("Análisis de variables categóricas")
colsCategoricas = df.select_dtypes(exclude="number").columns
st.write(f"Columnas categóricas: {', '.join(colsCategoricas) if len(colsCategoricas) else 'ninguna'}")
if len(colsCategoricas):
    resumen_cat = pd.DataFrame({
        "n_unique": [df[c].nunique() for c in colsCategoricas]
    }, index=colsCategoricas)
    st.dataframe(resumen_cat)

st.subheader("2) Eliminación de duplicados y valores inconsistentes + nulos")
df.drop_duplicates(inplace=True)

# Filtrar valores inconsistentes si existen las columnas
before_rows = len(df)
if "age" in df.columns:
    df = df[df["age"].between(10, 30)]
if "absences" in df.columns:
    df = df[(df["absences"] >= 0) & (df["absences"] <= 100)]
for nota_col in ["G1", "G2", "G3"]:
    if nota_col in df.columns:
        df = df[df[nota_col].between(0, 20)]
removed = before_rows - len(df)
st.write(f"Registros removidos por inconsistencias: {removed}")

colsNumericas = df.select_dtypes(include="number").columns
colsCategoricas = df.select_dtypes(exclude="number").columns

df[colsNumericas] = df[colsNumericas].fillna(df[colsNumericas].mean())
for col in colsCategoricas:
    if df[col].isna().any():
        df[col].fillna(df[col].mode().iloc[0], inplace=True)

st.write("Nulos restantes por columna:")
st.dataframe(df.isna().sum().to_frame("nulos"))

st.subheader("3) Codificación de variables categóricas")
to_encode = list(colsCategoricas)
if to_encode:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_array = ohe.fit_transform(df[to_encode])
    ohe_cols = ohe.get_feature_names_out(to_encode)
    df_ohe = pd.DataFrame(ohe_array, columns=ohe_cols, index=df.index)
    df_numeric = df.drop(columns=to_encode)
    df = pd.concat([df_numeric, df_ohe], axis=1)
st.write(f"Columnas codificadas (OneHot): {', '.join(to_encode) if to_encode else 'ninguna'}")

st.subheader("4) Normalización (estandarización) de variables numéricas específicas")
vars_a_escalar = ["age", "absences", "G1", "G2"]
scale_cols = [c for c in vars_a_escalar if c in df.columns]
if scale_cols:
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
st.write(f"Columnas estandarizadas: {', '.join(scale_cols) if scale_cols else 'ninguna'}")

st.subheader("Reto adicional: Correlación entre G1, G2 y G3")
corr_cols = [c for c in ["G1", "G2", "G3"] if c in df.columns]
if len(set(["G1", "G2", "G3"]).intersection(df.columns)) >= 2:
    corr_mat = df[corr_cols].corr()
    st.dataframe(corr_mat)
else:
    st.info("No hay suficientes columnas de notas (G1, G2, G3) para calcular correlaciones.")

st.subheader("5) División Train/Test (80/20)")
y = df["G3"].astype(float)
X = df.drop(columns=["G3"])  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write(f"Shape X_train: {X_train.shape}")
st.write(f"Shape X_test: {X_test.shape}")
st.write(f"Shape y_train: {y_train.shape}")
st.write(f"Shape y_test: {y_test.shape}")

st.subheader("6) Entrenamiento del Modelo: Regresión Lineal")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.3f}")
col2.metric("RMSE", f"{rmse:.3f}")
col3.metric("R²", f"{r2:.3f}")

st.subheader("Predicho vs Real (muestra)")
viz_df = pd.DataFrame({"Real": y_test.reset_index(drop=True), "Predicho": pd.Series(y_pred)})
st.dataframe(viz_df.head(10))
st.scatter_chart(viz_df.rename(columns={"Real": "y", "Predicho": "x"}))

st.subheader("Primeros 5 registros procesados")
st.dataframe(pd.concat([X, y], axis=1).head(5))

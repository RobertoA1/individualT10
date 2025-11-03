import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.header("Ejercicio 03: Dataset Iris")
st.caption("Objetivo: Preprocesar y visualizar resultados del dataset Iris.")

@st.cache_data(show_spinner=False)
def cargar_iris():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    target_names = iris.target_names
    return X, y, target_names

X, y, target_names = cargar_iris()

st.subheader("1) DataFrame con nombres de columnas")
st.dataframe(X.head())

st.subheader("2) Estandarización con StandardScaler")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
st.write("Estadísticas descriptivas (estandarizado):")
st.dataframe(X_scaled.describe().T)

st.subheader("3) División Train/Test (70/30)")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
st.write(f"Shape X_train: {X_train.shape} | X_test: {X_test.shape}")

st.subheader("4) Gráfico de dispersión: sepal length vs petal length por clase")
feat_x = "sepal length (cm)"
feat_y = "petal length (cm)"
fig, ax = plt.subplots(figsize=(6, 4))
for idx, name in enumerate(target_names):
    mask = (y == idx)
    ax.scatter(
        X.loc[mask, feat_x],
        X.loc[mask, feat_y],
        label=name,
        alpha=0.7,
    )
ax.set_xlabel(feat_x)
ax.set_ylabel(feat_y)
ax.legend(title="Clase")
ax.set_title("Iris: Sepal length vs Petal length")
st.pyplot(fig)

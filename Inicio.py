import streamlit as st
import importlib.util
import pathlib
from dotenv import load_dotenv

load_dotenv('./.env')

st.set_page_config(page_title="Ejercicios", page_icon="📚", layout="wide")

st.title("Ejercicios con Streamlit")
st.write("Roberto Quezada, Ing. Sistemas")
st.write("Utiliza el menú de la izquierda para navegar a cada ejercicio.")
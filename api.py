# app.py
import streamlit as st
import requests

# URL de la API de FastAPI
API_URL = "http://127.0.0.1:8000/predict"

# Título de la aplicación
st.title("Predicción de Clase con FastAPI y Streamlit")

# Instrucciones para el usuario
st.write("Ingresa las características del modelo para obtener una predicción")

# Entradas de características del modelo
Gender = st.selectbox("Género", ["Male", "Female"])
Age = st.number_input("Edad", min_value=18, max_value=100, step=1, value=30)
HasDrivingLicense = st.selectbox("Tiene Licencia de Conducir", [0, 1])
RegionID = st.number_input("ID de Región", min_value=0.0, step=1.0, value=1.0)
Switch = st.selectbox("Cambio de Compañía", [0, 1])
PastAccident = st.selectbox("Ha tenido Accidentes", ["Yes", "No"])
AnnualPremium = st.number_input("Prima Anual", min_value=0.0, step=0.01, value=1000.0)

# Convertir datos de entrada de acuerdo a la API
past_accident_binary = 1 if PastAccident == "Yes" else 0

# Botón para realizar la predicción
if st.button("Predecir"):
    # Crear el payload de datos para enviar a la API
    payload = {
        "Gender": Gender,
        "Age": Age,
        "HasDrivingLicense": HasDrivingLicense,
        "RegionID": RegionID,
        "Switch": Switch,
        "PastAccident": PastAccident,
        "AnnualPremium": AnnualPremium
    }
    
    try:
        # Realizar la solicitud POST a la API
        response = requests.post(API_URL, json=payload)
        
        # Verificar si la solicitud fue exitosa
        if response.status_code == 200:
            prediction = response.json()["predicted_class"]
            st.success(f"Predicción de Clase: {prediction}")
        else:
            st.error("Error en la predicción. Revisa los datos de entrada y vuelve a intentarlo.")
    except requests.exceptions.RequestException:
        st.error("Error al conectar con la API. Asegúrate de que está corriendo.")

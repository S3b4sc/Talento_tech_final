import streamlit as st
import pandas as pd
import numpy as np
from scripts.data_loader import loadForAll, loadNNForestData, loadDensityPlot, AllDfFinalLoad, NN_run7History
from scripts.data_cleaner import Allformat, AllAnnotations, densityDataProcess, rf_cleanTransform, rf_hourlyDis_process
from utils import forAllUtils
from scripts.visualizer import plot_map, desnityPlot, rf_plots, rf_hourlyDist, dailyPower, NN_learningPlot
import joblib

# Configuración inicial de la app
st.set_page_config(
    page_title="Dashboard de Análisis y Predicción",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo personalizado
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
    }
    .main-header {
        font-size: 2rem;
        color: #2C3E50;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .divider {
        border-top: 2px solid #BDC3C7;
        margin: 2rem 0;
    }
    .compact-form .stNumberInput {
        max-width: 200px;
        margin-right: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Título de la app
st.markdown('<div class="main-header">Dashboard de Análisis y Predicción</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📂 Navegación")
section = st.sidebar.radio(
    "Selecciona una sección:",
    ["Inicio", "Datos por Departamentos", "Modelos Predictivos"],
    index=0,
    help="Navega entre las diferentes funcionalidades de la aplicación."
)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if section == "Inicio":
    st.header("🏠 Bienvenido")
    st.markdown(
        """
        ¡Bienvenido a este dashboard interactivo! Explora análisis de datos, visualizaciones avanzadas y predicciones.
        
        ### Funcionalidades principales
        - **Análisis por Departamentos**: Gráficos y mapas interactivos.
        - **Modelos Predictivos**: Visualizaciones y predicciones basadas en Random Forest, Redes Neuronales y LSTM.
        """
    )
    st.image("./assets/intro_image.png", caption="Explora tus datos de forma interactiva.")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

elif section == "Datos por Departamentos":
    st.header("🌍 Análisis por Departamentos")
    st.markdown("Visualiza datos relevantes relacionados con diferentes departamentos.")

    st.subheader("1. Mapas Interactivos")
    st.markdown("""Visualiza información geográfica y distribuciones de datos a través de mapas dinámicos.""")
    with st.spinner("Procesando datos del mapa..."):
        col1, col2 = st.columns(2)
        with col1:
            if st.button('🌎 Generar Mapa'):
                try:
                    dataframes = loadForAll(path=forAllUtils['dataPath'], n=10, m=13)
                    dayDeparments = Allformat(dataframes=dataframes)
                    annot = AllAnnotations(dayDepartments=dayDeparments)
                    map_fig = plot_map(annot=annot)
                    st.session_state["map_fig"] = map_fig
                    st.success("¡Mapa generado exitosamente!")
                except Exception as e:
                    st.error(f"Error al generar el mapa: {e}")

        with col2:
            if st.button('📂 Cargar Mapa Guardado'):
                try:
                    annot = joblib.load('./data/processed/annotDict.joblib')
                    map_fig = plot_map(annot=annot)
                    st.session_state["map_fig"] = map_fig
                    st.success("¡Mapa cargado exitosamente!")
                except FileNotFoundError as e:
                    st.error(f"Archivo no encontrado: {e}")

        if "map_fig" in st.session_state:
            st.pyplot(st.session_state["map_fig"])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.subheader("2. Gráficos de Densidad")
    st.markdown("""Analiza la distribución de probabilidades y densidades de los datos.""")
    with st.spinner("Procesando datos de densidad..."):
        if st.button('📈 Generar Gráficos de Densidad'):
            try:
                data = loadDensityPlot('./data/raw/AllDepartments/')
                df_final = densityDataProcess(data=data)
                fig = desnityPlot(df_final=df_final)
                st.session_state["density_fig"] = fig
                st.success("¡Gráfico de densidad generado exitosamente!")
            except Exception as e:
                st.error(f"Error al generar gráficos de densidad: {e}")

        if st.button('📂 Cargar Gráficos Guardados'):
            try:
                df_final = AllDfFinalLoad(path='./data/processed/df_final.parquet')
                fig = desnityPlot(df_final=df_final)
                st.session_state["density_fig"] = fig
                st.success("¡Gráfico de densidad cargado exitosamente!")
            except FileNotFoundError as e:
                st.error(f"Archivo no encontrado: {e}")

        if "density_fig" in st.session_state:
            st.pyplot(st.session_state["density_fig"])

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

elif section == "Modelos Predictivos":
    st.header("🧠 Modelos Predictivos")

    st.subheader("1. Random Forest")
    st.markdown("Visualiza y analiza resultados del modelo Random Forest.")
    col1, col2 = st.columns(2)

    with col1:
        if st.button('🌲 Gráficos Random Forest'):
            try:
                y_test, predictions, indices, feature_importances, features = rf_cleanTransform(path='./data/processed/rf_test_data.csv')
                fig = rf_plots(y_test, predictions, indices, feature_importances, features)
                st.session_state["rf_fig"] = fig
                st.success("¡Gráficos de Random Forest generados exitosamente!")
            except Exception as e:
                st.error(f"Error al generar gráficos: {e}")

    with col2:
        if st.button('⏱️ Distribución Horaria'):
            try:
                rawData = loadNNForestData(path='./data/raw/Timeseries_11.573_-72.814_E5_3kWp_crystSi_16_v45deg_2005_2023.csv', n=10, m=13)
                data = rf_hourlyDis_process(data=rawData)
                fig = rf_hourlyDist(data=data)
                st.session_state["hourly_fig"] = fig
                st.success("¡Distribución horaria generada exitosamente!")
            except Exception as e:
                st.error(f"Error al generar distribución horaria: {e}")

    if "rf_fig" in st.session_state:
        st.pyplot(st.session_state["rf_fig"])
    if "hourly_fig" in st.session_state:
        st.pyplot(st.session_state["hourly_fig"])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.subheader("2. Red Neuronal")
    st.markdown("Visualiza el aprendizaje y realiza predicciones con una red neuronal.")
    col1, col2 = st.columns(2)

    with col1:
        if st.button('⚡ Producción Diaria de Potencia'):
            try:
                rawData = loadNNForestData(path='./data/raw/Timeseries_11.573_-72.814_E5_3kWp_crystSi_16_v45deg_2005_2023.csv', n=10, m=13)
                fig = dailyPower(data=rawData)
                st.session_state["daily_power_fig"] = fig
                st.success("¡Gráfico de potencia diaria generado exitosamente!")
            except Exception as e:
                st.error(f"Error al generar gráfico de potencia: {e}")

    with col2:
        if st.button('📉 Gráficos de Aprendizaje'):
            try:
                history = NN_run7History(path='./data/processed/history_DenseNN_run7.csv')
                fig = NN_learningPlot(history=history)
                st.session_state["learning_fig"] = fig
                st.success("¡Gráfico de aprendizaje generado exitosamente!")
            except Exception as e:
                st.error(f"Error al generar gráfico de aprendizaje: {e}")

    if "daily_power_fig" in st.session_state:
        st.pyplot(st.session_state["daily_power_fig"])
    if "learning_fig" in st.session_state:
        st.pyplot(st.session_state["learning_fig"])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.subheader("3. Predicciones con Red Neuronal")
    model = joblib.load('./models/NNModel')
    scaler = joblib.load('./models/NN_scaler.pkl')

    with st.form("prediction_form"):
        st.write("Ingrese los valores de las variables:")
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.number_input("Gb", min_value=0.0, max_value=1000.0, value=0.0, key='Gb', format='%f')
            var2 = st.number_input("Gd", min_value=0.0, max_value=1000.0, value=0.0, key='Gd', format='%f')
            var3 = st.number_input("Gr", min_value=0.0, max_value=1000.0, value=0.0, key='Gr', format='%f')
            var4 = st.number_input("T", min_value=0.0, max_value=1000.0, value=0.0)
            var5 = st.number_input("W", min_value=0.0, max_value=1000.0, value=0.0)
            submitted = st.form_submit_button("Predecir")

    if submitted:
        try:
            input_data = np.array([[var1, var2, var3, var4, var5]]).reshape(1, -1)
            scaled_input_data = scaler.transform(input_data)
            prediction = model.predict(scaled_input_data)
            st.success(f"El resultado de la predicción es: {prediction[0][0]}")
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

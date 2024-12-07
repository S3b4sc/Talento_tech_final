import streamlit as st
from scripts.data_loader import loadForAll, loadNNForestData, loadDensityPlot,AllDfFinalLoad
from scripts.data_cleaner import Allformat, AllAnnotations,densityDataProcess,rf_cleanTransform,rf_hourlyDis_process
from utils import forAllUtils
from scripts.visualizer import plot_map, desnityPlot, rf_plots, rf_hourlyDist
#from scripts.predictor import load_model, predict    # Aún por der definida

import joblib

# Título de la app
st.title("Pipeline de Datos y Predicciones")

# Cargar datos
st.sidebar.header("Cargar datos")
section = st.sidebar.radio("Selecciona una sección:", ["Inicio", "Departamentos", "Predicciones LST", "Random Forest - Neural Network"])

# Contenido dinámico según la sección seleccionada
if section == "Inicio":
    st.title("Inicio")
    st.write("Bienvenido a la aplicación.")
    
elif section == "Departamentos":
    st.title("Análisis de Datos")
    st.write("Aquí puedes cargar y visualizar datos.")
    
    # Si presiona el botón, se ejecuta el pipeline para todos los departamentos
    if st.button('Generate'):
        # Cargamos los datos para todos los departamentos
        dataframes = loadForAll(path=forAllUtils['dataPath'],n=10,m=13)
        # Agrupamos para obtneer registros por día
        dayDeparments = Allformat(dataframes=dataframes)
        # Obtenemos las anotaciones de cada departamento para el gráfico
        annot = AllAnnotations(dayDepartments=dayDeparments)
        # Realizamos el plot del mapa con las anotaciones
        map_fig = plot_map(annot=annot)
        st.pyplot(map_fig)
        
    if st.button('Load'):
        try:
            # Cargamos los resultados guardados
            annot = joblib.load('./data/processed/annotDict.joblib')
            # Realizamos el plot del mapa con las anotaciones
            map_fig = plot_map(annot=annot)
            #Realizamos el plot del mapa con las anotaciones
            st.pyplot(map_fig)
        except FileNotFoundError as e:
            st.write(e)
            
    if st.button('Density Gen'):
        data = loadDensityPlot('./data/raw/AllDepartments/')
        df_dinal = densityDataProcess(data=data)
        fig = desnityPlot(df_final=df_dinal)
        st.pyplot(fig)

    if st.button('Density Load'):
        df_dinal = AllDfFinalLoad(path='./data/processed/df_final.parquet')
        fig = desnityPlot(df_final=df_dinal)
        st.pyplot(fig)
    
    
elif section == "Random Forest - Neural Network":
    st.title("Predicciones")
    st.write("Sube datos para realizar predicciones.")
    
    if st.button('Start'):
        rawData = loadNNForestData(path='./data/raw/Timeseries_11.573_-72.814_E5_3kWp_crystSi_16_v45deg_2005_2023.csv',n=10,m=13)

        y_test, predictions, indices, feature_importances, features = rf_cleanTransform(path='./data/processed/rf_test_data.csv')
        fig = rf_plots(y_test, predictions, indices, feature_importances, features)
        st.pyplot(fig)
    if st.button('hourlyDist'):
        rawData = loadNNForestData(path='./data/raw/Timeseries_11.573_-72.814_E5_3kWp_crystSi_16_v45deg_2005_2023.csv',n=10,m=13)
        data = rf_hourlyDis_process(data=rawData)
        fig = rf_hourlyDist(data=data)
        st.pyplot(fig)
        
        
elif section == "Configuración":
    st.title("Configuración")
    st.write("Configura los parámetros del modelo.")
    # Agrega opciones de configuración
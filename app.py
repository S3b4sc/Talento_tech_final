import streamlit as st
import pandas as pd
import numpy as np
from scripts.data_loader import loadForAll, loadNNForestData, loadDensityPlot,AllDfFinalLoad, NN_run7History
from scripts.data_cleaner import Allformat, AllAnnotations,densityDataProcess,rf_cleanTransform,rf_hourlyDis_process
from utils import forAllUtils
from scripts.visualizer import plot_map, desnityPlot, rf_plots, rf_hourlyDist, dailyPower, NN_learningPlot
#from scripts.predictor import load_model, predict    # Aún por der definida

import joblib

# Título de la app
st.title("Pipeline de Datos y Predicciones")

# Cargar datos
st.sidebar.header("Seciones.")
section = st.sidebar.radio("Selecciona una sección:", ["Inicio", "Departamentos", "Predicciones LSTM", "Random Forest - Neural Network"])


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if section == "Inicio":
    st.title("Inicio")
    st.write("Bienvenido a la aplicación.")
    
elif section == "Departamentos":
    st.title("Análisis de Datos")
    st.write("Aquí puedes cargar y visualizar datos.")
    
    st.write('Para los gráficos del mapa')
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
    
    st.write('Para los gráficos de densidad de probabilidad')        
    if st.button('Density Gen'):
        data = loadDensityPlot('./data/raw/AllDepartments/')
        df_dinal = densityDataProcess(data=data)
        fig = desnityPlot(df_final=df_dinal)
        st.pyplot(fig)

    if st.button('Density Load'):
        df_dinal = AllDfFinalLoad(path='./data/processed/df_final.parquet')
        fig = desnityPlot(df_final=df_dinal)
        st.pyplot(fig)
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


elif section == "Predicciones LSTM":
    st.title("LSTM figures and predictions")
    st.write("Configura los parámetros del modelo.")
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
elif section == "Random Forest - Neural Network":
    st.title("Predicciones")
    st.write("Sube datos para realizar predicciones.")
    
    st.write('Para los gráficos de Random Forest')
    if st.button('Start'):
        #rawData = loadNNForestData(path='./data/raw/Timeseries_11.573_-72.814_E5_3kWp_crystSi_16_v45deg_2005_2023.csv',n=10,m=13)

        y_test, predictions, indices, feature_importances, features = rf_cleanTransform(path='./data/processed/rf_test_data.csv')
        fig = rf_plots(y_test, predictions, indices, feature_importances, features)
        st.pyplot(fig)
    if st.button('hourly Distribution'):
        rawData = loadNNForestData(path='./data/raw/Timeseries_11.573_-72.814_E5_3kWp_crystSi_16_v45deg_2005_2023.csv',n=10,m=13)
        data = rf_hourlyDis_process(data=rawData)
        fig = rf_hourlyDist(data=data)
        st.pyplot(fig)
    
    st.write('Para los gráficos de Red Neuronal Densa')
    if st.button('Daily Power'):
        
        # Gráfico de producción de potencia diaria
        rawData = loadNNForestData(path='./data/raw/Timeseries_11.573_-72.814_E5_3kWp_crystSi_16_v45deg_2005_2023.csv',n=10,m=13)
        fig = dailyPower(data=rawData)
        st.pyplot(fig)
        
    if st.button('Learning NN'):
        # Gráfico del aprendizaje de la red neuronal
        history = NN_run7History(path='./data/processed/history_DenseNN_run7.csv')
        fig = NN_learningPlot(history=history)
        st.pyplot(fig)
    
    if st.button('pred vs real'):
        data = pd.read_csv('./data/processed/NN_pred.csv').sample(100)
        # Mostrar el DataFrame con scroll
        st.dataframe(data, height=300)


    # Formulario para predcción con red neuronal
    model = joblib.load('./models/NNModel')
    scaler = joblib.load('./models/NN_scaler.pkl')
    # Crear un formulario
    with st.form("prediction_form"):
        st.write("Ingrese los valores de las variables:")
        # Supongamos que tu modelo necesita 3 características de entrada
        var1 = st.number_input("Gb", min_value=0.0, max_value=1000.0, value=0.0)
        var2 = st.number_input("Gd", min_value=0.0, max_value=1000.0, value=0.0)
        var3 = st.number_input("Gr", min_value=0.0, max_value=1000.0, value=0.0)
        var4 = st.number_input("T", min_value=0.0, max_value=1000.0, value=0.0)
        var5 = st.number_input("W", min_value=0.0, max_value=1000.0, value=0.0)
        # Botón para enviar el formulario
        submitted = st.form_submit_button("Predecir")
    # Procesar el formulario cuando se envía
    if submitted:
        # Crear un array con los datos ingresados
        input_data = np.array([[var1, var2, var3, var4, var5]]).reshape(1,-1)
        # Escalamos la entrada
        scaled_input_data = scaler.transform(input_data)
        # Hacer la predicción
        prediction = model.predict(scaled_input_data)
        # Mostrar el resultado
        st.write("El resultado de la predicción es:", prediction[0][0])
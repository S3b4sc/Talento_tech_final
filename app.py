import streamlit as st
import pandas as pd
import numpy as np
from scripts.data_loader import loadForAll, loadNNForestData, loadDensityPlot, AllDfFinalLoad, NN_run7History
from scripts.data_cleaner import Allformat, AllAnnotations, densityDataProcess, rf_cleanTransform, rf_hourlyDis_process, Oneformat, OneAnnotations
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
st.markdown("""
    <style>
    body {
        background-color: #a3e4d7;
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        font-size: 2rem;
        color: #2C3E50;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #2980B9;
        margin-bottom: 0.5rem;
    }
    .divider {
        border-top: 2px solid #BDC3C7;
        margin: 2rem 0;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    </style>
    """, unsafe_allow_html=True)


# Título de la app
st.markdown('<div class="main-header">Dashboard de Análisis y Predicción</div>', unsafe_allow_html=True)


# Sidebar
st.sidebar.header("📂 Navegación")
section = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["Inicio", "Paneles solares: departamentos", "Paneles solares: Modelos Predictivos", "Resultados"],
    index=0,
    help="Navega entre las diferentes funcionalidades de la aplicación."
)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if section == "Inicio":
    st.header("🏠 Bienvenido")
    st.markdown("""
        ¡Bienvenido a este dashboard interactivo! Explora los análisis realizados que buscan entender acerca de 
        la implementación energías renovables en Colombia: energía fotovoltaica y mareomotriz
    """)
    
    with st.expander("Ver funcionalidades principales"):
        st.markdown("""
        ### Funcionalidades principales
        - **Análisis por Departamentos**: Gráficos y análisis regional estático y personalizado.
        - **Modelos Predictivos**: Visualizaciones y predicciones basadas en Random Forest, Redes Neuronales y LSTM.
        - **Imágenes con insights importantes**: Imágenes que ayudan a comprender el comportamiento de dos tipos de generación de energía y los datos disponibles.
        """)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

elif section == "Paneles solares: departamentos":
    st.header("🌍 Análisis por Departamentos")
    st.markdown("""Visualiza datos relevantes relacionados con diferentes departamentos.
                En este apartado, encuentras los resultados de un análisis estadístico que busca comprender 
                cuáles regiones de Colombia son viables para la implementación de paneles solares basados en la eficiencia
                y estabilidad que pueden brindar estos sistemas a lo largo de un periodo definido.
                """)

    st.subheader("1. Análisis regional")
    st.markdown("""Visualiza información geográfica y distribuciones de datos bajo una segmentación regional.
                Puedes escoger si deseas generar o cargar los datos para visualizar el gráfico, ten en cuenta que generarlo 
                toma un poco más de tiempo.
                \n Para las visualización regional, se tomaron datos de radiación a lo largo de un mes para todos los departamentos, 
                y en base a la probabilidad promedio de producir una cantidad de energía diaria baja, normal o alta se categoriza
                como viable o inviable la implementación de los páneles solares.
                """)
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
    st.markdown("""Analiza la distribución de probabilidades y densidades de los datos.
                \n Para gestionar el análisis regional, fue necesario estimar la distribución de probabilidad que tiene
                un panel solar de generar cierta cantidad de potencia cuando es sometido a ciertas condiciones 
                ambientales, para esto, se consideran las componentes de la radiación y con ayuda de el método de estimación
                de densidad de probabilidad por kernerl gausianos, se hizo una estimación de la probabilidad que tiene un panel de generar
                en un día una cantidad de potencia baja, normal o alta.
                \n Para ejemplificar, se muestra una proyección 2D de la función densidad de probabilidad para las tres categorías.
                Nuevamente, puede generar los datos o cargarlos.
                """)
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
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("3. Predice tus datos")
    st.markdown("""Si desea conocer la viabilidad basada en datos histórico para un lugar en particular,
                tiene laposibilidad de usar este programa para montar su archivo csv.
                \nEl archvio csv debe tener el mismo formato y datos que hay en la página:
                 https://re.jrc.ec.europa.eu/pvg_tools/en/
                 \nSi no tiene datos, le recomendamo dirigirse a ese link, escoger el lugar de preferencia, y descargar datos
                 históricos con reporte de hora a hora de las condiciones climáticas, una vez ingrese el archvio csv se realizará la
                 predicción de viabilidad basa en el modelo estadístico preprogramado.
                 \n Asegurese de descargar datos por horas para mayor confibilidad, puede descargar y montar datos historicos de varios años,
                 Es importante que se asegure que descarge incluyendo las componentes de la radiación y mantenga el formato correcto.
                 \n Descarga y monta en el apartado estos datos de prueba, tomados de Antioquia de los años 2022 y 2023.
                """)
    # Leer el archivo para asegurarte de que exista y sea accesible
    try:
        with open("./data/raw/oneprueba.csv", "r") as file:
            csv_data = file.read()

        # Crear el botón de descarga
        st.download_button(
            label="Descargar archivo CSV de prueba",
            data=csv_data,
            file_name="oneprueba.csv",
            mime="text/csv"
        )
    except FileNotFoundError:
        st.error("El archivo no se encuentra en la ruta especificada.")
    # Crear un widget para subir archivos
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    # Verificar si se cargó un archivo
    if uploaded_file is not None:
        # Leer el archivo CSV en un DataFrame
        df = pd.read_csv(
            uploaded_file, 
            skiprows=8,  # Saltar las líneas iniciales
            skipfooter=13,  # Saltar las líneas finales
            engine='python'  # Necesario para usar skipfooter
        )
        
        
        dayPlace = Oneformat(data=df)
        annot = OneAnnotations(dayPlace=dayPlace)
        st.success("Resultados obtenidos...")
        #formatted_annot = annot.replace("\n", "  \n")
        #st.markdown(formatted_annot)
        st.markdown(annot['place'])
        
    else:
        st.write("Por favor, sube un archivo CSV para comenzar.")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

elif section == "Paneles solares: Modelos Predictivos":
    st.header("🧠 Modelos Predictivos")
    
    st.markdown(""" En esta sección, puede explorar sobre los modelos de Aprendizaje estadístico que permiten obtener insights importates
                sobre la viabilidad de implementación de generación de energía renovable através de paneles solares y mareomotriz.
                """)
    
    # Usar tabs para dividir el contenido
    tab1, tab2, tab3 = st.tabs(["Random Forest", "Red Neuronal", "Predicciones"])

    with tab1:
        st.subheader("1. Random Forest")
        st.markdown("""Presentamos una alternativa diferente para categorizar la posible cantidad de producción de potencia de un sistema de paneles solares
                    presentada en la sección de análisis regional, recordemos que el modelo analítico presentado en la otra sección, permite estimar 
                    mediante probabilidades la categoría de produción de un sistema bajo ciertas condiciones ambientales acotadas en intervalos
                    restringidos por la naturaleza de los datos, sin embargo, implementando un modelo de Machine Learning como RandomForest
                    podemos realizar clasificaciónes que extrapolan los intervalosque antes eran una barrera.
                    
                    \nEn esta sección puede encontrar las métricas relacionadas con el entrenamiento del modelo que permite encontrar
                    una opción viable para llevar acabo análisis estadísticos similares al análisis regional.
            
                    """)
        with st.spinner("Generando gráficos de Random Forest..."):
            if st.button('🌲 Random Forest: entrenamiento'):
                try:
                    y_test, predictions, indices, feature_importances, features = rf_cleanTransform(path='./data/processed/rf_test_data.csv')
                    fig = rf_plots(y_test, predictions, indices, feature_importances, features)
                    st.session_state["rf_fig"] = fig
                    
                    st.success("¡Gráficos de Random Forest generados exitosamente!")
                except Exception as e:
                    st.error(f"Error al generar gráficos: {e}")


            if st.button('🌲 Performance de modelo'):
                st.markdown("""
                    A continuación se muestran algunas imágenes de la precisión del modelo junto con la ROC.
                    """)
                st.image('./images/model_preci.png', use_container_width=False)
                st.success("¡Gráficos de Random Forest generados exitosamente!")
                
            if "rf_fig" in st.session_state:
                st.pyplot(st.session_state["rf_fig"])

        
        
    with tab2:
        st.subheader("2. Red Neuronal")
        st.markdown("""A parte de la posibilidad de generar modelos de predicción utilizando Machine Learning, podemos emplear 
                    modelos de regresión que puedes brindar información muy importante para la toma de decisiones.
                    \nEn este apartado, puede visualizar el comportamiento de una red neuronal densa (DNN) cuyo propśito es
                    ejecutar una regresión multivariada de la potencia que puede generar un array de 10 páneles solares
                    bajo ciertas condiciones ambientales.
                    \nComo se puede evidenciar en los gráficos de entrenamiento, el error cuadrádico medio de los datos predichos con los
                    reales es bastante bajo y permite obtener buenas predicciones.
                    \nEl modelo reportado cuenta con un error cuadrático medio de 6 Watts approx.
                    """)
        with st.spinner("Generando gráfico de aprendizaje..."):
            if st.button('📉 Gráficos de Aprendizaje'):
                try:
                    history = NN_run7History(path='./data/processed/history_DenseNN_run7.csv')
                    fig = NN_learningPlot(history=history)
                    st.session_state["learning_fig"] = fig
                    st.success("¡Gráfico de aprendizaje generado exitosamente!")
                except Exception as e:
                    st.error(f"Error al generar gráfico de aprendizaje: {e}")

            if "learning_fig" in st.session_state:
                st.pyplot(st.session_state["learning_fig"])
            
            if st.button("📉 Prediciones (P watts)"):
                df = pd.read_csv('./data/processed/NN_pred.csv').sample(100).reset_index(drop=True)
                st.dataframe(df, height=400)

    with tab3:
        st.subheader("3. Predicciones con Red Neuronal")
        model = joblib.load('./models/NNModel')
        scaler = joblib.load('./models/NN_scaler.pkl')

        st.markdown("""En este apartado, tiene disponible el modelo de red neuronal densa (DNN) para realizar predicciones
                    de cuánta potencia en Watts esperaría que un array de 10 páneles solares genere en una hora bajo ciertas condiciones ambientales.
                    """)
        with st.form("prediction_form"):
            st.write("Ingrese los valores de las variables:")
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.number_input("Gb", min_value=0.0, max_value=1000.0, value=0.0)
                var2 = st.number_input("Gd", min_value=0.0, max_value=1000.0, value=0.0)
                var3 = st.number_input("Gr", min_value=0.0, max_value=1000.0, value=0.0)
                var4 = st.number_input("T", min_value=10.0, max_value=1000.0, value=25.5)
                var5 = st.number_input("W", min_value=2.0, max_value=1000.0, value=3.7)
                submitted = st.form_submit_button("Predecir")

        if submitted:
            try:
                input_data = np.array([var1, var2, var3, var4, var5]).reshape(1, -1)
                scaled_input_data = scaler.transform(input_data)
                prediction = model.predict(scaled_input_data)
                st.success(f"El resultado de la predicción es: {prediction[0][0]}")
            except Exception as e:
                st.error(f"Error en la predicción: {e}")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

elif section == "Resultados":
    st.header("🖼️ Sección de Imágenes")
    st.markdown("""
    En esta sección, podrás ver algunas imágenes relacionadas con el análisis y las predicciones.
    """)

    st.subheader("1. Imágenes de Análisis")
    st.markdown("""
    Resultados de predicción de potencia basado en red neuronal.
    """)

    st.image('./images/compare_real_predict.png', use_container_width=True)

    st.markdown("""
    Distribución de producción diaria histórica en la guajira.
    """)
    rfhourlydata = loadNNForestData(path='./data/raw/Timeseries_11.573_-72.814_E5_3kWp_crystSi_16_v45deg_2005_2023.csv',
                                    n=10,
                                    m=13)
    rfhourlydist = rf_hourlyDis_process(data=rfhourlydata)
    fig1 = rf_hourlyDist(data=rfhourlydist)
    st.pyplot(fig1)
    st.markdown("""
    Potencia diaría generada en la Guajira desde 2005 para 10 paneles solares..
    """)
    fig2 = dailyPower(data=rfhourlydist)
    st.pyplot(fig2)
    
    st.markdown("""
    Convergencia del modelo de red neuronal Long Short Term Memory (LSTM) para los páneles solares
    """)

    st.image('./images/Loss_plot_effi.png', use_container_width=False)

    st.image('./images/Loss_plot_power.png', use_container_width=False)
    
    st.markdown("""
    Distribución de potencia de los páneles solares.
    """)

    st.image('./images/normal_dist_power.png', use_container_width=False)

    st.markdown("""
    Predición de red neuronal recurrente Long Short Term Memory para la potencia, resultados reales vs predicción. para
    la producción de energía mareomotriz en un area de 50km cuadrados.
    """)
    st.image('./images/Tidal_energy.png', use_container_width=False)

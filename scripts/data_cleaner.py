
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def toDay(data:pd.DataFrame):
    """
    Groups the input DataFrame by day and computes the daily mean for selected columns.
    Filters out rows where 'Gb(i)' <= 0 before grouping.

    Args:
        data (pd.DataFrame): Input DataFrame with a 'time' column (datetime) and columns ['Gb(i)', 'Gd(i)', 'Gr(i)'].

    Returns:
        pd.DataFrame: A grouped DataFrame with the daily mean of 'Gb(i)', 'Gd(i)', and 'Gr(i)'.
    """
    
    data['Dia'] = data['time'].dt.date
    
    # Filtrar las horas donde G(i) > 0 (ignorar radiación cero)
    df_filtered = data[data['Gb(i)'] > 0]

    # Agrupar por día, calculando el promedio de G(i) por día
    df_avg = df_filtered.groupby(['Dia'])[['Gb(i)','Gd(i)','Gr(i)']].mean().reset_index()

    return df_avg

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Allformat(*,dataframes:dict):
    """
    Processes a dictionary of DataFrames by correcting the 'time' column format and 
    grouping each DataFrame by day using the 'toDay' function.

    Args:
        dataframes (dict): Dictionary where keys are department names and values are DataFrames.

    Returns:
        dict: Dictionary with grouped DataFrames (daily averages) for each department.
    """
    # Agrupamos cada uno de los departamentos por días
    dayDepartments = {}
    for department in dataframes.keys():
        # Corregimos el formato
        dataframes[department]['time'] = pd.to_datetime(dataframes[department]['time'], format='%Y%m%d:%H%M')
        # Agrupamos por días
        dayDepartments[department] = toDay(dataframes[department])

    return dayDepartments

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def predProb(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates class probabilities for each row in the DataFrame using pre-trained Kernel Density Estimation (KDE) models.
    Adds the normalized probabilities as new columns to the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame with columns ['Gb(i)', 'Gd(i)', 'Gr(i)'].

    Returns:
        pd.DataFrame: The input DataFrame with added columns for normalized probabilities for each class.
    """
    # Lista de clases previamente identificadas ('low', 'normal', 'high')
    clases = ['low', 'normal', 'high']
    values = data[['Gb(i)', 'Gd(i)', 'Gr(i)']]
    
    # Crear un DataFrame para almacenar las probabilidades
    probabilidades = pd.DataFrame(index=values.index, columns=clases)

    # Calcular las probabilidades para cada clase
    for clase in clases:
        # Cargar el modelo guardado para cada clase
        modelo_path = f'./models/kernel_{clase}'
        kde = joblib.load(modelo_path)
        
        # Calcular la densidad logarítmica y convertir a probabilidad
        log_densidad = kde.score_samples(values)  # Usar el DataFrame 
        probabilidades[clase] = np.exp(log_densidad) # Convertir log-densidad a densidad

    # Normalizar probabilidades (como DataFrame)
    probabilidades_normalizadas = probabilidades.div(probabilidades.sum(axis=1), axis=0)

    # Agregar las probabilidades normalizadas al DataFrame original
    for clase in clases:
        data[f'Prob_{clase}'] = probabilidades_normalizadas[clase]

    return data


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def viability(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a global viability score based on normalized production probabilities and 
    determines whether the production is viable or not.

    Args:
        data (pd.DataFrame): DataFrame with columns ['Prob_low', 'Prob_normal', 'Prob_high'].

    Returns:
        str: Annotation with average probabilities for each class and the viability status ('Viable' or 'Not Viable').
    """
    # Asegurar que las columnas necesarias están presentes
    required_columns = ['Prob_low', 'Prob_normal', 'Prob_high']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"La columna '{col}' no está en el DataFrame.")
    
    # Calcular la frecuencia de cada categoría
    total_dias = len(data)
    
    meanProbHigh = data['Prob_high'].mean()
    meanProbNormal = data['Prob_normal'].mean()
    meanProbLow = data['Prob_low'].mean()
    
    if (meanProbHigh + meanProbNormal >= 2* meanProbLow):
        viability = 'Viable'
    else:
        viability = 'Not Viable'
        
    # Normalizar probabilidades promedio
    
    annotation = f""" 
    daily high prob: {np.round(meanProbHigh,3)}
    daily normal prob: {np.round(meanProbNormal,3)}
    daily low prob: {np.round(meanProbLow,3)}
            {viability}
    """

    return annotation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def AllAnnotations(*,dayDepartments:dict):
    """
    Processes a dictionary of DataFrames by predicting probabilities and calculating viability annotations for each department.

    Args:
        dayDepartments (dict): Dictionary where keys are department names and values are daily-aggregated DataFrames.

    Returns:
        dict: Dictionary with viability annotations for each department.
    """
    
    annot = {}

    for department in dayDepartments.keys():
        # Predecimos el score 
        dayDepartments[department] = predProb(dayDepartments[department])
        # Calculamos las anotaciones a mostrar en el gráfico
        annot[department] = viability(dayDepartments[department])
    
    joblib.dump(annot,'./data/processed/annotDict.joblib')
    
    return annot
  
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def densityDataProcess(*,data:pd.DataFrame):
    
    # Filtrar las horas donde G(i) > 0 (ignorar radiación cero)
    df_filtered = data[data['Gb(i)'] > 0]
    # Agrupar por departamento y día, y sumar la producción diaria (kWp)
    df_daily = data.groupby(['Departamento', 'Dia'])['kWh'].sum().reset_index()
    df_daily.rename(columns={'kWh': 'ProduccionDiaria(Kwh)'}, inplace=True)

    # Definir los umbrales para las categorías
    high_threshold = 24.9  
    low_threshold = 15.48   

    # Crear una nueva columna 'Producción' con la categoría
    def categorize_production(p):
        if p >= high_threshold:
            return 'high'
        elif p <= low_threshold:
            return 'low'
        else:
            return 'normal'

    df_daily['Produccion'] = df_daily['ProduccionDiaria(Kwh)'].apply(categorize_production)
    # Filtrar las horas donde G(i) > 0 (ignorar radiación cero)
    df_filtered = data[data['Gb(i)'] > 0]

    # Agrupar por departamento y día, calculando el promedio de G(i) por día
    df_avg_radiation = df_filtered.groupby(['Departamento', 'Dia'])[['Gb(i)','Gd(i)','Gr(i)','H_sun','T2m','P']].mean().reset_index()
    df_avg_radiation.rename(columns={'G(i)': 'PromedioRadiacion'}, inplace=True)

    # Combinar con el DataFrame que contiene las categorías de producción
    # (df_daily es el DataFrame con Producción creado anteriormente)
    df_final = pd.merge(df_daily, df_avg_radiation, on=['Departamento', 'Dia'], how='left')

    df_final.to_parquet('./data/processed/df_final.parquet', engine='pyarrow', compression='snappy') 
    
    return df_final

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def rf_cleanTransform(*,path:str):
    # Cargar los datos desde el archivo CSV
    loaded_test_df = pd.read_csv(path)

    # Separar nuevamente X_test y y_test
    X_test = loaded_test_df.drop(columns=['Target']).values
    y_test = loaded_test_df['Target'].values

    # Cargamos el modelo
    model = joblib.load('./models/randomForest')
    
    # Make predictions
    predictions = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, predictions)
    testReport = classification_report(y_test,predictions)

    # Feature importance
    feature_importances = model.feature_importances_
    features = ['H_sun','Gr(i)','Gb(i)','Gd(i)','hour','T2m']
    indices = np.argsort(feature_importances)

    return y_test, predictions, indices, feature_importances,features

def rf_hourlyDis_process(*,data:pd.DataFrame):
    data['time'] = pd.to_datetime(data['time'], format='%Y%m%d:%H%M')
    data['hour'] = data['time'].dt.hour
    mean = data['P'].mean()
    std = data['P'].std()
    
    def target(power):
        if power < (mean - mean/2):
            return 'Low'
        elif ( power > (mean + mean/2) ):
            return 'High'
        else:
            return 'Normal'
        
    data['Target'] = data['P'].apply(target)
    
    return data
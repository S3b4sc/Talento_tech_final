from utils import departments_coordinates

import pandas as pd
from io import StringIO
import os


# Functions to load data for random forest

def loadNNForestData(*,path:str, n:int,m:int) -> pd.DataFrame:
    """
    Load the data to be used in the RandomForest model and Dense Neural Network model
    
    args:
        path: Str   the relative location to the .csv file
        n:    int   Number of first rows to ignore
        m:    int   Number of last rows to ignore
    
    returns:
        rawData     pandas dataframe with the loded data
    """
    # Read the file
    with open(path, 'r') as file:
        lines = file.readlines()

    # Remove first nth and last mth lines
    csvLines = lines[n:-m]

    # Join the lines and load the to pandas
    csvData = StringIO("".join(csvLines))
    rawData = pd.read_csv(csvData)
    rawData['time'] = pd.to_datetime(rawData['time'], format='%Y%m%d:%H%M')

    return rawData

def loadForAll(*,path:str,n:int,m:int) -> pd.DataFrame:
    """
    Load the data to be used for the all departments production analysis
    the data is loaded one by one csv and concatenated to retunr onw single pd.DataFrame
    
    args:
        path: Str   the relative location to the .csv file
        n:    int   Number of first rows to ignore
        m:    int   Number of last rows to ignore
    
    returns:
        combined_df     pandas dataframe with the loded data
    """

    # Lista para almacenar los DataFrames
    dataframes = {}

    for filename in os.listdir(path):
        if filename.endswith(".csv"):  
            department_name = filename.replace(".csv", "")  # Obtener el nombre del departamento
            file_path = os.path.join(path, filename)  # Ruta completa del archivo

            # Leer el archivo CSV y asignarlo al diccionario
            dataframes[department_name] = pd.read_csv(file_path,skiprows=n,skipfooter=m,engine="python")   # Necesario para skipfooter
    
    return dataframes

def loadDensityPlot(path:str):
    #Lista para almacenar los DataFrames
    dataframes = []

    # Recorrer cada archivo CSV en la carpeta
    for file_name in os.listdir(path):
        if file_name.endswith(".csv"):  # Verificar que sea un archivo CSV
            # Ruta completa del archivo
            file_path = os.path.join(path, file_name)

            # Leer el archivo, ignorando las primeras N filas y últimas M
            df = pd.read_csv(
                file_path,
                skiprows=10,       
                skipfooter=13,     
                engine="python"   # Necesario para skipfooter
            )

            # Añadir el nombre del departamento como columna
            department_name = file_name.replace(".csv", "")
            df["Departamento"] = department_name

            # Limpiar columnas innecesarias 
            df = df[["time", "P", "Gb(i)","Gr(i)","Gd(i)", "H_sun","T2m","WS10m","Departamento"]]

            # Agregar el DataFrame limpio a la lista
            dataframes.append(df)

    # Combinar todos los DataFrames en uno solo
    data = pd.concat(dataframes, ignore_index=True)
    data['time'] = pd.to_datetime(data['time'], format='%Y%m%d:%H%M')
    data['latitude'] = data['Departamento'].map(lambda x: departments_coordinates.get(x, {}).get('latitude'))
    data['longitude'] = data['Departamento'].map(lambda x: departments_coordinates.get(x, {}).get('longitude'))
    data['kWh'] = data['P'] / 1000
    data['Dia'] = data['time'].dt.date
    
    return data

def AllDfFinalLoad(*,path:str):
    df_final = pd.read_parquet(path, engine='pyarrow')
    return df_final
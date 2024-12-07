import pandas as pd

def addDepartments(data:pd.DataFrame, coords:dict, threshold:list = [24.9,15.48]):
    
    # Set latitude and lingitude to each department
    data['latitude'] = data['Departamento'].map(lambda x: coords.get(x, {}).get('latitude'))
    data['longitude'] = data['Departamento'].map(lambda x: coords.get(x, {}).get('longitude'))
    
    # Find the produced power on Kwh
    data['kWh'] = data['P'] / 1000
    
    # Set dat column 
    data['Dia'] = data['time'].dt.date
    
    # Agrupar por departamento y día, y sumar la producción diaria (kWp)
    df_daily = data.groupby(['Departamento', 'Dia'])['kWh'].sum().reset_index()
    df_daily.rename(columns={'kWh': 'ProduccionDiaria(Kwh)'}, inplace=True)
    
    # Definir los umbrales para las categorías
    high_threshold = threshold[0]
    low_threshold = threshold[1] 

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
    
    return df_final
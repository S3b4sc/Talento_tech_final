from utils import departments_coordinates

import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_map(*,annot:dict):
    
    # Descargar datos directamente desde Natural Earth
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    colombia = world[world['ADMIN'] == 'Colombia']

    # Crear el plot
    fig, ax = plt.subplots(figsize=(20, 20))
    colombia.plot(ax=ax, color='lightgrey', edgecolor='black')

    # Agregar anotaciones
    for dept, coords in departments_coordinates.items():
        lat, lon = coords['latitude'], coords['longitude']
        ax.plot(lon, lat, 'ro')  # Punto rojo en las coordenadas
        annotation = f'{dept}\n' + annot[dept] 

        if 'Not' in annot[dept]:
            ax.text(lon + 0.1, lat, annotation, fontsize=6, color='blue')  # Texto cerca del punto
        else:
            ax.text(lon + 0.1, lat, annotation, fontsize=6, color='green')  # Texto cerca del punto

    # Personalizar el plot
    ax.set_title('Mapa de Colombia con Anotaciones', fontsize=16)
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    
    return fig

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def desnityPlot(*,df_final:pd.DataFrame):
    
    fig = plt.figure(figsize=(10,7))

    sns.kdeplot(
        data=df_final,
        x='Gb(i)',
        y='Gd(i)',
        hue='Produccion',  # Categoría para diferenciar las densidades
        fill=False,  # Densidades sin relleno
        common_norm=False  # Evitar normalizar entre categorías
    )

    plt.title("Densidad estimada por categoría")
    plt.xlabel("Gb(i)")
    plt.ylabel("Gd(i)")
    
    return fig

def rf_plots(y_test, predictions, indices, feature_importances, features):
    
    # Crear la figura y los subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Matriz de confusión
    cm_test = confusion_matrix(y_test, predictions, labels=['Low', 'Normal', 'High'])
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['Low', 'Normal', 'High'])
    disp_test.plot(ax=axes[0], colorbar=False)  # Dibujar en el primer subgráfico
    axes[0].set_title('Confusion Matrix')
    axes[0].grid(False)

    # Gráfico 2: Importancia de características
    axes[1].barh(range(len(indices)), feature_importances[indices], align="center")
    axes[1].set_yticks(range(len(indices)))
    axes[1].set_yticklabels([features[i] for i in indices])
    axes[1].set_xlabel("Relative Importance")
    axes[1].set_title("Feature Importance")

    # Ajustar diseño
    plt.tight_layout()
    return fig

def rf_hourlyDist(*,data:pd.DataFrame):
    
    # Hourly distribution of classes
    fig = plt.figure(figsize=(12,7))
    sns.countplot(data=data, x='hour', hue='Target')
    plt.title("Target Distribution by Hour\n", fontsize=15)

    plt.legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Mostrar la gráfica
    plt.tight_layout()  # Ajustar la figura para evitar recortes
    return fig
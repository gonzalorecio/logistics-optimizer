import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter


import utils
import algorithm as alg


# App title
st.title('Optimización logística')

# Cargar datos
df = pd.read_csv('data/input.csv', sep=';', dtype=str)

st.write(df.reset_index(drop=True).head())
# Download csv template
st.markdown("Descarga la plantilla de archivo csv:")
st.download_button(
    label="Descargar plantilla",
    data=df.to_csv(index=False, sep=';'),
    file_name="plantilla.csv",
    mime="text/csv"
)

# Leer un archivo csv como entrada usando st.file_uploader
st.markdown("")
st.title('Paso 1: Subir archivo csv') 
uploaded_file = st.file_uploader(label="Sube un archivo csv:", type="csv")

# Mostrar el contenido del archivo csv si se ha subido
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=';')
    st.write('Archivo subido:')
    st.write(data)
    df = utils.preprocess_data(data)

    

    st.title('Paso 2: Optimización') 
    
    num_stations = st.number_input(
        label="Introduce el número de estaciones:",
        min_value=2, max_value=100, value=4, step=1,
        format="%d"
    )
    
    alg.set_globals(data, num_stations)

    clicked = st.button('Optimizar reparto 🚀', type='primary')
    if clicked:
        with st.spinner("Optimizando reparto..."):
            best_solution = alg.optimize_logistics(max_iterations=20000)
        st.success("Optimización completada")

        # solution_str = alg.print_solution_str(best_solution)
        st.header('Resultados')    

        Q = 0
        table_items = []
        for i, station in enumerate(best_solution):
            capacity = alg.get_station_load(station)
            station_cost = alg.get_station_cost(station)
            station_routes = alg.get_station_routes_list(station)
            table_items += [
                [f'Estacion {i+1}', 
                 f'{capacity} tiendas', 
                 f'{station_cost:.4f}',
                 f'{dict(Counter(station_routes))}'
                ]
            ]
            # st.write(f'Estacion {i+1} ({capacity} shops), sum-Q = {station_cost:.4f}, routes: {Counter(station_routes)}')
            Q += station_cost

        table = pd.DataFrame(
            table_items,
            columns=('Estaciones', 'Total de tiendas', 'Suma de Q\'s', 'Recuento de rutas'))

        st.table(table)

        package_dispersion = alg.get_solution_dispersion(best_solution)//2
        # st.write(f'Dispersión = {package_dispersion}')
        st.metric(label='Dispersión paquetes', value=package_dispersion)
        print(alg.solution_cost(best_solution, weights=alg.weights))


        result = utils.export_result(data, best_solution)
        # st.markdown("Descarga la el resultado de archivo csv:")
        st.download_button(
            label="Descargar el resultado en csv ⬇",
            data=result.to_csv(index=False, sep=';', decimal=','),
            file_name="resultado.csv",
            mime="text/csv"
        )
        st.write(result)
        st.write(utils.plot_optimization_cost_history(alg.last_optimization_history))
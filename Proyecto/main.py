import Definitions
import folium
import folium.plugins as fp
import pandas as pd
import os.path as osp
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from keras.models import load_model
import tensorflow as tf

# from streamlit_folium import st_folium

from Parameters import Parameters
from back.DataController import DataController
from back.WindowGenerator import WindowGenerator

# Variables
param = Parameters()
controller = DataController()
CENTER_START = [4, -74]
ZOOM_START = 4

# Estados
if "center" not in st.session_state:
    st.session_state["center"] = [4, -74]
if "markers" not in st.session_state:
    st.session_state["markers"] = []
if "zoom" not in st.session_state:
    st.session_state["zoom"] = 4

# st.title(" 📰 {}".format(lbl.main_title))
st.set_page_config(page_icon="🌡️", layout="wide", initial_sidebar_state="expanded")

def get_prediction(data):
    title, model_dates, X_Real_val, trainPredictPlot, \
        testPredictPlot, metrics = controller.predict(data, station_code=station_sel, hour=hour_sel)

    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(x=model_dates, y=X_Real_val, mode='lines+markers', name='Real'))
    pred_fig.add_trace(go.Scatter(x=model_dates, y=trainPredictPlot, mode='lines+markers', name='Estimado'))
    pred_fig.add_trace(go.Scatter(x=model_dates, y=testPredictPlot, mode='lines+markers', name='Predecido'))
    pred_fig.update_layout(title=title, xaxis_title='Dia', yaxis_title='Temperatura')

    return pred_fig, metrics

# Visualizar las estaciones actuales
def show_stations(l1_sel, l2_sel, l3_sel):
    print("station_l3_options ->")


# Presentación de resultados
row_01_col1, row_01_col2 = st.columns(2)
row_02_col1, row_02_col2 = st.columns(2)
row_03_col1, row_03_col2 = st.columns(2)
row_04_col1, row_04_col2, row_04_col3 = st.columns([1, 2, 1])

with st.form(key='FilterForm'):
    with st.sidebar:
        subtitle_1 = st.sidebar.title(param.form_title_lbl)
        subtitle_2 = st.sidebar.caption(param.form_subtitle1_lbl)

        subtitle_3 = st.sidebar.caption(param.form_subtitle2_lbl)
        initial_date = st.sidebar.date_input(label=param.init_date_lbl, value=param.init_date_val,
                                             help=param.form_init_tooltip)
        ending_date = st.sidebar.date_input(label=param.end_date_lbl, value=param.ending_date_val,
                                            help=param.form_end_tooltip)

        submitted = st.sidebar.button('Ver predicciones 🔎')

        # Botón en la barra lateral
        if submitted:
            # st.caption("Presentando datos...")

            initial_date_sel = initial_date.strftime("%Y-%m-%d")
            ending_date_sel = ending_date.strftime("%Y-%m-%d")

            input_data_ok = True
            date_diff = ending_date - initial_date
            if date_diff.days < 90:
                input_data_ok = False

            print(f"input_data_ok: {input_data_ok}")

            if input_data_ok:
                with st.spinner("Procesando datos...."):
                    temp_df = controller.query_data(initial_date_sel, ending_date_sel)
                    INPUT_LENGTH = 24  # Registros de 24 horas consecutivas a la entrada
                    OUTPUT_LENGTH = 1  # El modelo va a predecir 1 hora a futuro
                    bm_window = WindowGenerator(temp_df, "MinTemp", INPUT_LENGTH, OUTPUT_LENGTH, multimodal=True)
                    model_path = osp.join(Definitions.ROOT_DIR, "resources/models", "temperaturas.h5")
                    print(model_path)
                    if not osp.exists(model_path):
                        raise "El modelo no ha sido entrenado"
                    model_prd = load_model(model_path)

                    with row_01_col1:
                        ":cold_face Temperatura :hot_face"
                        st.caption("Observaciones de temperatura")
                        temp_fig = px.line(temp_df, x=temp_df.index, y="MinTemp", title='Temperaturas')
                        st.plotly_chart(temp_fig)

                    # "Datos de las temperaturas"
                    with row_01_col2:
                        ":memo: Distribución de los datos de entrada"
                        st.pyplot(bm_window.plot_violin_dist())

                    with row_02_col1:
                        ":memo: Ejemplo de ventana de tiempo"
                        st.pyplot(bm_window.plot_train_sample())

                    with row_02_col2:
                        ":chart_with_upwards_trend: Métricas"

                        rmse_tr, rmse_vl, rmse_ts = bm_window.evaluate(model_prd)
                        st.text("row_03_col2")
                        st.text(f'  RMSE train:\t {rmse_tr:.3f}')
                        st.text(f'  RMSE val:\t {rmse_vl:.3f}')
                        st.text(f'  RMSE test:\t {rmse_ts:.3f}')

                    pred_error_msg = "No fue posible generar la predicción con los parámetros indicados"
                    with row_03_col1:
                        try:
                            ":chart_with_upwards_trend: Predicciones"
                            print("Predicciones")
                            y_pred_s = model_prd.predict(bm_window.X_test, verbose=0)
                            print(y_pred_s.flatten())
                            print("Predicciones OK")
                            column_values = ['value']
                            df = pd.DataFrame(data=y_pred_s, columns = column_values)
                            df.reset_index()
                            print(df.dtypes)
                            print(df)
                            fig = px.line(df, x=df.index, y="value", title='Temperatura')
                            st.pyplot(fig)
                        except:
                            pred_error_msg

                    with row_03_col2:
                        ":chart_with_upwards_trend: Métricas"
                        try:
                            #st.dataframe(metrics)
                            st.empty()
                        except:
                            pred_error_msg
            else:
                st.info('Parámetros incompletos', icon="ℹ")


    with row_04_col2:
        with st.expander("Mas Información"):
            f"""
            El aplicativo dispone la información de modelos entrenados para estaciones en Colombia en el periodo de tiempo
            anteriormente indicado, el mínimo de datos considerado es de {param.min_instances} instancias. 
            """

            st.info(
                f"""
                [Entrenamiento] Tomado de la información del año 2020 al año 2021 para los datos de temperatura mínima 
                del aire a 2 metros.
                © Datos de estaciones de IDEAM compartidos en {controller.open_data_host}
                """
            )

            """
            Los modelos fueron entrenados mediante redes recurrentes LSTM con una ventana de 1 día seoarado por intervalos de hora, permitiendo buscar
            sobre los datos distintas tendencias que pudiera presentar la temperatura mínima en el aite a 2 metros.
            """

            """
            Estos modelos tienen la capacidad de capturar patrones complejos en los datos y generar predicciones precisas. 
            Esto sugiere que los modelos LSTM pueden ser una herramienta útil para estudiar y predecir el clima en otras 
            regiones o para analizar diferentes variables climáticas.
            """

            """
            Los resultados obtenidos en este proyecto resaltan la importancia de monitorear y comprender los cambios en la 
            temperatura mínima en Colombia. Estos cambios pueden tener impactos significativos en diferentes aspectos, como 
            la agricultura, la salud pública y los ecosistemas. El análisis de la variación temporal de la temperatura 
            mínima proporciona información valiosa para comprender mejor el clima y tomar decisiones informadas 
            en áreas relacionadas.        
            """

            st.info(
                """
                [Código Fuente GitHub](https://github.com/fcastellanosp/UNIANDES_MINE_202310_AML/tree/main/Proyecto)
                """
            )

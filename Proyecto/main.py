import Definitions
import numpy as np
import pandas as pd
import os.path as osp
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from keras.models import load_model
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

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
            pred_error_msg = "No fue posible generar la predicción con los parámetros indicados"
            # prd_scaler = MinMaxScaler(feature_range=(-1, 1))

            input_data_ok = True
            date_diff = ending_date - initial_date
            if date_diff.days < param.min_instances:
                input_data_ok = False

            print(f"input_data_ok: {input_data_ok}")

            if input_data_ok:
                with st.spinner("Procesando datos...."):
                    temp_df = controller.query_data(initial_date_sel, ending_date_sel)
                    # prd_scaler.fit_transform(temp_df[["MinTemp"]])
                    INPUT_LENGTH = 24  # Registros de 24 horas consecutivas a la entrada
                    OUTPUT_LENGTH = 1  # El modelo va a predecir 1 hora a futuro
                    bm_window = WindowGenerator(temp_df, "MinTemp", INPUT_LENGTH, OUTPUT_LENGTH, multimodal=True)
                    model_path = osp.join(Definitions.ROOT_DIR, "resources/models", "temperaturas.h5")
                    print(model_path)
                    if not osp.exists(model_path):
                        raise "El modelo no ha sido entrenado"
                    model_prd = load_model(model_path)

                    with row_01_col1:
                        ":thermometer: Observaciones de temperatura"
                        temp_fig = px.line(temp_df, x=temp_df.index, y="MinTemp", title='Temperaturas')
                        st.plotly_chart(temp_fig)

                    # "Datos de las temperaturas"
                    with row_01_col2:
                        ":memo: Variables y covariables"
                        st.dataframe(temp_df)

                    with row_02_col1:
                        ":violin: Distribución de los datos de entrada"
                        # st.pyplot(bm_window.plot_violin_dist(True))

                        violin_fig = px.violin(bm_window.data_s_df, box=False)
                        st.plotly_chart(violin_fig)

                    with row_02_col2:
                        ":window: Ejemplo de ventana de tiempo :window:"
                        window_df, title = bm_window.plot_train_sample()

                        window_fig = go.Figure()
                        window_fig.add_trace(go.Scatter(x=window_df["X"], y=window_df["Y"],
                                                      mode='lines+markers',
                                                      name='Valores'))
                        window_fig.add_trace(go.Scatter(x=window_df.index, y=window_df["Mean"],
                                                      mode='lines+markers',
                                                      name='Promedio'))

                        window_fig.update_layout(title=title,
                                               xaxis_title=param.x_label,
                                               yaxis_title=param.y_label)
                        st.plotly_chart(window_fig)

                    with row_03_col1:
                        try:
                            ":chart_with_upwards_trend: Predicciones :chart_with_upwards_trend:"
                            print("Predicciones")
                            y_pred = bm_window.predict(model_prd)
                            print(y_pred.shape)
                            print(bm_window.Y.shape)
                            print("Datos escalados")
                            y = bm_window.Y.flatten()

                            data = {"Y": y, "Prediccion": y_pred}
                            pred_df = pd.DataFrame(data)
                            pred_df["Hora"] = list(range(0, len(y_pred)))
                            pred_df["Error"] = pred_df["Y"] - pred_df["Prediccion"]

                            pred_fig = go.Figure()
                            pred_fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Y"],
                                                     mode='lines+markers',
                                                     name='Real'))
                            pred_fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Prediccion"],
                                                     mode='lines+markers',
                                                     name='Predicción'))

                            pred_fig.update_layout(title='Prediction to 1 hour',
                                              xaxis_title=param.x_label,
                                              yaxis_title=param.y_label)

                            st.plotly_chart(pred_fig)
                        except Exception as error:
                            #print("An exception occurred:", error)
                            st.text(pred_error_msg)
                            st.text(error)

                    with row_03_col2:
                        ""
                        try:
                            #st.dataframe(metrics)
                            ":warning: Error en las predicciones :warning:"
                            print("Inicia predicción")
                            bm_window.predict(model_prd)
                            print("Visualiza predicción")
                            errors = bm_window.get_error_predictions(model_prd)

                            error_data = {"Error": errors}
                            pred_error_df = pd.DataFrame(error_data)
                            pred_error_df["Hora"] = list(range(0, len(errors)))
                            pred_error_fig = px.line(pred_error_df, x="Hora", y="Error", title='Temperatura')
                            pred_error_fig.update_layout(title='Prediction to 1 hour',
                                              xaxis_title=param.x_label,
                                              yaxis_title=param.y_error_label)
                            st.plotly_chart(pred_error_fig)
                        except Exception as error:
                            st.text(pred_error_msg)
                            st.text(error)

                    with row_04_col1:
                        ":chart_with_downwards_trend: Métricas :chart_with_downwards_trend:"
                        metric_df = bm_window.evaluate(model_prd)
                        st.dataframe(metric_df)
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
                [Entrenamiento] Tomado de la información del año 2021 al año 2022 para los datos de temperatura mínima 
                del aire a 2 metros.
                © Datos de estaciones de IDEAM compartidos en {controller.open_data_host}
                """
            )

            """
            Los modelos fueron entrenados mediante una red recurrentes LSTM multivariada unistep con una ventanas separada por intervalos de hora, permitiendo buscar
            sobre los datos distintas tendencias que pudiera presentar la temperatura mínima en el aite a 2 metros.
            """

            st.info(
                """
                [Código Fuente GitHub](https://github.com/fcastellanosp/UNIANDES_MINE_202320_ADL/tree/main/Proyecto)
                """
            )

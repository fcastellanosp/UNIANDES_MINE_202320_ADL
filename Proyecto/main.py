import Definitions
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from Parameters import Parameters
from back.DataController import DataController

# Variables
param = Parameters()
controller = DataController()

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
                                             min_value=param.default_init_date_val,
                                             max_value=param.default_ending_date_val,
                                             help=param.form_init_tooltip)
        ending_date = st.sidebar.date_input(label=param.end_date_lbl, value=param.ending_date_val,
                                            min_value=param.default_init_date_val,
                                            max_value=param.default_ending_date_val,
                                            help=param.form_end_tooltip)

        submitted = st.sidebar.button('Ver predicciones 🔎')

        # Botón en la barra lateral
        if submitted:

            initial_date_sel = initial_date.strftime("%Y-%m-%d")
            ending_date_sel = ending_date.strftime("%Y-%m-%d")
            pred_error_msg = "No fue posible generar la predicción con los parámetros indicados"

            input_data_ok = True
            date_diff = ending_date - initial_date
            if date_diff.days < param.min_instances:
                input_data_ok = False

            print(f"input_data_ok: {input_data_ok}")

            if not input_data_ok:
                st.warning(f'Las fechas seleccionadas son válidas, debe seleccionar un rango con mínimo ({param.min_instances}) día(s) de diferencia', icon="🚨")
            else:
                with st.spinner("Procesando datos...."):
                    controller.prepare_window(initial_date_sel, ending_date_sel)
                    try:
                        data_df = controller.get_current_window().data_df

                        y_pred = controller.predict()
                        pred_df = controller.get_prediction_result()
                        metric_df = controller.evaluate()

                        with row_01_col1:
                            ":thermometer: Observaciones de temperatura"
                            temp_fig = px.line(data_df, x=data_df.index, y="MinTemp", title='Temperaturas')
                            st.plotly_chart(temp_fig)

                        # "Datos de las temperaturas"
                        with row_01_col2:
                            ":memo: Variables y covariables"
                            st.dataframe(data_df)

                        with row_02_col1:
                            ":violin: Distribución de los datos de entrada"
                            violin_fig = px.violin(controller.get_current_window().data_s_df, box=False)
                            st.plotly_chart(violin_fig)

                        with row_02_col2:
                            ":window: Ejemplo de ventana de tiempo :window:"
                            window_df, title = controller.get_current_window().plot_train_sample()

                            window_fig = go.Figure()
                            window_fig.add_trace(go.Scatter(x=window_df["X"], y=window_df["Y"],
                                                            mode='lines+markers',
                                                            name='Valores'))
                            window_fig.add_trace(go.Scatter(x=window_df.index, y=window_df["Mean"],
                                                            mode='lines+markers',
                                                            name='Promedio'))

                            window_fig.update_layout(title=title, xaxis_title=param.x_label, yaxis_title=param.y_label)
                            st.plotly_chart(window_fig)

                        with row_03_col1:
                            try:
                                ":chart_with_upwards_trend: Predicciones :chart_with_upwards_trend:"
                                pred_fig = go.Figure()
                                pred_fig.add_trace(go.Scatter(x=pred_df["Hora"], y=pred_df["Y"],
                                                              mode='lines+markers',
                                                              name='Real'))
                                pred_fig.add_trace(go.Scatter(x=pred_df["Hora"], y=pred_df["Prediccion"],
                                                              mode='lines+markers',
                                                              name='Predicción'))

                                pred_fig.update_layout(title='Prediction to 1 hour', xaxis_title=param.x_label,
                                                       yaxis_title=param.y_label)

                                st.plotly_chart(pred_fig)
                            except Exception as error:
                                print("An exception occurred:", error)
                                st.text(pred_error_msg)
                                st.text(error)

                        with row_03_col2:
                            ""
                            try:
                                ":warning: Error en las predicciones :warning:"
                                pred_error_fig = px.line(pred_df, x="Hora", y="Error", title='Temperatura')
                                pred_error_fig.update_layout(title='Prediction to 1 hour', xaxis_title=param.x_label,
                                                             yaxis_title=param.y_error_label)
                                st.plotly_chart(pred_error_fig)
                            except Exception as error:
                                st.text(pred_error_msg)
                                st.text(error)

                        with row_04_col1:
                            ":chart_with_downwards_trend: Métricas :chart_with_downwards_trend:"
                            st.dataframe(metric_df)
                    except:
                        st.warning('No hay datos para las fechas seleccionadas', icon="🚨")

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

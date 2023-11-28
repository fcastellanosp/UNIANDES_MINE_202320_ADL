import Definitions
import math
import numpy as np
import pandas as pd
import os
import os.path as osp

from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sodapy import Socrata


class DataController:

    def __init__(self, load_data=False):
        self.stations_ds_id = "n6vw-vkfe"
        self.temperature_ds_id = "sbwg-7ju4"
        self.open_data_host = "www.datos.gov.co"
        self.app_token = "qgHbkdSlEXEA5tsCEFeDngbpa"
        self.stations_df = None
        self.temperature_fields = ["codigoestacion", "fechaobservacion", "valorobservado"]
        # self.prd_scaler = MinMaxScaler(feature_range=(0, 1))

        if load_data:
            self.query_all_stations()

    def get_y_coordinate(self, x):
        obj = x["ubicaci_n"]
        return obj["latitude"]

    def get_x_coordinate(self, x):
        obj = x["ubicaci_n"]
        return obj["longitude"]

    def fix_code(self, x):
        cod = str(x["codigo"]).zfill(10)
        x["codigo"] = cod
        return x

    def fix_code2(self, x):
        return x[:10]

    # Obtener el listado de estaciones
    def query_all_stations(self):
        client = Socrata(self.open_data_host, self.app_token)
        query = "categoria LIKE 'Climática%' AND estado = 'Activa'"
        results = client.get(self.stations_ds_id, where=query, limit=10000)

        model_path = osp.join(Definitions.ROOT_DIR, "resources/models")
        model_list = []
        for path in os.listdir(model_path):
            if osp.isfile(osp.join(model_path, path)):
                model_list.append(path)

        model_df = pd.DataFrame( { "codigo": model_list } )
        model_df["codigo_"] = model_df["codigo"].apply(self.fix_code2)
        model_df["codigo"] = model_df["codigo_"]

        # model_df = model_df[["codigo_"]].value_counts()
        model_df = model_df.groupby(['codigo'])['codigo_'].count().reset_index()
        model_df = model_df.drop("codigo_", axis=1)

        # Convertir a pandas DataFrame
        self.stations_df = pd.DataFrame.from_records(results)
        # self.stations_df['codigo'] = self.stations_df['codigo'][-8:]
        self.stations_df = self.stations_df.apply(self.fix_code, axis=1)
        self.stations_df['lon'] = self.stations_df.apply(self.get_x_coordinate, axis=1)
        self.stations_df['lat'] = self.stations_df.apply(self.get_y_coordinate, axis=1)
        self.stations_df["lon"] = self.stations_df["lon"].astype("float64")
        self.stations_df["lat"] = self.stations_df["lat"].astype("float64")
        self.stations_df["altitud"] = self.stations_df["altitud"].astype("float64")

        self.stations_df = pd.merge(model_df, self.stations_df, on="codigo")
        # print(self.stations_df)

        return self.stations_df

    # Obtener el listado de departamentos
    def query_dep(self):
        if self.stations_df is None:
            self.query_all_stations()

        results = self.stations_df.groupby(['departamento'])['estado'].count().reset_index()
        results = results.sort_values(by='departamento', ascending=True)
        results = results.drop("estado", axis=1)
        # print(results.dtypes)
        return results["departamento"]

    # Obtener el listado de municipios dado un departamento
    def query_mun(self, dpto=""):
        if dpto == "":
            return

        if self.stations_df is None:
            self.query_all_stations()

        results = self.stations_df.copy()
        results = results[(results['departamento'] == dpto)]
        results = results.groupby(['municipio'])['estado'].count().reset_index()
        results = results.sort_values(by='municipio', ascending=True)
        results = results.drop("estado", axis=1)
        return results["municipio"]

    # Obtener el listado de estaciones dado un departamento
    def query_stations_by_dep(self, dpto=""):
        if dpto == "":
            return

        if self.stations_df is None:
            self.query_all_stations()

        results = self.stations_df.copy()
        results = results[(results['departamento'] == dpto)]
        results = results.groupby(['nombre'])['estado'].count().reset_index()
        results = results.sort_values(by='nombre', ascending=True)
        results = results.drop("estado", axis=1)
        return results["nombre"]

    # Obtener el listado de estaciones dado un municipio
    def query_stations_by_mun(self, mun=""):
        if mun == "":
            return

        if self.stations_df is None:
            self.query_all_stations()

        results = self.stations_df.copy()
        results = results[(results['municipio'] == mun)]
        results = results.groupby(['nombre'])['estado'].count().reset_index()
        results = results.sort_values(by='nombre', ascending=True)
        results = results.drop("estado", axis=1)
        return results["nombre"]


    # Obtener los valores de las estaciones mediante servicio
    def query_temp_station_values(self, station_code="0021205012", start_date="2020-01-01", ending_date="2020-04-30"):
        client = Socrata(self.open_data_host, self.app_token)
        query = f"codigoestacion='{station_code}' AND fechaobservacion BETWEEN '{start_date}' AND '{ending_date}'"
        results = client.get(self.temperature_ds_id, select=",".join(self.temperature_fields), where=query,
                             limit=200000)

        # Convertir a pandas DataFrame
        temp_station_df = pd.DataFrame.from_records(results)

        if temp_station_df.shape[0] == 0:
            return temp_station_df

        temp_station_df["fecha"] = pd.to_datetime(temp_station_df["fechaobservacion"]).dt.date
        temp_station_df["hora"] = pd.to_datetime(temp_station_df["fechaobservacion"]).dt.hour.astype('int32')
        temp_station_df["observacion"] = temp_station_df["valorobservado"].astype('float64')
        temp_station_df = temp_station_df.drop(['fechaobservacion', 'valorobservado'], axis=1)
        # print(temp_station_df)
        return temp_station_df

    def predict(self, data, station_code="0021205012", hour=12):
        model_name = f"{station_code}_h{hour}"
        model_path = osp.join(Definitions.ROOT_DIR, "resources/models", f"{model_name}.h5")
        # print(model_path)

        # data["observacion_normalizada"] = self.prd_scaler.fit_transform(data[["observacion"]])
        prd_scaler = MinMaxScaler(feature_range=(0, 1))
        prd_scaler.fit_transform(data[["observacion"]])

        # Asegurar tratamiento de datos con la hora seleccionada
        # data = data[data["hora"] == hour]
        data_v_df = pd.pivot_table(data, aggfunc='sum', columns='fecha', index=['hora'],
                                   values='observacion', fill_value=np.nan)

        # Imputación
        data_v_df = data_v_df.fillna(method='ffill', axis=1)
        data_v_df = data_v_df.fillna(method='bfill', axis=1)

        # Fechas
        input_dates = data_v_df.columns

        x_real_val = data_v_df.loc[hour, input_dates].values.astype('float32')

        dataset = prd_scaler.fit_transform(np.reshape(x_real_val, (-1, 1)))
        dataset = np.reshape(dataset, (-1))
        print(dataset)
        train, test = dataset[:-31], dataset[-46:]

        past, future = 8, 1
        x_train, y_train = self.create_dataset(train, past)
        print(x_train)
        print(y_train)
        x_test, y_test = self.create_dataset(test, past)
        print(x_test)
        print(y_test)

        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        if not osp.exists(model_path):
            raise "El modelo no ha sido entrenado"
        model_prd = load_model(model_path)

        # Predicciones
        train_predict = model_prd.predict(x_train)
        test_predict = model_prd.predict(x_test)
        metrics = pd.DataFrame(index=['Error Cuadrático Medio - MSE', 'Desviación media cuadrática - RMSE', 'Error absoluto medio - MAE', 'R2'], columns=['Entrenamiento', 'Prueba'])
        # Los datos estaban escalados
        train_predict = prd_scaler.inverse_transform(train_predict)
        train_y = prd_scaler.inverse_transform([y_train])
        test_predict = prd_scaler.inverse_transform(test_predict)
        test_y = prd_scaler.inverse_transform([y_test])
        # Calcular MSE
        train_score = mean_squared_error(train_y[0], train_predict[:, 0])
        metrics.at['Error Cuadrático Medio - MSE', 'Entrenamiento'] = '{:.2f}'.format(train_score)
        test_score = mean_squared_error(test_y[0], test_predict[:, 0])
        metrics.at['Error Cuadrático Medio - MSE', 'Prueba'] = '{:.2f}'.format(test_score)
        # Calcular RMSE
        train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
        metrics.at['Desviación media cuadrática - RMSE', 'Entrenamiento'] = '{:.2f}'.format(train_score)
        test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
        metrics.at['Desviación media cuadrática - RMSE', 'Prueba'] = '{:.2f}'.format(test_score)
        # Calcular r2
        train_score = r2_score(train_y[0], train_predict[:, 0])
        metrics.at['R2', 'Entrenamiento'] = '{:.2f}'.format(train_score)
        test_score = r2_score(test_y[0], test_predict[:, 0])
        metrics.at['R2', 'Prueba'] = '{:.2f}'.format(test_score)
        # Calcular MAE
        train_score = mean_absolute_error(train_y[0], train_predict[:, 0])
        metrics.at['Error absoluto medio - MAE', 'Entrenamiento'] = '{:.2f}'.format(train_score)
        test_score = mean_absolute_error(test_y[0], test_predict[:, 0])
        metrics.at['Error absoluto medio - MAE', 'Prueba'] = '{:.2f}'.format(test_score)

        # Datos de entrenamiento para presentación de resultados
        train_predict_plot = np.empty_like(dataset)
        train_predict_plot[:] = np.nan
        train_predict_plot[past:len(train_predict) + past] = np.reshape(train_predict, -1)
        # Datos de prueba para presentación de resultados
        test_predict_plot = np.empty_like(dataset)
        test_predict_plot[:] = np.nan
        test_predict_plot[len(train_predict):len(dataset) - 1] = np.reshape(test_predict, -1)

        title = f"Prediccción con {model_prd.name}, ventana [{past} días]"
        return title, input_dates, x_real_val, train_predict_plot, test_predict_plot, metrics

    """ Función encargada de generar los dataset como línea de tiempo  """
    def create_dataset(self, dataset, look_back=1):
        data_x, data_y = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back)]
            data_x.append(a)
            data_y.append(dataset[i + look_back])
        return np.array(data_x), np.array(data_y)

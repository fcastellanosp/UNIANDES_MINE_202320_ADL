import Definitions
import math
import numpy as np
import pandas as pd
import os
import os.path as osp

from back.WindowGenerator import WindowGenerator

from datetime import datetime
from datetime import timedelta
from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sodapy import Socrata


class DataController:

    def __init__(self):
        self.stations_ds_id = "n6vw-vkfe"
        
        self.mint_ds_id = "sbwg-7ju4"
        self.maxt_ds_id = "ccvq-rp9s"
        self.hum_ds_id = "uext-mhny"
        self.prec_ds_id = "s54a-sgyg"
        self.wind_ds_id = "sgfv-3yp8"        
        self.open_data_host = "www.datos.gov.co"
        self.app_token = "qgHbkdSlEXEA5tsCEFeDngbpa"
        self.stations_df = None
        self.temperature_fields = ["codigoestacion", "fechaobservacion", "valorobservado"]
        self.query_columns = [ "fechaobservacion", "valorobservado" ]
        self.max_results_count = 200000
        self.date_format = '%Y-%m-%d'
        self.date_hour_format = '%Y-%m-%d %H'
        self.datetime_format = '%m/%d/%y %H:%M:%S'
        # self.prd_scaler = MinMaxScaler(feature_range=(0, 1))

    # Obtener el query a Open Data
    def query_values(self, dataset_id="", start_date="2022-08-01", ending_date="2023-02-02"):
        print(f"query_values(start_date={start_date},ending_date={ending_date})")
        if dataset_id == "":
            return 
            
        client = Socrata(self.open_data_host, self.app_token)
        
        start_date_ = datetime.strptime(start_date, "%Y-%m-%d")
        ending_date_ = datetime.strptime(ending_date, "%Y-%m-%d")
        city = 'BOGOTA, D.C'
        
        current_time = start_date_

        mint_df = None
        mint_df_ = None
        while (current_time < ending_date_):
            try:
              #Prepara el intervalo de fecha
              from_time = current_time
              current_time = current_time + timedelta(days=60)
              to_time = current_time
              #Genera la consulta al servicio
              query = f"municipio = '{city}' AND fechaobservacion >= '{from_time.strftime(self.date_format)}' AND fechaobservacion < '{to_time.strftime(self.date_format)}'"
              results = client.get(dataset_id, select=",".join(self.query_columns), where=query, limit=self.max_results_count)

              # Convertir a pandas DataFrame
              mint_df_ = pd.DataFrame.from_records(results)

              if mint_df_.shape[0] > 0:
                mint_df_["ValorObservado"] = mint_df_["valorobservado"].astype('float64')
                mint_df_["FechaObservacion"] = pd.to_datetime(mint_df_["fechaobservacion"]).dt.strftime(self.date_hour_format) + ":00"
                mint_df_["FechaObservacion"] = pd.to_datetime(mint_df_["FechaObservacion"])

                if str(mint_df) == 'None':
                  mint_df = mint_df_
                else:
                  mint_df = pd.concat([mint_df, mint_df_])
            except Exception as error:
                print("An exception occurred:", error)

        mint_df = mint_df[["FechaObservacion", "ValorObservado"]].copy()
        del mint_df_
        return mint_df
        
    # Consultar las temperaturas mínimas
    def query_mint_values(self, start_date="2022-08-01", ending_date="2023-02-02"):
        print(f"query_mint_values(start_date={start_date},ending_date={ending_date})")
        dataset_id = self.mint_ds_id
        
        new_mint_df = self.query_values(dataset_id, start_date, ending_date)        
        new_mint_df["MinTemp"] = new_mint_df[["ValorObservado"]]
        new_mint_df["MinTemp"] = new_mint_df["MinTemp"].astype('float64')
        new_mint_df["Date"] = pd.to_datetime(new_mint_df["FechaObservacion"])
        new_mint_rs_df = new_mint_df.groupby('Date').agg(MinTemp=("MinTemp", "min"))
        new_mint_rs_df.reset_index().set_index('Date') 
        
        return new_mint_rs_df
    
    # Consultar las temperaturas máximas
    def query_maxt_values(self, start_date="2022-08-01", ending_date="2023-02-02"):
        print(f"query_maxt_values(start_date={start_date},ending_date={ending_date})")
        dataset_id = self.maxt_ds_id
        
        new_maxt_df = self.query_values(dataset_id, start_date, ending_date)        
        new_maxt_df["MaxTemp"] = new_maxt_df[["ValorObservado"]]
        new_maxt_df["MaxTemp"] = new_maxt_df["MaxTemp"].astype('float64')
        new_maxt_df["Date"] = pd.to_datetime(new_maxt_df["FechaObservacion"])
        new_maxt_rs_df = new_maxt_df.groupby('Date').agg(MaxTemp=("MaxTemp", "max"))
        new_maxt_rs_df.reset_index().set_index('Date')         
        
        return new_maxt_rs_df
    
    # Consultar las humedad
    def query_hum_values(self, start_date="2022-08-01", ending_date="2023-02-02"):
        print(f"query_hum_values(start_date={start_date},ending_date={ending_date})")
        dataset_id = self.hum_ds_id
        
        new_hum_df = self.query_values(dataset_id, start_date, ending_date)        
        new_hum_df["Hum"] = new_hum_df[["ValorObservado"]]
        new_hum_df["Hum"] = new_hum_df["Hum"].astype('float64')
        new_hum_df["Date"] = pd.to_datetime(new_hum_df["FechaObservacion"])        
        new_hum_rs_df = new_hum_df.groupby('Date').agg(Hum=("Hum", "mean"))
        new_hum_rs_df.reset_index().set_index('Date')
        
        return new_hum_rs_df
       
    # Consultar la velocidad del viento
    def query_wind_values(self, start_date="2022-08-01", ending_date="2023-02-02"):
        print(f"query_wind_values(start_date={start_date},ending_date={ending_date})")
        dataset_id = self.wind_ds_id
        
        new_wind_df = self.query_values(dataset_id, start_date, ending_date)        
        new_wind_df["Wind"] = new_wind_df[["ValorObservado"]]
        new_wind_df["Wind"] = new_wind_df["Wind"].astype('float64')
        new_wind_df["Date"] = pd.to_datetime(new_wind_df["FechaObservacion"])        
        new_wind_rs_df = new_wind_df.groupby('Date').agg(Wind=("Wind", "mean"))
        new_wind_rs_df.reset_index().set_index('Date')        
        
        return new_wind_rs_df
    
    # Consultar la precipitación
    def query_prec_values(self, start_date="2022-08-01", ending_date="2023-02-02"):
        print(f"query_prec_values(start_date={start_date},ending_date={ending_date})")
        dataset_id = self.prec_ds_id
        
        new_prec_df = self.query_values(dataset_id, start_date, ending_date)        
        new_prec_df["Prec"] = new_prec_df[["ValorObservado"]]
        new_prec_df["Prec"] = new_prec_df["Prec"].astype('float64')
        new_prec_df["Date"] = pd.to_datetime(new_prec_df["FechaObservacion"])        
        new_prec_rs_df = new_prec_df.groupby('Date').agg(Prec=("Prec", "mean"))
        new_prec_rs_df.reset_index().set_index('Date')        
        
        return new_prec_rs_df
        
    def query_data(self, start_date="2022-08-01", ending_date="2023-02-02"):
        print(f"query_data(start_date={start_date},ending_date={ending_date})")
        start_date_ = datetime.strptime(start_date, "%Y-%m-%d")        
        ending_date_ = datetime.strptime(ending_date, "%Y-%m-%d")    
        
        train_df = pd.DataFrame(index=pd.date_range(start=start_date_, end=ending_date_, freq='H'))
        train_df = train_df.rename_axis("Date")        
        
        new_hum_rs_df = self.query_hum_values(start_date, ending_date)
        new_wind_rs_df = self.query_wind_values(start_date, ending_date)
        new_prec_rs_df = self.query_prec_values(start_date, ending_date)
        new_mint_rs_df = self.query_mint_values(start_date, ending_date)
        new_maxt_rs_df = self.query_maxt_values(start_date, ending_date)
        
        train_df_ = train_df.join(new_hum_rs_df)
        train_df_ = train_df_.join(new_wind_rs_df)
        train_df_ = train_df_.join(new_prec_rs_df)
        train_df_ = train_df_.join(new_mint_rs_df)
        train_df_ = train_df_.join(new_maxt_rs_df)

        new_train_df = train_df_.copy()
        new_train_df = new_train_df.drop_duplicates()
        new_train_df = new_train_df[(new_train_df["Hum"].isna() == False) & (new_train_df["Wind"].isna() == False) & (new_train_df["Prec"].isna() == False)
          & (new_train_df["MinTemp"].isna() == False) & (new_train_df["MaxTemp"].isna() == False)]
        new_train_df        
        
        return new_train_df
        
    def plot_violin_dist(self, data_df):
        INPUT_LENGTH = 24    # Registros de 24 horas consecutivas a la entrada
        OUTPUT_LENGTH = 1    # El modelo va a predecir 1 hora a futuro
        bm_window = WindowGenerator(data_df, "MinTemp", INPUT_LENGTH, OUTPUT_LENGTH, multimodal=True)
        return bm_window.plot_violin_dist()

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

import Definitions
import pandas as pd
import os.path as osp

from back.WindowGenerator import WindowGenerator

from datetime import datetime
from datetime import timedelta
from keras.models import load_model
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
        self.model_prd = None
        self.bm_window = None
        self.data_df = None
        self.y_pred = None

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
              # Prepara el intervalo de fecha
              from_time = current_time
              current_time = current_time + timedelta(days=60)
              to_time = current_time
              # Genera la consulta al servicio
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
        
    # Consultar las temperaturas mínimas en un rango de fechas
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
    
    # Consultar las temperaturas máximas en un rango de fechas
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
    
    # Consultar las humedad en un rango de fechas
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
       
    # Consultar la velocidad del viento en un rango de fechas
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
    
    # Consultar la precipitación en un rango de fechas
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

    # Prepara el set de datos con variables y covariables
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

    # Llamado local a la información de datos abiertos
    def query_local_data(self, start_date="2022-08-01", ending_date="2023-02-02"):
        print(f"query_local_data(start_date={start_date},ending_date={ending_date})")
        start_date_ = datetime.strptime(start_date, "%Y-%m-%d")
        ending_date_ = datetime.strptime(ending_date, "%Y-%m-%d")
        excel_path = osp.join(Definitions.ROOT_DIR, 'resources/InputData.xlsx')
        if osp.exists(excel_path):
            result = pd.read_excel(excel_path, sheet_name='Sheet1', index_col=0)
            new_result = result[(result.index >= start_date_) & (result.index <= ending_date_)]
            return new_result

    #Inicia el procesamiento
    def prepare_window(self, start_date="2022-08-01", ending_date="2023-02-02"):
        print(f"prepare_window(start_date={start_date},ending_date={ending_date})")
        # Registros de 24 horas consecutivas a la entrada
        INPUT_LENGTH = 24
        # El modelo va a predecir 1 hora a futuro
        OUTPUT_LENGTH = 1
        # Variable a predecir
        Y_FEATURE = "MinTemp"

        model_path = osp.join(Definitions.ROOT_DIR, "resources/models", "temperaturas.h5")
        print(model_path)
        self.model_prd = load_model(model_path)
        print("Model loaded!")
        #data_df = self.query_data(start_date, ending_date)
        data_df = self.query_local_data(start_date, ending_date)
        print("Data ok!")

        if not osp.exists(model_path):
            raise "El modelo no ha sido entrenado"
        try:
            self.bm_window = WindowGenerator(data_df, Y_FEATURE, INPUT_LENGTH, OUTPUT_LENGTH, multimodal=True)
            print("Window ok!")
        except Exception as error:
            print("An exception occurred:", error)
            return None

    # Generar la predicción con la información de entrada
    def predict(self):
        if self.bm_window is not None:
            self.y_pred = self.bm_window.predict(self.model_prd)
            return self.y_pred

    # Obtener la ventana actual
    def get_current_window(self):
        print("get_current_window -> ")
        if self.bm_window is not None:
            print("ok")
            return self.bm_window

    # Obtener el resultado de la predicción
    def get_prediction_result(self):
        if self.bm_window is not None:
            y = self.bm_window.Y.flatten()

            data = {"Y": y, "Prediccion": self.y_pred}
            pred_df = pd.DataFrame(data)
            pred_df["Hora"] = list(range(0, len(self.y_pred)))
            pred_df["Error"] = pred_df["Y"] - pred_df["Prediccion"]

            return pred_df

    # Evaluar el resultado de la predicción mediante las métricas
    def evaluate(self):
        if self.bm_window is not None:
            metric_df = self.bm_window.evaluate(self.model_prd)
            return metric_df

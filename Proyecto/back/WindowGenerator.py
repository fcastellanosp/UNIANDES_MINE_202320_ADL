import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import MinMaxScaler


# Compilamos las funciones en comentario de las celdas anteriores para generar una clase
# que permitirá más fácilmente trabajar todo el conjunto de datos en diferentes escenarios
class WindowGenerator:
    def __init__(self, data_df: pd.DataFrame, target_feature, input_length, output_length, multimodal=False):
        ''' Prepara la información con las entradas y salidas para la red LSTM
    - data_df: pandas dataframe con el set de datos completo
    - target_feature: nombre de la variable (feature) a predecir
    - input_length: instantes de tiempo consecutivos de las series de tiempo
      usados para alimentar el modelo.
    - output_length: instantes de tiempo a pronosticar (salida del modelo)
    '''

        my_seed = 19
        random.seed(my_seed)

        self.data_df = data_df.copy()
        self.data_s_df = data_df.copy()
        self.target_feature = target_feature
        self.target_feature_index = self.data_df.columns.get_loc(self.target_feature)
        self.input_length = input_length
        self.output_length = output_length
        self.multimodal = multimodal

        self.data_column_names = self.data_df.columns.tolist()
        self.record_count = 0
        self.scalers = []
        self.feature_count = 0

        self.Y_s = None
        self.X_s = None

        self.Y = None
        self.X = None

        self.create_supervised_datasets()
        self.scale_datasets()

        self.column_indexes = {name: i for i, name in enumerate(data_df.columns)}

        self.total_window_size = input_length

        self.input_slice = slice(0, input_length)
        self.input_indexes = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.output_length
        self.labels_slice = slice(self.label_start, None)
        self.label_indexes = np.arange(self.total_window_size)[self.labels_slice]

        self.model = None
        self.y_pred = None

    def __repr__(self):
        '''
    Visualización string de la clase
    '''
        return '\n'.join([
            f'Input Data Size: {self.data_df.shape}',
            f'target_feature: {self.target_feature}',
            f'input_length: {self.input_length}',
            f'output_length: {self.output_length}',
            f'column_indexes: {self.column_indexes}',
            f'total_window_size: {self.total_window_size}',
            f'input_slice: {self.input_slice}',
            f'input_indices: {self.input_indexes}',
            f'label_start: {self.label_start}',
            f'labels_slice: {self.labels_slice}',
            f'data_column_names: {self.data_column_names}',
            f'label_indices: {self.label_indexes}'])

    def scale_datasets(self):
        print("scale_datasets ->")
        self.scalers = [MinMaxScaler(feature_range=(-1, 1)) for i in range(self.feature_count)]

        self.scale_x_dataset()
        self.scale_y_dataset()

    def create_supervised_datasets(self):
        print("create_supervised_datasets ->")

        self.X, self.Y = self.create_supervised_dataset(self.data_df.values)

        self.feature_count = self.X.shape[2]  # Número de variables del modelo
        print(f"feature_count: {self.feature_count}")

        # Imprimir información en pantalla
        print('\nResúmen de información de datasets:')
        print(f'Tamaño entrada (INPUT_LENGTH x FEATURES): ({self.X.shape[0]} x {self.feature_count})')

    def plot_train_sample(self, shift=10, normalized=False):
        print("plot_train_sample ->")
        ''' Imprime una muestra de los datos de entrenamiento
        - shift: número de días a expander para la ventana
        '''

        # Indicar si valores normalizados o no
        if self.multimodal:
            print(f"index: {self.target_feature_index}")
            y_values = (self.X_s[:, :, self.target_feature_index] if normalized else self.X[:, :, self.target_feature_index])
        else:
            y_values = (self.X_s if normalized else self.X)

        # Preparar un dataframe con el resultado
        y_values = y_values.flatten()

        window_data = {"Y": y_values}
        result_df = pd.DataFrame(window_data)
        result_df["X"] = result_df.index

        custom_hour = random.randint(shift, result_df.shape[0] - shift)

        result_df = result_df.loc[result_df.index[custom_hour - shift * 24:custom_hour]]
        result_df["Mean"] = result_df["Y"].mean()

        norm_text = "normalizada" if normalized == True else ""
        title = f'Visualización {norm_text} de ({shift}) ventanas'

        return result_df, title

    def create_supervised_dataset(self, array):
        ''' Prepara la información con las entradas y salidas para la red LSTM
    - array: arreglo numpy de tamaño N x features (N: cantidad de datos,
      f: cantidad de features)
    '''

        X, Y = [], []
        shape = array.shape

        if len(shape) == 1:
            rows, cols = array.shape[0], 1
            array = array.reshape(rows, cols)
        # Multivariado
        else:
            rows, cols = array.shape

        # Los arreglos
        for i in range(rows - self.input_length - self.output_length):
            # Entrada al modelo
            X.append(array[i:i + self.input_length, 0:cols])
            # Salida (el índice 1 corresponde a la columna con la variable a predecir)
            Y.append(array[i+self.input_length:(i+self.input_length+self.output_length),-1].reshape(self.output_length,1))

        # Arreglos de numpy
        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def scale_x_dataset(self):
        print("scale_x_dataset ->")
        self.X_s = np.zeros(self.X.shape)
        print(self.X_s.shape)

        # Escalamiento: se usarán los min/max del set de entrenamiento para
        # escalar la totalidad de los datasets

        # Escalamiento Xs: en este caso debemos garantizar que cada dato de entrada
        # a fit_transform o transform debe ser de tamaño nsamples x nfeatures
        # (en este caso 24x5)

        for i in range(self.feature_count):
            sc = MinMaxScaler(feature_range=(-1, 1))
            column_name = self.data_column_names[i]
            self.data_s_df[column_name] = sc.fit_transform(self.data_s_df[[column_name]].values)

        # for i in range(FEATURE_COUNT):
        for i in range(self.feature_count):
            self.X_s[:, :, i] = self.scalers[i].fit_transform(self.X[:, :, i])

        # Verificación
        print('\nResúmen de escalamiento en X:')
        print(f'- Min X sin escalamiento: {self.X.min()}')
        print(f'* Min X con escalamiento: {self.X_s.min()}')
        print(f'- Max X sin escalamiento: {self.X.max()}')
        print(f'* Max X con escalamiento: {self.X_s.max()}')

    def scale_y_dataset(self):
        self.Y_s = np.zeros(self.Y.shape)
        self.Y_s[:, :, 0] = self.scalers[-1].fit_transform(self.Y[:, :, 0])

        # Verificación
        print('\nResúmen de escalamiento en Y:')
        print(f'- Min Y sin escalamiento: {self.Y.min()}')
        print(f'* Min Y con escalamiento: {self.Y_s.min()}')
        print(f'- Max Y sin escalamiento: {self.Y.max()}')
        print(f'* Max Y con escalamiento: {self.Y_s.max()}')

    def set_model(self, model):
        self.model = None

        if model is None:
            return

        self.model = model

    def predict(self, model):
        print("predict ->")
        self.y_pred = None
        self.set_model(model)

        # Calcular predicción escalada en el rango de -1 a 1
        y_pred_s = self.model.predict(self.X_s, verbose=0)
        print("self.model.predict OK->")

        # Llevar la predicción a la escala original
        y_pred = self.get_scaler().inverse_transform(y_pred_s)
        self.y_pred = y_pred.flatten()

        return self.y_pred

    def get_error_predictions(self, model):
        ''' Visualiza una gráfica de los errores simples que presentan las predicciones
    '''
        # if str(self.model) == 'None' or str(self.y_pred) == 'None':
        self.set_model(model)
        if self.model is None:
            return

        pred_length = len(self.y_pred)  # Número de predicciones (tamaño del set de prueba)
        # ndato = np.linspace(1, pred_length, pred_length)

        # Cálculo de errores
        pred_errors = self.Y.flatten() - self.y_pred
        # return plt.plot(pred_errors)
        return pred_errors

    def evaluate(self, model=None):
        self.set_model(model)

        metric_names = ["RMSE"]
        metrics = [np.nan]

        if self.model is not None:
            rmse_ts = self.model.evaluate(x=self.X_s, y=self.Y_s, verbose=0)
            metrics = [rmse_ts]

        result_df = pd.DataFrame({"Name": metric_names, "Metric": metrics})

        return result_df

    def get_scaler(self):
        return self.scalers[-1]

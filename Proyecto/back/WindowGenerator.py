
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import MinMaxScaler

# Compilamos las funciones en comentario de las celdas anteriores para generar una clase
# que permitirá más fácilmente trabajar todo el conjunto de datos en diferentes escenarios
class WindowGenerator():
  def __init__(self, data_df:pd.DataFrame, target_feature, input_length, output_length, multimodal=False):
    ''' Prepara la información con las entradas y salidas para la red LSTM
    - data_df: pandas dataframe con el set de datos completo
    - target_feature: nombre de la variable (feature) a predecir
    - input_length: instantes de tiempo consecutivos de las series de tiempo
      usados para alimentar el modelo.
    - output_length: instantes de tiempo a pronosticar (salida del modelo)
    '''

    my_seed = 19
    random.seed(my_seed)

    self.data_df = data_df
    self.target_feature = target_feature
    self.input_length = input_length
    self.output_length = output_length
    self.multimodal = multimodal

    self.train_df, self.test_df, self.val_df, self.data_column_names = self.series_train_test_split(data_df) if multimodal else self.series_train_test_split(data_df[target_feature])
    self.feature_count = self.create_supervised_datasets()
    self.scale_datasets()

    self.column_indexes = {name: i for i, name in enumerate(data_df.columns)}

    self.total_window_size = input_length

    self.input_slice = slice(0, input_length)
    self.input_indexes = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.output_length
    self.labels_slice = slice(self.label_start, None)
    self.label_indexes = np.arange(self.total_window_size)[self.labels_slice]

    self.model = None

    self.X_train = self.x_tr_s
    self.Y_train = self.y_tr_s
    self.X_val = self.x_vl_s
    self.Y_val = self.y_vl_s
    self.X_test = self.x_ts_s
    self.Y_test = self.y_ts_s

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
            f'label_indices: {self.label_indexes}'])

  def scale_datasets(self):
    print("scale_datasets ->")
    self.scalers = [MinMaxScaler(feature_range=(-1,1)) for i in range(self.feature_count)]

    self.x_tr_s, self.x_vl_s, self.x_ts_s, self.scalers = self.scale_x_dataset(self.x_tr, self.x_vl, self.x_ts, self.scalers, self.feature_count)

    self.y_tr_s, self.y_vl_s, self.y_ts_s = self.scale_y_dataset(self.y_tr, self.y_vl, self.y_ts, self.scalers)
    #self.scaler = self.scalers[self.data_df.columns.get_loc(self.target_feature)]
    #print(self.scaler)
    #self.scaler = self.scalers[0] if self.feature_count == 1 else self.scalers[self.data_df.columns.get_loc(self.target_feature)]
    #self.scaler = self.scalers[0] if self.feature_count == 1 else self.scalers[self.data_df.columns.get_loc("MinTemp")]
    self.scaler = self.scalers[3]
    print(self.scaler)

  def create_supervised_datasets(self):
    # Datasets supervisados para entrenamiento (x_tr, y_tr), validación
    # (x_vl, y_vl) y prueba (x_ts, y_ts)
    self.x_tr, self.y_tr = self.create_supervised_dataset(self.train_df.values)
    self.x_vl, self.y_vl = self.create_supervised_dataset(self.val_df.values)
    self.x_ts, self.y_ts = self.create_supervised_dataset(self.test_df.values)

    feature_count = self.x_tr.shape[2] # Número de variables del modelo
    print(f"feature_count: {feature_count}")

    # Imprimir información en pantalla
    print('\nResúmen de información de datasets:')
    print('Tamaños entrada (BATCHES x INPUT_LENGTH x FEATURES) y de salida (BATCHES x OUTPUT_LENGTH x FEATURES)')
    print(f'Set de entrenamiento - x_tr: {self.x_tr.shape}, y_tr: {self.y_tr.shape}')
    print(f'Set de validación - x_vl: {self.x_vl.shape}, y_vl: {self.y_vl.shape}')
    print(f'Set de prueba - x_ts: {self.x_ts.shape}, y_ts: {self.y_ts.shape}')

    return feature_count

  def series_train_test_split(self, df, train_size=0.8, val_size=0.1, test_size=0.1):
    if (train_size + val_size + test_size) != 1:
      raise "Las particiones no son válidas"

    count = df.shape[0]
    self.record_count = count
    train_length = int(count * train_size)
    val_length = int(count * val_size)
    #test_length = int(len(count) * test_size)
    test_length = count - train_size - val_size
    print('\nResúmen de partición de datos:')
    print(f"Núm. Registros: {count}")
    print(f"Entrenamiento ({train_size*100})%: {train_length}")
    print(f"Validación ({val_size*100})%: {val_length}")
    print(f"Prueba ({test_size*100})%: {test_length}")

    train = df[0:train_length]
    validation = df[train_length:train_length + val_length]
    test = df[train_length + val_length :]

    return train, validation, test, self.data_df.columns.values if self.multimodal else [self.target_feature]

  def plot_train_sample(self, shift=10, normalized=False):
    ''' Imprime una muestra de los datos de entrenamiento
    - shift: número de días a expander para la ventana
    '''
    #sns.set(rc={'figure.figsize':(8, 4)})
    #plt.figure(figsize=(8, 4))
    fig, ax = plt.subplots(figsize=(10, 5))

    #Indicar si valores normalizados o no
    if self.multimodal:
      index = np.where(self.data_column_names==self.target_feature)[0][0]
      print(f"index: {index}")
      y_values = (self.x_tr_s[:,:,index] if normalized else self.x_tr[:,:,index])
    else:
      y_values = (self.x_tr_s if normalized else self.x_tr)

    custom_day = random.randint(shift, y_values.shape[0] - shift)
    #custom_day = random.randint(shift, len(y_values) - shift)

    #Trae los set de datos
    x_values = [*range(0, 24 * shift)]
    y_values = np.concatenate(y_values[custom_day-shift:custom_day])
    mean_values = np.repeat(y_values.mean(), 24 * shift)

    plt.plot(x_values, y_values,
                      label='Observaciones', marker='.', c="darkblue", zorder=-10)
    plt.plot(x_values, mean_values,
                      label='Promedio', marker='_', c="red", zorder=-10)

    norm_text = "normalizada" if normalized == True else ""
    #Etiquetas de los ejes
    plt.xlabel('Tiempo [h]')
    plt.ylabel(f'{self.target_feature}')
    plt.title(f'Visualización {norm_text} a ({shift}) días')
    plt.legend()

    return fig

  def plot_violin_dist(self, normalized=False):
    '''Visualizar una gráfica de violín de la información para conocer su distribución
    '''
    fig, ax = plt.subplots(figsize=(12,4))
    if self.feature_count == 1:
      columns = [self.target_feature]
    #Multivariado
    else:
      columns = self.train_df.columns

    train = self.x_tr_s if normalized else self.x_tr
    val = self.x_vl_s if normalized else self.x_vl
    test = self.x_ts_s if normalized else self.x_ts

    for i in range(self.feature_count):
        ax.violinplot(dataset=train[:,:,i].flatten(), positions=[i])
        ax.violinplot(dataset=val[:,:,i].flatten(), positions=[i])
        ax.violinplot(dataset=test[:,:,i].flatten(), positions=[i])

    norm_text = "normalizada" if normalized == True else ""

    # Etiquetas eje horizontal
    ax.set_xticks(list(range(self.feature_count)))
    ax.set_xticklabels(columns, rotation=90)
    plt.title(f'Distribución de la información {norm_text}')
    plt.xlabel('Variable')
    plt.ylabel('Valor')
    ax.autoscale()

    return fig

  def plot_train_test_val(self):
    #sns.set(rc={'figure.figsize':(10, 5)})
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(self.train_df[self.target_feature] if self.multimodal else self.train_df, label="Train")
    ax.plot(self.test_df[self.target_feature] if self.multimodal else self.test_df, label="Validation")
    ax.plot(self.val_df[self.target_feature] if self.multimodal else self.val_df, label="Test")
    plt.legend()

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
    #Multivariado
    else:
      rows, cols = array.shape

    #Los arreglos
    for i in range(rows - self.input_length - self.output_length):
      # Entrada al modelo
      X.append(array[i:i + self.input_length,0:cols])
      # Salida (el índice 1 corresponde a la columna con la variable a predecir)
      #Y.append(array[i + self.input_length:(i + self.input_length + self.output_length), -1].reshape(self.output_length, 1))
      Y.append(array[i + self.input_length:(i + self.input_length + self.output_length), 1].reshape(self.output_length, 1))

    #Arreglos de numpy
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

  def scale_x_dataset(self, x_tr, x_vl, x_ts, scalers, feature_count):
    x_tr_s = np.zeros(x_tr.shape)
    x_vl_s = np.zeros(x_vl.shape)
    x_ts_s = np.zeros(x_ts.shape)

    # Escalamiento: se usarán los min/max del set de entrenamiento para
    # escalar la totalidad de los datasets

    # Escalamiento Xs: en este caso debemos garantizar que cada dato de entrada
    # a fit_transform o transform debe ser de tamaño nsamples x nfeatures
    # (en este caso 24x13)

    #for i in range(FEATURE_COUNT):
    for i in range(feature_count):
        x_tr_s[:,:,i] = scalers[i].fit_transform(x_tr[:,:,i])
        x_vl_s[:,:,i] = scalers[i].transform(x_vl[:,:,i])
        x_ts_s[:,:,i] = scalers[i].transform(x_ts[:,:,i])

    # Verificación
    print('\nResúmen de escalamiento en X:')
    print(f'- Min x_tr/x_vl/x_ts sin escalamiento: {x_tr.min()}/{x_vl.min()}/{x_ts.min()}')
    print(f'* Min x_tr/x_vl/x_ts con escalamiento: {x_tr_s.min()}/{x_vl_s.min()}/{x_ts_s.min()}')
    print(f'- Max x_tr/x_vl/x_ts sin escalamiento: {x_tr.max()}/{x_vl.max()}/{x_ts.max()}')
    print(f'* Max x_tr/x_vl/x_ts con escalamiento: {x_tr_s.max()}/{x_vl_s.max()}/{x_ts_s.max()}')

    return x_tr_s, x_vl_s, x_ts_s, scalers

  def scale_y_dataset(self, y_tr, y_vl, y_ts, scalers):
    y_tr_s = np.zeros(y_tr.shape)
    y_vl_s = np.zeros(y_vl.shape)
    y_ts_s = np.zeros(y_ts.shape)

    y_tr_s[:,:,0] = scalers[-1].fit_transform(y_tr[:,:,0])
    y_vl_s[:,:,0] = scalers[-1].transform(y_vl[:,:,0])
    y_ts_s[:,:,0] = scalers[-1].transform(y_ts[:,:,0])

    # Verificación
    print('\nResúmen de escalamiento en Y:')
    print(f'- Min y_tr/y_vl/y_ts sin escalamiento: {y_tr.min()}/{y_vl.min()}/{y_ts.min()}')
    print(f'* Min y_tr/y_vl/y_ts con escalamiento: {y_tr_s.min()}/{y_vl_s.min()}/{y_ts_s.min()}')

    print(f'- Max y_tr/y_vl/y_ts sin escalamiento: {y_tr.max()}/{y_vl.max()}/{y_ts.max()}')
    print(f'* Max y_tr/y_vl/y_ts con escalamiento: {y_tr_s.max()}/{y_vl_s.max()}/{y_ts_s.max()}')

    return y_tr_s, y_vl_s, y_ts_s

  def get_input_shape(self):
    return (self.x_tr_s.shape[1], self.x_tr_s.shape[2])

  def set_model(self, model):
    self.model = None

    if model is None:
      return

    self.model = model

  def predict(self, model):
    print("predict ->")
    self.y_pred = None
    self.set_model(model)
    print(self.model)

    # Calcular predicción escalada en el rango de -1 a 1
    # self.X_test = self.x_ts_s
    y_pred_s = self.model.predict(self.X_test, verbose=0)

    return y_pred_s

    ## Llevar la predicción a la escala original
    #y_pred = self.scaler.inverse_transform(y_pred_s)
    #self.y_pred = y_pred.flatten()

    return self.y_pred

  def plot_error_predictions(self):
    ''' Visualiza una gráfica de los errores simples que presentan las predicciones
    '''
    #if str(self.model) == 'None' or str(self.y_pred) == 'None':
    if self.model is None:
      return

    pred_length = len(self.y_pred)  # Número de predicciones (tamaño del set de prueba)
    #ndato = np.linspace(1, pred_length, pred_length)

    # Cálculo de errores
    pred_errors = self.y_ts.flatten() - self.y_pred
    return plt.plot(pred_errors)

  def evaluate(self, model=None):
    self.set_model(model)

    if self.model is None:
      return None, None, None

    rmse_tr = self.model.evaluate(x=self.x_tr_s, y=self.y_tr_s, verbose=0)
    rmse_vl = self.model.evaluate(x=self.x_vl_s, y=self.y_vl_s, verbose=0)
    rmse_ts = self.model.evaluate(x=self.x_ts_s, y=self.y_ts_s, verbose=0)

    return rmse_tr, rmse_vl, rmse_ts
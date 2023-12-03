import Definitions
import os.path as osp
import pandas as pd


from Parameters import Parameters
from back.DataController import DataController

from back.WindowGenerator import WindowGenerator

# Variables
controller = DataController()


temp_df = controller.query_data("2022-10-01", "2022-10-31")
INPUT_LENGTH = 24  # Registros de 24 horas consecutivas a la entrada
OUTPUT_LENGTH = 1  # El modelo va a predecir 1 hora a futuro
bm_window = WindowGenerator(temp_df, "MinTemp", INPUT_LENGTH, OUTPUT_LENGTH, multimodal=True)
print(bm_window.__repr__())
model_path = osp.join(Definitions.ROOT_DIR, "resources/models", "temperaturas.h5")

# print("X escalado")
# print(bm_window.X_s[0:24])

# print("X")
# print(bm_window.X[0:24])

# print("Y")
# print(bm_window.Y[0:24])
import pandas as pd
import yaml
import os

class Ingestion:
    def __init__(self, config_path="config.yml"):
        # Configurar la ruta al archivo de configuración
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Carga la configuración desde un archivo YAML."""
        try:
            with open(self.config_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: El archivo de configuración {self.config_path} no se encuentra.")
            raise
        except yaml.YAMLError as e:
            print(f"Error: Hubo un problema al leer el archivo de configuración. {e}")
            raise

    def load_data(self):
        """Carga los datos de entrenamiento y prueba desde los archivos definidos en el archivo de configuración."""
        try:
            # Obtener las rutas de los datos desde la configuración
            train_data_path = self.config['data']['train_path']
            test_data_path = self.config['data']['test_path']
            
            # Verificar si las rutas son absolutas o relativas
            print(f"Ruta de entrenamiento: {train_data_path}")
            print(f"Ruta de prueba: {test_data_path}")
            
            # Verificar si los archivos existen antes de intentar cargarlos
            if not os.path.exists(train_data_path):
                raise FileNotFoundError(f"El archivo de entrenamiento no se encuentra: {train_data_path}")
            if not os.path.exists(test_data_path):
                raise FileNotFoundError(f"El archivo de prueba no se encuentra: {test_data_path}")

            # Cargar los datos CSV usando pandas
            df_train = pd.read_csv(train_data_path)
            df_test = pd.read_csv(test_data_path)
            print(f"Datos de entrenamiento cargados: {df_train.shape[0]} registros.")
            print(f"Datos de prueba cargados: {df_test.shape[0]} registros.")
            return df_train, df_test
        
        except pd.errors.EmptyDataError:
            print("Error: Uno de los archivos está vacío o no tiene un formato válido.")
            raise
        except pd.errors.ParserError:
            print("Error: Hubo un problema al analizar los archivos CSV.")
            raise
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            raise

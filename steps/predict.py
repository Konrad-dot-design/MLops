import os
import joblib
import yaml
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class Predictor:
    def __init__(self, config_path="config.yml"):
        """Inicializa el Predictor, carga el modelo y la configuración"""
        self.config_path = config_path
        self.model_path = self.load_config()['model']['store_path']
        self.pipeline = self.load_model()

    def load_config(self):
        """Carga la configuración desde un archivo YAML"""
        try:
            with open(self.config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except FileNotFoundError:
            print(f"Error: El archivo de configuración {self.config_path} no se encuentra.")
            raise
        except yaml.YAMLError as e:
            print(f"Error al leer el archivo de configuración: {e}")
            raise

    def load_model(self):
        """Carga el modelo previamente entrenado desde el archivo"""
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        try:
            model = joblib.load(model_file_path)
            print(f"Modelo cargado desde: {model_file_path}")
            return model
        except FileNotFoundError:
            print(f"Error: El archivo de modelo no se encuentra en {model_file_path}")
            raise
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            raise

    def feature_target_separator(self, data):
        """Separa las características (X) y el objetivo (y)"""
        X = data.iloc[:, :-1]  # Asume que la última columna es el objetivo
        y = data.iloc[:, -1]   # Última columna es el objetivo (y)
        return X, y

    def evaluate_model(self, X_test, y_test):
        """Evalúa el modelo usando precisión, reporte de clasificación y ROC-AUC"""
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        try:
            roc_auc = roc_auc_score(y_test, y_pred)
        except ValueError:
            roc_auc = None  # Si no se puede calcular ROC-AUC, devolvemos None
            print("No se pudo calcular ROC-AUC, probablemente porque la variable objetivo es binaria o constante.")
        
        return accuracy, class_report, roc_auc

    def predict(self, df_test):
        """Realiza la predicción y evalúa el modelo"""
        X_test, y_test = self.feature_target_separator(df_test)
        
        # Evaluación del modelo en el conjunto de prueba
        accuracy, class_report, roc_auc = self.evaluate_model(X_test, y_test)
        
        print(f"Precisión del modelo: {accuracy:.4f}")
        print("Reporte de clasificación:")
        print(class_report)
        
        if roc_auc is not None:
            print(f"ROC-AUC Score: {roc_auc:.4f}")
        else:
            print("ROC-AUC no disponible debido a la naturaleza del problema.")

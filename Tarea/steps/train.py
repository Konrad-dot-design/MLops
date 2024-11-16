import os
import joblib
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline  # Cambié aquí a Pipeline de sklearn

class Trainer:
    def __init__(self, config_path="config.yml"):
        """Inicializa el entrenador, carga la configuración y prepara el pipeline"""
        self.config_path = config_path
        self.config = self.load_config()
        self.model_name = self.config['model']['name']
        self.model_params = self.config['model']['params']
        self.model_path = self.config['model']['store_path']
        self.pipeline = self.create_pipeline()

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

    def create_pipeline(self):
        """Crea el pipeline de preprocesamiento y modelado"""
        # Definimos el preprocesador con las transformaciones necesarias
        preprocessor = ColumnTransformer(transformers=[
            # Normalizar `Total Spend` usando MinMaxScaler
            ('minmax', MinMaxScaler(), ['Total Spend']),  # Ajusta con el nombre correcto de la columna
            # Estandarizar `Age`, `Usage Frequency`, `Support Calls` y `Payment Delay`
            ('standardize', StandardScaler(), ['Age', 'Usage Frequency', 'Support Calls', 'Payment Delay']),
            # Codificar variables categóricas con OneHotEncoder
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Subscription Type', 'Contract Length'])
        ])

        # Mapeo de modelos
        model_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'LogisticRegression': LogisticRegression
        }

        # Selección del modelo
        if self.model_name not in model_map:
            raise ValueError(f"El modelo '{self.model_name}' no es válido. Elija uno de los siguientes: {list(model_map.keys())}")
        
        model_class = model_map[self.model_name]
        model = model_class(**self.model_params)

        # Crear el pipeline de transformación + entrenamiento sin SMOTE
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        return pipeline

    def feature_target_separator(self, data):
        """Separa las características (X) y el objetivo (y)"""
        X = data.drop(columns=['Churn'])  # Usamos 'Churn' como columna objetivo
        y = data['Churn']
        return X, y

    def train_model(self, X_train, y_train):
        """Entrena el modelo usando los datos de entrenamiento"""
        print("Entrenando el modelo...")
        self.pipeline.fit(X_train, y_train)

    def save_model(self):
        """Guarda el modelo entrenado en un archivo .pkl"""
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        joblib.dump(self.pipeline, model_file_path)
        print(f"Modelo guardado en: {model_file_path}")

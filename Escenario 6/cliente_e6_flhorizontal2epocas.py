# ============================================================
# CLIENTE - ESCENARIO 6
# Nombre del script: cliente_e6_flhorizontal2epocas.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
# Escenario: Eficiencia de Comunicación
#
# Descripción:
# Este script implementa un cliente federado para un escenario de aprendizaje federado horizontal, con un enfoque en la eficiencia de comunicación al aumentar el número de épocas de entrenamiento local a 2.
# ============================================================


# ==========================
# SECCIÓN: IMPORTACIÓN DE LIBRERÍAS
# ==========================
import flwr as fl  # librería para construir clientes y servidores de aprendizaje federado
import tensorflow as tf  # librería para definir y entrenar modelos de deep learning
import numpy as np  # librería para manejo de arreglos y operaciones numéricas
from sklearn.metrics import precision_recall_fscore_support  # métricas de clasificación binaria
import psutil  # librería para monitorear el uso de CPU y memoria

# ==========================
# SECCIÓN: CONFIGURACIÓN Y PARÁMETROS DEL CLIENTE
# ==========================

# Identificador único del cliente en el experimento federado.
CLIENT_ID = "1"

# Dirección del servidor Flower que coordina las rondas federadas.
SERVER_IP = "10.0.2.2:8080"

# Ruta local del dataset de este cliente.
DATASET_PATH = "./Dataset_cliente1"

# Parámetros de entrenamiento
BATCH_SIZE = 16
IMG_SIZE = (224, 224) 

# ==========================
# SECCIÓN: CARGA Y PREPROCESAMIENTO DE DATOS
# ==========================
print(f"Cargando dataset local para Cliente {CLIENT_ID}...")

# Cargar datos de entrenamiento desde el dataset local
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,  # Separa el 20% de los datos para validación
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Cargar datos de validación desde el dataset local
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,  # Separa el 20% de los datos para validación
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Usar autotuning de TensorFlow para optimizar la lectura de datos
AUTOTUNE = tf.data.AUTOTUNE

# Normalizar los píxeles de la imagen a [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)


def preprocess(x, y):
    """Normaliza un lote de imágenes y devuelve las etiquetas sin cambios.

    Args:
        x: Tensor con un lote de imágenes.
        y: Tensor con las etiquetas correspondientes.

    Returns:
        Tupla (imágenes normalizadas, etiquetas).
    """
    return normalization_layer(x), y


# Aplicar la normalización y preparar los conjuntos de datos para un acceso eficiente
train_ds = train_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)

# Calcular el número aproximado de ejemplos de entrenamiento y validación
num_train_examples = int(tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE)
num_val_examples = int(tf.data.experimental.cardinality(val_ds).numpy() * BATCH_SIZE)

# ==========================
# SECCIÓN: DEFINICIÓN DEL MODELO
# ==========================


def get_model():
    """Construye y compila el modelo de clasificación binaria.

    Se utiliza MobileNetV2 preentrenado como extractor de características,
    seguido de un pooling global, dropout y una capa densa de salida.

    Returns:
        tf.keras.Model compilado listo para entrenar y evaluar.
    """
    # Cargar el modelo preentrenado MobileNetV2 como extractor de características
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,  # No incluir capas densas finales
        weights="imagenet",  # Usar pesos preentrenados en ImageNet
    )

    # Permitir ajuste fino de todas las capas del modelo base
    base_model.trainable = True

    # Construir el modelo secuencial con capas adicionales para clasificación binaria
    model = tf.keras.Sequential([
        base_model,  # Extractor de características preentrenado
        tf.keras.layers.GlobalAveragePooling2D(),  # Reducir dimensionalidad
        tf.keras.layers.Dropout(0.2),  # Prevenir overfitting
        tf.keras.layers.Dense(1, activation="sigmoid"),  # Capa de salida para clasificación binaria
    ])

    # Compilar el modelo con el optimizador Adam y la función de pérdida
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# Instanciar el modelo local del cliente
model = get_model()

# ==========================
# SECCIÓN: CLIENTE FLOWER PARA APRENDIZAJE FEDERADO (CON TELEMETRÍA)
# ==========================


class PlacasClient(fl.client.NumPyClient):
    """Cliente federado que interactúa con el servidor Flower.

    Hereda de `fl.client.NumPyClient` para intercambiar parámetros en formato NumPy.
    Incluye telemetría de hardware (CPU, RAM) para estudiar el impacto en la comunicación.
    """

    def get_parameters(self, config):
        """Devuelve los pesos actuales del modelo.

        Args:
            config: Diccionario de configuración enviado por el servidor.

        Returns:
            Lista de arrays NumPy que representan los pesos del modelo.
        """
        return model.get_weights()

    def fit(self, parameters, config):
        """Entrena el modelo local usando los parámetros globales recibidos.

        Args:
            parameters: Pesos del modelo recibidos del servidor.
            config: Configuración de la ronda de entrenamiento.

        Returns:
            tuple: (pesos actualizados, número de ejemplos usados, métricas de entrenamiento)
        """
        # Establecer los pesos globales en el modelo local
        model.set_weights(parameters)
        
        # Entrenar durante 2 épocas para ver mejor la evolución
        history = model.fit(train_ds, epochs=2, verbose=1)
        
        # Registrar pérdida y precisión de la primera época
        results = {
            "loss": float(history.history["loss"][0]),
            "accuracy": float(history.history["accuracy"][0]),
        }
        
        return model.get_weights(), int(num_train_examples), results

    def evaluate(self, parameters, config):
        """Evalúa el modelo local y devuelve métricas extendidas.

        Además de pérdida y precisión, calcula precisión/recall/f1 y recoge
        métricas del sistema (CPU y RAM) para estudiar el impacto de la comunicación federada.
        """
        # Establecer los pesos globales en el modelo local
        model.set_weights(parameters)
        
        # Evaluación básica de pérdida y precisión en el conjunto de validación
        loss, accuracy = model.evaluate(val_ds, verbose=0)

        # Obtener predicciones en todo el conjunto de validación para calcular métricas adicionales
        y_true, y_pred_probs = [], []
        for images, labels in val_ds:
            preds = model.predict(images, verbose=0)
            y_pred_probs.extend(preds)
            y_true.extend(labels.numpy())

        # Convertir probabilidades a clases binarias (umbral: 0.5)
        y_pred_classes = (np.array(y_pred_probs) > 0.5).astype("int32").flatten()
        y_true = np.array(y_true).astype("int32").flatten()

        # Calcular métricas adicionales de clasificación
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_classes, average="binary", zero_division=0
        )

        # Capturar métricas de hardware en el cliente
        cpu_percent = psutil.cpu_percent(interval=1)
        ram_percent = psutil.virtual_memory().percent

        # Diccionario de telemetría que viaja al servidor por gRPC
        metrics = {
            "client_id": CLIENT_ID,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "cpu_usage": float(cpu_percent),
            "ram_usage": float(ram_percent),
        }

        return float(loss), int(num_val_examples), metrics

# ==========================
# SECCIÓN: EJECUCIÓN PRINCIPAL
# ==========================
if __name__ == "__main__":
    print(f"Iniciando Cliente {CLIENT_ID} en {SERVER_IP}...")
    
    # Iniciar el cliente de Flower y conectarlo con el servidor federado
    fl.client.start_client(
        server_address=SERVER_IP,
        client=PlacasClient().to_client(),
    )
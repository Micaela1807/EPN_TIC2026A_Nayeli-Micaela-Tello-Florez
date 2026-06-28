# ============================================================
# CLIENTE - ESCENARIO 3
# Nombre del script: cliente_e3_latencia.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
# Escenario: Impacto de la Latencia en la Red
#
# Descripción:
# Este script implementa un cliente federado para un escenario de aprendizaje federado horizontal, con un enfoque en la medición del impacto de la latencia en la red durante el entrenamiento y evaluación del modelo.
# ============================================================

# ==========================
# SECCIÓN: IMPORTACIÓN DE LIBRERÍAS
# ==========================
import flwr as fl # librería para construir clientes y servidores de aprendizaje federado
import tensorflow as tf # librería para definir y entrenar modelos de deep learning
import numpy as np # librería para manejo de arreglos y operaciones numéricas
import mlflow # librería para el seguimiento de experimentos y métricas
import psutil  # Para control exacto de métricas por cliente
from sklearn.metrics import precision_recall_fscore_support # métricas de clasificación binaria


# ==========================
# SECCIÓN: CONFIGURACIÓN Y PARÁMETROS DEL CLIENTE
# ==========================

# Identificador del cliente en el experimento. 
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
print(f"Cargando y optimizando dataset local para Cliente {CLIENT_ID}...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

AUTOTUNE = tf.data.AUTOTUNE
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


train_ds = train_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)

# Número aproximado de ejemplos de entrenamiento/validación
num_train_examples = int(tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE)
num_val_examples = int(tf.data.experimental.cardinality(val_ds).numpy() * BATCH_SIZE)


# ==========================
# SECCIÓN: DEFINICIÓN DEL MODELO
# ==========================


def get_model():
    """Construye y compila el modelo de clasificación binaria.

    Se utiliza MobileNetV2 preentrenado como extractor de características,
    seguido de pooling global, dropout y una capa densa de salida.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = True

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

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
        model.set_weights(parameters)
        history = model.fit(train_ds, epochs=1, verbose=1)

        results = {"loss": float(history.history["loss"][0]), "accuracy": float(history.history["accuracy"][0])}
        return model.get_weights(), int(num_train_examples), results

    def evaluate(self, parameters, config):
        """Evalúa el modelo local y devuelve métricas extendidas.

        Además de pérdida y precisión, calcula precisión/recall/f1 y recoge
        métricas del sistema (CPU y RAM) para estudiar el impacto de la latencia.
        """
        model.set_weights(parameters)

        # Evaluación básica de pérdida y precisión
        loss, accuracy = model.evaluate(val_ds, verbose=0)

        # Obtener predicciones para calcular métricas adicionales
        y_true, y_pred_probs = [], []
        for images, labels in val_ds:
            preds = model.predict(images, verbose=0)
            y_pred_probs.extend(preds)
            y_true.extend(labels.numpy())

        y_pred_classes = (np.array(y_pred_probs) > 0.5).astype("int32").flatten()
        y_true = np.array(y_true).astype("int32").flatten()

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average="binary", zero_division=0)

        # Métricas del sistema del cliente para monitoreo de recursos
        cpu_percent = psutil.cpu_percent(interval=1)
        ram_percent = psutil.virtual_memory().percent

        # Diccionario de telemetría que viaja al servidor
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
    print(f"Iniciando Cliente {CLIENT_ID} con Latencia Simulada apuntando a {SERVER_IP}...")
    fl.client.start_client(server_address=SERVER_IP, client=PlacasClient().to_client())
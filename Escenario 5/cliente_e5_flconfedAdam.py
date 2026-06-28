# ============================================================
# CLIENTE - ESCENARIO 5
# Nombre del script: cliente_e5_flconfedAdam.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
# Escenario: Comparación de Estrategias de Agregación
#
# Descripción:
# Cliente federado que participa en un experimento comparando optimizadores
# (ej. Adam vs SGD). El cliente entrena localmente y reporta métricas del
# modelo y del hardware al servidor.
# ============================================================


# ==========================
# SECCIÓN: IMPORTACIÓN DE LIBRÍAS
# ==========================
# Librerías principales: Flower (FL), TensorFlow para modelos, sklearn para métricas,
# y psutil para monitorizar CPU/RAM.
import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import psutil


# ==========================
# SECCIÓN: CONFIGURACIÓN
# ==========================
# Identificador del cliente y parámetros del dataset/entrenamiento.
CLIENT_ID = "1"
SERVER_IP = "10.0.2.2:8080"
DATASET_PATH = "./Dataset_cliente1"

BATCH_SIZE = 16
IMG_SIZE = (224, 224)


# ==========================
# SECCIÓN: CARGA Y PREPROCESADO DE DATOS
# ==========================
print(f"Cargando dataset local para Cliente {CLIENT_ID}...")

# Carga datasets de entrenamiento y validación desde carpetas organizadas por clase.
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

# Normalización y optimizaciones de entrada de datos.
AUTOTUNE = tf.data.AUTOTUNE
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

def preprocess(x, y):
    """Aplica normalización a las imágenes."""
    return normalization_layer(x), y

train_ds = train_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)

# Estimación del número de ejemplos de entrenamiento/validación.
num_train_examples = tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE
num_val_examples = tf.data.experimental.cardinality(val_ds).numpy() * BATCH_SIZE


# ==========================
# SECCIÓN: DEFINICIÓN DEL MODELO
# ==========================
def get_model():
    """Construye y compila el modelo local (MobileNetV2 + cabeza binaria)."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = True  # permitimos fine-tuning

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    # Compilamos con Adam por defecto; en el experimento se pueden comparar optimizadores.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


model = get_model()


# ==========================
# SECCIÓN: CLIENTE FLOWER
# ==========================
class PlacasClient(fl.client.NumPyClient):
    """Cliente compatible con Flower que implementa get_parameters, fit, evaluate."""

    def get_parameters(self, config):
        """Devuelve los pesos locales del modelo al servidor."""
        return model.get_weights()

    def fit(self, parameters, config):
        """Recibe parámetros globales, entrena localmente y devuelve pesos actualizados.

        Retorna: (pesos, num_examples, dict_metrics_train)
        """
        # 1) Aplicar parámetros globales
        model.set_weights(parameters)

        # 2) Entrenamiento local (1 época por defecto)
        history = model.fit(train_ds, epochs=1, verbose=1)

        # 3) Preparar métricas de entrenamiento para enviar al servidor
        results = {
            "loss": float(history.history["loss"][0]),
            "accuracy": float(history.history["accuracy"][0]),
        }

        # 4) Retornar pesos, número de ejemplos y métricas
        return model.get_weights(), int(num_train_examples), results

    def evaluate(self, parameters, config):
        """Evalúa el modelo localmente y retorna (loss, num_examples, metrics_dict).

        Además de loss/accuracy, calcula precision/recall/f1 y reporta uso de CPU/RAM.
        """
        # Aplicar parámetros globales
        model.set_weights(parameters)

        # Evaluación estándar para obtener loss y accuracy
        loss, accuracy = model.evaluate(val_ds, verbose=0)

        # Recolectar predicciones para calcular precision/recall/f1
        y_true, y_pred_probs = [], []
        for images, labels in val_ds:
            preds = model.predict(images, verbose=0)
            y_pred_probs.extend(preds)
            y_true.extend(labels.numpy())

        y_pred_classes = (np.array(y_pred_probs) > 0.5).astype("int32").flatten()
        y_true = np.array(y_true).astype("int32").flatten()

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_classes, average="binary", zero_division=0
        )

        # Métricas de hardware del cliente (útiles para análisis de rendimiento)
        cpu_percent = psutil.cpu_percent(interval=1)
        ram_percent = psutil.virtual_memory().percent

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
# SECCIÓN: BLOQUE PRINCIPAL
# ==========================
if __name__ == "__main__":
    print(f"Iniciando Cliente {CLIENT_ID} en {SERVER_IP}...")
    # Conecta el cliente al servidor federado y comienza el bucle de Flower.
    fl.client.start_client(server_address=SERVER_IP, client=PlacasClient().to_client())
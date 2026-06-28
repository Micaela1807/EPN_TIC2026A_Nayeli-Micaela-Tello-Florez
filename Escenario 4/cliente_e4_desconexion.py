# ============================================================
# CLIENTE - ESCENARIO 4
# Nombre del script: cliente_e4_desconexion.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
# Escenario: Tolerancia a Fallos y Desconexión
#
# Descripción:
# Este script implementa un cliente federado para un escenario de aprendizaje federado horizontal, con un enfoque en la tolerancia a fallos y desconexión de clientes durante el entrenamiento y evaluación del modelo.
# ============================================================


# ==========================
# SECCIÓN: IMPORTACIÓN DE LIBRERÍAS
# ==========================
# Librerías principales: Flower para FL, TensorFlow para modelos, MLflow para tracking.
import flwr as fl  # cliente/servidor federado
import tensorflow as tf  # modelado y entrenamiento
import mlflow  # opcional: tracking (aquí se evita logging automático de sistema)
import os  # manejo de entorno y variables
import numpy as np  # manipulación de arrays
import psutil  # medir CPU/RAM/red (pip install psutil)
from sklearn.metrics import precision_recall_fscore_support  # métricas adicionales

# Desactivamos el logging automático de hardware de MLflow porque lo hacemos manualmente con psutil
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

# ==========================
# SECCIÓN: CONFIGURACIÓN
# ==========================
# Parámetros básicos del cliente: ID, ruta del dataset y tamaño de batch/épocas.
CLIENT_ID = "VM_Cliente_1"
DATASET_PATH = "./Dataset_cliente1"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 1

# ==========================
# SECCIÓN: CARGA Y PREPROCESADO DE DATOS
# ==========================
print("Cargando dataset local...")
# Crea datasets de entrenamiento y validación a partir de carpetas con imágenes.
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

# Mejoras de rendimiento: cache + prefetch para pipelines de tf.data
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# ==========================
# SECCIÓN: DEFINICIÓN DEL MODELO
# ==========================
# Usamos MobileNetV2 como backbone preentrenado en ImageNet.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = True  # permitimos fine-tuning del backbone

# Cabeza simple para clasificación binaria
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

# Función para extraer métricas de hardware
# ==========================
# SECCIÓN: MÉTRICAS DEL SISTEMA
# ==========================
def get_system_metrics():
    """
    Lee y devuelve métricas del sistema: CPU %, RAM % y tráfico de red en MB.
    Se usa en la evaluación para registrar el estado del cliente justo al evaluar.
    """
    cpu_usage = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    ram_usage = ram.percent
    net_io = psutil.net_io_counters()
    bytes_sent = net_io.bytes_sent / (1024 * 1024)  # Convertir a MB
    bytes_recv = net_io.bytes_recv / (1024 * 1024)  # Convertir a MB
    return cpu_usage, ram_usage, bytes_sent, bytes_recv

# --- 4. CLASE DEL CLIENTE FLOWER ---
# ==========================
# SECCIÓN: CLIENTE FLOWER (NumPyClient)
# ==========================
class PlacasClient(fl.client.NumPyClient):
    """
    Implementa los métodos requeridos por Flower:
    - get_parameters: devuelve pesos locales
    - fit: entrenamiento local
    - evaluate: evaluación local y reporte de métricas (modelo + hardware)
    """

    def get_parameters(self, config):
        # Devuelve los pesos actuales del modelo al servidor.
        return model.get_weights()

    def fit(self, parameters, config):
        # Aplica parámetros globales y entrena localmente durante EPOCHS.
        model.set_weights(parameters)
        model.fit(train_ds, epochs=EPOCHS, verbose=1)

        # Número de ejemplos de entrenamiento utilizados (estimado por cardinalidad * batch)
        num_examples = int(tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE)

        # Devuelve pesos actualizados, número de ejemplos y diccionario de métricas de entrenamiento (vacío aquí).
        return model.get_weights(), num_examples, {}

    def evaluate(self, parameters, config):
        # 1) Aplica pesos recibidos
        model.set_weights(parameters)

        # 2) Evalúa en el conjunto de validación
        loss, accuracy = model.evaluate(val_ds, verbose=0)

        # 3) Recolecta predicciones para calcular precision/recall/f1
        y_true = []
        y_pred_probs = []
        for images, labels in val_ds:
            y_true.extend(labels.numpy())
            preds = model.predict(images, verbose=0)
            y_pred_probs.extend(preds.flatten())

        y_pred = (np.array(y_pred_probs) > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        # 4) Captura métricas del sistema justo al evaluar (CPU, RAM, tráfico de red)
        cpu, ram, net_sent, net_recv = get_system_metrics()

        # 5) Devuelve la tupla esperada por Flower: (loss, num_examples, metrics_dict)
        # El diccionario incluye identificador del cliente, métricas del modelo y del sistema.
        return float(loss), len(y_true), {
            "client_id": CLIENT_ID,
            "accuracy": float(accuracy),
            "loss": float(loss),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "cpu": float(cpu),
            "ram": float(ram),
            "net_sent_mb": float(net_sent),
            "net_recv_mb": float(net_recv),
        }

# ==========================
# SECCIÓN: BLOQUE PRINCIPAL (STARTUP)
# ==========================
if __name__ == "__main__":
    # Mensaje informativo
    print("Conectando al servidor Flower...")

    # Nota: el cliente no abre un run de MLflow porque los logs de sistema se centralizan en el servidor.
    fl.client.start_numpy_client(server_address="10.0.2.2:8080", client=PlacasClient())
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
import flwr as fl # librería para construir clientes y servidores de aprendizaje federado
import tensorflow as tf # librería para definir y entrenar modelos de deep learning
import mlflow # librería para el seguimiento de experimentos y métricas
import os # libreria para manejo de archivos y rutas
import numpy as np # librería para manejo de arreglos y operaciones numéricas
import psutil # IMPORTANTE: Instalar con pip install psutil
from sklearn.metrics import precision_recall_fscore_support

# Desactivamos el logging automático de hardware de MLflow porque lo haremos manualmente con psutil
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

# --- 1. CONFIGURACIÓN ---
# Identificador del cliente en el experimento. 
CLIENT_ID = "VM_Cliente_1"
DATASET_PATH = "./Dataset_cliente1" 
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 1 

# --- 2. CARGA DE DATOS OPTIMIZADA ---
print("Cargando dataset local...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 3. ARQUITECTURA DEL MODELO ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
base_model.trainable = True

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Función para extraer métricas de hardware
def get_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    ram_usage = ram.percent
    net_io = psutil.net_io_counters()
    bytes_sent = net_io.bytes_sent / (1024 * 1024) # Convertir a MB
    bytes_recv = net_io.bytes_recv / (1024 * 1024) # Convertir a MB
    return cpu_usage, ram_usage, bytes_sent, bytes_recv

# --- 4. CLASE DEL CLIENTE FLOWER ---
class PlacasClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(train_ds, epochs=EPOCHS, verbose=1)
        num_examples = int(tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE)
        return model.get_weights(), num_examples, {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(val_ds, verbose=0)
        
        y_true = []
        y_pred_probs = []
        for images, labels in val_ds:
            y_true.extend(labels.numpy())
            preds = model.predict(images, verbose=0)
            y_pred_probs.extend(preds.flatten())
            
        y_pred = (np.array(y_pred_probs) > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        # Capturamos hardware justo en este momento (el interval=0.5 es perfecto aquí)
        cpu, ram, net_sent, net_recv = get_system_metrics()
        
        # DICCIONARIO CORREGIDO: Empata 100% con el servidor
        return float(loss), len(y_true), {
            "client_id": CLIENT_ID, 
            "accuracy": float(accuracy), 
            "loss": float(loss), 
            "precision": float(precision), 
            "recall": float(recall),      # ¡Añadido!
            "f1_score": float(f1),
            "cpu": float(cpu),            # Renombrado para que el servidor lo entienda
            "ram": float(ram),            # Renombrado para que el servidor lo entienda
            "net_sent_mb": float(net_sent),
            "net_recv_mb": float(net_recv)
        }

if __name__ == "__main__":
    print("Conectando al servidor Flower...")
    # Ya no abrimos un "run" de MLflow aquí, el cliente solo entrena y envía datos. 
    # El Servidor centralizará TODOS los logs para que queden en una sola gráfica.
    fl.client.start_numpy_client(server_address="10.0.2.2:8080", client=PlacasClient())
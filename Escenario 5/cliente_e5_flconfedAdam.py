# ============================================================
# CLIENTE - ESCENARIO 5
# Nombre del script: cliente_e5_flconfedAdam.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
# Escenario: Comparación de Estrategias de Agregación
#
# Descripción:
# Este script implementa un cliente federado para un escenario de aprendizaje federado horizontal, con un enfoque en la comparación de estrategias de agregación, específicamente entre el optimizador Adam y el optimizador SGD.
# ============================================================


# ==========================
# SECCIÓN: IMPORTACIÓN DE LIBRERÍAS
# ==========================
import flwr as fl # librería para construir clientes y servidores de aprendizaje federado
import tensorflow as tf # librería para definir y entrenar modelos de deep learning
import numpy as np # librería para manejo de arreglos y operaciones numéricas
from sklearn.metrics import precision_recall_fscore_support # métricas de clasificación binaria
import psutil  # librería para monitorear el uso de CPU y memoria

# --- 1. CONFIGURACIÓN ---
# Identificador del cliente en el experimento. 
CLIENT_ID = "1"
SERVER_IP = "10.0.2.2:8080" 
DATASET_PATH = "./Dataset_cliente1" 

BATCH_SIZE = 16
IMG_SIZE = (224, 224) 

# --- 2. CARGAR Y NORMALIZAR DATOS LOCALES ---
# Se carga el dataset desde la ruta local, con una partición de validación.
print(f"Cargando dataset local para Cliente {CLIENT_ID}...")

# Genera dos tf.data.Dataset: training y validation usando la estructura de carpetas.
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

# Preparación adicional: normalización y optimizaciones de perf.
AUTOTUNE = tf.data.AUTOTUNE
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

def preprocess(x, y):
	# Normaliza los píxeles y devuelve (inputs, labels)
	return normalization_layer(x), y

train_ds = train_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)

# Calcula el número aproximado de ejemplos (cardinalidad * batch_size).
# Nota: tf.data.experimental.cardinality puede devolver tf.int64, por eso .numpy().
num_train_examples = tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE
num_val_examples = tf.data.experimental.cardinality(val_ds).numpy() * BATCH_SIZE

# --- 3. DEFINIR EL MODELO ---
def get_model():
	# Usamos MobileNetV2 como feature extractor inicial (preentrenado en ImageNet).
	base_model = tf.keras.applications.MobileNetV2(
		input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
		include_top=False,
		weights="imagenet",
	)

	# Permitimos que el backbone sea entrenable para fine-tuning.
	base_model.trainable = True

	# Construimos una cabeza simple para clasificación binaria.
	model = tf.keras.Sequential([
		base_model,
		tf.keras.layers.GlobalAveragePooling2D(),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(1, activation="sigmoid"),
	])

	# Compilamos el modelo con Adam (esto es el caso 'Adam' del experimento).
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
		loss="binary_crossentropy",
		metrics=["accuracy"],
	)

	return model

model = get_model()

# --- 4. DEFINIR EL CLIENTE FLOWER ---
class PlacasClient(fl.client.NumPyClient):
    def get_parameters(self, config):
		# Devuelve los pesos actuales del modelo local al servidor cuando se lo solicitan.
		return model.get_weights()

    def fit(self, parameters, config):
		# 1) Recibe parámetros globales y los aplica al modelo local.
		model.set_weights(parameters)

		# 2) Entrena localmente una época (aquí se podría ajustar epochs según necesidad).
		history = model.fit(train_ds, epochs=1, verbose=1)

		# 3) Prepara un diccionario con métricas de entrenamiento para enviar al servidor.
		results = {
			"loss": float(history.history["loss"][0]),
			"accuracy": float(history.history["accuracy"][0]),
		}

		# 4) Devuelve (pesos actualizados, número de ejemplos usados, métricas).
		return model.get_weights(), int(num_train_examples), results

    def evaluate(self, parameters, config):
		# 1) Aplica los parámetros globales recibidos al modelo local.
		model.set_weights(parameters)

		# 2) Evalúa sobre el conjunto de validación para obtener loss y accuracy.
		loss, accuracy = model.evaluate(val_ds, verbose=0)

		# 3) Calcula predicciones para obtener precision/recall/f1.
		y_true, y_pred_probs = [], []
		for images, labels in val_ds:
			preds = model.predict(images, verbose=0)
			y_pred_probs.extend(preds)
			y_true.extend(labels.numpy())

		# Convertimos probabilidades a clases binarias con umbral 0.5.
		y_pred_classes = (np.array(y_pred_probs) > 0.5).astype("int32").flatten()
		y_true = np.array(y_true).astype("int32").flatten()

		# 4) Calcula precision, recall y f1 a nivel local.
		precision, recall, f1, _ = precision_recall_fscore_support(
			y_true, y_pred_classes, average="binary", zero_division=0
		)

		# 5) Captura métricas de hardware del cliente para análisis conjunto.
		cpu_percent = psutil.cpu_percent(interval=1)
		ram_percent = psutil.virtual_memory().percent

		# 6) Empaqueta todas las métricas (modelo + hardware) en un diccionario.
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

		# 7) Devuelve (loss, número de ejemplos de validación, métricas) como espera Flower.
		return float(loss), int(num_val_examples), metrics

# --- 5. INICIAR EL CLIENTE ---
if __name__ == "__main__":
	# Mensaje informativo de inicio.
	print(f"Iniciando Cliente {CLIENT_ID} en {SERVER_IP}...")

	# Se inicia el cliente y se conecta al servidor federado.
	fl.client.start_client(server_address=SERVER_IP, client=PlacasClient().to_client())
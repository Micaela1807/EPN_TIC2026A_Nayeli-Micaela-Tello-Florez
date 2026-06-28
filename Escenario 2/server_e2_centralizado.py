# ============================================================
# ESCENARIO 2
# Nombre del script: server_e2_centralizado.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
# Escenario: Aprendizaje Centralizado
#
# Descripción:
# Este script implementa un servidor centralizado para un escenario de aprendizaje centralizado, donde todos los datos están disponibles en un solo lugar y se entrena un único modelo.
# ============================================================


# ==========================
# SECCIÓN: IMPORTACIÓN DE LIBRERÍAS
# ==========================
import tensorflow as tf  # librería para definir y entrenar modelos de deep learning
import time  # librería para medir tiempos de ejecución
import mlflow  # librería para el seguimiento de experimentos y métricas
import numpy as np  # librería para manejo de arreglos y operaciones numéricas
import psutil  # librería para monitorear el uso de CPU y memoria
from sklearn.metrics import precision_recall_fscore_support  # métricas de clasificación binaria

# ==========================
# SECCIÓN: CONFIGURACIÓN DE MLFLOW
# ==========================
# Establece la ruta donde se guardará el seguimiento de experimentos de MLflow
mlflow.set_tracking_uri("file:./mlruns")

# Define el nombre del experimento para organizar y clasificar las ejecuciones
mlflow.set_experiment("Aprendizaje_Centralizado_Deteccion_Placas")

# ==========================
# SECCIÓN: CONFIGURACIÓN DE ENTRENAMIENTO
# ==========================
# Ruta donde se encuentra el dataset centralizado (combina datos de todos los clientes)
DATASET_PATH = r"K:\Escritorio\TIC\dataset\Dataset_centralizado"

# Dimensiones de las imágenes (alto, ancho) que espera el modelo
IMG_SIZE = (224, 224)

# Tamaño de lote utilizado en el entrenamiento y validación
BATCH_SIZE = 16

# Número total de épocas para entrenar el modelo
EPOCHS = 30 


# ==========================
# SECCIÓN: CARGA Y PREPROCESAMIENTO DE DATOS
# ==========================
print("Cargando dataset unificado...")

# Cargar datos de entrenamiento desde el dataset centralizado
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,  # Separa el 20% de los datos para validación
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Cargar datos de validación desde el dataset centralizado
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,  # Separa el 20% de los datos para validación
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Normalizar los píxeles de la imagen a [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)

# Aplicar la normalización a los datos de entrenamiento
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Aplicar la normalización a los datos de validación
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ==========================
# SECCIÓN: DEFINICIÓN DEL MODELO
# ==========================
# Cargar el modelo preentrenado MobileNetV2 como extractor de características
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
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

# ==========================
# SECCIÓN: CALLBACK PARA LOGUEAR MÉTRICAS (ML Y HARDWARE)
# ==========================
class ComprehensiveLoggingCallback(tf.keras.callbacks.Callback):
    """Callback personalizado para registrar métricas de entrenamiento y hardware en cada época.
    
    Calcula métricas de clasificación (precision, recall, f1) y monitorea uso de recursos
    (CPU y RAM) para estudiar cómo afecta el entrenamiento centralizado a los recursos.
    """

    def on_epoch_end(self, epoch, logs=None):
        """Se ejecuta al final de cada época de entrenamiento.
        
        Args:
            epoch: Número de la época actual.
            logs: Diccionario con métricas de la época (loss, accuracy, etc.).
        """
        # 1. Obtener predicciones en todo el conjunto de validación
        y_true = []
        y_pred_probs = []
        for images, labels in val_ds:
            preds = self.model.predict(images, verbose=0)
            y_pred_probs.extend(preds)
            y_true.extend(labels.numpy())

        # Convertir probabilidades a clases binarias (umbral: 0.5)
        y_pred_classes = (np.array(y_pred_probs) > 0.5).astype("int32").flatten()

        # Calcular métricas adicionales de clasificación
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_classes, average="binary", zero_division=0
        )

        # 2. Medir uso de recursos del sistema
        cpu_usage = psutil.cpu_percent(interval=None)
        ram_usage = psutil.virtual_memory().percent

        # 3. Registrar todas las métricas en MLflow
        mlflow.log_metric("loss", logs["val_loss"], step=epoch)
        mlflow.log_metric("accuracy", logs["val_accuracy"], step=epoch)
        mlflow.log_metric("precision", float(precision), step=epoch)
        mlflow.log_metric("recall", float(recall), step=epoch)
        mlflow.log_metric("f1_score", float(f1), step=epoch)
        mlflow.log_metric("cpu_usage_percent", cpu_usage, step=epoch)
        mlflow.log_metric("ram_usage_percent", ram_usage, step=epoch)

        # Mostrar resumen de la época en consola
        print(f" - Época {epoch + 1} | CPU: {cpu_usage}% | RAM: {ram_usage}% | Registrado en MLflow.")

# ==========================
# SECCIÓN: EJECUCIÓN PRINCIPAL DEL ENTRENAMIENTO CENTRALIZADO
# ==========================
# Crear una nueva ejecución (run) en MLflow para registrar este experimento
with mlflow.start_run(run_name="Entrenamiento_Centralizado_Completo"):
    print("Iniciando cronómetro, sensores de recursos y registro en MLflow...")
    
    # Registrar el tiempo de inicio
    start_time = time.time()
    
    # Inicializar psutil para asegurar lecturas precisas desde la época 1
    psutil.cpu_percent(interval=None)

    # Entrenar el modelo con los datos centralizados
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ComprehensiveLoggingCallback()],  # Callback personalizado para logging
    )

    # Calcular tiempo total de entrenamiento
    total_time = time.time() - start_time

    # Registrar parámetros del experimento en MLflow
    mlflow.log_param("tipo_entrenamiento", "Centralizado")
    mlflow.log_param("learning_rate", "1e-5")
    
    # Registrar métrica final: tiempo total de ejecución
    mlflow.log_metric("tiempo_total_seg", total_time)

    # Mostrar resultado final
    print(f"Completado en: {total_time:.2f} seg.")
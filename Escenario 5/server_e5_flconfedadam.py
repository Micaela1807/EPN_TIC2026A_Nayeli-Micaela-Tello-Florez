# ============================================================
# SERVIDOR - ESCENARIO 5
# Nombre del script: server_e5_flconfedadam.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
# Escenario: Aprendizaje Federado Horizontal
#
# Descripción: 
# Este script implementa un servidor federado para un escenario de aprendizaje federado horizontal, con un enfoque en la comparación de estrategias de agregación, específicamente entre el optimizador Adam y el optimizador SGD.
# ============================================================


# ==========================
# SECCIÓN: IMPORTACIÓN DE LIBRERÍAS
# ==========================
import flwr as fl  # librería principal para construir servidores federados
import mlflow  # librería para el seguimiento de experimentos y métricas
import psutil  # librería para monitorear el uso de CPU y memoria
import tensorflow as tf  # librería para definir y manipular modelos de deep learning
from typing import List, Tuple, Dict, Optional  # anotaciones de tipos para mayor claridad
from flwr.common import Metrics, ndarrays_to_parameters  # tipos y utilitarios de Flower para conversión de parámetros

# ==========================
# SECCIÓN: CONFIGURACIÓN DE MLFLOW
# ==========================
# Establece la ubicación local donde MLflow almacenará los datos de experimentos
mlflow.set_tracking_uri("file:./mlruns")

# Define el nombre del experimento para agrupar las ejecuciones de esta prueba
mlflow.set_experiment("Prueba_5_Estrategias_Agregacion") 

# ==========================
# SECCIÓN: FUNCIÓN DE AGREGACIÓN DE MÉTRICAS (GLOBAL)
# ==========================

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calcula el promedio ponderado de métricas de clasificación de todos los clientes.
    
    Esta función agrega las métricas locales de los clientes en una única métrica global,
    ponderando por la cantidad de ejemplos que cada cliente usó en su evaluación.
    Utilizadas en la estrategia FedAdam como función de agregación de evaluación.
    
    Args:
        metrics: Lista de tuplas (número_ejemplos, dict_métricas) para cada cliente que reportó.
        
    Returns:
        Diccionario con métricas globales ponderadas (accuracy, precision, recall, f1_score).
    """
    # Extraer el número de ejemplos de cada cliente
    examples = [num_examples for num_examples, _ in metrics]
    
    # Sumar el total de ejemplos usados por todos los clientes
    total_examples = sum(examples)
    
    # Si ningún cliente reportó métricas, devolver valores por defecto (0.0)
    if total_examples == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    # Calcular el promedio ponderado de accuracy
    weighted_acc = sum(n * m["accuracy"] for n, m in metrics) / total_examples
    
    # Calcular el promedio ponderado de precision
    weighted_prec = sum(n * m["precision"] for n, m in metrics) / total_examples
    
    # Calcular el promedio ponderado de recall
    weighted_rec = sum(n * m["recall"] for n, m in metrics) / total_examples
    
    # Calcular el promedio ponderado de f1_score
    weighted_f1 = sum(n * m["f1_score"] for n, m in metrics) / total_examples

    # Devolver diccionario con todas las métricas globales ponderadas
    return {
        "accuracy": weighted_acc,
        "precision": weighted_prec,
        "recall": weighted_rec,
        "f1_score": weighted_f1,
    }

# ==========================
# SECCIÓN: INICIALIZACIÓN DE PARÁMETROS PARA FEDADAM
# ==========================

def get_initial_parameters():
    """Genera los parámetros iniciales del modelo para la estrategia FedAdam.
    
    FedAdam requiere parámetros iniciales en formato de Flower (ndarrays) para
    inicializar la primera ronda de coordinación. Creamos la misma arquitectura
    que los clientes para asegurar compatibilidad.
    
    Returns:
        Objeto Parameters de Flower contiendo los pesos iniciales del modelo.
    """
    # Crear el modelo base MobileNetV2 preentrenado (sin entrenar en el servidor)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",  # Usar pesos preentrenados en ImageNet
    )
    
    # Construir el modelo completo con las capas finales
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    
    # Convertir los pesos de NumPy al formato de Flower (ndarrays_to_parameters)
    # Esto es requerido por FedAdam para inicializar correctamente
    return ndarrays_to_parameters(model.get_weights())

# ==========================
# SECCIÓN: ESTRATEGIA FEDADAM PERSONALIZADA CON MLFLOW
# ==========================

class MLflowStrategy(fl.server.strategy.FedAdam):
    """Estrategia de agregación FedAdam personalizada con logging en MLflow.
    
    FedAdam (Federated Adam) es una estrategia de optimización adaptativa que utiliza
    momentum y adaptación de tasa de aprendizaje para mejorar la convergencia del modelo
    federado. Esta clase extiende FedAdam para registrar métricas detalladas en MLflow.
    
    Ventajas sobre FedAvg:
    - Converge más rápido gracias al momentum
    - Adapta automáticamente la tasa de aprendizaje por parámetro
    - Mejor manejo de datos no-IID (no idénticamente distribuidos)
    """

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Agrega los resultados de evaluación de todos los clientes y registra en MLflow.
        
        Args:
            server_round: Número de la ronda federada actual.
            results: Lista de tuplas (cliente_proxy, EvaluateRes) con resultados de cada cliente.
            failures: Lista de excepciones en caso de fallos de clientes.
            
        Returns:
            Tupla (pérdida_global, dict_métricas_globales) del modelo agregado.
        """
        # 1. Ejecutar la agregación FedAdam para obtener pérdida y métricas globales
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # 2. Medir recursos del servidor en este momento
        server_cpu = psutil.cpu_percent(interval=None)  # CPU en el servidor
        server_ram = psutil.virtual_memory().percent  # RAM en el servidor

        # 3. Registrar métricas en MLflow si la agregación fue exitosa
        if aggregated_loss is not None and aggregated_metrics is not None:
            # Mostrar resumen en consola
            print(f"Ronda {server_round} | Global Acc: {aggregated_metrics['accuracy']:.4f} | Servidor CPU: {server_cpu}%")
            
            # A. REGISTRAR MÉTRICAS GLOBALES DEL MODELO (Agregadas con FedAdam)
            mlflow.log_metric("Global_Loss", aggregated_loss, step=server_round)
            mlflow.log_metric("Global_Accuracy", aggregated_metrics["accuracy"], step=server_round)
            mlflow.log_metric("Global_Precision", aggregated_metrics["precision"], step=server_round)
            mlflow.log_metric("Global_Recall", aggregated_metrics["recall"], step=server_round)
            mlflow.log_metric("Global_F1_Score", aggregated_metrics["f1_score"], step=server_round)
            
            # B. REGISTRAR MÉTRICAS DE HARDWARE DEL SERVIDOR
            mlflow.log_metric("Servidor_CPU", server_cpu, step=server_round)
            mlflow.log_metric("Servidor_RAM", server_ram, step=server_round)

            # C. REGISTRAR MÉTRICAS INDIVIDUALES POR CLIENTE
            for client_proxy, eval_res in results:
                # Extraer el diccionario de métricas que envió el cliente
                c_metrics = eval_res.metrics
                
                # Obtener el ID del cliente
                c_id = c_metrics.get("client_id", "Desconocido")
                
                # Registrar métricas del modelo del cliente
                mlflow.log_metric(f"Cliente_{c_id}_Loss", c_metrics.get("loss", 0), step=server_round)
                mlflow.log_metric(f"Cliente_{c_id}_Accuracy", c_metrics.get("accuracy", 0), step=server_round)
                mlflow.log_metric(f"Cliente_{c_id}_Precision", c_metrics.get("precision", 0), step=server_round)
                mlflow.log_metric(f"Cliente_{c_id}_Recall", c_metrics.get("recall", 0), step=server_round)
                mlflow.log_metric(f"Cliente_{c_id}_F1_Score", c_metrics.get("f1_score", 0), step=server_round)
                
                # Registrar métricas de hardware del cliente
                mlflow.log_metric(f"Cliente_{c_id}_CPU", c_metrics.get("cpu_usage", 0), step=server_round)
                mlflow.log_metric(f"Cliente_{c_id}_RAM", c_metrics.get("ram_usage", 0), step=server_round)
            
        # Devolver la pérdida y métricas agregadas al servidor de Flower
        return aggregated_loss, aggregated_metrics

# ==========================
# SECCIÓN: INICIALIZACIÓN Y EJECUCIÓN DEL SERVIDOR
# ==========================

if __name__ == "__main__":
    """Bloque principal que configura y lanza el servidor Flower con estrategia FedAdam."""
    
    # Instanciar FedAdam con parámetros calibrados para entrenamiento de MobileNetV2
    strategy = MLflowStrategy(
        initial_parameters=get_initial_parameters(),  # Parámetros iniciales del modelo
        eta=1e-4,  # Tasa de aprendizaje global del servidor (reducida a 1e-4 para pasos finos)
        eta_l=1e-5,  # Tasa de aprendizaje local de los clientes (se mantiene baja)
        tau=1e-4,  # Parámetro de escala para adaptación (controla el momentum)
        fraction_fit=1.0,  # Proporción de clientes disponibles para entrenar (100%)
        fraction_evaluate=1.0,  # Proporción de clientes disponibles para evaluar (100%)
        min_fit_clients=2,  # Número mínimo de clientes para entrenar
        min_evaluate_clients=2,  # Número mínimo de clientes para evaluar
        min_available_clients=2,  # Número mínimo de clientes disponibles antes de iniciar
        evaluate_metrics_aggregation_fn=weighted_average,  # Función personalizada para agregación
    )

    # Mostrar banner de inicio del servidor
    print("\n" + "="*50)
    print("SERVIDOR FLOWER (FEDADAM) INICIADO")
    print("="*50 + "\n")

    # Envolver toda la ejecución en UN SOLO Run de MLflow
    # Esto agrupa todas las rondas bajo una única ejecución para mejor trazabilidad
    with mlflow.start_run(run_name="Sesion_Completa_FedAdam"):
        # Iniciar el servidor de Flower con la estrategia FedAdam personalizada
        fl.server.start_server(
            server_address="0.0.0.0:8080",  # Escuchar en todas las interfaces en puerto 8080
            config=fl.server.ServerConfig(num_rounds=30),  # Ejecutar 30 rondas de coordinación
            strategy=strategy,  # Usar estrategia FedAdam con logging personalizado en MLflow
        )
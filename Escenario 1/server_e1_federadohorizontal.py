# ============================================================
# SERVIDOR - ESCENARIO 1
# Nombre del script: server_e1_federadohorizontal.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
# Escenario: Aprendizaje Federado Horizontal
#
# Descripción: 
# Este script implementa un servidor federado para un escenario de aprendizaje federado horizontal, coordinando la comunicación y agregación de modelos entre múltiples clientes.
# ============================================================


# ==========================
# SECCIÓN: IMPORTACIÓN DE LIBRERÍAS
# ==========================
import flwr as fl  # librería principal para construir servidores y clientes de aprendizaje federado
import mlflow  # librería para el seguimiento de experimentos, métricas y parámetros
import psutil  # librería para monitorear el uso de CPU, memoria y otros recursos del sistema
from typing import List, Tuple, Dict, Optional  # anotaciones de tipos para mayor claridad y verificación estática
from flwr.common import Metrics  # tipo de dato que define las métricas intercambiadas entre cliente-servidor

# ==========================
# SECCIÓN: CONFIGURACIÓN DE MLFLOW
# ==========================
# Establece la ubicación local donde MLflow almacenará los datos de experimentos y artefactos
mlflow.set_tracking_uri("file:./mlruns")

# Define el nombre del experimento para organizar y agrupar las ejecuciones de pruebas
mlflow.set_experiment("Piloto_Deteccion_Placas")

# ==========================
# SECCIÓN: FUNCIÓN DE AGREGACIÓN DE MÉTRICAS (GLOBAL)
# ==========================

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calcula el promedio ponderado de métricas de clasificación de todos los clientes.
    
    Esta función agrega las métricas locales de los clientes en una única métrica global,
    ponderando por la cantidad de ejemplos que cada cliente usó en su evaluación.
    Es útil para obtener un resumen representativo del desempeño global del modelo federado.
    
    Args:
        metrics: Lista de tuplas (número_ejemplos, dict_métricas) para cada cliente que reportó.
        
    Returns:
        Diccionario con métricas globales ponderadas (accuracy, precision, recall, f1_score).
    """
    # Extraer el número de ejemplos de cada cliente (primera parte de la tupla)
    examples = [num_examples for num_examples, _ in metrics]
    
    # Sumar el total de ejemplos usados por todos los clientes
    total_examples = sum(examples)
    
    # Si ningún cliente reportó métricas, devolver valores por defecto (0.0)
    if total_examples == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    # Calcular el promedio ponderado de accuracy: suma(ejemplos * accuracy) / total_ejemplos
    weighted_acc = sum(n * m["accuracy"] for n, m in metrics) / total_examples
    
    # Calcular el promedio ponderado de precision
    weighted_prec = sum(n * m["precision"] for n, m in metrics) / total_examples
    
    # Calcular el promedio ponderado de recall
    weighted_rec = sum(n * m["recall"] for n, m in metrics) / total_examples
    
    # Calcular el promedio ponderado de f1_score
    weighted_f1 = sum(n * m["f1_score"] for n, m in metrics) / total_examples

    # Devolver un diccionario con todas las métricas globales ponderadas
    return {
        "accuracy": weighted_acc,
        "precision": weighted_prec,
        "recall": weighted_rec,
        "f1_score": weighted_f1,
    }

# ==========================
# SECCIÓN: ESTRATEGIA PERSONALIZADA CONECTADA A MLFLOW
# ==========================

class MLflowStrategy(fl.server.strategy.FedAvg):
    """Estrategia de agregación federada personalizada basada en FedAvg (Federated Averaging).
    
    Extiende la estrategia estándar FedAvg de Flower para registrar métricas detalladas
    en MLflow después de cada ronda de evaluación. Captura tanto métricas globales del modelo
    como métricas individuales de cada cliente y recursos del servidor.
    """

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Agrega los resultados de evaluación de todos los clientes y registra en MLflow.
        
        Este método se invoca automáticamente al final de cada ronda para agregar
        las métricas de evaluación de los clientes, calcular métricas globales y registrar
        toda la información en MLflow para su posterior análisis.
        
        Args:
            server_round: Número de la ronda federada actual.
            results: Lista de tuplas (cliente_proxy, EvaluateRes) con resultados de cada cliente.
            failures: Lista de excepciones en caso de fallos de clientes (puede estar vacía).
            
        Returns:
            Tupla (pérdida_global, dict_métricas_globales) del modelo agregado.
        """
        # 1. Ejecutar la agregación estándar FedAvg para obtener pérdida y métricas globales
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # 2. Medir recursos de CPU y RAM del servidor en este momento
        server_cpu = psutil.cpu_percent(interval=None)  # Porcentaje de uso de CPU en el servidor
        server_ram = psutil.virtual_memory().percent  # Porcentaje de uso de RAM en el servidor

        # 3. Registrar métricas en MLflow si la agregación fue exitosa
        if aggregated_loss is not None and aggregated_metrics is not None:
            # Mostrar en consola un resumen rápido de la ronda
            print(f"Ronda {server_round} | Global Acc: {aggregated_metrics['accuracy']:.4f} | Servidor CPU: {server_cpu}%")
            
            # Crear una sub-ejecución anidada en MLflow para esta ronda
            with mlflow.start_run(run_name=f"Ronda_{server_round}", nested=True):
                # A. REGISTRAR MÉTRICAS GLOBALES DEL MODELO (Agregadas del servidor)
                mlflow.log_metric("Global_Loss", aggregated_loss, step=server_round)
                mlflow.log_metric("Global_Accuracy", aggregated_metrics["accuracy"], step=server_round)
                mlflow.log_metric("Global_Precision", aggregated_metrics["precision"], step=server_round)
                mlflow.log_metric("Global_Recall", aggregated_metrics["recall"], step=server_round)
                mlflow.log_metric("Global_F1_Score", aggregated_metrics["f1_score"], step=server_round)
                
                # B. REGISTRAR MÉTRICAS DE HARDWARE DEL SERVIDOR
                mlflow.log_metric("Servidor_CPU", server_cpu, step=server_round)
                mlflow.log_metric("Servidor_RAM", server_ram, step=server_round)

                # C. REGISTRAR MÉTRICAS INDIVIDUALES POR CLIENTE (Hardware + Modelo)
                for client_proxy, eval_res in results:
                    # Extraer el diccionario de métricas que envió el cliente en eval_res.metrics
                    c_metrics = eval_res.metrics
                    
                    # Obtener el ID del cliente (o "Desconocido" si no viene en las métricas)
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
    """Bloque principal que configura y lanza el servidor Flower con la estrategia personalizada."""
    
    # Instanciar la estrategia FedAvg personalizada con parámetros de coordinación federada
    strategy = MLflowStrategy(
        fraction_fit=1.0,  # Proporción de clientes disponibles a usar en cada ronda de entrenamiento (100%)
        fraction_evaluate=1.0,  # Proporción de clientes disponibles a usar en cada ronda de evaluación (100%)
        min_fit_clients=2,  # Número mínimo de clientes requeridos para participar en el entrenamiento
        min_evaluate_clients=2,  # Número mínimo de clientes requeridos para participar en la evaluación
        min_available_clients=2,  # Número mínimo de clientes disponibles antes de iniciar una ronda
        evaluate_metrics_aggregation_fn=weighted_average,  # Función personalizada para agregar métricas de evaluación
    )

    # Mostrar banner de inicio del servidor en consola
    print("\n" + "="*50)
    print("SERVIDOR FLOWER MONITOREADO INICIADO")
    print("Esperando conexión de 2 Clientes (VMs)...")
    print("="*50 + "\n")

    # Iniciar el servidor de Flower con la estrategia personalizada
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Escuchar en todas las interfaces de red en puerto 8080
        config=fl.server.ServerConfig(num_rounds=30),  # Configurar 30 rondas de entrenamiento federado
        strategy=strategy,  # Usar la estrategia personalizada con logging en MLflow
    )
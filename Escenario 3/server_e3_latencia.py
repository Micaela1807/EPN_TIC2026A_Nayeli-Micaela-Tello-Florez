# ============================================================
# SERVIDOR - ESCENARIO 3
# Nombre del script: server_e3_latencia.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
# Escenario: Impacto de la Latencia en la Red
#
# Descripción:
# Este script implementa un servidor federado para un escenario de aprendizaje
# federado horizontal, con un enfoque en la medición del impacto de la latencia
# en la red durante el entrenamiento y evaluación del modelo.
# ============================================================

# ==========================
# SECCIÓN: IMPORTACIÓN DE LIBRERÍAS
# ==========================
# Aquí se cargan las librerías necesarias para el servidor Flower,
# el registro de métricas con MLflow, y la medición de recursos del servidor.
import flwr as fl # librería principal para construir servidores federados
import mlflow # librería para el seguimiento de experimentos y métricas
import psutil  # librería para medir el uso de CPU y RAM del servidor
import time # librería para medir la duración de las rondas de entrenamiento
from typing import List, Tuple, Dict, Optional #librerías para tipado de funciones y estructuras de datos
from flwr.common import Metrics #librería para definir el tipo de métricas que se manejarán en el servidor

# --- CONFIGURACIÓN DE MLFLOW ---
# Se establece la ruta local donde se guardarán los resultados de MLflow.
mlflow.set_tracking_uri("file:./mlruns")
# Se define el experimento con nombre específico para este escenario.
mlflow.set_experiment("Prueba3_FL_Latencia_Red")

# ==========================
# SECCIÓN: FUNCIÓN DE AGREGACIÓN DE MÉTRICAS
# ==========================

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calcula el promedio ponderado de métricas entre clientes."""
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    # Si el total de ejemplos es cero, devolvemos métricas neutras.
    if total_examples == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    # Calcula cada métrica global como promedio ponderado por número de ejemplos.
    weighted_acc = sum(n * m["accuracy"] for n, m in metrics) / total_examples
    weighted_prec = sum(n * m["precision"] for n, m in metrics) / total_examples
    weighted_rec = sum(n * m["recall"] for n, m in metrics) / total_examples
    weighted_f1 = sum(n * m["f1_score"] for n, m in metrics) / total_examples

    return {
        "accuracy": weighted_acc,
        "precision": weighted_prec,
        "recall": weighted_rec,
        "f1_score": weighted_f1,
    }

# ==========================
# SECCIÓN: ESTRATEGIA PERSONALIZADA DE FLOWER
# ==========================
class MLflowStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_start_time = 0.0

    def configure_fit(self, server_round: int, parameters, client_manager):
        # Capturamos el tiempo exacto en que inicia la ronda para medir latencia.
        self.round_start_time = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Agrega métricas de evaluación y registra datos en MLflow."""

        # 1. Obtiene la pérdida y las métricas agregadas de Flower.
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # 2. Calcula la duración total de la ronda.
        round_duration = time.time() - self.round_start_time

        # 3. Mide uso de hardware del servidor.
        server_cpu = psutil.cpu_percent(interval=None)
        server_ram = psutil.virtual_memory().percent

        if aggregated_loss is not None and aggregated_metrics is not None:
            print(f"📊 [RONDA {server_round}] -> Duración: {round_duration:.2f}s | Global Acc: {aggregated_metrics['accuracy']:.4f}")

            # 4. Guarda métricas en MLflow para esta ronda.
            with mlflow.start_run(run_name=f"Ronda_{server_round}", nested=True):
                # 4.1. Métrica de latencia.
                mlflow.log_metric("duracion_ronda_segundos", round_duration, step=server_round)
                
                # 4.2. Métricas globales del servidor.
                mlflow.log_metric("Global_Loss", aggregated_loss, step=server_round)
                mlflow.log_metric("Global_Accuracy", aggregated_metrics["accuracy"], step=server_round)
                mlflow.log_metric("Global_Precision", aggregated_metrics["precision"], step=server_round)
                mlflow.log_metric("Global_Recall", aggregated_metrics["recall"], step=server_round)
                mlflow.log_metric("Global_F1_Score", aggregated_metrics["f1_score"], step=server_round)
                
                # 4.3. Consumo de hardware del servidor.
                mlflow.log_metric("Servidor_CPU", server_cpu, step=server_round)
                mlflow.log_metric("Servidor_RAM", server_ram, step=server_round)

                # 4.4. Registra las métricas individuales de cada cliente.
                for client_proxy, eval_res in results:
                    c_metrics = eval_res.metrics
                    c_id = c_metrics.get("client_id", "Desconocido")
                    
                    # Métricas de rendimiento del modelo del cliente.
                    mlflow.log_metric(f"Cliente_{c_id}_Loss", c_metrics.get("loss", 0), step=server_round)
                    mlflow.log_metric(f"Cliente_{c_id}_Accuracy", c_metrics.get("accuracy", 0), step=server_round)
                    mlflow.log_metric(f"Cliente_{c_id}_Precision", c_metrics.get("precision", 0), step=server_round)
                    mlflow.log_metric(f"Cliente_{c_id}_Recall", c_metrics.get("recall", 0), step=server_round)
                    mlflow.log_metric(f"Cliente_{c_id}_F1_Score", c_metrics.get("f1_score", 0), step=server_round)
                    
                    # Métricas de hardware del cliente.
                    mlflow.log_metric(f"Cliente_{c_id}_CPU", c_metrics.get("cpu_usage", 0), step=server_round)
                    mlflow.log_metric(f"Cliente_{c_id}_RAM", c_metrics.get("ram_usage", 0), step=server_round)
            
        return aggregated_loss, aggregated_metrics

if __name__ == "__main__":
    # 1. Define la estrategia con los parámetros de federated learning.
    strategy = MLflowStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # 2. Imprime en consola información de inicio del servidor.
    print("\n" + "=" * 60)
    print("SERVIDOR FLOWER - PRUEBA DE LATENCIA Y HARDWARE")
    print("=" * 60 + "\n")

    # 3. Inicia un run global en MLflow para agrupar las métricas de todas las rondas.
    with mlflow.start_run(run_name="Servidor_Escenario_Latencia_Completo"):
        # 4. Arranca el servidor Flower y espera clientes en 0.0.0.0:8080.
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=30),
            strategy=strategy,
        )
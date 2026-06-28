# ============================================================
# SERVIDOR - ESCENARIO 6
# Nombre del script: server_e6_flhorizontal2epocas.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
# Escenario: Aprendizaje Federado Horizontal
#
# Descripción: 
# Este script implementa un servidor federado para un escenario de aprendizaje federado horizontal, ejecutando 2 épocas de entrenamiento local en los clientes.
# ============================================================


# ==========================
# SECCIÓN: IMPORTACIÓN DE LIBRERÍAS
# ==========================
import flwr as fl # librería principal para construir servidores federados
import mlflow # librería para el seguimiento de experimentos y métricas
import psutil  # librería para medir el uso de CPU y RAM del servidor
from typing import List, Tuple, Dict, Optional #librerías para tipado de funciones y estructuras de datos
from flwr.common import Metrics #librería para definir el tipo de métricas que se manejarán en el servidor

# ==========================
# SECCIÓN: CONFIGURACIÓN DE MLFLOW
# ==========================
# Se define la carpeta local donde MLflow guardará los resultados.
mlflow.set_tracking_uri("file:./mlruns")
# Se selecciona el experimento que agrupará las ejecuciones de este servidor.
mlflow.set_experiment("Prueba6_FlHorizontal_2Epocas")

# ==========================
# SECCIÓN: FUNCIÓN DE AGREGACIÓN DE MÉTRICAS
# ==========================
# Calcula métricas globales a partir de resultados de evaluación de todos los clientes.
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    
    # Si no hay ejemplos en ninguno de los clientes, devolvemos métricas neutras.
    if total_examples == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    # Promedio ponderado por número de ejemplos de cada cliente.
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
# SECCIÓN: ESTRATEGIA PERSONALIZADA
# ==========================
# Se extiende FedAvg para capturar métricas adicionales y enviarlas a MLflow.
class MLflowStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        
        # 1. Ejecutar FedAvg para obtener la pérdida y métricas globales.
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # 2. Medir hardware del servidor en el momento de agregación.
        server_cpu = psutil.cpu_percent(interval=None)
        server_ram = psutil.virtual_memory().percent

        # 3. Registrar las métricas agregadas en MLflow.
        if aggregated_loss is not None and aggregated_metrics is not None:
            print(f"Ronda {server_round} | Global Acc: {aggregated_metrics['accuracy']:.4f} | Servidor CPU: {server_cpu}%")
            
            with mlflow.start_run(run_name=f"Ronda_{server_round}", nested=True):
                # A. Métricas globales agregadas por el servidor.
                mlflow.log_metric("Global_Loss", aggregated_loss, step=server_round)
                mlflow.log_metric("Global_Accuracy", aggregated_metrics["accuracy"], step=server_round)
                mlflow.log_metric("Global_Precision", aggregated_metrics["precision"], step=server_round)
                mlflow.log_metric("Global_Recall", aggregated_metrics["recall"], step=server_round)
                mlflow.log_metric("Global_F1_Score", aggregated_metrics["f1_score"], step=server_round)
                
                # B. Métricas de hardware del servidor.
                mlflow.log_metric("Servidor_CPU", server_cpu, step=server_round)
                mlflow.log_metric("Servidor_RAM", server_ram, step=server_round)

                # C. Métricas individuales enviadas por cada cliente.
                for client_proxy, eval_res in results:
                    # eval_res.metrics contiene el diccionario que envió el cliente.
                    c_metrics = eval_res.metrics
                    c_id = c_metrics.get("client_id", "Desconocido")
                    
                    # 1) Métricas de rendimiento del modelo del cliente.
                    mlflow.log_metric(f"Cliente_{c_id}_Loss", c_metrics.get("loss", 0), step=server_round)
                    mlflow.log_metric(f"Cliente_{c_id}_Accuracy", c_metrics.get("accuracy", 0), step=server_round)
                    mlflow.log_metric(f"Cliente_{c_id}_Precision", c_metrics.get("precision", 0), step=server_round)
                    mlflow.log_metric(f"Cliente_{c_id}_Recall", c_metrics.get("recall", 0), step=server_round)
                    mlflow.log_metric(f"Cliente_{c_id}_F1_Score", c_metrics.get("f1_score", 0), step=server_round)
                    
                    # 2) Métricas de hardware del cliente.
                    mlflow.log_metric(f"Cliente_{c_id}_CPU", c_metrics.get("cpu_usage", 0), step=server_round)
                    mlflow.log_metric(f"Cliente_{c_id}_RAM", c_metrics.get("ram_usage", 0), step=server_round)
            
        return aggregated_loss, aggregated_metrics

# ==========================
# SECCIÓN: INICIO DEL SERVIDOR
# ==========================
if __name__ == "__main__":
    # Configura la estrategia federada con el número mínimo de clientes.
    strategy = MLflowStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Mensajes de inicio para el operador.
    print("\n" + "=" * 50)
    print("SERVIDOR FLOWER MONITOREADO INICIADO")
    print("Esperando conexión de 2 Clientes (VMs)...")
    print("=" * 50 + "\n")

    # Inicia el servidor de Flower en la dirección local y ejecuta 30 rondas.
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )
# ============================================================
# SERVIDOR - ESCENARIO 4
# Nombre del script: server_e4_desconexion.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
# Escenario: Aprendizaje Federado Horizontal
#
# Descripción: 
# Este script implementa un servidor federado para un escenario de aprendizaje federado horizontal, con un enfoque en la tolerancia a fallos y la detección de desconexiones de clientes durante el entrenamiento y evaluación del modelo.
# ============================================================


# ==========================
# SECCIÓN: IMPORTACIÓN DE LIBRERÍAS
# ==========================
import flwr as fl  # librería principal para construir servidores federados
import mlflow  # librería para el seguimiento de experimentos y métricas
import os  # librería para operaciones del sistema operativo y variables de entorno
import psutil # librería para medir el uso de CPU y RAM del servidor

# ==========================
# SECCIÓN: CONFIGURACIÓN GLOBAL
# ==========================

# Deshabilitar el logging automático de MLflow para hardware. 
# Lo haremos manualmente extrayéndolo del diccionario para poder forzarlo a 0.0 cuando se caiga.
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

# Contador global para el eje X de MLflow
ronda_contador = 0
CLIENTES_ESPERADOS = ["VM_Cliente_1", "VM_Cliente_2"]

def weighted_average(metrics):
    """
    Agregación con Tolerancia a Fallos. 
    Registra métricas individuales y globales completas (Acc, Loss, Prec, Recall, F1, CPU, RAM).
    """
    global ronda_contador
    ronda_contador += 1

    # 1. CONSTRUIR DICCIONARIO DE CLIENTES VIVOS EN ESTA RONDA
    clientes_reportados = {}
    for num_examples, m in metrics:
        cid = m.get("client_id", "Desconocido")
        clientes_reportados[cid] = m
        clientes_reportados[cid]["num_examples"] = num_examples

    # 2. REGISTRAR EN MLFLOW (CLIENTES ACTIVOS VS DESCONECTADOS)
    for cid in CLIENTES_ESPERADOS:
        if cid in clientes_reportados:
            # EL CLIENTE ESTÁ VIVO: Extraer sus métricas (usamos .get() por seguridad)
            m = clientes_reportados[cid]
            mlflow.log_metric(f"{cid}_accuracy", m.get("accuracy", 0.0), step=ronda_contador)
            mlflow.log_metric(f"{cid}_loss", m.get("loss", 0.0), step=ronda_contador)
            mlflow.log_metric(f"{cid}_precision", m.get("precision", 0.0), step=ronda_contador)
            mlflow.log_metric(f"{cid}_recall", m.get("recall", 0.0), step=ronda_contador)
            mlflow.log_metric(f"{cid}_f1_score", m.get("f1_score", 0.0), step=ronda_contador)
            # Métricas de hardware
            mlflow.log_metric(f"{cid}_cpu_usage", m.get("cpu", 0.0), step=ronda_contador)
            mlflow.log_metric(f"{cid}_ram_usage", m.get("ram", 0.0), step=ronda_contador)
        else:
            # EL CLIENTE SE CAYÓ: Forzamos TODAS las métricas a 0.0
            mlflow.log_metric(f"{cid}_accuracy", 0.0, step=ronda_contador)
            mlflow.log_metric(f"{cid}_loss", 0.0, step=ronda_contador)
            mlflow.log_metric(f"{cid}_precision", 0.0, step=ronda_contador)
            mlflow.log_metric(f"{cid}_recall", 0.0, step=ronda_contador)
            mlflow.log_metric(f"{cid}_f1_score", 0.0, step=ronda_contador)
            mlflow.log_metric(f"{cid}_cpu_usage", 0.0, step=ronda_contador)
            mlflow.log_metric(f"{cid}_ram_usage", 0.0, step=ronda_contador)
            
            print(f"ALERTA: {cid} no reportó en la ronda {ronda_contador}. Gráficas caen a 0.")

    # 3. CÁLCULO DE MÉTRICAS GLOBALES (SERVIDOR)
    if not clientes_reportados:
        return {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0, "f1_score": 0}
    
    total_examples = sum([m["num_examples"] for m in clientes_reportados.values()])

    # Calcular promedios ponderados para el modelo global
    agg_accuracy = sum([m["num_examples"] * m.get("accuracy", 0.0) for m in clientes_reportados.values()]) / total_examples
    agg_loss = sum([m["num_examples"] * m.get("loss", 0.0) for m in clientes_reportados.values()]) / total_examples
    agg_precision = sum([m["num_examples"] * m.get("precision", 0.0) for m in clientes_reportados.values()]) / total_examples
    agg_recall = sum([m["num_examples"] * m.get("recall", 0.0) for m in clientes_reportados.values()]) / total_examples
    agg_f1 = sum([m["num_examples"] * m.get("f1_score", 0.0) for m in clientes_reportados.values()]) / total_examples

    # 4. REGISTRAR CONSOLIDADO GLOBAL EN MLFLOW
    mlflow.log_metric("Global_Server_accuracy", agg_accuracy, step=ronda_contador)
    mlflow.log_metric("Global_Server_loss", agg_loss, step=ronda_contador)
    mlflow.log_metric("Global_Server_precision", agg_precision, step=ronda_contador)
    mlflow.log_metric("Global_Server_recall", agg_recall, step=ronda_contador)
    mlflow.log_metric("Global_Server_f1_score", agg_f1, step=ronda_contador)

    # Medición de Hardware del Servidor
    server_cpu = psutil.cpu_percent(interval=None)
    server_ram = psutil.virtual_memory().percent

    mlflow.log_metric("Server_cpu_usage", server_cpu, step=ronda_contador)
    mlflow.log_metric("Server_ram_usage", server_ram, step=ronda_contador)

    print(f"Ronda {ronda_contador} -> Acc: {agg_accuracy:.4f} | Loss: {agg_loss:.4f} | F1: {agg_f1:.4f} | Clientes vivos: {len(clientes_reportados)}")
    
    return {"accuracy": agg_accuracy, "loss": agg_loss, "precision": agg_precision, "recall": agg_recall, "f1_score": agg_f1}

# ==========================
# ESTRATEGIA FEDAVG - TOLERANCIA A FALLOS
# ==========================
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0, # Permite que el modelo se entrene si queda al menos 1 cliente
    fraction_evaluate=1.0, # Permite que el modelo se evalúe si queda al menos 1 cliente
    min_fit_clients=1,       # Permite que el modelo se entrene si queda al menos 1
    min_evaluate_clients=1,  # Permite que el modelo se evalúe si queda al menos 1
    min_available_clients=1, # El servidor no colapsa al desconectarse un cliente
    evaluate_metrics_aggregation_fn=weighted_average,
)

# ==========================
# EJECUCIÓN DEL SERVIDOR
# ==========================
if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Prueba4_Tolerancia_Fallos")

    print("Iniciando Servidor Federado (Monitor de Caídas y Hardware)...")
    
    # Todo el proceso ocurre dentro de este run principal. Las 30 rondas se agruparán 
    # visualmente en las gráficas usando el parámetro 'step' configurado arriba.
    with mlflow.start_run(run_name="Servidor_Escenario4_Desconexion"):
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=30),
            strategy=strategy,
        )
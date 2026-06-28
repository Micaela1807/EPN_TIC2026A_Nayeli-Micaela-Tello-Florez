# ============================================================
# Nombre del script: preparar_datos.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
#
# Descripción:
# Este script organiza y divide un dataset de imágenes en dos conjuntos para dos clientes diferentes, enumerando las imágenes de manera secuencial.
# ============================================================

# Importa la librería os para trabajar con rutas y carpetas.
import os
# Importa shutil para copiar archivos de una carpeta a otra.
import shutil

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Coloca aquí la ruta exacta donde tienes tus carpetas originales con las 649 imágenes.
CARPETA_ORIGEN_PLACA = "path"
CARPETA_ORIGEN_NO_PLACA = r"path"

# Nombres de las carpetas principales destino.
DESTINO_C1 = "./path"
DESTINO_C2 = "./path"

# --- 2. FUNCIÓN PARA ENUMERAR Y DIVIDIR ---
# Define una función que organiza, renombra y reparte las imágenes en dos carpetas.
def organizar_y_enumerar(clase, ruta_origen):
    # Crea la estructura de carpetas para cada cliente y cada categoría si no existe.
    os.makedirs(os.path.join(DESTINO_C1, clase), exist_ok=True)
    os.makedirs(os.path.join(DESTINO_C2, clase), exist_ok=True)

    # Verifica si la carpeta origen existe antes de continuar.
    if not os.path.exists(ruta_origen):
        # Muestra un mensaje de error si la ruta no existe.
        print(f"Error: No se encontró la carpeta origen: {ruta_origen}")
        return

    # Lee todos los archivos de la carpeta origen y los ordena alfabéticamente.
    archivos = sorted([f for f in os.listdir(ruta_origen) if os.path.isfile(os.path.join(ruta_origen, f))])

    # Muestra cuántas imágenes se encontraron en esa categoría.
    print(f"Encontradas {len(archivos)} imágenes en la categoría '{clase}'. Procesando...")

    # Recorre cada archivo para renombrarlo y repartirlo.
    for i, archivo in enumerate(archivos):
        # Extrae la extensión original del archivo.
        extension = os.path.splitext(archivo)[1]

        # Crea un nuevo nombre enumerado con el formato: clase_1.jpg.
        nuevo_nombre = f"{clase}_{i + 1}{extension}"

        # Guarda la ruta actual del archivo en la carpeta origen.
        ruta_actual = os.path.join(ruta_origen, archivo)

        # Lógica de división: las primeras 324 imágenes van al Cliente 1.
        if i < 324:
            # Define la ruta de destino para el Cliente 1.
            ruta_nueva = os.path.join(DESTINO_C1, clase, nuevo_nombre)
        else:
            # El resto, desde la imagen 325, van al Cliente 2.
            ruta_nueva = os.path.join(DESTINO_C2, clase, nuevo_nombre)

        # Copia el archivo con su nuevo nombre a la carpeta destino correspondiente.
        shutil.copy2(ruta_actual, ruta_nueva)

# --- 3. EJECUCIÓN ---
# Muestra un mensaje inicial antes de comenzar.
print("Iniciando el proceso de partición y enumeración...")
# Ejecuta la función para la clase placa.
organizar_y_enumerar("placa", CARPETA_ORIGEN_PLACA)
# Ejecuta la función para la clase no_placa.
organizar_y_enumerar("no_placa", CARPETA_ORIGEN_NO_PLACA)

# Muestra un mensaje final cuando el proceso termina.
print("\n ¡Proceso completado con éxito!")
# Indica en qué carpetas quedaron los resultados.
print(f"Revisa las carpetas '{DESTINO_C1}' y '{DESTINO_C2}'.")
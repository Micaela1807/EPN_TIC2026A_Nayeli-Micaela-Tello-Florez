# ============================================================
# Nombre del script: preprocesar_yolo.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
#
# Descripción:
# Este script procesa un dataset en formato YOLO, recortando las placas de matrícula y generando imágenes de fondo aleatorias para entrenar un modelo MobileNetV2.
# ============================================================

# Importa la librería os para trabajar con rutas y carpetas del sistema.
import os
# Importa OpenCV para leer, modificar y guardar imágenes.
import cv2
# Importa random para generar posiciones aleatorias del fondo.
import random

# 1. Configura tus rutas aquí 
# Define la carpeta donde se encuentran las imágenes originales.
YOLO_IMAGES_DIR = r"path"
# Define la carpeta donde se encuentran los archivos de etiquetas en formato YOLO.
YOLO_LABELS_DIR = r"path"
# Define la carpeta donde se guardarán las imágenes procesadas.
OUTPUT_DIR = r"path"

# Crea la ruta completa para guardar las imágenes que contienen placas.
PLACA_DIR = os.path.join(OUTPUT_DIR, "placa")
# Crea la ruta completa para guardar las imágenes que no contienen placas.
NO_PLACA_DIR = os.path.join(OUTPUT_DIR, "no_placa")

# Crea la carpeta de placas si no existe.
os.makedirs(PLACA_DIR, exist_ok=True)
# Crea la carpeta de no placas si no existe.
os.makedirs(NO_PLACA_DIR, exist_ok=True)

# Muestra un mensaje inicial en la consola.
print("Iniciando diagnóstico de procesamiento de dataset YOLO...")
# Muestra la ruta donde se buscarán las imágenes.
print(f"Buscando imágenes en: {YOLO_IMAGES_DIR}")

# Comprobar si las carpetas existen antes de empezar.
if not os.path.exists(YOLO_IMAGES_DIR):
    # Muestra un error y termina el programa si la carpeta no existe.
    print("ERROR: No encuentro la carpeta de imágenes. Revisa la ruta.")
    exit()

if not os.path.exists(YOLO_LABELS_DIR):
    # Muestra un error y termina el programa si la carpeta no existe.
    print("ERROR: No encuentro la carpeta de etiquetas (labels). Revisa la ruta.")
    exit()

# Lista todos los archivos dentro de la carpeta de imágenes.
imagenes_encontradas = os.listdir(YOLO_IMAGES_DIR)
# Muestra cuántos archivos se encontraron.
print(f"Encontré {len(imagenes_encontradas)} archivos en la carpeta de imágenes.")

# Inicializa el contador de placas procesadas.
imagenes_procesadas = 0

# 2. Recorremos todas las imágenes.
for filename in imagenes_encontradas:
    # Ignora archivos que no tengan extensión de imagen válida, sin importar si está en mayúsculas o minúsculas.
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        # Salta al siguiente archivo si no es una imagen.
        continue

    # Construye la ruta completa de la imagen actual.
    img_path = os.path.join(YOLO_IMAGES_DIR, filename)
    # Construye el nombre del archivo de texto de etiquetas correspondiente.
    txt_filename = filename.rsplit('.', 1)[0] + '.txt'
    # Construye la ruta completa del archivo de etiquetas.
    txt_path = os.path.join(YOLO_LABELS_DIR, txt_filename)

    # Lee la imagen desde el disco.
    img = cv2.imread(img_path)
    # Verifica si la imagen se cargó correctamente.
    if img is None:
        # Muestra un aviso si no se pudo leer la imagen.
        print(f"No pude leer la imagen: {filename}")
        continue

    # Obtiene la altura y el ancho de la imagen.
    h, w, _ = img.shape

    # 3. Verificamos si tiene su archivo de etiquetas YOLO.
    if os.path.exists(txt_path):
        # Abre el archivo de etiquetas y lo lee línea por línea.
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        # Verifica si el archivo de texto está vacío.
        if len(lines) == 0:
            # Muestra un aviso si no hay anotaciones.
            print(f"El archivo de texto está vacío: {txt_filename}")

        # Recorre cada línea del archivo de etiquetas.
        for i, line in enumerate(lines):
            # El formato YOLO es: class x_center y_center width height (normalizados de 0 a 1).
            parts = line.strip().split()
            # Solo procesa líneas que tengan al menos 5 elementos.
            if len(parts) >= 5:
                # Convierte los valores de la etiqueta a números decimales.
                x_center, y_center, box_w, box_h = map(float, parts[1:5])

                # Des-normaliza las coordenadas para convertirlas a píxeles reales.
                x_center_abs = x_center * w
                y_center_abs = y_center * h
                box_w_abs = box_w * w
                box_h_abs = box_h * h

                # Calcula las coordenadas de las esquinas para recortar la placa.
                x_min = max(0, int(x_center_abs - box_w_abs / 2))
                y_min = max(0, int(y_center_abs - box_h_abs / 2))
                x_max = min(w, int(x_center_abs + box_w_abs / 2))
                y_max = min(h, int(y_center_abs + box_h_abs / 2))

                # --- RECORTAR PLACA ---
                # Extrae la región de la imagen que corresponde a la placa.
                placa_crop = img[y_min:y_max, x_min:x_max]
                # Verifica que el recorte no esté vacío.
                if placa_crop.size > 0:
                    # Redimensiona la placa a 128x128 píxeles.
                    placa_crop_resized = cv2.resize(placa_crop, (128, 128))
                    # Guarda la imagen recortada en la carpeta de placas.
                    cv2.imwrite(os.path.join(PLACA_DIR, f"placa_{i}_{filename}"), placa_crop_resized)
                    # Incrementa el contador de placas procesadas.
                    imagenes_procesadas += 1

                # --- RECORTAR NO PLACA (Fondo aleatorio) ---
                # Toma un recorte aleatorio de la misma imagen para que la red aprenda qué NO es una placa.
                bg_size = max(int(box_w_abs), 128)
                # Verifica que haya espacio suficiente para extraer un fondo aleatorio.
                if w > bg_size and h > bg_size:
                    # Genera una posición aleatoria dentro de la imagen.
                    rand_x = random.randint(0, w - bg_size)
                    rand_y = random.randint(0, h - bg_size)
                    # Extrae un recorte aleatorio de fondo.
                    no_placa_crop = img[rand_y:rand_y + bg_size, rand_x:rand_x + bg_size]

                    # Verifica que el recorte de fondo no esté vacío.
                    if no_placa_crop.size > 0:
                        # Redimensiona el fondo a 128x128 píxeles.
                        no_placa_crop_resized = cv2.resize(no_placa_crop, (128, 128))
                        # Guarda la imagen de fondo en la carpeta de no placas.
                        cv2.imwrite(os.path.join(NO_PLACA_DIR, f"noplaca_{i}_{filename}"), no_placa_crop_resized)
    else:
        # Muestra un aviso si la imagen no tiene su archivo de etiquetas.
        print(f"Imagen sin etiqueta: {filename} (buscaba {txt_filename})")

# Muestra el total de placas extraídas al finalizar.
print(f"¡Proceso terminado! Se extrajeron {imagenes_procesadas} placas.")
# Muestra la ruta donde quedaron guardados los datos.
print(f"Tus datos están listos en la carpeta: {OUTPUT_DIR}")
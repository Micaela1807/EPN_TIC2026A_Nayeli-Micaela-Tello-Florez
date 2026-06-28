# ============================================================
# Nombre del script: preprocesar_xml.py
# Elaborado por: Nayeli Micaela Tello Florez
# TRABAJO DE INTEGRACIÓN CURRICULAR - TIC 2026A
#
# Descripción:
# Este script procesa un dataset en formato PASCAL VOC (XML), recortando las placas de matrícula y generando imágenes de fondo aleatorias para entrenar un modelo MobileNetV2.
# ============================================================

# Importa la librería os para trabajar con rutas y carpetas del sistema.
import os
# Importa OpenCV para leer, modificar y guardar imágenes.
import cv2
# Importa random para generar posiciones aleatorias del fondo.
import random
# Importa ElementTree para leer archivos XML de anotaciones.
import xml.etree.ElementTree as ET

# 1. Configura tus rutas absolutas aquí 
# Define la carpeta donde se encuentran las imágenes originales.
XML_IMAGES_DIR = r"path"
# Define la carpeta donde se encuentran los archivos XML de anotaciones.
XML_LABELS_DIR = r"path"
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
print("Iniciando procesamiento de dataset PASCAL VOC (XML)...")
# Muestra la ruta donde se buscarán las imágenes.
print(f"Buscando imágenes en: {XML_IMAGES_DIR}")

# Comprueba si la carpeta de imágenes existe.
if not os.path.exists(XML_IMAGES_DIR):
    # Muestra un error y termina el programa si la carpeta no existe.
    print("ERROR: No encuentro la carpeta de imágenes. Revisa la ruta.")
    exit()

# Comprueba si la carpeta de XML existe.
if not os.path.exists(XML_LABELS_DIR):
    # Muestra un error y termina el programa si la carpeta no existe.
    print("ERROR: No encuentro la carpeta de XMLs. Revisa la ruta.")
    exit()

# Lista todos los archivos dentro de la carpeta de imágenes.
imagenes_encontradas = os.listdir(XML_IMAGES_DIR)
# Muestra cuántos archivos se encontraron.
print(f"Encontré {len(imagenes_encontradas)} archivos en la carpeta de imágenes.")

# Inicializa el contador de placas procesadas.
imagenes_procesadas = 0

# 2. Recorremos todas las imágenes.
for filename in imagenes_encontradas:
    # Solo procesa archivos con extensión de imagen válida.
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Salta al siguiente archivo si no es una imagen.
        continue

    # Construye la ruta completa de la imagen actual.
    img_path = os.path.join(XML_IMAGES_DIR, filename)
    # Construye el nombre del archivo XML correspondiente.
    xml_filename = filename.rsplit('.', 1)[0] + '.xml'
    # Construye la ruta completa del XML.
    xml_path = os.path.join(XML_LABELS_DIR, xml_filename)

    # Lee la imagen desde disco.
    img = cv2.imread(img_path)
    # Verifica si la imagen se cargó correctamente.
    if img is None:
        # Muestra un aviso si no se pudo leer la imagen.
        print(f"No pude leer la imagen: {filename}")
        continue

    # Obtiene la altura y el ancho de la imagen.
    h, w, _ = img.shape

    # 3. Verificamos si tiene su archivo XML.
    if os.path.exists(xml_path):
        try:
            # Lee y parsea el archivo XML de anotaciones.
            tree = ET.parse(xml_path)
            # Obtiene la raíz del árbol XML.
            root = tree.getroot()

            # Busca todos los objetos anotados dentro del XML.
            for i, obj in enumerate(root.findall('object')):
                # En algunos datasets el nombre del objeto puede variar, por lo que se asume que todo objeto es una placa.
                # Aquí recortaremos cualquier objeto marcado, asumiendo que son placas.
                # Busca la caja delimitadora del objeto.
                bndbox = obj.find('bndbox')
                # Verifica si la caja delimitadora existe.
                if bndbox is not None:
                    # Convierte los valores de las coordenadas a enteros.
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))

                    # Asegura que las coordenadas no salgan de los límites de la imagen.
                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(w, xmax), min(h, ymax)

                    # Calcula el ancho y alto del recorte de la placa.
                    box_w_abs = xmax - xmin
                    box_h_abs = ymax - ymin

                    # --- RECORTAR PLACA ---
                    # Extrae la región de la imagen que corresponde a la placa.
                    placa_crop = img[ymin:ymax, xmin:xmax]
                    # Verifica que el recorte no esté vacío.
                    if placa_crop.size > 0:
                        # Redimensiona la placa a 128x128 píxeles.
                        placa_crop_resized = cv2.resize(placa_crop, (128, 128))
                        # Guarda la imagen recortada en la carpeta de placas.
                        cv2.imwrite(os.path.join(PLACA_DIR, f"placa_{i}_{filename}"), placa_crop_resized)
                        # Incrementa el contador de placas procesadas.
                        imagenes_procesadas += 1

                    # --- RECORTAR NO PLACA (Fondo aleatorio) ---
                    # Define el tamaño del recorte de fondo basado en el tamaño de la placa.
                    bg_size = max(int(box_w_abs), 128)  # Tamaño del recorte de fondo
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
        except Exception as e:
            # Muestra un mensaje si ocurre un error al leer el XML.
            print(f"Error leyendo el XML de {filename}: {e}")
    else:
        # Muestra un aviso si la imagen no tiene su archivo XML correspondiente.
        print(f"Imagen sin XML: {filename} (buscaba {xml_filename})")

# Muestra el total de placas extraídas al finalizar.
print(f"¡Proceso terminado! Se extrajeron {imagenes_procesadas} placas.")
# Muestra la ruta donde quedaron guardados los datos.
print(f"Tus datos están listos en la carpeta: {OUTPUT_DIR}")
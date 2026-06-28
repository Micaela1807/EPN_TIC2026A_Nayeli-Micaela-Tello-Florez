# EPN_TIC2026A_Nayeli-Micaela-Tello-Florez

Este es un repositorio donde se encuentran los scripts correspondientes a todos los escenarios de pruebas de estrés de mi Trabajo de Integración Curricular.

## 📋 Descripción del Proyecto

Repositorio que contiene los scripts en Python para la ejecución y análisis de pruebas de estrés en diferentes escenarios. Este trabajo incluye la implementación, ejecución y visualización de resultados de rendimiento en arquitecturas cliente-servidor.

## 🛠️ Requisitos

- Python 3.x
- Las dependencias específicas se encuentran documentadas en cada escenario

## 📁 Estructura del Repositorio

```
├── Escenario 1/                       # Scripts del escenario 1
├── Escenario 2/                       # Scripts del escenario 2
├── Escenario 3/                       # Scripts del escenario 3
├── Escenario 4/                       # Scripts del escenario 4
├── Escenario 5/                       # Scripts del escenario 5
├── Escenario 6/                       # Scripts del escenario 6
├── Scripts Preprocesamiento Dataset/  # Scripts para preparación de datos
├── Graficas de Resultados/            # Visualizaciones y gráficos de resultados
└── README.md                          # Este archivo
```

## 🔍 Descripción de Componentes

### 📊 Escenarios de Pruebas (Escenario 1-6)
- Cada carpeta contiene scripts para un escenario diferente de pruebas de estrés
- Incluyen componentes cliente-servidor para simular y medir el rendimiento
- Los datos se recopilan para análisis posterior

### 🔧 Scripts Preprocesamiento Dataset
- `preparar_datos.py`: Orquestación general de preparación de datos
- `preprocesar_xml.py`: Transforma archivos XML a formato procesable
- `preprocesar_yolo.py`: Convierte datos al formato YOLO para detección de objetos

### 📈 Gráficas de Resultados
- 9 visualizaciones PNG que muestran métricas de rendimiento
- **Métricas de sistema**: CPU y RAM (Cliente 1, Cliente 2, Servidor)
- **Métricas de modelo**: Accuracy, Precisión, F1-Score

## 🚀 Guía de Uso

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/Micaela1807/EPN_TIC2026A_Nayeli-Micaela-Tello-Florez.git
   ```

2. **Preparar los datos** (opcional)
   - Navega a `Scripts Preprocesamiento Dataset/`
   - Ejecuta `preparar_datos.py` para procesar tu dataset

3. **Ejecutar un escenario**
   - Navega al escenario que desees ejecutar (Escenario 1-6)
   - Ejecuta los scripts correspondientes según las instrucciones de cada carpeta

4. **Revisar resultados**
   - Los gráficos y análisis se encuentran en `Graficas de Resultados/`

## 👤 Autor

Nayeli Micaela Tello Florez

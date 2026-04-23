# P1 - Smartports Visual Classification
# Enrique Pardo García

# Correo - enrique.pardo.garcia@udc.es

## Datos necesarios (no incluidos en el ZIP)

Las imágenes y los CSV **no están incluidos** en el ZIP de entrega para reducir su
tamaño. Antes de ejecutar cualquier script, coloca los archivos del material de la
práctica en la siguiente estructura dentro de `P1-Material/`:

```
P1-Material/
├── images/          ← carpeta con todas las imágenes (.jpg)
├── ship.csv         ← etiquetas para detección de barco
└── docked.csv       ← etiquetas para detección de atracado
```

---

## Estructura del proyecto

```
P1_PardoGarcia_Enrique/
├── P1-Material/
│   ├── images/          ← AÑADIR MANUALMENTE
│   ├── ship.csv         ← AÑADIR MANUALMENTE
│   └── docked.csv       ← AÑADIR MANUALMENTE
├── src/
│   ├── dataset.py       # PortDataset, transforms, dataloaders
│   ├── models.py        # SmallCNN (scratch) + ResNet18 (pretrained)
│   ├── trainer.py       # Bucle de entrenamiento
│   ├── evaluation.py    # Métricas y visualizaciones
│   ├── logger.py        # Logger en tiempo real a .txt
│   └── run_manager.py   # Gestión de carpetas por timestamp
├── train_ship.py        # Tarea 2: Ship/No-ship (4 configs)
├── train_docked.py      # Tarea 4: Docked/Undocked (6 configs)
├── test.py              # Evaluación sobre datos de test
├── visualize.py         # Visualizaciones a la carta
├── runs/                # Una carpeta por ejecución
│   └── ship_20260402_185035/
│       ├── pretrained_aug/
│       │   └── aug10_val10/
│       │       ├── models/      ← pesos .pt del mejor modelo de cada run
│       │       ├── results/     ← gráficas PNG, JSONs de historial y métricas
│       │       └── logs/        ← log actualizado en tiempo real
│       ├── pretrained_noaug/
│       ├── scratch_aug/
│       └── scratch_noaug/
└── requirements.txt
```

## Instalación

```bash
conda create -n aida python=3.10 -y
conda activate aida
pip install -r requirements.txt
```

## Entrenamiento

```bash
python train_ship.py      # Tarea 2
python train_docked.py    # Tarea 4 (ejecutar después de train_ship.py)
```

Cada ejecución crea una carpeta en runs/ con timestamp. Dentro se generan subcarpetas por combinación de configuración (tipo de modelo, aug_level, val_split), cada una con:
- models/            — pesos .pt del mejor modelo de cada run
- results/           — gráficas PNG, JSONs de historial y métricas
- logs/training.txt  — log actualizado en tiempo real epoch a época

## test.py

Evalúa un modelo entrenado sobre un CSV de test externo y guarda métricas y visualizaciones.

### Argumentos

| Argumento | Obligatorio | Valores | Descripción |
|---|---|---|---|
| `--task` | Sí | `ship`, `docked` | Tarea de clasificación |
| `--model_type` | Sí | `scratch`, `pretrained` | Arquitectura del modelo |
| `--model_path` | Sí | ruta `.pt` | Pesos del modelo entrenado |
| `--csv` | Sí | ruta `.csv` | CSV del conjunto de test |
| `--images_dir` | Sí | ruta carpeta | Carpeta con las imágenes de test |
| `--image_size` | No | entero (def. `224`) | Tamaño de imagen de entrada |
| `--batch_size` | No | entero (def. `32`) | Tamaño de batch |
| `--config_name` | No | cadena | Nombre del experimento en los resultados |
| `--results_dir` | No | ruta carpeta | Carpeta de salida (por defecto `runs/test_<task>_<timestamp>/results/`) |
| `--save_visuals` | No | flag | Genera gráficas (ROC, matriz de confusión, predicciones, errores, histograma de confianza) |

### Salida

- Por consola: `Accuracy`, `F1` y `AUC` sobre el test set
- `<config_name>_test_results.json` con las tres métricas
- Si se usa `--save_visuals`: gráficas PNG en la carpeta de resultados

### Ejemplos

> Los paths de modelo son ilustrativos. La config ganadora se determina comparando
> la media y std de AUC/accuracy entre runs al final del entrenamiento.

**Tarea ship — uso mínimo:**
```bash
python test.py \
  --task ship \
  --model_type pretrained \
  --model_path runs/ship_20260402_185035/pretrained_aug/aug15_val10/models/pretrained_aug15_val10_run1_best.pt \
  --csv P1-Material/ship.csv \
  --images_dir P1-Material/images
```

**Tarea ship — con visualizaciones y carpeta de salida personalizada:**
```bash
python test.py \
  --task ship \
  --model_type pretrained \
  --model_path runs/ship_20260402_185035/pretrained_aug/aug15_val10/models/pretrained_aug15_val10_run1_best.pt \
  --csv P1-Material/ship.csv \
  --images_dir P1-Material/images \
  --results_dir resultados_defensa/ship \
  --config_name ship_mejor \
  --save_visuals
```

**Tarea docked — con visualizaciones:**
```bash
python test.py \
  --task docked \
  --model_type pretrained \
  --model_path runs/docked_20260403_095434/pretrained_aug/aug10_val10/models/docked_aug10_val10_run1_best.pt \
  --csv P1-Material/docked.csv \
  --images_dir P1-Material/images \
  --save_visuals
```

**Modelo scratch (sin preentrenamiento):**
```bash
python test.py \
  --task ship \
  --model_type scratch \
  --model_path runs/ship_20260402_185035/scratch_noaug/aug00_val20/models/scratch_noaug_val20_run1_best.pt \
  --csv P1-Material/ship.csv \
  --images_dir P1-Material/images \
  --save_visuals
```

## visualize.py

Visuals disponibles: predictions, errors, roc, confusion_matrix,
                     confidence_hist, training_curves, comparison, all

**Tarea ship:**
```bash
python visualize.py \
  --task ship --visual predictions \
  --model_type pretrained \
  --model_path runs/ship_20260402_185035/pretrained_aug/aug10_val10/models/pretrained_aug10_val10_run1_best.pt \
  --csv P1-Material/ship.csv \
  --images_dir P1-Material/images

python visualize.py \
  --task ship --visual comparison \
  --results_json runs/ship_20260402_185035/pretrained_aug/aug10_val10/results/results_summary.json

python visualize.py \
  --task ship --visual training_curves \
  --history_json runs/ship_20260402_185035/pretrained_aug/aug10_val10/results/pretrained_aug10_val10_run1_history.json
```

**Tarea docked:**
```bash
python visualize.py \
  --task docked --visual predictions \
  --model_type pretrained \
  --model_path runs/docked_20260403_095434/pretrained_aug/aug10_val10/models/docked_aug10_val10_run1_best.pt \
  --csv P1-Material/docked.csv \
  --images_dir P1-Material/images

python visualize.py \
  --task docked --visual comparison \
  --results_json runs/docked_20260403_095434/pretrained_aug/aug10_val10/results/results_summary.json

python visualize.py \
  --task docked --visual training_curves \
  --history_json runs/docked_20260403_095434/pretrained_aug/aug10_val10/results/docked_aug10_val10_run1_history.json
```

---

## Resultados — Mejores configuraciones

Métricas calculadas como media ± std sobre 3 runs independientes en el conjunto de validación.

### Tarea 2 — Ship/No-ship

| Criterio | Config | Accuracy | F1 | AUC |
|---|---|---|---|---|
| **Mejor accuracy** | `pretrained_noaug / val30` | **0.9773** ± 0.0093 | 0.9824 ± 0.0079 | 0.9972 ± 0.0015 |
| **Mejor F1** | `pretrained_noaug / val30` | 0.9773 ± 0.0093 | **0.9824** ± 0.0079 | 0.9972 ± 0.0015 |
| **Mejor AUC** | `pretrained_aug15 / val10` | 0.9655 ± **0.0000** | 0.9716 ± 0.0032 | **1.0000** ± 0.0000 |
| **Más estable** | `pretrained_aug15 / val10` | 0.9655 ± **0.0000** | 0.9716 ± 0.0032 | **1.0000** ± **0.0000** |

> `pretrained_aug15 / val10` es la configuración recomendada para inicializar docked:
> AUC perfecto con varianza cero en los 3 runs, lo que indica características muy robustas.

Los modelos scratch quedan ~12 puntos por debajo en accuracy y ~6 en AUC respecto a los preentrenados.

### Tarea 4 — Docked/Undocked (fine-tuning)

| Criterio | Config | Accuracy | F1 | AUC |
|---|---|---|---|---|
| **Mejor accuracy** | `pretrained_noaug / val20` | **0.8056** ± 0.0227 | 0.7366 ± 0.0454 | 0.8265 ± 0.0667 |
| **Mejor F1** | `pretrained_aug10 / val30` | 0.8000 ± 0.0148 | **0.7642** ± 0.0455 | 0.8468 ± 0.0338 |
| **Mejor AUC** | `pretrained_aug10 / val10` | 0.7778 ± 0.0786 | 0.6707 ± 0.1040 | **0.9126** ± 0.0520 |
| **Más estable** | `pretrained_aug10 / val20` | 0.7870 ± 0.0131 | 0.7071 ± 0.0186 | 0.8862 ± **0.0207** |

> La tarea docked es significativamente más difícil que ship (AUC ~0.83–0.91 vs ~1.0).
> La augmentation moderada (`aug10`) supera a la agresiva (`aug15`) en todas las métricas.

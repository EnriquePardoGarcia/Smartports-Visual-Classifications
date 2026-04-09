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
│   └── ship_20260323_101742/
│       ├── hyperparams.json
│       ├── models/
│       ├── results/
│       └── logs/training.txt
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

Cada ejecución crea una carpeta en runs/ con timestamp, que contiene:
- hyperparams.json   — toda la configuración usada
- models/            — pesos .pt del mejor modelo de cada run
- results/           — gráficas PNG, JSONs de historial y métricas
- logs/training.txt  — log actualizado en tiempo real epoch a época

## test.py

```bash
python test.py \
  --task ship \
  --model_type pretrained \
  --model_path runs/ship_20260323_101742/models/pretrained_aug_run1_best.pt \
  --csv P1-Material/ship.csv \
  --images_dir P1-Material/images \
  --save_visuals
```

## visualize.py

Visuals disponibles: predictions, errors, roc, confusion_matrix,
                     confidence_hist, training_curves, comparison, all

```bash
python visualize.py \
  --task ship --visual predictions \
  --model_type pretrained \
  --model_path runs/ship_20260323_101742/models/pretrained_aug_run1_best.pt \
  --csv P1-Material/ship.csv \
  --images_dir P1-Material/images

python visualize.py \
  --task ship --visual comparison \
  --results_json runs/ship_20260323_101742/results/results_summary.json

python visualize.py \
  --task ship --visual training_curves \
  --history_json runs/ship_20260323_101742/results/pretrained_aug_run1_history.json
```

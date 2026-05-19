# Plant Disease Classifier

ResNet50 transfer learning til klassifikation af plantesygdomme på tværs af 35 klasser og 7 planter.

KEA 4. semester — Machine Learning | Eksamen: juni 2026

## Model

- **Arkitektur**: ResNet50 (ImageNet, frosset) + custom head
- **Custom head**: Flatten → Dense(30, relu) → Dropout(0.2) → Dense(30, relu) → Dense(35, softmax)
- **Input**: 100×100 px RGB
- **Klasser**: 35 (Cauliflower, Corn, EggPlant, Potato, Rice, Tomato, Wheat)
- **Optimizer**: Adam (lr=0.0001) | **Epochs**: 20

## Datasæt

Roboflow — bd_plant_diseases (~25.000 træningsbilleder, kraftigt ubalanceret 1:143 ratio).  
Datasæt ikke inkluderet i repoet — ligger lokalt og på Google Drive.

## Lokal kørsel

Kræver `plant_disease_model.keras` i projektets rod (ikke inkluderet i repoet — hentes fra Google Drive).

```bash
# Første gang
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Enkelt prediktion
.venv/bin/python app/predict.py sti/til/billede.jpg

# Gradio frontend
.venv/bin/python app/app.py
```

## Struktur

```
plant-disease-classifier/
├── Plant_Disease_Classifier.ipynb
├── requirements.txt
├── README.md
├── CLAUDE.md
├── app/
│   ├── app.py
│   ├── predict.py
│   └── class_names.json
└── docs/
    └── EKSAMEN_REFLEKSIONER.md
```

## Resultater

| Metric | Værdi |
|--------|-------|
| Val accuracy (epoch 20) | 87.85% |
| Val loss (epoch 20) | 0.38 |
| Test accuracy | 87.82% |
| Test loss | 0.40 |

Val og test accuracy er næsten identiske — modellen generaliserer godt på tværs af alle 35 klasser.

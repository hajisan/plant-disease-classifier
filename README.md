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

## Struktur

```
plant-disease-classifier/
├── Plant_Disease_Classifier.ipynb
├── CLAUDE.md
├── EKSAMEN_REFLEKSIONER.md
├── README.md
└── docs/
```

## Resultater

| Metric | Værdi |
|--------|-------|
| Val accuracy (epoch 20) | 87.85% |
| Val loss (epoch 20) | 0.38 |
| Test accuracy | 87.82% |
| Test loss | 0.40 |

Val og test accuracy er næsten identiske — modellen generaliserer godt på tværs af alle 35 klasser.

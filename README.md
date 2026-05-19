# Plant Disease Classifier

ResNet50 transfer learning til klassifikation af plantesygdomme på tværs af 35 klasser og 7 planter.

KEA 4. semester — Machine Learning | Eksamen: juni 2026

## Model

- **Arkitektur**: ResNet50 (ImageNet, frosset) + custom head
- **Input**: 100×100 px RGB
- **Klasser**: 35 (Cauliflower, Corn, EggPlant, Potato, Rice, Tomato, Wheat)
- **Val accuracy**: 87% (20 epochs, Adam lr=0.0001)

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
    ├── accuracy_over_epochs.png
    ├── loss_over_epochs.png
    ├── confusion_matrix.png
    ├── model-summary_output.png
    ├── model_evaluate_output.png
    └── class_weights.md
```

## Resultater

| Metric | Værdi |
|--------|-------|
| Val accuracy (epoch 20) | 87% |
| Val loss (epoch 20) | 0.39 |
| Test accuracy | 24.5%* |

*Test accuracy er misvisende pga. index-mismatch — `Rice_Bacterial_leaf_blight` mangler i test-splittet og forskydes alle efterfølgende klasser. Se EKSAMEN_REFLEKSIONER.md.

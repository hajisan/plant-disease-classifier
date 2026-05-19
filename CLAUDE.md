# CLAUDE.md — Plant Disease Classifier (ML Eksamensprojekt)

## Projekt
ResNet50 transfer learning til klassifikation af plantesygdomme.
KEA 4. semester — Machine Learning, Jon Eikholm (jone@ek.dk)
Eksamen: 11–16. juni 2026 | Aflevering Wiseflow: senest 27. maj 2026 kl. 12.00

---

## Datasæt — bd_plant_diseases.v1i.folder

**Kilde**: Roboflow — plantdiseasesdataset/bd_plant_diseases  
**Format**: folder (klassemapper klar til `flow_from_directory`)  
**Splits**: train (~25.000), valid (~6.700), test (~3.000)  
**Klasser**: 35 klasser fordelt på 7 planter (Cauliflower, Corn, EggPlant, Potato, Rice, Tomato, Wheat)

### Kendte datasætproblemer

**Index-mismatch (løst):**
- `Rice_Bacterial_leaf_blight` fandtes ikke i test-splittet i den originale Roboflow-download
- Løsning: 3 billeder flyttet fra `valid/Rice_Bacterial_leaf_blight` til `test/Rice_Bacterial_leaf_blight`
- Alle tre splits har nu 35 klasser med alignede indices

**Kritisk ubalance (1:143 ratio):**
- Mindste klasse: `Rice_Brown_spot` — 26 billeder i train
- Største klasse: `Tomato_Yellow_Leaf_Curl_Virus` — 3.721 billeder i train
- Rice-klasserne (Brown spot, Leaf smut, Bacterial leaf blight) har 26–33 billeder — for lidt til solid læring
- Løsning: class weighting via `compute_class_weight` fra sklearn

**Omdøbte mappenavne (potentielle konflikter fjernet):**
- `Tomato_Spider_mites Two-spotted_spider_mite` → `Tomato_Spider_mites`
- `Corn_Cercospora_leaf_spot_(Gray_leaf_spot)` → `Corn_Cercospora_leaf_spot`
- Skal være konsistent på tværs af alle tre splits (train, valid, test)

---

## Arkitektur

- **Base model**: ResNet50 pretrained på ImageNet, alle lag frosset
- **Custom head**: Flatten → Dense(30, relu) → Dropout(0.2) → Dense(30, relu) → Dense(35, softmax)
- **Loss**: sparse_categorical_crossentropy
- **Optimizer**: Adam (lr=0.0001)
- **Epochs**: 20
- **Input**: 100x100 px, RGB
- **Val accuracy**: 87.85% | **Test accuracy**: 87.82%

Samme arkitektur som `RestNet_Vegetable_Scanner.ipynb` — tilpasset til 35 output-klasser.

---

## Class weighting

Skal bruges pga. ubalancen. Eksempel:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.array(sorted(training_set.class_indices.values()))
weights = compute_class_weight(class_weight='balanced', classes=classes, y=training_set.classes)
class_weight_dict = dict(zip(classes, weights))

model.fit(training_set, epochs=epochs, validation_data=valid_set, class_weight=class_weight_dict)
```

---

## Evalueringsmetrikker

- Accuracy og loss på valid_set per epoch
- `model.evaluate(test_set)` på test
- Confusion matrix — forventer at Rice-klasserne klarer sig dårligst
- Peg på hvilke klasser der forveksles og hvorfor (visuel lighed, f.eks. Early blight på tomat vs. kartoffel)

---

---

## Lokal prediktion

- `predict.py` — tager en billedsti som argument, printer klasse og confidence
- `class_names.json` — 35 klasser indekseret alfabetisk (matcher træningsdata)
- `requirements.txt` — tensorflow==2.21.0, numpy, pillow
- Kræver Python 3.12 (TensorFlow understøtter ikke 3.13+)
- Model-filen `plant_disease_model.keras` er gitignored — hentes fra Google Drive

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python predict.py billede.jpg
```

---

## Eksamensfokus

Se EKSAMEN_REFLEKSIONER.md for detaljerede svar på alle eksamensspørgsmål.

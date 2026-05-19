# Class Weights — Plant Disease Classifier

Beregnet med `compute_class_weight('balanced')` fra sklearn.
Høj vægt = få træningsbilleder. Lav vægt = mange træningsbilleder.

| Index | Klasse | Vægt | Træningsbilleder (ca.) |
|-------|--------|------|------------------------|
| 0 | Cauliflower_Bacterial_spot_rot | 5.23 | 117 |
| 1 | Cauliflower_Black_Rot | 8.74 | 70 |
| 2 | Cauliflower_Downy_Mildew | 5.51 | 111 |
| 3 | Cauliflower_Healthy | 4.00 | 153 |
| 4 | Corn_Cercospora_leaf_spot | 1.72 | 356 |
| 5 | Corn_Common_rust | 0.76 | 808 |
| 6 | Corn_Northern_Leaf_Blight | 0.86 | 714 |
| 7 | Corn_healthy | 0.76 | 809 |
| 8 | EggPlant_Healthy_Leaf | 4.56 | 134 |
| 9 | EggPlant_Insect_Pest_Disease | 4.63 | 132 |
| 10 | EggPlant_Leaf_Spot_Disease | 4.31 | 142 |
| 11 | EggPlant_Mosaic_Virus_Disease | 4.34 | 141 |
| 12 | EggPlant_Small_Leaf_Disease | 4.31 | 142 |
| 13 | EggPlant_White_Mold_Disease | 4.53 | 135 |
| 14 | EggPlant_Wilt_Disease | 4.05 | 151 |
| 15 | Potato_Early_blight | 0.86 | 709 |
| 16 | Potato_Late_blight | 0.90 | 682 |
| 17 | Potato_healthy | 6.00 | 102 |
| 18 | Rice_Bacterial_leaf_blight | **18.54** | 33 |
| 19 | Rice_Brown_spot | **23.53** | 26 |
| 20 | Rice_Healthy | 1.66 | 369 |
| 21 | Rice_Leaf_smut | **20.39** | 30 |
| 22 | Tomato_Bacterial_spot | 0.42 | 1472 |
| 23 | Tomato_Early_blight | 0.85 | 716 |
| 24 | Tomato_Late_blight | 0.46 | 1335 |
| 25 | Tomato_Leaf_Mold | 0.93 | 658 |
| 26 | Tomato_Septoria_leaf_spot | 0.50 | 1220 |
| 27 | Tomato_Spider_mites | 0.52 | 1185 |
| 28 | Tomato_Target_Spot | 0.62 | 979 |
| 29 | Tomato_Yellow_Leaf_Curl_Virus | **0.16** | 3721 |
| 30 | Tomato_healthy | 0.54 | 1131 |
| 31 | Tomato_mosaic_virus | 2.36 | 259 |
| 32 | Wheat_Brown_Rust | 0.78 | 781 |
| 33 | Wheat_Healthy | 0.57 | 1071 |
| 34 | Wheat_Yellow_Rust | 0.75 | 814 |

## Nøgleobservationer

**Højeste vægte (mindst data):**
- Rice_Brown_spot: **23.5×** — kun 26 billeder
- Rice_Leaf_smut: **20.4×** — kun 30 billeder
- Rice_Bacterial_leaf_blight: **18.5×** — kun 33 billeder

**Laveste vægt (mest data):**
- Tomato_Yellow_Leaf_Curl_Virus: **0.16×** — 3.721 billeder

**Ratio**: 23.5 / 0.16 = **~147×** forskel mellem højeste og laveste vægt.

En fejl på Rice_Brown_spot vægter altså 147 gange mere end en fejl på Tomato_Yellow_Leaf_Curl_Virus under backpropagation.

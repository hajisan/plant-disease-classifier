# Eksamenrefleksioner — Plant Disease Classifier

## Hvorfor 87% val accuracy — og hvad begrænser modellen?

87% val accuracy på 35 ubalancerede klasser er et stærkt resultat. Til eksamen skal du kunne forklare både hvad der driver resultatet og hvad der begrænser det yderligere.

**1. Kritisk ubalance (1:143 ratio)**
- Mindste klasse: `Rice_Brown_spot` — 26 billeder i train
- Største klasse: `Tomato_Yellow_Leaf_Curl_Virus` — 3.721 billeder i train
- Modellen ser Rice-billeder ekstremt sjældent per epoch — selv med class weighting er 26 billeder for lidt til solid læring
- Sammenlign med Vegetable Scanner: 15 klasser, alle velbалancerede med ~1.000 billeder per klasse → 95% accuracy

**2. 35 klasser er komplekst**
- Vegetable Scanner: 15 klasser → 95% accuracy
- Her: 35 klasser → markant sværere klassifikationsopgave

**3. Visuel lighed mellem klasser**
- `Tomato_Early_blight` og `Potato_Early_blight` ligner hinanden meget — samme sygdom, forskellig plante
- `Tomato_Late_blight` og `Potato_Late_blight` samme problem
- Modellen ser kun billedet, ikke plantetype — det skaber forvirring

**4. Rice_Bacterial_leaf_blight og index-mismatch på test**
- Klassen fandtes ikke i test-splittet i den originale Roboflow-download
- Keras nummererer klasser alfabetisk — når én klasse mangler i test, forskydes alle efterfølgende klasser med 1 index
- Fra klasse 18 og frem evalueres modellen mod forkerte labels (Rice_Brown_spot får index 18 i test, men 19 i træning osv.)
- 17 ud af 35 klasser er systematisk misalignede — det forklarer test accuracy på 24.5% vs val accuracy på 87% i første evaluering

**Løsning — 3 billeder flyttet fra valid til test:**
- 3 billeder fra `valid/Rice_Bacterial_leaf_blight` (7 billeder) blev flyttet til `test/Rice_Bacterial_leaf_blight`
- Test har nu 35 klasser alignet med train og valid
- Valid beholder 4 billeder — stadig brugbart til validering
- Modellen retrænes på det opdaterede datasæt — test accuracy afspejler nu den reelle ydeevne

---

## Hvorfor transfer learning med ResNet50?

- ResNet50 er trænet på ImageNet (1.2 millioner billeder, 1.000 klasser)
- De frosne lag har allerede lært at genkende kanter, teksturer, former — lavniveau features der er relevante for plantebilleder
- Alternativet (træne fra bunden) ville kræve langt flere billeder og meget mere beregningstid
- Vi tilpasser kun den custom head til vores 35 klasser

## Hvorfor fryse ResNet-lagene?

- Vores datasæt er for lille til at retræne 25 millioner parametre uden massiv overfitting
- De ImageNet-features der allerede er lært er generelle nok til at være nyttige for plantebilleder
- Hurtigere træning — kun custom head opdateres

## Hvad gør class weighting?

- Under backpropagation multipliceres loss for hver observation med klassens vægt
- En fejl på `Rice_Brown_spot` (26 billeder) vægter ~50x mere end en fejl på `Tomato_Yellow_Leaf_Curl_Virus`
- Forhindrer at modellen bare lærer at gætte de store klasser hele tiden
- Bruges via `class_weight=class_weight_dict` i `model.fit`

## Hvorfor sparse_categorical_crossentropy frem for categorical_crossentropy?

- `categorical_crossentropy` kræver one-hot encoded labels: `[0, 0, 1, 0, ...]`
- `sparse_categorical_crossentropy` arbejder direkte med heltalsindeks: `2`
- Med 35 klasser og ~25.000 billeder sparer sparse-versionen markant hukommelse
- Derfor også `class_mode='sparse'` i `flow_from_directory`

## Hvad beskytter Dropout(0.2) mod?

- Overfitting — under træning slukkes 20% af neuroner tilfældigt per batch
- Tvinger netværket til ikke at blive afhængigt af specifikke neuroner
- Gør modellen mere robust og generaliserbar

## Hvad fortæller confusion matrixen?

- Diagonalen (mørk blå) = korrekte forudsigelser
- Off-diagonal felter = fejlklassifikationer
- Forventning: Rice-klasserne har lyse rækker (få korrekte)
- Forventning: Tomat/Kartoffel Early+Late blight forveksles med hinanden
- Forventning: Tomato_Yellow_Leaf_Curl_Virus klarer sig godt (mange træningsbilleder)

## Første kørsel og hvad vi lærte af den

**Første kørsel: 10 epochs, standard Adam (lr=0.001)**

Resultaterne var ustabile og utilfredsstillende:
- Train accuracy *faldt* over tid (0.25 → 0.22) — unormalt
- Val accuracy hoppede vildt (0.30 → 0.43 → 0.29) — ingen konvergens
- Val loss faldt stadig ved epoch 10 — modellen var ikke færdig med at lære

**Diagnose — to årsager:**

1. **Learning rate for høj**: Standard Adam lr=0.001 er aggressiv når class weights er ekstreme (op til 24×). De store gradienter fra Rice-klasserne fik optimizeren til at overshoot og destabilisere træningen.

2. **For få epochs**: Val loss faldt stadig ved epoch 10 — modellen havde ikke konvergeret og havde brug for mere tid.

**Anden kørsel: 20 epochs, Adam lr=0.0001**

- Learning rate sænket 10× for at give mere stabil træning med de aggressive class weights
- Epochs fordoblet til 20 for at give modellen tid til at konvergere
- Samme arkitektur og class weighting — kun optimizerens hastighed ændret
- Val accuracy 87% — men test accuracy kun 24.5% pga. index-mismatch (se punkt 4 ovenfor)

**Tredje kørsel (endelig): 20 epochs, Adam lr=0.0001, rettet datasæt**

- `Rice_Bacterial_leaf_blight` tilføjet til test — alle tre splits har nu 35 klasser
- Samme hyperparametre som anden kørsel
- Resultater:
  - Epoch 1: val_accuracy 0.53 → Epoch 10: 0.83 → Epoch 20: **0.8785**
  - Bedste val accuracy: epoch 18 — **0.8859**
  - Train accuracy (0.83) < val accuracy (0.88) — som forventet pga. augmentering
  - Val loss faldt konsistent fra 1.70 → 0.38 — ingen overfitting
  - **Test accuracy: 87.82%** — næsten identisk med val accuracy

**Til eksamen**: Dette er et godt eksempel på hyperparameter-tuning og datasætkvalitet i praksis — man observerer ustabil træning, identificerer årsagen (for høj lr + aggressive weights + manglende klasse i test) og løser begge problemer systematisk. 87.82% test accuracy på 35 ubalancerede klasser er et stærkt resultat der kan forsvares fagligt.

---

## Hvorfor augmentering på train men ikke valid/test?

- Augmentering (rotation, zoom, flip, shear) simulerer variation i den virkelige verden — en syg plante kan fotograferes fra mange vinkler og afstande
- Gør modellen mere robust overfor billeder den ikke har set før
- Valid og test må **aldrig** augmenteres — de skal repræsentere virkeligheden præcist, ellers giver evalueringen et forkert billede af modellens faktiske ydeevne

## Hvorfor er train_accuracy lavere end val_accuracy?

- Augmenteringen gør træningsbillederne sværere at klassificere — modellen ser forvrængede, roterede og zoomede versioner
- Valid_set har ingen augmentering, så billederne er nemmere
- Det er normalt og et **godt tegn** — det betyder ikke underfitting
- Hvis train_accuracy var meget højere end val_accuracy ville det derimod være et tegn på overfitting

## Hvorfor shuffle=False på valid og test?

- Confusion matrixen kræver at `test_set.classes` (sande labels) og `model.predict(test_set)` (forudsigelser) er i **nøjagtig samme rækkefølge**
- Med `shuffle=True` ville rækkefølgen være tilfældig og forskellig for hver kørsel — labels og forudsigelser ville ikke matche hinanden
- `shuffle=False` garanterer konsistent rækkefølge så confusion matrixen bliver korrekt

## Parallel til AI Agenter projektet

- PlotPlanner RAG (hajisan/plotplanner-rag): rådgiver om *hvad der skal plantes og ved siden af hvad*
- Dette projekt: detekterer *om planten er syg*
- Samme domæne (markplanlægning), forskellig ML-teknik og problemtype
- Eksempel på tværfaglig anvendelse af ML i landbrugsdomænet

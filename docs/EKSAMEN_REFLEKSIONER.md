# Eksamenrefleksioner — Plant Disease Classifier

## Hvorfor 87% val accuracy — og hvad begrænser modellen?

87% val accuracy på 35 ubalancerede klasser er et stærkt resultat. Til eksamen skal du kunne forklare både hvad der driver resultatet og hvad der begrænser det yderligere.

**1. Kritisk ubalance (1:143 ratio)**
- Mindste klasse: `Rice_Brown_spot` — 26 billeder i train
- Største klasse: `Tomato_Yellow_Leaf_Curl_Virus` — 3.721 billeder i train
- Modellen ser Rice-billeder ekstremt sjældent per epoch — selv med class weighting er 26 billeder for lidt til solid læring
- Sammenlign med Vegetable Scanner: 15 klasser, alle velbalancerede med ~1.000 billeder per klasse → 95% accuracy

**Hvorfor er datasættet ubalanceret?**
Ubalancen afspejler virkeligheden: nogle plantesygdomme er sjældne eller svære at dokumentere i felten, mens andre er udbredte og veldokumenterede. `Tomato_Yellow_Leaf_Curl_Virus` er en af de mest udbredte tomatvirusser globalt — der findes tusindvis af billeder online. Rice-sygdommene derimod forekommer mere sæsonbetinget og geografisk afgrænset, og kræver at man fysisk er på markerne på det rigtige tidspunkt. Datasættet er indsamlet fra Roboflow og afspejler hvad der var tilgængeligt — ikke hvad der ville være ideelt for en ML-model.

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

## Hvad er en CNN — og hvorfor bruger vi en til dette projekt?

En Convolutional Neural Network (CNN) er en type DNN designet til at arbejde med billeddata. I stedet for at forbinde alle pixels direkte til alle neuroner (som en standard DNN ville), bruger en CNN **conv-lag** der scanner billedet med et lille filter og lærer lokale mønstre.

- **Conv-lag**: Et filter (f.eks. 3×3 pixels) glider hen over billedet og aktiverer når det genkender et bestemt mønster (kant, tekstur, farveovergang)
- **Max Pooling**: Reducerer billedets dimensioner ved at beholde det maksimale signal i hvert område — gør modellen mere robust overfor forskydninger og reducerer beregning
- **Hierarkisk læring**: Tidlige lag lærer kanter og farver, senere lag kombinerer dem til former og til sidst sygdomsmønstre

ResNet50 er en CNN med 50 lag og residual connections (skip connections) der løser problemet med vanishing gradients i meget dybe netværk. Den er pretrænet på ImageNet og kan genkende komplekse visuelle features — præcis hvad vi har brug for til plantesygdomme.

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

---

## Grundlæggende DNN-spørgsmål (kurrikulumkrav)

### Hvad er en Deep Neural Network model?

En DNN er et netværk af lag med neuroner der er forbundet med vægte. Data strømmer fra input-laget gennem ét eller flere skjulte lag til output-laget. "Deep" refererer til at der er flere skjulte lag.

- **Input**: Rå data — her 100×100 px RGB-billeder flattened til en vektor af pixelværdier som ResNet50 modtager
- **Output**: En sandsynlighedsfordeling over 35 klasser via softmax — den klasse med højest sandsynlighed er modellens forudsigelse
- **Vægte (weights)**: Talværdier på forbindelserne mellem neuroner — det er vægtene der justeres under træning så modellen lærer. ResNet50 har ~25 millioner vægte, men de er frosset — kun custom head's ~3.76 MB trænes
- **Bias**: En ekstra talværdi per neuron der lægges til summen inden aktivering — giver modellen mulighed for at forskyde aktiveringen uafhængigt af input. Uden bias kan netværket kun lære funktioner der går igennem origo
- **Hidden layer**: Lag mellem input og output — her Dense(30, relu) × 2. Disse lag lærer abstrakte repræsentationer af data (kanter → teksturer → former → sygdomsmønstre)

### Hvad er forward propagation?

Data bevæger sig fremad igennem netværket lag for lag. For hver neuron: tag inputs, multiplicer med vægte, læg bias til, send resultatet gennem aktiveringsfunktionen. Gentag til output-laget. Resultatet er en forudsigelse.

### Hvad er summerings-funktion (Sigma)?

$$z = \sum (x_i \cdot w_i) + b$$

For hver neuron summeres alle inputs (x) ganget med deres tilhørende vægte (w), plus bias (b). Dette tal sendes videre til aktiveringsfunktionen.

### Hvad er aktiveringsfunktion — og findes der forskellige?

Aktiveringsfunktionen bestemmer om og hvor meget en neuron "fyrer". Uden aktivering ville et netværk med mange lag kun kunne lære lineære sammenhænge.

- **ReLU** (`relu`): `f(x) = max(0, x)` — bruges i de skjulte lag. Hurtig, simpel, undgår vanishing gradient
- **Softmax**: Konverterer output-lagets rå tal til sandsynligheder der summer til 1 — bruges til multi-class klassifikation (vores 35 klasser)
- Andre: Sigmoid (binær klassifikation), Tanh, Leaky ReLU

### Hvad er target og error?

- **Target**: Den korrekte label for et træningsbillede — f.eks. klasse 7 (`Corn_Common_rust`)
- **Error (loss)**: Forskellen mellem modellens forudsigelse og target — måles med `sparse_categorical_crossentropy`. Jo lavere loss, jo tættere er modellen på de korrekte svar. Val loss faldt fra 1.70 → 0.38 over 20 epochs

### Hvad er back propagation?

Når forward propagation har givet en forudsigelse, beregnes fejlen (loss). Back propagation sender fejlsignalet baglæns gennem netværket og beregner hvor meget hver vægt bidrog til fejlen — dette bruges til at opdatere vægtene.

- **Gradient**: Den afledede af loss med hensyn til en vægt — angiver retning og størrelse af den ændring der vil reducere fejlen mest. Stor gradient = vægt har stor indflydelse på fejlen
- **Learning rate**: Hvor store skridt vi tager i retning af gradienten. Her `lr=0.0001` — vi sænkede det fra standard `0.001` fordi de aggressive class weights (op til ~50×) gav ustabil træning med høj lr
- **Momentum**: En mekanisme der husker tidligere gradienter og "ruller videre" i samme retning — undgår at optimizeren hopper rundt og hjælper med at komme ud af lokale minima. Adam-optimizeren inkluderer momentum automatisk (standardværdi β₁=0.9)
- **Ny vægt**: `w_ny = w_gammel - learning_rate × gradient` — subtraktion fordi vi vil minimere loss (bevæge os ned ad fejlfladen)

---

## Hvordan ville vi forbedre modellen yderligere?

**Større input (224×224 i stedet for 100×100)**
ResNet50 er designet til 224×224 px — vi nedskalerede til 100×100 for at spare beregningstid. Med native inputstørrelse ville modellen bevare flere detaljer i billederne, særligt subtile sygdomsmønstre der forsvinder ved nedskalering. Det ville sandsynligvis give en mærkbar forbedring i accuracy.

**Unfreeze de øverste ResNet-lag (fine-tuning)**
I stedet for at fryse hele ResNet50 kunne man optøe de sidste 10–20 lag og træne dem med en meget lav learning rate (f.eks. 1e-5). De øverste lag er de mest domænespecifikke — ved at tilpasse dem til plantebilleder ville modellen lære mere relevante features end de generelle ImageNet-features.

**Mere data til de mindste klasser**
Rice-klasserne med 26–33 billeder er for lidt til solid læring uanset class weighting. Løsningen ville være at indsamle eller syntetisere flere billeder — f.eks. via aggressiv augmentering kun på disse klasser, eller ved at finde supplerende datasæt.

---

## Hvordan kan data og modeloutput repræsenteres bedre?

**Classification report (per-klasse metrics)**
`model.evaluate` giver kun samlet accuracy. Med sklearns `classification_report` får man precision, recall og F1-score per klasse — langt mere informativt når datasættet er ubalanceret. En Rice-klasse med 26 billeder kan have 0% recall selvom samlet accuracy er 87%.

**Grad-CAM (Gradient-weighted Class Activation Mapping)**
En teknik der visualiserer hvilke dele af billedet modellen kigger på når den laver en forudsigelse — producerer et heatmap oven på billedet. Gør modellens beslutning transparent og er stærkt visuelt til en demo. Viser f.eks. at modellen fokuserer på bladpletten og ikke baggrunden.

**Confidence-fordeling**
I stedet for bare at vise top-1 forudsigelse kan man plotte sandsynlighedsfordelingen over alle 35 klasser for et givet billede — Gradio-appen viser allerede top-5. Det afslører om modellen er sikker (én klasse dominerer) eller usikker (sandsynlighederne er spredte).

---

## Parallel til AI Agenter projektet

- PlotPlanner RAG (hajisan/plotplanner-rag): rådgiver om *hvad der skal plantes og ved siden af hvad*
- Dette projekt: detekterer *om planten er syg*
- Samme domæne (markplanlægning), forskellig ML-teknik og problemtype
- Eksempel på tværfaglig anvendelse af ML i landbrugsdomænet

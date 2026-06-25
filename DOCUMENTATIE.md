# Detecție Păsări vs Drone — Documentație Tehnică

**Proiect de Licență**
Un studiu comparativ între detectoare de obiecte bazate pe învățare automată
clasică și pe învățare profundă (deep learning), pentru discriminarea binară între
**păsări** și **drone** în imagini aeriene/terestre.

---

## Cuprins
1. [Obiectiv și Domeniu de Aplicare](#1-obiectiv-și-domeniu-de-aplicare)
2. [Arhitectura Sistemului](#2-arhitectura-sistemului)
3. [Setul de Date](#3-setul-de-date)
4. [Metodologie](#4-metodologie)
5. [Hiperparametri și Configurația de Antrenare](#5-hiperparametri-și-configurația-de-antrenare)
6. [Configurația Experimentală](#6-configurația-experimentală)
7. [Rezultate](#7-rezultate)
8. [Concluzii și Analiză](#8-concluzii-și-analiză)
9. [Limitări](#9-limitări)
10. [Direcții Viitoare](#10-direcții-viitoare)
11. [Reproductibilitate](#11-reproductibilitate)
12. [Concluzie](#12-concluzie)

---

## 1. Obiectiv și Domeniu de Aplicare

Scopul acestui proiect este detectarea și clasificarea a două clase similare
vizual, dar distincte semantic — **păsări** și **drone** — și **compararea a trei
metodologii de detecție** ce acoperă două paradigme:

| Paradigmă | Metodă | Familie |
|-----------|--------|---------|
| ML clasic | **HOG + MLP** | Trăsături manuale + clasificator neuronal |
| ML clasic | **HOG + SVM** | Trăsături manuale + clasificator cu nucleu (kernel) |
| Învățare profundă | **YOLOv8n** | Detector CNN într-o singură etapă (bazat pe ancore, NMS) |
| Învățare profundă | **YOLOv10n** | Detector CNN într-o singură etapă (fără NMS, cap dual) |

Produsul livrabil este o aplicație interactivă **Streamlit** care permite:
- antrenarea tuturor modelelor printr-o singură acțiune („Train All Models"),
- rularea **simultană a tuturor celor patru detectoare** pe o imagine încărcată
  (test live),
- vizualizarea unei **comparații de evaluare alăturate** pe setul de test reținut.

Problema de discriminare este motivată de aplicații reale de tip counter-UAS
(Unmanned Aerial System — sisteme aeriene fără pilot), unde un sistem de percepție
trebuie să evite alarmele false declanșate de păsări, detectând în același timp în
mod fiabil dronele.

---

## 2. Arhitectura Sistemului

```
LiceentaBogdan/
├── app.py                       # UI Streamlit: Train All | Results | Live Detection
├── src/core/
│   ├── data_loader.py           # Parsare etichete YOLO → vectori de patch-uri (pipeline HOG)
│   ├── feature_extractor.py     # Calculul descriptorului HOG
│   ├── model.py                 # BirdDroneModel: wrapper MLP/SVM (3 clase)
│   ├── detector.py              # Detector cu fereastră glisantă multi-scală (HOG)
│   ├── yolov8_detector.py       # Wrapper de inferență YOLOv8 (Ultralytics)
│   └── yolov10_detector.py      # Wrapper de inferență YOLOv10 (Ultralytics)
├── src/utils/metrics.py         # Metrici de clasificare + IoU/AP
├── scripts/
│   ├── relabel_by_filename.py   # Corecția etichetelor corupte (B→pasăre, D→dronă)
│   ├── train_hog.py             # Antrenare CLI pentru HOG+MLP și HOG+SVM
│   ├── train_yolov8.py          # Punct de intrare antrenare YOLOv8
│   ├── train_yolov10.py         # Punct de intrare antrenare YOLOv10
│   ├── evaluate_all.py          # Evaluează toate cele 4 modele → results/all_metrics.json
│   └── merge_datasets.py        # Construiește setul de date echilibrat (merged)
├── datasets/merged/data.yaml    # Configurație set de date YOLO (nc=2; 0=dronă, 1=pasăre)
├── models/                      # hog_mlp.pkl, hog_svm.pkl (modele clasice salvate)
├── runs/train/                  # Rulări de antrenare YOLO (ponderi + results.csv)
└── results/all_metrics.json     # Comparația pe 4 căi (persistată)
```

Două pipeline-uri paralele folosesc același set de date:

- **Pipeline-ul clasic** (`data_loader → feature_extractor → model → detector`)
  operează pe **patch-uri de imagine**: obiectele sunt decupate din adnotări,
  redimensionate la 64×64 și descrise cu HOG. Inferența folosește o **fereastră
  glisantă** multi-scală.
- **Pipeline-ul profund** (`yolov8_detector` / `yolov10_detector` prin Ultralytics)
  operează **end-to-end** pe imagini complete, prezicând direct casetele și clasele.

Ambele wrapper-e de detector returnează un tuplu identic de 6 elemente
`(x, y, w, h, încredere, etichetă)`, astfel încât UI-ul le poate trata
interschimbabil.

---

## 3. Setul de Date

### 3.1 Sursă și Structură

Setul de date respectă convenția YOLO: fiecare imagine are un fișier de etichetă
`.txt` corespunzător, cu o adnotare pe linie, `id_clasă  x_centru  y_centru
lățime  înălțime` (normalizat). Sunt prezente atât casete delimitatoare standard,
cât și adnotări poligonale.

Imaginile respectă o convenție strictă de denumire care codifică clasa reală:

| Prefix | Semnificație | Subset |
|--------|--------------|--------|
| `BT*`  | **B**ird (pasăre), **T**rain/Test | train, test |
| `BV*`  | **B**ird (pasăre), **V**alidation | valid |
| `DT*`  | **D**rone (dronă), **T**rain/Test | train, test |
| `DV*`  | **D**rone (dronă), **V**alidation | valid |

Sunt utilizate două rădăcini de set de date:
- `Data/` — subsetul brut (`train/`, `valid/`, `test/`) folosit de pipeline-ul
  clasic (HOG).
- `datasets/merged/` — o îmbinare echilibrată pe clase (`train/`, `val/`, `test/`)
  folosită pentru antrenarea YOLO, declarată în `datasets/merged/data.yaml`
  (`nc: 2`, `names: {0: drone, 1: bird}`).

### 3.2 Constatare Critică — Coruperea Etichetelor (cauza-rădăcină)

O constatare centrală a acestei lucrări este că **identificatorii de clasă din
adnotările setului de date au fost corupți sistematic**. O tabulare încrucișată a
*prefixului numelui de fișier* (clasa reală) versus *id-ul de clasă adnotat* a
relevat:

| Subset | Imagini pasăre (`B*`) etichetate clasa 0 (dronă) | Imagini pasăre etichetate clasa 1 (pasăre) | % imagini pasăre greșit etichetate |
|--------|--------------------------------------------------|--------------------------------------------|------------------------------------|
| train | 6 119 | 73 | **98,8 %** |
| val   | 1 142 | 12 | 98,9 % |
| test  | 385   | 4  | 99,0 % |

Cu alte cuvinte, **aproape fiecare pasăre a fost adnotată ca dronă (clasa 0)**.
*Coordonatele* casetelor delimitatoare erau corecte — doar tokenul de clasă era
greșit. Acest singur defect explică fiecare mod de eșec observat înainte de
corecție:

- **HOG+MLP/SVM** au fost antrenate pe date în care clasa „Pasăre" practic nu
  exista (numărul de patch-uri „Pasăre" ≈ 0), deci nu o puteau prezice niciodată —
  fiecare pasăre era raportată ca dronă.
- Modelele **YOLO** au învățat o distribuție aproape degenerată, cu o singură
  clasă („totul este o dronă"), producând detecții lipsite de sens.

Aceasta este o ilustrare clasică a faptului că **calitatea datelor domină calitatea
modelului**: nicio alegere de arhitectură sau reglare de hiperparametri nu poate
compensa etichetele corupte.

### 3.3 Metodologia de Re-etichetare

Corecția (`scripts/relabel_by_filename.py`) rescrie **tokenul de clasă al fiecărei
linii de adnotare** folosind prefixul fiabil al numelui de fișier:

```
numele fișierului începe cu 'B'  →  clasa 1 (pasăre)
numele fișierului începe cu 'D'  →  clasa 0 (dronă)
```

Proprietăți ale procedurii:
- **Păstrează coordonatele** — se modifică doar primul token pe linie; casetele
  delimitatoare și poligoanele rămân neatinse.
- **Idempotentă** — rularea repetată produce același rezultat.
- **Reversibilă** — toate etichetele originale sunt arhivate în `labels_backup.zip`
  (34,9 MB) înainte de orice scriere.
- Aplicată **ambelor** seturi `Data/` și `datasets/merged/` (41 858 fișiere,
  74 652 linii rescrise).

După re-etichetare, toate cache-urile de patch-uri HOG (`*/.cache`) și cache-urile
de etichete YOLO (`*/labels.cache`) au fost invalidate și reconstruite, astfel
încât etichetele corecte să se propage.

### 3.4 Distribuția Finală a Claselor (după re-etichetare)

**Numărul de imagini (setul de date merged):**

| Subset | Imagini pasăre | Imagini dronă | Total |
|--------|----------------|---------------|-------|
| train | 6 759 | 10 002 | 16 761 |
| val   | 1 265 | 1 877  | 3 142  |
| test  | 427   | 622    | 1 049  |

**Distribuția patch-urilor folosite pentru antrenarea/testarea clasificatoarelor
HOG** (3 clase: Fundal / Pasăre / Dronă), cu `max_train=1500`, `max_test=400`
imagini eșantionate:

| Subset | Fundal | Pasăre | Dronă |
|--------|--------|--------|-------|
| train  | 3 861 | 1 574 | 2 286 |
| test   | 409   | 172   | 235   |

> Pipeline-ul clasic este o problemă cu **3 clase** (trebuie să respingă în plus
> „Fundalul" ferestrelor glisante goale), în timp ce YOLO este o problemă cu
> **2 clase** (localizează direct obiectele și nu are o clasă explicită de fundal).

---

## 4. Metodologie

### 4.1 Extracția Trăsăturilor HOG (`feature_extractor.py`)

Fiecare patch este redimensionat la o fereastră fixă de **64×64** și descris cu
descriptorul **Histogram of Oriented Gradients** (Histograma Gradienților
Orientați):

| Parametru HOG | Valoare |
|---------------|---------|
| Dimensiune fereastră | 64 × 64 px |
| Orientări | 9 |
| Pixeli per celulă | 8 × 8 |
| Celule per bloc | 2 × 2 |
| Normalizare bloc | L2-Hys (implicit skimage) |
| Culoare | RGB (`channel_axis = -1`) |
| Corecție gamma | `transform_sqrt = True` |
| Lungime descriptor rezultat | **5 292** (9 orient. × 7×7 blocuri × 2×2 celule × 3 canale) |

HOG codifică structura locală a orientării muchiilor/gradienților și este robust la
variații moderate de iluminare — istoric eficient pentru detecția formelor rigide.

### 4.2 Clasificatorul Clasic (`model.py` — `BirdDroneModel`)

O etapă comună de preprocesare și echilibrare precede ambele clasificatoare:

1. **StandardScaler** — normalizare la medie zero / varianță unitară a vectorului
   HOG de 5 292 dimensiuni.
2. **PCA → 256 componente** — reducerea dimensionalității pentru separabilitatea
   claselor și viteză.
3. **SMOTE** — supraeșantionarea sintetică a minorității pentru echilibrarea
   Fundal/Pasăre/Dronă înainte de antrenare.

#### 4.2.1 HOG + MLP

Un perceptron multistrat antrenat epocă-cu-epocă prin `partial_fit`:

| Parametru | Valoare |
|-----------|---------|
| Straturi ascunse | 3 × 128 unități |
| Activare | ReLU |
| Solver | Adam |
| Rată de învățare inițială | 0,001 (adaptivă) |
| Epoci | 30–40 |
| Dimensiune lot (batch) | 32 |
| Clase de ieșire | 3 (Fundal, Pasăre, Dronă) |

#### 4.2.2 HOG + SVM

Un clasificator cu vectori suport (kernel) cu calibrare de probabilitate:

| Parametru | Valoare |
|-----------|---------|
| Nucleu (kernel) | RBF |
| C | 10,0 |
| Ponderarea claselor | `balanced` |
| Probabilitate | scalare Platt (`probability=True`) |
| Clase de ieșire | 3 (Fundal, Pasăre, Dronă) |

### 4.3 Detectorul cu Fereastră Glisantă (`detector.py`)

Deoarece modelele clasice clasifică patch-uri fixe, detecția pe o imagine completă
se realizează printr-o **fereastră glisantă multi-scală**:

| Componentă | Configurație |
|------------|--------------|
| Piramidă de imagine | factor de scală 0,8, până la 6 niveluri, latură min. 64 px |
| Dimensiuni fereastră | 64×64, 96×96, 128×128 |
| Pas (stride) | `max(4, 32 − seq_length·2)` (UI `seq_length` ∈ [1,16]) |
| Filtru de fundal | poartă pe energie HOG ce respinge ferestrele goale/cer |
| Încredere | scor obiect = `1 − P(fundal)`, cu prag (implicit UI 0,72) |
| Eliminarea duplicatelor | **Non-Maximum Suppression** pe clase, IoU = 0,3 |

### 4.4 YOLOv8n (`yolov8_detector.py`, `train_yolov8.py`)

YOLOv8 nano — un detector CNN într-o singură etapă, bazat pe ancore (Ultralytics).
Wrapper-ul încarcă `best.pt`, rulează `model.predict` și convertește rezultatele în
tuplul comun de 6 elemente. Pragul de încredere la inferență, implicit: **0,35**
(reglabil în UI). Cap cu 2 clase (`0=dronă, 1=pasăre`). ~3,0 M parametri.

### 4.5 YOLOv10n (`yolov10_detector.py`, `train_yolov10.py`)

YOLOv10 nano — un detector într-o singură etapă, mai recent, cu un design
**dual-head fără NMS** (atribuire consistentă unu-la-unu + unu-la-mulți), astfel
încât emite cel mult o casetă per obiect, fără un pas ulterior de NMS. ~2,27 M
parametri, 6,5 GFLOPs. Pragul de încredere la inferență, implicit: 0,45 (reglabil).

---

## 5. Hiperparametri și Configurația de Antrenare

### 5.1 Modele Clasice

A se vedea §4.2. Ambele modele partajează StandardScaler → PCA(256) → SMOTE;
diferă doar prin estimatorul final (MLP vs SVC).

### 5.2 Antrenarea YOLO (`hyp.drone.yaml` + valorile implicite ale scriptului)

Ambele modele YOLO au fost antrenate cu setări identice pe setul de date merged:

| Parametru | Valoare | Justificare |
|-----------|---------|-------------|
| Dimensiune imagine | 416 | Intrare mai mică → antrenare fezabilă pe CPU |
| Dimensiune lot | 8 | Limitat de memorie pe CPU |
| Epoci | 10 | Limitat de timp (termen prezentare) |
| Dispozitiv | CPU | Fără GPU disponibil |
| Workers | 0 | Necesar pentru a evita blocajul DataLoader pe Windows |
| Optimizator | auto → AdamW (lr≈0,00167) | Auto-selecție Ultralytics |
| `lr0` / `lrf` | 0,01 / 0,01 | Program rată de bază/finală |
| Programator | cosinus (`cos_lr=True`) | Descreștere lină |
| `momentum` | 0,937 | — |
| `weight_decay` | 0,0005 | — |
| `warmup_epochs` | 3,0 | — |
| Ponderi pierdere box / cls / dfl | 7,5 / **0,3** / 1,5 | `cls` redus pentru a descuraja supraîncrederea |
| `label_smoothing` | 0,1 | Atenuează supraînvățarea cu încredere de 100% |
| `mixup` / `copy_paste` | 0,15 / 0,1 | Augmentare pentru a ajuta clasa minoritară (pasăre) |
| `scale` | 0,9 | Interval larg de scală (obiecte la multe distanțe) |
| `fliplr` / HSV | 0,5 / (0,015, 0,7, 0,4) | Augmentare fotometrică/geometrică standard |
| **`multi_scale`** | **False** | **Dezactivat după o eroare — vezi §8.2** |
| `save_period` | 5 | Checkpoint la fiecare 5 epoci |
| `exist_ok` | True | Director de rulare stabil (calea ponderilor nu se schimbă) |

---

## 6. Configurația Experimentală

| Componentă | Detaliu |
|------------|---------|
| Sistem de operare | Windows 11 Pro (10.0.26200) |
| Hardware | Doar CPU (fără GPU NVIDIA) |
| Mediu Python | `.venv` local |
| Stivă deep-learning | Ultralytics 8.4.54, PyTorch (build CPU) |
| Stivă clasică | scikit-learn, scikit-image, imbalanced-learn (SMOTE), OpenCV, joblib |
| UI | Streamlit |

**Timp de antrenare observat (CPU, rulări paralele):**
- YOLOv8n, 10 epoci: ≈ 13 147 s (≈ 3 h 39 m)
- YOLOv10n, 10 epoci: ≈ 23 782 s (≈ 6 h 36 m)
- HOG+MLP și HOG+SVM (combinat): câteva minute.

---

## 7. Rezultate

### 7.1 Comparație Cantitativă (set de test reținut)

Sursă: `results/all_metrics.json`, produs de `scripts/evaluate_all.py`. Modelele
clasice sunt evaluate la **nivel de patch** (clasificare cu 3 clase); modelele YOLO
sunt evaluate cu **metrici de detecție** (`model.val`, mAP@IoU 0,50).

| Model | Acuratețe | Precizie | Recall | F1 | mAP@50 |
|-------|-----------|----------|--------|------|--------|
| **HOG + MLP** | 0,806 | 0,817 | 0,806 | 0,807 | — |
| **HOG + SVM** | **0,827** | **0,833** | **0,827** | **0,826** | — |
| **YOLOv8n** | — | 0,668 | 0,512 | 0,579 | **0,601** |
| **YOLOv10n** | — | 0,694 | 0,503 | 0,583 | 0,578 |

> ⚠️ **Metricile nu sunt direct comparabile între paradigme.** Acuratețea/F1
> clasice sunt calculate pe **patch-uri de obiecte pre-decupate** (o sarcină mai
> ușoară, localizată), în timp ce mAP@50 al YOLO măsoară **detecția completă**
> (localizare *și* clasificare pe întreaga imagine). Cele două numere răspund la
> întrebări diferite; vezi §8.3.

### 7.2 Metrici de Antrenare YOLO (epoca finală, subset de validare)

Din `runs/train/*/results.csv`, epoca 10:

| Model | Precizie(B) | Recall(B) | mAP@50 | mAP@50–95 |
|-------|-------------|-----------|--------|-----------|
| YOLOv8n | 0,716 | 0,531 | 0,623 | 0,405 |
| YOLOv10n | 0,710 | 0,521 | 0,597 | 0,388 |

Ambele componente de pierdere (box/cls/dfl) au scăzut monoton de-a lungul celor 10
epoci, indicând o convergență sănătoasă; mAP era încă în creștere la epoca 10
(adică modelele sunt **sub-antrenate**, limitate de bugetul de timp CPU — vezi §9).

### 7.3 Matrici de Confuzie ale Clasificatoarelor HOG (set de test pe patch-uri)

Rândurile = adevărul de bază, coloanele = predicția; ordinea **[Fundal, Pasăre,
Dronă]**. Suport patch-uri test: 409 Fundal, 172 Pasăre, 235 Dronă.

**HOG + MLP** (acuratețe 0,806):
```
            pred Fundal  pred Pasăre  pred Dronă
adev Fundal [   307         43           59    ]
adev Pasăre [    18        136           18    ]   → recall pasăre = 136/172 = 79,1 %
adev Dronă  [    12         16          207    ]   → recall dronă  = 207/235 = 88,1 %
```

**HOG + SVM** (acuratețe 0,827):
```
            pred Fundal  pred Pasăre  pred Dronă
adev Fundal [   327         42           40    ]
adev Pasăre [    15        146           11    ]   → recall pasăre = 146/172 = 84,9 %
adev Dronă  [     9          6          220    ]   → recall dronă  = 220/235 = 93,6 %
```

**Interpretare.** Pe patch-uri de obiecte decupate strâns, ambele modele clasice
discriminează bine între păsări și drone. În mod crucial, **confuzia Pasăre→Dronă
care afecta setul de date corupt este rezolvată**: doar 18/172 (MLP) și 11/172
(SVM) păsări sunt greșit etichetate ca drone, față de ~100 % înainte de corecția de
re-etichetare. SVM este modelul clasic mai puternic pe fiecare axă.

### 7.4 Detecție Live Calitativă (imagine completă, fereastră glisantă vs end-to-end)

Două imagini complete reprezentative au fost procesate de toate cele patru
detectoare în UI.

**Imaginea A — un pănțăruș albastru (clasă reală: pasăre), fundal de frunziș:**

| Detector | Etichetă principală | Nr. casete | Verdict |
|----------|---------------------|------------|---------|
| HOG + MLP | **Pasăre 100 %** | 40 | ✅ clasă corectă, foarte zgomotos (multe casete suprapuse) |
| HOG + SVM | Dronă 99,8 % | 34 | ❌ clasă greșită, zgomotos |
| YOLOv8n | **pasăre 0,42** | 2 | ✅ corect, curat |
| YOLOv10n | dronă 0,40 | 1 | ❌ clasă greșită, dar o singură casetă curată |

**Imaginea B — un quadcopter deasupra unei plaje (clasă reală: dronă), fundal cer:**

| Detector | Etichetă principală | Nr. casete | Verdict |
|----------|---------------------|------------|---------|
| HOG + MLP | Pasăre 100 % | 14 | ❌ clasă greșită, zgomotos |
| HOG + SVM | **Dronă 99,4 %** | 9 | ✅ casetă principală corectă, zgomotos |
| YOLOv8n | **dronă 0,85** | 1 | ✅ corect, curat, încredere ridicată |
| YOLOv10n | **dronă 0,87** | 1 | ✅ corect, curat, încredere ridicată |

**Observații.**
- **YOLOv8n a fost cel mai consistent** detector pe ambele imagini (pasăre→pasăre,
  dronă→dronă, o singură casetă cu încredere ridicată). YOLOv10n l-a egalat pe
  dronă, dar a etichetat greșit pasărea (în concordanță cu mAP-ul său ușor mai
  scăzut la 10 epoci).
- **YOLO produce casete unice, curate**; detectoarele clasice emit **zeci de casete
  suprapuse** chiar și după NMS pe clase, deoarece fiecare fereastră de grilă peste
  un obiect texturat se poate declanșa.
- Modelele HOG prezintă un **decalaj de performanță patch↔imagine completă** (vezi
  §8.3): acuratețea lor pe patch este ridicată, dar pe ferestrele glisante laxe
  eticheta dominantă se poate inversa (de ex., SVM raportează cu încredere „Dronă"
  pe o imagine cu pasăre).

---

## 8. Concluzii și Analiză

### 8.1 Coruperea etichetelor a fost defectul dominant (§3.2)
Cel mai important rezultat empiric al proiectului este metodologic: un singur defect
de calitate a datelor (≈99 % dintre păsări greșit etichetate ca drone) a fost
responsabil pentru *toate* eșecurile dinainte de corecție, la *toate* cele patru
modele. Corectarea etichetelor — fără a schimba vreun model — a restabilit
detectabilitatea păsărilor peste tot. **Etichete proaste la intrare, modele proaste
la ieșire.**

### 8.2 Eroarea de antrenare `multi_scale` (o capcană reproductibilă de deep-learning)
Primele rulări de antrenare YOLO au eșuat cu:
```
ValueError: Expected more than 1 value per channel when training,
            got input size torch.Size([1, 256, 1, 1])
```
**Cauză:** cu `multi_scale=True`, Ultralytics redimensionează aleatoriu fiecare
lot; o reducere agresivă poate micșora o hartă de trăsături la **1×1** spațial.
Simultan, dimensiunea setului de antrenare (16 761) **nu este divizibilă cu
dimensiunea lotului (8)** — rămâne un ultim lot de **o** imagine
(`16761 mod 8 = 1`). BatchNorm nu poate calcula statistici dintr-o singură valoare
pe canal → generează eroarea. Eroarea este intermitentă deoarece necesită ambele
condiții (ultim lot de dimensiune 1 *și* o redimensionare nefericită la 1×1) să
coincidă.
**Corecție:** setarea `multi_scale=False` în ambele scripturi de antrenare. Aceasta
elimină complet condiția hărții de trăsături 1×1 (la 416 px, harta cea mai adâncă
este 13×13, oferind 169 de valori pe canal chiar și pentru un lot de unul) și, ca
beneficiu secundar, **accelerează antrenarea pe CPU**. După corecție, ambele modele
s-au antrenat până la final.

### 8.3 Discrepanța nivel-patch vs fereastră glisantă
Tensiunea principală din rezultate este că modelele HOG obțin **0,81–0,83**
acuratețe pe patch-uri, dar se comportă **eratic** în detecția live pe imagine
completă (§7.4). Acest lucru este de așteptat și instructiv:
- Metricile pe patch evaluează clasificatorul pe **decupaje strânse, bine
  centrate** — sarcina pentru care a fost antrenat.
- Fereastra glisantă prezintă clasificatorului mii de **ferestre arbitrare, slab
  aliniate** (obiecte parțiale, amestecuri obiect+fundal, scale multiple).
  Ferestrele din afara distribuției pot declanșa predicții greșite cu încredere, iar
  clasa care se declanșează cel mai des câștigă imaginea. Trăsăturile HOG clasice nu
  au niciun mecanism de *localizare* — clasifică doar fereastra care le este dată.
- YOLO, în schimb, este antrenat **end-to-end pentru localizare**: învață unde se
  află obiectele și emite o singură casetă calibrată. De aici ieșirea sa curată.
Acest decalaj este în sine o constatare-cheie a lucrării: **o acuratețe ridicată de
clasificare pe patch nu implică o detecție puternică** — protocolul de detecție
contează la fel de mult ca și clasificatorul.

### 8.4 Capcana ponderilor învechite (reproductibilitate inginerească)
O rulare YOLOv10 anterioară (dinainte de corecție) scrisese ponderile într-o cale
*imbricată* veche (`runs/detect/runs/train/...`) din cauza unei particularități a
parametrului `task` din Ultralytics. Atât aplicația, cât și evaluatorul preferau
acea cale, astfel încât evaluau în tăcere un **model defect de 2 epoci** (mAP@50 ≈
0,09) în loc de cel proaspăt antrenat. Eliminarea directorului învechit și
inversarea priorității căii către locația curată `runs/train/drone_v10_cpu/...` a
ridicat mAP@50 măsurat pentru YOLOv10 de la **0,09 → 0,58**. Lecție: căi
deterministe și neambigue ale artefactelor sunt esențiale pentru o evaluare demnă
de încredere.

### 8.5 Clasic vs profund — judecată de sinteză
- **Cel mai bun clasificator pe patch:** HOG + SVM (acuratețe 0,827).
- **Cel mai bun detector practic:** YOLOv8n (cele mai curate și consistente
  rezultate pe imagine completă; mAP@50 0,601).
- **Cea mai modernă arhitectură:** YOLOv10n (fără NMS; competitiv, dar puțin în urmă
  față de v8n la acest buget scurt de antrenare).
- **Metodele clasice** rămân utile ca **baze de referință interpretabile** și sunt
  ieftine de antrenat, dar nu localizează și sunt fragile sub protocolul ferestrei
  glisante. Detectoarele profunde sunt net superioare pentru sarcina end-to-end.

---

## 9. Limitări

1. **Doar CPU, buget scurt de antrenare.** Modelele YOLO au fost antrenate doar 10
   epoci și încă se îmbunătățeau; mAP ar crește substanțial cu un GPU și mai multe
   epoci.
2. **Dezechilibru de clase.** Dronele depășesc numeric păsările ~1,5:1; combinat cu
   coruperea istorică a etichetelor, clasa pasăre rămâne cea mai dificilă (recall
   mai scăzut).
3. **Nepotrivirea metricilor între paradigme.** Acuratețea clasică (patch) și mAP
   YOLO (detecție) nu pot fi mediate sau clasate pe o singură scală (§7.1).
4. **Costul și zgomotul ferestrei glisante.** Detectorul HOG este lent (secunde pe
   imagine) și produce multe casete redundante; este o referință, nu un detector de
   producție.
5. **Eterogenitatea adnotărilor.** Etichete mixte casetă/poligon; Ultralytics
   ignoră segmentele poligonale și folosește doar casetele delimitatoare.

---

## 10. Direcții Viitoare

- **Antrenare pe GPU** cu programe mai lungi (100–300 epoci) și oprire timpurie
  (early stopping).
- **Backbone-uri mai mari** (YOLOv8s/m, YOLOv10s/m) pentru un studiu al
  compromisului acuratețe/viteză.
- **Reechilibrarea clasei pasăre** prin augmentare țintită sau imagini suplimentare
  cu păsări.
- **Înlocuirea ferestrei glisante HOG** cu o propunere de regiuni adecvată, sau
  formularea bazei de referință clasice ca *doar clasificare* pe decupaje YOLO,
  pentru o comparație mai echitabilă.
- **Raportarea mAP@50–95 și AP pe clasă** pentru pipeline-ul clasic folosind
  `compute_iou_metrics`, pentru o comparație de detecție complet aliniată.
- **Studiu de calibrare a încrederii** (diagrame de fiabilitate) pentru toate cele
  patru modele.

---

## 11. Reproductibilitate

### 11.1 Corecția unică a datelor
```bash
python scripts/relabel_by_filename.py          # re-etichetează ambele seturi (+backup)
# cache-urile sunt reconstruite automat la următoarea încărcare
```

### 11.2 Antrenare
```bash
# Clasic (rapid):
python scripts/train_hog.py --max-train 1500 --max-test 400 --mlp-epochs 40

# Profund (CPU, ~ore):
python scripts/train_yolov8.py  --merged --epochs 10 --device cpu
python scripts/train_yolov10.py --merged --epochs 10 --device cpu --batch 8 \
       --model yolov10n.pt --name drone_v10_cpu
```
Sau, din UI: **`streamlit run app.py` → „Train All Models"** (un singur clic rulează
HOG sincron, apoi lansează ambele antrenări YOLO în fundal).

### 11.3 Evaluare și inspecție
```bash
python scripts/evaluate_all.py                 # scrie results/all_metrics.json
streamlit run app.py                           # tab Results = tabel comparativ
                                               # tab Live Detection = demo pe 4 căi
```

### 11.4 Artefacte cheie
| Artefact | Cale |
|----------|------|
| Modele HOG | `models/hog_mlp.pkl`, `models/hog_svm.pkl` |
| Ponderi YOLOv8 | `runs/train/drone_v8/weights/best.pt` |
| Ponderi YOLOv10 | `runs/train/drone_v10_cpu/weights/best.pt` |
| Metrici comparative | `results/all_metrics.json` |
| Backup etichete | `labels_backup.zip` |
| Jurnale de antrenare | `logs/train_yolov8.log`, `logs/train_yolov10.log` |

---

## 12. Concluzie

Acest proiect a livrat o comparație funcțională pe patru căi a detectoarelor de
păsări vs drone, acoperind abordări clasice (HOG+MLP, HOG+SVM) și profunde
(YOLOv8n, YOLOv10n), integrate într-o singură aplicație interactivă.

Rezultatul decisiv nu a fost unul arhitectural, ci unul de **calitate a datelor**:
corectarea unei corupții sistematice a etichetelor (≈99 % dintre păsări greșit
etichetate ca drone) a restabilit funcționalitatea la toate modelele — o
demonstrație concretă că integritatea datelor este o precondiție pentru, și adesea
mai impactantă decât, selecția modelului.

Pe datele corectate, **SVM-ul clasic** a obținut cea mai bună acuratețe la nivel de
patch (0,827), în timp ce **YOLOv8n** a oferit cea mai fiabilă detecție end-to-end
(mAP@50 0,601, predicții curate cu o singură casetă). **Decalajul patch-vs-detecție**
observat la pipeline-ul HOG evidențiază faptul că acuratețea de clasificare și
calitatea detecției sunt proprietăți distincte — protocolul de evaluare este la fel
de important ca și modelul. Cu antrenare pe GPU și programe mai lungi, se așteaptă
ca detectoarele profunde să-și mărească substanțial avantajul.

---

*Document generat din codul sursă al proiectului, jurnalele de antrenare și
rezultatele evaluării. Toate metricile sunt reproductibile prin comenzile din §11.*

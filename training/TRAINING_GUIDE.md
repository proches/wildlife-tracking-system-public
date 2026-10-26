# Jinsi ya kufundisha model yako (Training Guide)

Lengo: model inayoona **ndege na wanyama wadogo** kwenye *camera yako, pembe yako, maji yako*. Pretrained models (MegaDetector, SpeciesNet) hazijawahi kuona mtazamo huu — hapo ndipo faida kubwa ilipo.

## Mzunguko (active learning)

```
main.py inaendesha  ──►  logs/review/  (picha nzima + draft boxes + json)
        ▲                      │
        │                      ▼
  models/melia_wildlife.pt   review_labels.py  (unathibitisha/unarekebisha, dakika chache)
        ▲                      │
        │                      ▼
   train_custom_yolo.py ◄── build_dataset.py (--tile, split kwa muda)
```

Kila wiki: review picha mpya → train upya → model inaboreka. Hard negatives (picha ambazo hakuna mnyama: vigogo, mng'ao, maandishi) ni muhimu kama positives — `c` kwenye reviewer.

## Hatua

### 1. Kusanya picha (≥ 300 za kuanzia, 1500+ ni nzuri)
```bash
python training/harvest_frames.py --video logs/highlights/*.mp4 --out dataset_raw/images --every 2
python training/harvest_frames.py --live --minutes 180 --out dataset_raw/images --every 20   # camera iwe huru
```
Pia tumia `logs/review/` (system inajikusanyia). Hakikisha una: **maji, jioni/asubuhi, mbali na karibu, ndege wadogo, picha tupu**.

### 2. Draft ya labels (inaokoa 90% ya kazi)
```bash
python training/prelabel.py dataset_raw/images --classify
```

### 3. Review (kibodi)
```bash
python training/review_labels.py dataset_raw/images
```
`SPACE` inayofuata · `a` ongeza box · `d` futa ya mwisho · `c` futa zote (hard negative) · `f` animal↔bird · `z` kuza · `q` toka.
**Usifundishe kwa drafts ambazo hujazikagua** — model ingejifunza makosa ya MegaDetector.

### 4. Dataset (tiles + split kwa muda)
```bash
python training/build_dataset.py dataset_raw/images logs/review --out dataset --tile
```
* `--tile`: tiles za 640 px kwa **pixel asilia**. Ndege wa px 14 hawafutiki kama ilivyo picha ikipunguzwa.
* Split kwa *muda* (vizuizi vya dakika 10) — frames jirani hazigawanywi train/val (la sivyo val inadanganya).
* Val ni **picha nzima** (ndivyo inavyoendeshwa shambani).

### 5. Train — Google Colab (GPU ya bure)
PC hii (i5-7200U, bila GPU) ingechukua masaa mengi.  Tumia `training/colab_train.ipynb`:
1. Zip folder `dataset` → `dataset.zip`, weka kwenye Google Drive: `MyDrive/melia/dataset.zip`.
2. Fungua https://colab.research.google.com → *Upload* → `training/colab_train.ipynb`.
3. *Runtime → Change runtime type → T4 GPU*, kisha endesha cells zote (dakika ~20–40).
4. `melia_wildlife_YYYYMMDD.pt` inahifadhiwa kwenye Drive → nakili kuwa `models/melia_wildlife.pt`.

Ukiwa na GPU yako mwenyewe:
```bash
pip install -r requirements-custom-yolo.txt
python training/train_custom_yolo.py --data dataset/data.yaml --preset small --device 0
```
`small` = yolo11s, tiles 640, haraka. Una picha nyingi zaidi? `--preset large`.

### 6. Pima kwa ukubwa (hapa ndipo ndege wanapotea)
```bash
yolo predict model=runs/wildlife/melia/weights/best.pt source=dataset/images/val imgsz=1920 conf=0.15 save_txt=True save_conf=True
python training/eval_by_size.py --gt dataset/labels/val --pred runs/detect/predict/labels
```
Utaona recall kwa `tiny / small / medium / large` na idadi ya false positives. Kama `tiny` ni ya chini: ongeza picha za ndege wadogo + `--tile-size 512`.

### 7. Washa kwenye system
```text
# .env
WILDLIFE_CUSTOM_YOLO_ENABLED=true
WILDLIFE_CUSTOM_YOLO_MODEL_PATH=models/melia_wildlife.pt
WILDLIFE_CUSTOM_YOLO_IMGSZ=1920
```
Model yako inaunganishwa (fusion) na MegaDetector; species bado zinatoka SpeciesNet.

## Madarasa
Anza na `animal`, `bird` (`training/labels_io.py: CLASSES`). Baadaye unaweza kugawa species (zebra, wildebeest…) — lakini SpeciesNet tayari inatoa majina; detector inahitaji kujua *wapi*, si *nani*.

## Dataset ya nje (hiari)
LILA *Snapshot Serengeti* ina picha nyingi za camera-trap (mwanga/pembe tofauti). Inasaidia kama mazoezi ya awali, lakini **picha za camera yako ndizo zinazopunguza false positives** — usiziache.

## Ukweli kuhusu idadi
Hakuna idadi ya "accuracy %" ya kuaminika hadi upime kwenye val yako uliyoikagua mwenyewe. 300 picha → uboreshaji unaonekana; 1500+ → thabiti.

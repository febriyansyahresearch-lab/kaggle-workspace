# NOTEBOOK-KAGGLE — Kaggle Notebooks by Collection

**Febriyansyah** — MTI, IT Security Leader (15+ yrs, Banking)

Kumpulan 13 Kaggle notebooks terorganisir per collection (mirror dari Kaggle Collections). Setiap notebook berada di `<COLLECTION>/<slug>/` berisi `<slug>.ipynb` + `kernel-metadata.json`.

## Structure

```
NOTEBOOK-KAGGLE/
├── AUTOMATION/           # 1 notebook
│   └── auto-github-runner/
├── CLASSIFICATION/       # 1 notebook
│   └── classify-breast-cancer/
├── COMPUTER-VISION/      # 3 notebooks (Collection: Computer Vision - 19123799)
│   ├── classify-malimg/
│   ├── vision-malevis-generator/
│   └── vision-image-size-check/
├── EDA/                  # 4 notebooks
│   ├── eda-iris/
│   ├── eda-titanic/
│   ├── eda-housing/
│   └── eda-customer-segmentation/
├── EXPERIMENTS/          # 2 notebooks
│   ├── exp-transformer-mamba/
│   └── exp-transformer-test/
├── NLP/                  # 2 notebooks
│   ├── nlp-retail-chatbot-rag/
│   └── nlp-retail-cs-ai/
├── LICENSE
└── README.md
```

## Collections

| Collection | Notebooks | Description |
|---|---|---|
| `AUTOMATION` | `auto-github-runner` | GitHub Runner Automation |
| `CLASSIFICATION` | `classify-breast-cancer` | Breast Cancer Classification |
| `COMPUTER-VISION` | `classify-malimg`, `vision-malevis-generator`, `vision-image-size-check` | Malware & Vision |
| `EDA` | `eda-iris`, `eda-titanic`, `eda-housing`, `eda-customer-segmentation` | Exploratory Data Analysis |
| `EXPERIMENTS` | `exp-transformer-mamba`, `exp-transformer-test` | Transformer Experiments |
| `NLP` | `nlp-retail-chatbot-rag`, `nlp-retail-cs-ai` | Retail Chatbot RAG |

Total: **13 notebooks** — semua `is_private: true`, `enable_internet: true`.

## Kaggle Sync

Setiap subfolder memiliki `kernel-metadata.json` (`id: febriyansyahresearch/<slug>`).

### Push ke Kaggle

```bash
# Auth (kaggle CLI 2.x)
export KAGGLE_API_TOKEN=$(kaggle auth print-access-token)

# Push satu notebook
cd COMPUTER-VISION/classify-malimg
kaggle kernels push

# Push semua notebook (per collection)
for d in */*/*/; do [ -f "$d/kernel-metadata.json" ] && (cd "$d" && kaggle kernels push); done
# atau
find . -name "kernel-metadata.json" -execdir kaggle kernels push \;
```

> Catatan: Kaggle membatasi 5 sesi CPU bersamaan — push berurutan dengan jeda bila perlu.

### Pull dari Kaggle

```bash
kaggle kernels pull febriyansyahresearch/eda-iris -p EDA/eda-iris/
kaggle kernels list --mine
```

### Collections

Kaggle Collections hanya via web UI (tidak ada CLI):
- `https://www.kaggle.com/work/collections/19123799` — Computer Vision
- Buat collection baru di `Your Work > Collections > New Collection` lalu Add notebook `febriyansyahresearch/<slug>`

## Profile

- Kaggle: `https://www.kaggle.com/febriyansyahresearch`
- GitHub: `https://github.com/febriyansyah-id/KAGGLE-WORKSPACE`

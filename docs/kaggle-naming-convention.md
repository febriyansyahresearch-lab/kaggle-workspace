# Kaggle Notebook Naming Convention

**Format:** `<kategori>-<topik>-<tugas>`
**Aturan:**
- Huruf kecil semua (lowercase)
- Kebab-case (pisah dengan `-`)
- Bahasa Inggris
- Prefix kategori wajib di depan

## Kategori

| Kategori | Prefix | Contoh |
|----------|--------|--------|
| EDA (Exploratory Data Analysis) | `eda-` | `eda-iris` |
| Classification | `classify-` | `classify-breast-cancer` |
| NLP / LLM | `nlp-` | `nlp-retail-chatbot-rag` |
| Model Experiment | `exp-` | `exp-transformer-mamba` |
| Vision / Image | `vision-` | `vision-malevis-generator` |
| Automation / Utility | `auto-` | `auto-github-runner` |

## Daftar Notebook & Nama Usulan

| # | Slug Saat Ini | Judul Saat Ini | Kategori | Nama Baru Usulan |
|---|---------------|----------------|----------|------------------|
| 1 | `malimg-tes` | Malimg Tes | Classification | `classify-malimg` |
| 2 | `transformer-mamba` | Transformer Mamba | Model Experiment | `exp-transformer-mamba` |
| 3 | `breast-cancer-classification-benchmark` | Breast Cancer Classification Benchmark | Classification | `classify-breast-cancer` |
| 4 | `customer-segmentation-eda` | Customer Segmentation EDA | EDA | `eda-customer-segmentation` |
| 5 | `california-housing-eda` | California Housing EDA | EDA | `eda-housing` |
| 6 | `titanic-eda-feature-engineering` | Titanic EDA & Feature Engineering | EDA | `eda-titanic` |
| 7 | `iris-eda` | Iris EDA | EDA | `eda-iris` |
| 8 | `mini-malevis-generator` | Mini MaleVis Generator | Vision | `vision-malevis-generator` |
| 9 | `ceksizeimage300x300` | CekSizeImage300x300 | Vision | `vision-image-size-check` |
| 10 | `retail-chatbot-rag` | Retail Chatbot RAG | NLP | `nlp-retail-chatbot-rag` |
| 11 | `retail-cs-ai` | Retail CS AI | NLP | `nlp-retail-cs-ai` |
| 12 | `transformer-model-test` | Transformer Model Test | Model Experiment | `exp-transformer-test` |
| 13 | `github-runner-ipynb` | github_runner.ipynb | Automation | `auto-github-runner` |

## Catatan

- Kaggle tidak punya fitur rename langsung. Mengubah nama = buat kernel baru + hapus kernel lama (URL berubah).
- Eksekusi rename hanya jika sudah disetujui.
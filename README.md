<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://readme-typing-svg.demolab.com?font=Fira+Mono&size=13&pause=1000&color=64FFDA&center=true&vCenter=true&width=600&lines=Topology+%C3%97+Contrastive+Learning+%C3%97+Genomics">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Mono&size=13&pause=1000&color=0A7A5A&center=true&vCenter=true&width=600&lines=Topology+%C3%97+Contrastive+Learning+%C3%97+Genomics" alt="Typing SVG">
</picture>

# PolyCLIP-T

### *Topological Deep Learning for Polygenic Variant Discovery in Familial Multimorbid Disorders*

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3b82f6?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GUDHI](https://img.shields.io/badge/TDA-GUDHI-8b5cf6?style=flat-square)](https://gudhi.inria.fr/)
[![bioRxiv](https://img.shields.io/badge/Paper-bioRxiv-b91c1c?style=flat-square&logo=arxiv&logoColor=white)](https://github.com/MorillaLab/PolyCLIP-T)
[![Status](https://img.shields.io/badge/Status-Active-16a34a?style=flat-square)]()

<br>

> *From millions of genomic variants to a handful of coherent polygenic candidates —*  
> *by treating variant prioritisation as a topological discovery problem.*

<br>

</div>

---

## The Problem

Clinical whole-genome sequencing yields **4–5 million variants per individual**. Rule-based ACMG/AMP pipelines excel at Mendelian alleles but systematically fail at the variants that matter most in complex familial disease:

| Challenge | Why it fails today |
|---|---|
| Low-penetrance alleles | Too common for frequency filters; too weak for effect-size thresholds |
| Non-coding variation | Regulatory context too complex for categorical scoring |
| Oligogenic interactions | Single-variant logic is blind to combinatorial burden |
| Germline–somatic interplay | Siloed germline and tumour pipelines miss two-hit evidence |

**PolyCLIP-T** reframes the question: instead of *classifying* each variant, it *discovers* geometrically coherent variant groups — sets that share sequence context, functional profile and familial segregation — using the mathematics of shape.

---

## How It Works

```
WGS (families)
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  VARIANT REPRESENTATION                                 │
│  • DNA sequence context  ±100–500 bp (DNABERT-2)        │
│  • Functional annotations (CADD, SpliceAI, gnomAD …)    │
│  • Familial context  (segregation, transmission, LOH)   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  DUAL-ENCODER CONTRASTIVE LEARNING                      │
│  Transformer (seq) ──┐                                  │
│                      ├──► shared latent space ℝ²⁵⁶      │
│  MLP (annotations) ──┘                                  │
│  Loss: symmetric InfoNCE  +  topological diffusion reg  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  PERSISTENT HOMOLOGY  (Vietoris–Rips, GUDHI)            │
│  Lifetime ℓ ≥ 0.25  →  stable topological component     │
│  + density enrichment  +  functional coherence filter   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
              Prioritised polygenic clusters
         (O(100) candidates from ~1 million variants)
```

### Core innovations

**Topological Diffusion Regularisation (TDR)** — a teacher–student EMA framework that preserves the multi-scale geometry of the variant manifold during training, preventing the geometric collapse that erases low-penetrance signals.

**Mixed-norm variant selection** — a principled filter combining L² profile distance with Wasserstein distance to a benign-variant reference, grounded in two theorems (compactness of polygenic profile space; finite covering property).

**Five-axis integrated scoring** — a deterministic, auditable evidence aggregation across germline pathogenicity, somatic two-hit evidence, family segregation, pathway plausibility and multi-compartment origin. Scores map to four clinical tiers (LOW / MODERATE / HIGH / TOP).

---

## Results at a Glance

Applied to five families with multimorbid cancer, autoimmune and cardiovascular disease:

| Metric | PolyCLIP-T | ACMG/AMP filter | CADD > 20 |
|---|---|---|---|
| Precision (ClinVar recovery) | **0.68** | 0.51 | 0.44 |
| Recall | **0.65** | 0.48 | 0.40 |
| F1 (full model) | **0.83** | — | — |
| ARI at 1–5% labels | **≥ 0.70** | n/a | n/a |
| Variants retained from 4.2M | **~104 per family** | ~3,500 | ~12,000 |

PolyCLIP-T retains **3× more variants in the 0.1–1% AF "twilight zone"** — the low-penetrance range systematically excluded by PM2 frequency filters — while producing a far smaller final candidate set.

### Representative discovery: Family F5

Persistent homology (lifetime ℓ = 0.42) identified a 15-variant cluster spanning *ERBB3*, *HLA-A* and *LPA* — genes canonically associated with breast cancer, autoimmune disease and cardiovascular risk respectively — co-segregating in a proband diagnosed with all three conditions. The *ERBB3* p.Val104Met variant (CADD 28.5, gnomAD AF 0.0003, LOH in tumour) scored TOP-tier (integrated score 9), having been classified as VUS by standard ACMG analysis.

---

## Installation

```bash
git clone https://github.com/MorillaLab/PolyCLIP-T.git
cd PolyCLIP-T
pip install -e .
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0, GUDHI, DNABERT-2, scikit-learn, ripser — see [`requirements.txt`](requirements.txt).

---

## Quick Start

```python
from pipeline.main_pipeline import PolyCLIPTPipeline
from data.config.preprocessing_config import load_config

# 1. Load family configuration (VCF paths, pedigree, phenotypes)
config = load_config('configs/base_config.yaml')

# 2. Run the full pipeline
pipeline = PolyCLIPTPipeline(config)
results  = pipeline.analyze_family(
    family_id  = "F1",
    vcf_paths  = ["data/F1/germline.vcf.gz",
                  "data/F1/tumour.vcf.gz"],
    pedigree   = "data/F1/pedigree.csv"
)

# 3. Retrieve prioritised variant clusters
top = results.get_prioritized_variants(tier="HIGH")
print(top[["gene", "variant", "integrated_score", "evidence_string"]])
```

Output columns include `INTEGRATED_SCORE`, `TIER`, `EVIDENCE_STRING` (e.g. `GERM_PASS|TWO_HIT_STRICT|SEG+2|PATHWAY_HIT`) and the persistent-homology component lifetime for each variant.

---

## Repository Structure

```
PolyCLIP-T/
├── Code/
│   ├── models/
│   │   ├── dual_encoder.py        # DNABERT-2 + MLP contrastive architecture
│   │   ├── tdr.py                 # Topological Diffusion Regularisation
│   │   └── scoring.py             # Five-axis integrated scoring
│   ├── topology/
│   │   ├── persistent_homology.py # Vietoris–Rips via GUDHI
│   │   └── mixed_norm.py          # Mixed-norm variant selection
│   ├── pipeline/
│   │   └── main_pipeline.py       # End-to-end family analysis
│   └── data/
│       ├── preprocessing.py       # VCF harmonisation & annotation
│       └── config/
├── Data/                          # Example family configs (anonymised)
├── result_new/                    # Precomputed embeddings & cluster outputs
├── requirements.txt
└── setup.py
```

---

## Method Details

### Contrastive pre-training

Reference and alternate DNA sequences (±100 bp, extended to ±500 bp for splice variants) are tokenised with a DNABERT-2 BPE tokeniser and encoded by a 6-layer, 12-head transformer (768-dim), fine-tuned at lr = 1×10⁻⁵. Tabular annotations pass through a two-hidden-layer MLP (512 → 256 units, ReLU, dropout 0.2). Both branches project to a shared ℝ²⁵⁶ space. Training uses symmetric InfoNCE (temperature τ = 0.07, 100 epochs, AdamW).

### Topological Diffusion Regularisation

A k-NN affinity graph (k=15, Gaussian kernel) yields Markov transition matrix P = D⁻¹A. Diffusion operators P^t for t ∈ {1, 3, 5} capture local and global manifold structure. A teacher network (EMA, α = 0.99) provides a stable reference; the TDR loss minimises squared discrepancy in diffusion distances between student and teacher across scales.

### Persistent homology

Fused embeddings z = [z_seq, z_bio] ∈ ℝ⁵¹² form a cosine-distance matrix per family. Vietoris–Rips filtration (GUDHI, max dim 2, sparse approximation for >10k variants) produces persistence diagrams converted to 20×20 persistence images. Components with lifetime ≥ 0.25, local density ≥ 2× background and functional coherence (CADD ≥ 20 enrichment or gnomAD AF < 0.01% or LOH overlap) are selected as candidate polygenic modules.

---

## Cohort

| Family | Members (Aff/Unaff) | Disease categories | WGS coverage (G/S) | Candidates output |
|---|---|---|---|---|
| F1 (Discovery) | 2 (2/0) | Cancer, Autoimmune | 35× / 85× | 127 |
| F2 (Discovery) | 2 (1/1) | Cancer, Autoimmune, Cardiovascular | 38× / 90× | 94 |
| F3 (Discovery) | 2 (1/1) | Cancer, Autoimmune, Cardiovascular | 32× / 78× | 88 |
| F4 (Discovery) | 3 (2/1) | Cancer, Autoimmune, Cardiovascular | 36× / 82× | 101 |
| F5 (Discovery) | 2 (1/1) | Cancer, Autoimmune | 40× / 80× | 112 |
| F6 (Test)      | 3 (2/1) | Cancer, Autoimmune, Cardiovascular | 40× / 80× | 112 |

G/S = Germline / Somatic. Starting from an average of 1.0M variants per family post-QC.

---

## Citation

If you use PolyCLIP-T in your research, please cite:

```bibtex
@article{vomodonfack2026polyclip,
  title   = {Topological Deep Learning Identifies Polygenic Variant Clusters
             Across Familial Multimorbid Disorders},
  author  = {Vomo-Donfack, Kelly Larissa and Bousquet, Guilhem and
             Falgarone, G{\'e}raldine and Ginot, Gr{\'e}gory and Morilla, Ian},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.1101/2026.xx.xx.xxxxxx}
}
```

---

## Acknowledgements

Supported by the **Laboratoire d'excellence Infibrex** (ANR-11-LABX-0011), the **Spanish National Research Council** (CSIC), and the **French National Research Agency** (ANR) through the SynKoMIC project. Computational resources provided by the Université Sorbonne Paris Nord computing cluster.

We thank the patients and families who participated in this study, and the clinical teams at Hôpital Avicenne (APHP) for sample collection and annotation.

---

## Contact

| | |
|---|---|
| **Kelly Larissa Vomo-Donfack** | vomodonfack@math.univ-paris13.fr |
| **Ian Morilla** *(corresponding)* | ian.morilla@ihsm.uma-csic.es |

*LAGA – Université Sorbonne Paris Nord / IHSM – Universidad de Málaga–CSIC*

---

<div align="center">
<sub>
© 2026 MorillaLab · MIT License · Université Sorbonne Paris Nord & Universidad de Málaga–CSIC
</sub>
</div>

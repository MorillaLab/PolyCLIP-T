# PolyCLIP-T: Topological Deep Learning for Polygenic Variant Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2501.xxxxx)

PolyCLIP-T is a topological representation learning framework for identifying polygenic variant clusters across familial multimorbid disorders. The method integrates contrastive learning with topological data analysis to discover biologically coherent variant groups in whole-genome sequencing data.

## Key Features

- **Dual-encoder contrastive learning** for aligning DNA sequence context with biological annotations
- **Topological diffusion regularization** to preserve geometric structure of variant manifolds
- **Persistent homology analysis** for identifying stable variant clusters
- **Five-axis scoring system** integrating germline pathogenicity, two-hit evidence, segregation, and pathway information
- **Mixed norm filtering** for robust variant selection

## Installation

```bash
# Clone the repository
git clone https://github.com/MorillaLab/PolyCLIP-T.git
cd PolyCLIP-T

# Install with pip
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

```bash
from pipeline.main_pipeline import PolyCLIPTPipeline
from data.config.preprocessing_config import load_config

# Load configuration
config = load_config('configs/base_config.yaml')

# Initialize pipeline
pipeline = PolyCLIPTPipeline(config)

# Process a family's WGS data
results = pipeline.analyze_family(
    family_id="FAM001",
    vcf_paths=["data/FAM001/*.vcf"],
    pedigree="data/FAM001/pedigree.csv"
)

# Get prioritized variants
top_variants = results.get_prioritized_variants(threshold=0.8)
```


## Documentation
Full documentation is available in the docs/ directory:

- [Installation Guide](docs/installation.md)
- [Usage Tutorial](docs/usage.md)
- [API Reference](docs/api.md)
- [Theoretical Background](docs/theory.md)



<!-- ============================================== -->
<div align="left">
  <h1 id="citation">🎈 Citation</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

If you use PolyCLIP-T in your research, please cite:
```bibtex
@article{vomodorfack2026polyclip,
  title={Topological Deep Learning Identifies Polygenic Variant Clusters Across Familial Multimorbid Disorders},
  author={Vomo Donfack, Kelly Larissa and Bousquet, Guilhem and Falgarrone, Géraldine and Ginot, Grégory and Morilla, Ian},
  journal={bioRxiv},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This work was supported by the Laboratoire d'excellence Infibrex (ANR-11-LABX-0011), the Spanish National Research Council (CSIC), and the French National Research Agency (ANR) through the SynKoMIC project.

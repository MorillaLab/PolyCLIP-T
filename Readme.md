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

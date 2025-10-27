# YOYO-Fusion: Plug-and-Play Merging of Arbitrary Fine-Tunes with Shared Architecture

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

YOYO-Fusion is an efficient merging technique for large language models (LLMs). Its core advantage lies in realizing a "three-no" merging paradigm—no additional data required, no parameter tuning needed, and no dependence on pre-trained models.

This method can efficiently absorb the high-value knowledge and capabilities of multiple fine-tuned models while maintaining the model's strong robustness, providing a new approach for building high-performance models at low cost.

---

## Key Features

- Robust Centering: Uses coordinate median or geometric median (Weiszfeld algorithm) as fusion center—resistant to outlier models.
- Anchor Mode: Optionally lock to a specific model as the reference (e.g., preserve instruction-following behavior).
- Subspace Truncation: Projects weight differences into a low-rank subspace (rank ≤ K−1 for K models) to remove consensus noise.
- Outlier Suppression: Applies Tukey’s biweight weighting in the subspace to downweight anomalous models per dimension.
- Norm Preservation: Automatically rescales output to match the average norm statistics of input models.
- Full Compatibility: Supports both single-file (`model.safetensors`) and sharded (`model.safetensors.index.json`) Hugging Face–style models.
- Memory Efficient: Processes one tensor at a time; no need to load all models fully into GPU memory.

---

## Quick Start

### Prerequisites

- Python >= 3.9
- PyTorch >= 2.0
- `safetensors`, `numpy`, `tqdm`

Install dependencies:
```bash
pip install torch safetensors numpy tqdm
```

### Basic Usage

```python
from yoyo_fusion import run_merge

run_merge(
    model_paths=[
        "path/to/model_A",
        "path/to/model_B",
        "path/to/model_C"
    ],
    output_dir="path/to/merged_model",
    anchor_index=0,  # n=0: no anchor n>=1: use n-th model as anchor
    config_dir=1,  # m>=1 use m-th as config
    use_k_minus_one_truncation=True,  # True: truncation + energy scaling False: full SVD (no truncation)
    use_geometric_median=True,  # True: use geometric median False: use lower median
)
```

Tip: For most cases, keep `anchor_index=0`, `use_k_minus_one_truncation=True`, and `use_geometric_median=True`.

---

## Algorithm Overview

YOYO-Fusion merges tensors through the following steps:

1. Normalize each input tensor by its RMS.
2. Compute a robust center:
   - If `anchor_index=0`: use geometric median (or coordinate median).
   - Else: use the specified anchor model.
3. Form residuals = normalized tensors − center.
4. SVD decomposition of residuals; optionally truncate to rank K−1.
5. Project residuals into dominant subspace.
6. Robustly average projected coordinates using Tukey’s biweight (resistant to outliers).
7. Reconstruct merged tensor and rescale to preserve original norm statistics.

This ensures the merged model retains meaningful capabilities from all inputs while discarding inconsistent or noisy directions.

---

## Recommended Use Cases

| Scenario | Recommended Settings |
|--------|----------------------|
| Merge many fine-tuned variant | `anchor_index=0`, `use_geometric_median=True/False`, `use_k_minus_one_truncation=True/False` |
| Preserve behavior of a specific model | `anchor_index=1`, `use_k_minus_one_truncation=True/False` |

Best Practice: Always validate merged models with task-specific benchmarks.

---

## Directory Structure

Your model directories should follow Hugging Face conventions:

```
model_A/
├── model.safetensors          # (single-file) OR
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
└── model.safetensors.index.json

model_B/
├── ...
```

The script auto-detects whether models are sharded or single-file and handles both.

---

## Parameters Explained

| Parameter | Type | Default | Description |
|---------|------|--------|-------------|
| `model_paths` | List[str] | — | Paths to input model directories (>=2) |
| `output_dir` | str | — | Output directory for merged model |
| `anchor_index` | int | 0 | 0: no anchor (robust center); n>=1: use n-th model as anchor (1-based) |
| `config_dir` | int | 1 | Which model’s config/index files to copy (1-based) |
| `use_k_minus_one_truncation` | bool | True | Truncate SVD to rank K−1 and apply energy scaling (recommended) |
| `use_geometric_median` | bool | False | Use geometric median instead of coordinate median (only if `anchor_index=0`) |

Note: `use_geometric_median` and `use_k_minus_one_truncation` are ignored when `anchor_index != 0`.

---

## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

## Acknowledgements

- Uses Weiszfeld’s algorithm for geometric median.
- Leverages Tukey’s biweight for outlier-resistant aggregation.

---

## Feedback & Contributions

Found a bug? Have an idea? Open an issue or PR.  
YOYO-Fusion is research-friendly and production-ready for model merging experiments.

---

Note: This tool merges weights only. It does not merge tokenizers, configs, or generation settings—those are copied from the `config_dir` model. Always verify compatibility of input models (same architecture, vocab size, etc.).

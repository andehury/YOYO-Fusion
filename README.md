# YOYO-Fusion: Plug-and-Play Merging of Arbitrary Fine-Tunes with Shared Architecture

[![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)

YOYO-Fusion is an efficient merging technique for large language models (LLMs). Its core advantage lies in realizing a "three-no" merging paradigm—no additional data required, no parameter tuning needed, and no dependence on pre-trained models.

This method can efficiently absorb the high-value knowledge and capabilities of multiple fine-tuned models while maintaining the model's strong robustness, providing a new approach for building high-performance models at low cost.

---

## Key Features

- Consensus Center: Determine the center (select a fine-tuned model) or estimate the center (lower median / geometric median)
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
    config_dir=1,    # m>=1 use m-th as config
    use_k_minus_one_truncation=True,  # True: truncation + energy scaling False: full SVD
    use_geometric_median=True,        # True: use geometric median False: use standard median
)
```

---

## Algorithm Steps


### Step 0: Inputs

A set of tensors:
```
𝒯 = {t₁, t₂, ..., tₖ}
```
where K ≥ 2 and each tᵢ ∈ ℝᴰ

**Parameters:**
- `anchor_index` ∈ {0, 1, ..., K}
  - If 0: no anchor; use a robust center (median or geometric median)
  - If n ≥ 1: use model n as anchor (i.e., tₙ)
- `use_geometric_median` ∈ {True, False} (only effective when anchor_index == 0)
- `use_k_minus_one_truncation` ∈ {True, False}

### Step 1: Normalize Input Tensors

Compute RMS normalization for each tensor:
```
rᵢ = RMS(tᵢ) = √[(1/D) ∑ⱼ₌₁ᴰ tᵢⱼ² + ε], where ε = 10⁻⁸
```
```
uᵢ = tᵢ / (rᵢ + ε)
```

Obtain normalized tensor matrix:
```
U = [u₁, u₂, ..., uₖ]ᵀ ∈ ℝᴷ×ᴰ
```

### Step 2: Determine Center Point m ∈ ℝᴰ

**Case A: anchor_index = n (n ≥ 1) (anchor mode)**
```
m = uₙ
```

Note: anchor_index = 1 corresponds to the first model (model_dirs[0]) due to 1-based indexing in the parameter.

**Case B: anchor_index = 0 (no anchor)**

**Subcase B1: use_geometric_median = True**
Compute the geometric median via the Weiszfeld algorithm:
```
m = argminᵧ ∑ᵢ₌₁ᴷ ||uᵢ - y||₂
```

Initialized with the coordinate-wise median and iterated to convergence.

**Subcase B2: use_geometric_median = False**
Use coordinate-wise median:
```
mⱼ = median(u₁ⱼ, u₂ⱼ, ..., uₖⱼ), ∀ j = 1,...,D
```

### Step 3: Compute Residual Matrix
```
R = U - 1ₖmᵀ ∈ ℝᴷ×ᴰ
```

where 1ₖ is a column vector of ones.

If ||R||_F < 10⁻⁷ (models are nearly identical), set:
```
y' = m
```
and skip to Step 6.

### Step 4: SVD Decomposition and Subspace Truncation

Perform SVD on Rᵀ ∈ ℝᴰ×ᴷ (in float64 for numerical stability):
```
Rᵀ = UΣVᵀ
```

where U ∈ ℝᴰ×ʳ, Σ ∈ ℝʳ×ʳ, and r = min(K, D)

Determine target rank r_target and scaling flag:
- If use_k_minus_one_truncation = True:
  ```
  r_target = min(K - 1, r)
  ```
  and energy scaling is enabled
- If use_k_minus_one_truncation = False:
  ```
  r_target = min(K, r)
  ```
  and no scaling is applied
- If r_target ≤ 0, return y' = m

If energy scaling is enabled (use_k_minus_one_truncation = True):
```
p = [∑ᵢ₌₁ʳ_target σᵢ²] / [∑ᵢ₌₁ʳ σᵢ² + ε]
```
```
α_scale = min(1/(p + ε), 10.0)
```

Extract top r_target left singular vectors:
```
U_m = U[:, :r_target] ∈ ℝᴰ×ʳ_target
```

Project residuals onto subspace:
```
Z = RU_m ∈ ℝᴷ×ʳ_target
```

### Step 5: Robust Weighted Averaging (M-estimator with Tukey Biweight)

For each dimension j = 1, ..., r_target:

Compute MAD-based scale estimate:
```
sⱼ = 1.4826 · median(|Z₁ⱼ|, ..., |Zₖⱼ|)
```

Ensure numerical stability:
```
sⱼ = max(sⱼ, 10⁻¹²)
```

Compute global row norms:
```
||zᵢ||₂ = √[∑ⱼ₌₁ʳ_target Zᵢⱼ²], i = 1,...,K
```
```
s_global = 1.4826 · median(||z₁||₂, ..., ||zₖ||₂)
```

Tukey biweight weights (with tuning constant c = 4.685):

Coordinate-wise weights:
```
wᵢⱼ^coord = 
{ [1 - (|Zᵢⱼ|/(c·sⱼ))²]² if |Zᵢⱼ| < c·sⱼ
{ 0 otherwise
```

Global weights:
```
wᵢ^global = 
{ [1 - (||zᵢ||₂/(c·s_global))²]² if ||zᵢ||₂ < c·s_global
{ 0 otherwise
```

Combined weights:
```
Wᵢⱼ = wᵢⱼ^coord · wᵢ^global
```

Compute weighted average per dimension:
```
zⱼ* = [∑ᵢ₌₁ᴷ WᵢⱼZᵢⱼ] / [∑ᵢ₌₁ᴷ Wᵢⱼ + ε]
```

yielding z* ∈ ℝʳ_target

Map back to original space:
```
r* = α_scale · U_mz*
```

Preliminary merged tensor:
```
y' = m + r*
```

### Step 6: Restore Original Scale and Normalize

Mean RMS of original tensors:
```
r̄ = (1/K) ∑ᵢ₌₁ᴷ rᵢ
```
```
y₁ = y' · r̄
```

Mean L2 norm of original tensors:
```
n̄ = (1/K) ∑ᵢ₌₁ᴷ ||tᵢ||₂
```

Norm of current merged tensor:
```
n_y = ||y₁||₂
```

Final scaling to match average norm:
```
α = n̄ / (n_y + ε), y = α · y₁
```

### Step 7: Output

**Merged Tensor = y ∈ ℝᴰ**

Reshape to original tensor shape.
---

## Recommended Use Cases

| Scenario | Recommended Settings |
|--------|----------------------|
| Absorb multiple models in a balanced manner | `anchor_index=0`,`use_geometric_median=True/False`,`use_k_minus_one_truncation=True` |
| Preserve behavior of a specific model | `anchor_index=1`,`use_k_minus_one_truncation=True` |

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

|Parameter|Type|Description|
|---|---|---|
|model_paths|List[str]|Paths to input model directories (>=2)|
|output_dir|str|Output directory for merged model|
|anchor_index|int|0: no anchor (robust center); n>=1: use n-th model as anchor (1-based)|
|config_dir|int|Which model’s config/index files to copy|
|use_k_minus_one_truncation|bool|True: truncation + energy scaling False: full SVD (no truncation)|
|use_geometric_median|bool|True: use geometric median False: use lower median (only if `anchor_index=0`) |

---

## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
---
Note: This tool merges weights only. It does not merge tokenizers, configs, or generation settings—those are copied from the config_dir model. Always verify compatibility of input models (same architecture, vocab size, etc.).

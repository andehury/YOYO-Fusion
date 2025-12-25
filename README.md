# YOYO-Fusion: Robust Merging in Residual Subspace

[![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)

YOYO-Fusion is an efficient merging technique for large language models (LLMs). Its core advantage lies in realizing a "three-no" merging paradigm—no additional data required, no parameter tuning needed, and no dependence on pre-trained models.

This method can efficiently absorb the high-value knowledge and capabilities of multiple fine-tuned models while maintaining the model's strong robustness, providing a new approach for building high-performance models at low cost.

---

## Key Features

- **Consensus Center**: Determine the center (select a fine-tuned model) or estimate the center (standard median / geometric median).
- **Subspace Truncation**: Projects weight differences into a low-rank subspace using adaptive rank via principle rank to remove consensus noise.
- **IRLS Option**: Supports both IRLS-based Welsch weighting and Tukey biweight for outlier suppression in the subspace.
- **Matrix Boost**: Enhances residual components for linear/attention layers by equalizing singular values to the maximum.
- **Norm Preservation**: Restores output tensor norm to match either the average or a specific input model’s norm.
- **Sign Alignment**: Optional coordinate-wise sign flipping to align directions with a reference model.
- **Full Compatibility**: Supports both single-file (`model.safetensors`) and sharded (`model.safetensors.index.json`) Hugging Face–style models.
- **Memory Efficient**: Processes one tensor at a time; no need to load all models fully into CPU memory.

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
    anchor_index=0,                # 0: robust center; n≥1: use n-th model as anchor
    config_dir=1,                  # use config from the n-th model (1-based)
    use_geometric_median=True,     # only used if anchor_index=0
    use_matrix_boost=False,        # apply Matrix Boost for linear/attention layers
    sign_reference_mode=0,         # 0: no alignment; n≥1: align signs to n-th model
    norm_restore_mode=0,           # 0: average norm; n≥1: use n-th model’s norm
    use_irls=True,                 # True: Welsch IRLS; False: Tukey biweight
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
- `use_geometric_median` ∈ {True, False} (only effective when `anchor_index == 0`)
- `use_irls` ∈ {True, False}: selects between Welsch IRLS (iterative) or Tukey biweight (non-iterative) robust fusion
- `use_matrix_boost` ∈ {True, False}: applies singular-value equalization for 2D layers
- `sign_reference_mode` ∈ {0, 1, ..., K}: enables coordinate-wise sign alignment to a reference model
- `norm_restore_mode` ∈ {0, 1, ..., K}: selects norm target for final scaling

### Step 1: Sign Alignment (Optional)

If `sign_reference_mode = r ≥ 1`:
- For each element j, flip sign of tᵢⱼ if sign(tᵢⱼ) ≠ sign(tᵣⱼ) and tᵣⱼ ≠ 0.

Output aligned tensors: {t̃₁, ..., t̃ₖ}

### Step 2: Normalize Input Tensors

Compute RMS normalization for each tensor:
```
rᵢ = RMS(t̃ᵢ) = √[(1/D) ∑ⱼ₌₁ᴰ t̃ᵢⱼ² + ε], ε = 10⁻⁸
```
```
uᵢ = t̃ᵢ / (rᵢ + ε)
```

Form normalized matrix:
```
U = [u₁, u₂, ..., uₖ]ᵀ ∈ ℝᴷ×ᴰ
```

### Step 3: Determine Center Point m ∈ ℝᴰ

**Case A: anchor_index = n ≥ 1**
```
m = uₙ
```

**Case B: anchor_index = 0**
- If `use_geometric_median = True`:  
  m = geometric median of {u₁, ..., uₖ} via Weiszfeld-style iteration.
- Else:  
  mⱼ = median(u₁ⱼ, ..., uₖⱼ), ∀ j

### Step 4: Compute Residual Matrix

```
R = U - 1ₖ mᵀ ∈ ℝᴷ×ᴰ
```

If ||R||_F < 10⁻⁷, set y' = m and skip to Step 7.

### Step 5: SVD and Subspace Projection

Perform SVD on Rᵀ (in float64):
```
Rᵀ = U Σ Vᵀ
```

Compute total energy E = ∑ σᵢ².  
Estimate effective rank via principle rank:
```
PR = (∑ σᵢ²)² / (∑ σᵢ⁴ + 10⁻¹⁶)
r_target = max(1, min(round(PR), K, rank(R)))
```

Compute energy-based scale factor:
```
E_retained = ∑_{i=1}^{r_target} σᵢ²
α_scale = min(√(E / (E_retained + 10⁻¹⁶)), 10.0)
```

Project into subspace:
```
U_m = U[:, :r_target] ∈ ℝᴰ×ʳ_target
Z = R U_m ∈ ℝᴷ×ʳ_target
```

### Step 6: Robust Weighted Fusion in Subspace

#### If `use_irls = True` (Welsch IRLS):
- Initialize z* = median(Z, dim=0)
- Iterate up to `irls_max_iter`:
  - Compute residual Δ = Z − z*
  - Per-dimension scale: sⱼ = 1.4826 · median(|Δ₁ⱼ|, ..., |Δₖⱼ|)
  - Global scale: s_global = 1.4826 · median(||Δ₁||₂, ..., ||Δₖ||₂)
  - Welsch weights (c = 2.985):
    ```
    wᵢⱼ = exp(−( |Δᵢⱼ| / (c sⱼ) )² ) · exp(−( ||Δᵢ||₂ / (c s_global) )² )
    ```
  - Update: z* = (∑ wᵢⱼ Zᵢⱼ) / (∑ wᵢⱼ + ε)
  - Stop if ||z*ₙₑ𝓌 − z*|| < tol

#### If `use_irls = False` (Tukey Biweight):
- Single-step computation with c = 4.685:
  ```
  wᵢⱼ^coord = [max(0, 1 − (|Δᵢⱼ|/(c sⱼ))²)]²
  wᵢ^global = [max(0, 1 − (||Δᵢ||₂/(c s_global))²)]²
  Wᵢⱼ = wᵢⱼ^coord · wᵢ^global
  z* = (∑ Wᵢⱼ Zᵢⱼ) / (∑ Wᵢⱼ + ε)
  ```

Reconstruct residual:
```
r* = α_scale · U_m z*
```

### Step 7: Optional Matrix Boost

If `use_matrix_boost = True`, and tensor is 2D and not embedding/lm_head:
- Reshape r* → R* ∈ ℝ^{m×n}
- Compute SVD: R* = U_R Σ_R V_Rᵀ
- If Σ_R non-empty, set all singular values to σ_max = Σ_R[0]
- Reconstruct: R_boost = U_R diag(σ_max, ..., σ_max) V_Rᵀ
- Update r* = vec(R_boost)

Final preliminary tensor:
```
y' = m + r*
```

### Step 8: Restore RMS Scale

```
r̄ = (1/K) ∑ rᵢ
y₁ = y' · r̄
```

### Step 9: Norm Restoration

Original L2 norms: nᵢ = ||t̃ᵢ||₂

- If `norm_restore_mode = 0`: n_target = (1/K) ∑ nᵢ
- If `norm_restore_mode = m ≥ 1`: n_target = nₘ₋₁

Final scaling:
```
α = n_target / (||y₁||₂ + ε)
y = α · y₁
```

### Step 10: Output

**Merged Tensor = y ∈ ℝᴰ**, reshaped to original dimensions.

---

## Recommended Use Cases

| Scenario | Recommended Settings |
|--------|----------------------|
| Balanced fusion of multiple models | `anchor_index=0`, `use_geometric_median=True`, `use_irls=True` |
| Preserve base model behavior | `anchor_index=1`, `sign_reference_mode=1`, `norm_restore_mode=1` |
| Maximize robustness against outliers | `use_irls=True`, `use_geometric_median=True` |
| Fast fusion with strong noise suppression | `use_irls=False`, `use_matrix_boost=False` |

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

| Parameter | Type | Description |
|---|---|---|
| `model_paths` | List[str] | Paths to input model directories (≥2) |
| `output_dir` | str | Output directory for merged model |
| `anchor_index` | int | 0: robust center; n≥1: use n-th model as anchor (1-based) |
| `config_dir` | int | Which model’s config/index files to copy (1-based) |
| `use_geometric_median` | bool | Use geometric median instead of coordinate-wise median (only if `anchor_index=0`) |
| `use_matrix_boost` | bool | Apply Matrix Boost to 2D linear/attention layers |
| `sign_reference_mode` | int | 0: no alignment; n≥1: align signs to n-th model |
| `norm_restore_mode` | int | 0: match average L2 norm; n≥1: match n-th model’s norm |
| `use_irls` | bool | True: use Welsch IRLS (iterative); False: use Tukey biweight (single-step) |

---

## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

**Note**: This tool merges weights only. It does not merge tokenizers, configs, or generation settings—those are copied from the config_dir model. Always verify compatibility of input models (same architecture, vocab size, etc.).

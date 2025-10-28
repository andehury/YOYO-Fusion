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
    config_dir=1,  # m>=1 use m-th as config
    use_k_minus_one_truncation=True,  # True: truncation + energy scaling False: full SVD (no truncation)
    use_geometric_median=True,  # True: use geometric median False: use lower median
)
```

---

## Algorithm Overview

YOYO-Fusion merges tensors through the following steps:

## **Inputs**

- A set of tensors:

$$ \mathcal{T} = \{ \mathbf{t}_1, \mathbf{t}_2, \dots, \mathbf{t}_K \} $$

where $K \geq 2$ and each $\mathbf{t}_i \in \mathbb{R}^D$

- Parameters:
  - `anchor_index` $\in \{0, 1, \dots, K\}$  
    - If 0: **no anchor**; use a robust center (median or geometric median)  
    - If $n \geq 1$: **use model $n$ as anchor** (i.e., $\mathbf{t}_n$)
  - `use_geometric_median` $\in \{\text{True}, \text{False}\}$ (only effective when `anchor_index == 0`)
  - `use_k_minus_one_truncation` $\in \{\text{True}, \text{False}\}$

---

## **Algorithm Steps**

### **Step 1: Normalize Input Tensors**

Compute RMS normalization for each tensor:

$$
r_i = \text{RMS}(\mathbf{t}_i) = \sqrt{ \frac{1}{D} \sum_{j=1}^D t_{i,j}^2 + \epsilon }, \quad \epsilon = 10^{-8}
$$

$$
\mathbf{u}_i = \frac{\mathbf{t}_i}{r_i + \epsilon}
$$

Obtain normalized tensor matrix:

$$ \mathbf{U} = [\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_K]^\top \in \mathbb{R}^{K \times D} $$

---

### **Step 2: Determine Center Point $\mathbf{m} \in \mathbb{R}^D$**

#### **Case A: `anchor_index = n (n>=1)` (anchor mode)**

$$
\mathbf{m} = \mathbf{u}_n
$$

> Note: `anchor_index = 1` corresponds to the first model (`model_dirs[0]`) due to 1-based indexing in the parameter.

#### **Case B: `anchor_index = 0` (no anchor)**

- **Subcase B1: `use_geometric_median = True`**  
  Compute the geometric median via the Weiszfeld algorithm:

$$
\mathbf{m} = \arg\min_{\mathbf{y}} \sum_{i=1}^K \| \mathbf{u}_i - \mathbf{y} \|_2
$$

  Initialized with the coordinate-wise median and iterated to convergence.

- **Subcase B2: `use_geometric_median = False`**  
  Use coordinate-wise lower median:

$$
m_j = \text{median}(u_{1,j}, u_{2,j}, \dots, u_{K,j}), \quad \forall j=1,\dots,D
$$

---

### **Step 3: Compute Residual Matrix**

$$
\mathbf{R} = \mathbf{U} - \mathbf{1}_K \mathbf{m}^\top \in \mathbb{R}^{K \times D}
$$

where $\mathbf{1}_K$ is a column vector of ones.

If $\|\mathbf{R}\|_F < 10^{-7}$ (models are nearly identical), set:

$$
\mathbf{y}' = \mathbf{m}
$$

and skip to **Step 6**.

---

### **Step 4: SVD Decomposition and Subspace Truncation**

Perform SVD on $\mathbf{R}^\top \in \mathbb{R}^{D \times K}$ (in float64 for numerical stability):

$$
\mathbf{R}^\top = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top
$$

where $\mathbf{U} \in \mathbb{R}^{D \times r}$, $\mathbf{\Sigma} \in \mathbb{R}^{r \times r}$, and $r = \min(K, D)$.

#### **Determine target rank $r_{\text{target}}$ and scaling flag:**

- If `use_k_minus_one_truncation = True`:  
$r_{\text{target}} = \min(K - 1, r)$, and energy scaling is enabled.
- If `use_k_minus_one_truncation = False`:  
$r_{\text{target}} = \min(K, r)$, and no scaling is applied.

If 
$r_{\text{target}} \leq 0$, return $\mathbf{y}' = \mathbf{m}$.

- **If energy scaling is enabled** (`use_k_minus_one_truncation = True`):
  
$$
p = \frac{ \sum_{i=1}^{r_{\text{target}}} \sigma_i^2 }{ \sum_{i=1}^{r} \sigma_i^2 + \epsilon }
$$

$$
\alpha_{\text{scale}} = \min\left( \frac{1}{p + \epsilon},\ 10.0 \right)
$$

- Extract top $r_{\text{target}}$ left singular vectors:

$$
\mathbf{U}_m = \mathbf{U}[:, :r_{\text{target}}] \in \mathbb{R}^{D \times r_{\text{target}}}
$$

- Project residuals onto subspace:
  
$$
\mathbf{Z} = \mathbf{R} \mathbf{U}_m \in \mathbb{R}^{K \times r_{\text{target}}}
$$

---

### **Step 5: Robust Weighted Averaging (M-estimator with Tukey Biweight)**

For each dimension $j = 1, \dots, r_{\text{target}}$:

- Compute MAD-based scale estimate:

$$
s_j = 1.4826 \cdot \text{median}\left( |Z_{1j}|, \dots, |Z_{Kj}| \right)
$$

  Ensure numerical stability: 

$s_j = \max(s_j, 10^{-12})$

- Compute global row norms:

$$
\|\mathbf{z}_i\|_2 = \sqrt{ \sum_{j=1}^{r_{\text{target}}} Z_{ij}^2 }, \quad i=1,\dots,K
$$

$$
s_{\text{global}} = 1.4826 \cdot \text{median}( \|\mathbf{z}_1\|_2, \dots, \|\mathbf{z}_K\|_2 )
$$

- Tukey biweight weights (with tuning constant $c = 4.685$):

  Coordinate-wise weights:

$$
w_{ij}^{\text{coord}} = 
\begin{cases}
\left( 1 - \left( \dfrac{|Z_{ij}|}{c \cdot s_j} \right)^2 \right)^2 & \text{if } |Z_{ij}| < c \cdot s_j \\
0 & \text{otherwise}
\end{cases}
$$

  Global weights:
  

  Combined weights:

$$
W_{ij} = w_{ij}^{\text{coord}} \cdot w_i^{\text{global}}
$$

- Compute weighted average per dimension:

$$
z_j^* = \frac{ \sum_{i=1}^K W_{ij} Z_{ij} }{ \sum_{i=1}^K W_{ij} + \epsilon }
$$

 yielding $\mathbf{z}^* \in \mathbb{R}^{r_{\text{target}}}$

- Map back to original space:

$$
\mathbf{r}^* = \alpha_{\text{scale}} \cdot \mathbf{U}_m \mathbf{z}^*
$$

- Preliminary merged tensor:

$$
\mathbf{y}' = \mathbf{m} + \mathbf{r}^*
$$

---

### **Step 6: Restore Original Scale and Normalize**

- Mean RMS of original tensors:

$$
\bar{r} = \frac{1}{K} \sum_{i=1}^K r_i
$$

$$
\mathbf{y}_1 = \mathbf{y}' \cdot \bar{r}
$$

- Mean L2 norm of original tensors:

$$
\bar{n} = \frac{1}{K} \sum_{i=1}^K \| \mathbf{t}_i \|_2
$$

- Norm of current merged tensor:

$$
n_y = \| \mathbf{y}_1 \|_2
$$

- Final scaling to match average norm:

$$
\alpha = \frac{ \bar{n} }{ n_y + \epsilon }, \quad \mathbf{y} = \alpha \cdot \mathbf{y}_1
$$

---

### **Step 7: Output**

$$
\boxed{ \text{Merged Tensor} = \mathbf{y} \in \mathbb{R}^D }
$$

Reshape to original tensor shape.

---

## Recommended Use Cases

| Scenario | Recommended Settings |
|--------|----------------------|
| Absorb multiple models in a balanced manner | `anchor_index=0`, `use_geometric_median=True/False`, `use_k_minus_one_truncation=True` |
| Preserve behavior of a specific model | `anchor_index=1`, `use_k_minus_one_truncation=True` |

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

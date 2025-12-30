import json
import shutil
import gc
import os
from pathlib import Path
from typing import List, Dict, Callable, Optional

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


# ---------------------------
# Key normalization
# ---------------------------

def normalize_key(key: str) -> str:
    """Normalize tensor keys: remove '.language_model.' if present."""
    if ".language_model." in key:
        return key.replace(".language_model.", ".")
    return key


# ---------------------------
# Math helpers
# ---------------------------

def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    D = x.numel()
    return torch.sqrt(torch.sum(x.float() ** 2) / (D + 1e-12) + eps)


def standard_median(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    sorted_vals, _ = torch.sort(x, dim=dim)
    n = sorted_vals.shape[dim]
    if n % 2 == 1:
        return sorted_vals.select(dim, n // 2)
    else:
        mid1 = sorted_vals.select(dim, n // 2 - 1)
        mid2 = sorted_vals.select(dim, n // 2)
        return (mid1 + mid2) / 2.0


def geometric_median(X: torch.Tensor, eps: float = 1e-8, max_iter: int = 100, tol: float = 1e-6) -> torch.Tensor:
    K, D = X.shape
    if K == 1:
        return X[0].clone()

    y = standard_median(X, dim=0).clone()
    for _ in range(max_iter):
        diff = X - y.unsqueeze(0)
        distances = torch.linalg.norm(diff, dim=1)
        weights = 1.0 / (distances + eps)
        weights = weights / weights.sum()
        y_new = torch.sum(weights.unsqueeze(1) * X, dim=0)
        if torch.linalg.norm(y_new - y) < tol:
            break
        y = y_new
    return y


def mad_scale(vec: torch.Tensor) -> float:
    return 1.4826 * torch.median(torch.abs(vec)).item()


# ---------------------------
# YOYO Fusion
# ---------------------------

def is_linear_or_attention_layer(key: str) -> bool:
    key_lower = key.lower()
    if 'embed' in key_lower or 'lm_head' in key_lower:
        return False
    return True


@torch.no_grad()
def subspace_robust_merge(
        tensors: List[torch.Tensor],
        eps: float = 1e-8,
        anchor_index: int = 0,
        use_geometric_median: bool = False,
        use_matrix_boost: bool = True,
        tensor_key: str = None,
        sign_reference_mode: int = 0,
        use_z_median: bool = True,
) -> torch.Tensor:
    assert len(tensors) >= 2, "Need at least two tensors to merge."
    K = len(tensors)
    original_shape = tensors[0].shape

    xs_raw = [t.detach().cpu().contiguous().float() for t in tensors]

    # --- Sign alignment ---
    if sign_reference_mode == 0:
        xs_aligned = xs_raw
    else:
        ref_idx = sign_reference_mode - 1
        if not (0 <= ref_idx < K):
            raise ValueError(f"sign_reference_mode={sign_reference_mode} out of range for {K} models.")
        ref_tensor = xs_raw[ref_idx]
        ref_sign = torch.sign(ref_tensor)
        xs_aligned = []
        for x in xs_raw:
            x_sign = torch.sign(x)
            flip_mask = (ref_sign != 0) & (x_sign * ref_sign < 0)
            x_aligned = x.clone()
            x_aligned[flip_mask] *= -1.0
            xs_aligned.append(x_aligned)

    xs = [x.view(-1) for x in xs_aligned]

    # --- Normalize by RMS ---
    rms_vals = torch.stack([rms(x) for x in xs])
    rms_vals = torch.clamp(rms_vals, min=1e-6)
    us = [x / (r + eps) for x, r in zip(xs, rms_vals)]
    X = torch.stack(us, dim=0)

    # --- Compute M (median or anchor) ---
    if anchor_index == 0:
        if use_geometric_median:
            M = geometric_median(X, eps=eps)
        else:
            M = standard_median(X, dim=0)
    else:
        anchor_pos = anchor_index - 1
        if not (0 <= anchor_pos < K):
            raise ValueError(f"anchor_index={anchor_index} out of range for {K} models.")
        M = us[anchor_pos].clone()

    R = X - M.unsqueeze(0)
    res_norms = torch.linalg.vector_norm(R, dim=1)
    if torch.max(res_norms) < 1e-7:
        y_prime = M.clone()
    else:
        V64 = R.transpose(0, 1).contiguous().numpy().astype(np.float64, copy=False)
        U_np, S_np, VT_np = np.linalg.svd(V64, full_matrices=False)

        total_energy = np.sum(S_np ** 2)
        if total_energy < 1e-16:
            y_prime = M.clone()
        else:
            lambdas = S_np ** 2
            numerator = (lambdas.sum()) ** 2
            denominator = (lambdas ** 2).sum() + 1e-16
            pr = numerator / denominator
            target_rank = int(round(pr))
            target_rank = max(1, min(target_rank, K, len(S_np)))

            retained_energy = np.sum(S_np[:target_rank] ** 2)
            scale_factor = np.sqrt(total_energy / (retained_energy + 1e-16))
            scale_factor = min(scale_factor, 10.0)

            U_m_np = U_np[:, :target_rank]
            U_m = torch.from_numpy(U_m_np).to(torch.float32)
            Z = torch.matmul(R, U_m)

            if use_z_median:
                z_star = standard_median(Z, dim=0).clone()
            else:
                z_star = torch.zeros(Z.shape[1], dtype=Z.dtype, device=Z.device)

            c_tukey = 4.685

            delta = Z - z_star.unsqueeze(0)
            z_row_norms = torch.linalg.vector_norm(delta, dim=1)

            s_coord_vals = []
            for j in range(delta.shape[1]):
                s_j = mad_scale(delta[:, j])
                s_coord_vals.append(max(s_j, 1e-12))
            s_coord = torch.tensor(s_coord_vals, dtype=torch.float32)

            s_global = mad_scale(z_row_norms)
            s_global = max(s_global, 1e-12)

            coord_ratio = torch.abs(delta) / (c_tukey * s_coord + eps)
            w_coord = torch.clamp(1.0 - coord_ratio ** 2, min=0.0) ** 2
            global_ratio = z_row_norms / (c_tukey * s_global + eps)
            w_global = torch.clamp(1.0 - global_ratio ** 2, min=0.0) ** 2
            w_global = w_global.view(-1, 1)
            W = w_coord * w_global

            numerator_w = torch.sum(W * Z, dim=0)
            denom_w = torch.sum(W, dim=0) + eps
            z_star = numerator_w / denom_w

            r_star = torch.matmul(U_m, z_star) * scale_factor

            if (use_matrix_boost and
                    len(original_shape) == 2 and
                    tensor_key is not None and
                    is_linear_or_attention_layer(tensor_key)):
                R_star = r_star.view(original_shape).to(torch.float64)
                U_R, S_R, V_Rt = torch.linalg.svd(R_star, full_matrices=False)
                if S_R.numel() > 0:
                    sigma_max = S_R[0]
                    S_boosted = torch.full_like(S_R, sigma_max)
                    R_boosted = U_R @ torch.diag(S_boosted) @ V_Rt
                    r_star = R_boosted.to(torch.float32).view(-1)

            y_prime = M + r_star
            del V64, U_np, S_np, VT_np, U_m, Z, delta, w_coord, w_global, W

    avg_rms = torch.mean(rms_vals)
    y = y_prime * avg_rms

    # --- Norm restoration (always average norm) ---
    orig_norms = torch.stack([torch.linalg.vector_norm(x) for x in xs])
    target_norm = torch.mean(orig_norms)

    y_norm = torch.linalg.vector_norm(y)
    alpha = (target_norm / (y_norm + eps)).item()
    y = y * alpha

    del xs, us, X, M, R, rms_vals, orig_norms, y_prime, xs_raw, xs_aligned
    return y.view(*original_shape).contiguous().float()


# ---------------------------
# Tensor Loader
# ---------------------------

class SmartModelLoader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.is_sharded = (model_dir / "model.safetensors.index.json").exists()
        if self.is_sharded:
            with open(model_dir / "model.safetensors.index.json", "r", encoding="utf-8") as f:
                index = json.load(f)
            self.weight_map = {normalize_key(k): v for k, v in index["weight_map"].items()}
            self.shard_handles = {}  # cache open shards
        else:
            self.weight_map = {"__SINGLE_FILE__": "model.safetensors"}
            self.single_file = model_dir / "model.safetensors"
            if not self.single_file.exists():
                raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
            self.shard_handles = {}

    def _get_shard_handle(self, shard_name: str):
        if shard_name not in self.shard_handles:
            path = self.model_dir / shard_name
            self.shard_handles[shard_name] = safe_open(str(path), framework="pt", device="cpu")
        return self.shard_handles[shard_name]

    def get_tensor(self, key: str) -> torch.Tensor:
        normalized_key = normalize_key(key)
        if normalized_key not in self.weight_map:
            raise KeyError(f"Tensor {key} (normalized: {normalized_key}) not found in model {self.model_dir}")
        shard_name = self.weight_map[normalized_key]
        if shard_name == "__SINGLE_FILE__":
            with safe_open(str(self.single_file), framework="pt", device="cpu") as f:
                return f.get_tensor(normalized_key)
        else:
            handle = self._get_shard_handle(shard_name)
            return handle.get_tensor(normalized_key)

    def release_shard(self, shard_name: str):
        """Release a specific shard's file handle to free memory."""
        if shard_name in self.shard_handles:
            del self.shard_handles[shard_name]
            gc.collect()  # help release memory

    def close(self):
        self.shard_handles.clear()
        gc.collect()


# ---------------------------
# Tensor Resize
# ---------------------------

def _resize_tensors_to_common_shape(tensors: List[torch.Tensor], target_shape: tuple) -> List[torch.Tensor]:
    if not tensors:
        return tensors
    if all(t.shape == target_shape for t in tensors):
        return tensors

    resized_tensors = []
    for t in tensors:
        if t.shape == target_shape:
            resized_tensors.append(t)
        else:
            new_tensor = torch.zeros(target_shape, dtype=t.dtype, device=t.device)
            slices = tuple(slice(0, min(t.shape[i], target_shape[i])) for i in range(len(target_shape)))
            new_tensor[slices] = t[slices]
            resized_tensors.append(new_tensor)
    return resized_tensors


# ---------------------------
# Main Entry
# ---------------------------

def run_merge(
        model_paths: List[str],
        output_dir: str,
        anchor_index: int = 0,
        config_dir: int = 1,
        use_geometric_median: bool = False,
        use_matrix_boost: bool = True,
        sign_reference_mode: int = 0,
        use_z_median: bool = True,
):
    assert len(model_paths) >= 2, "At least two models required."
    model_dirs = [Path(p) for p in model_paths]
    for p in model_dirs:
        if not p.exists():
            raise FileNotFoundError(f"Model path not found: {p}")

    if not (1 <= config_dir <= len(model_dirs)):
        raise ValueError(f"config_dir must be 1..{len(model_dirs)}")
    config_idx = config_dir - 1

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy config JSONs
    cfg_src = model_dirs[config_idx]
    print(f"📋 Copying config files from: {cfg_src}")
    for json_file in cfg_src.glob("*.json"):
        shutil.copy2(json_file, out_dir)

    # 2. Initialize loaders
    loaders = [SmartModelLoader(d) for d in model_dirs]

    # 3. Build shard-to-tensor map from config model
    config_loader = loaders[config_idx]
    if config_loader.is_sharded:
        shard_to_tensors = {}
        for key, shard in config_loader.weight_map.items():
            shard_to_tensors.setdefault(shard, []).append(key)
        index_src = cfg_src / "model.safetensors.index.json"
        if index_src.exists():
            shutil.copy2(index_src, out_dir / "model.safetensors.index.json")
    else:
        shard_to_tensors = {"model.safetensors": list(config_loader.weight_map.keys())}

    # 4. Prepare cache dir
    cache_dir = out_dir / "_temp_cache"
    cache_dir.mkdir(exist_ok=True)

    # 5. Define merge function
    def merge_func(tensors: List[torch.Tensor], tensor_name: str) -> torch.Tensor:
        return subspace_robust_merge(
            tensors,
            anchor_index=anchor_index,
            use_geometric_median=use_geometric_median,
            use_matrix_boost=use_matrix_boost,
            tensor_key=tensor_name,
            sign_reference_mode=sign_reference_mode,
            use_z_median=use_z_median,
        )

    # 6. Main loop: shard by shard
    sorted_shards = sorted(shard_to_tensors.keys())
    for shard_name in sorted_shards:
        final_shard_path = out_dir / shard_name
        if final_shard_path.exists():
            print(f"⏭️  Skipping completed shard: {shard_name}")
            # Release this shard from all loaders to free memory
            for loader in loaders:
                loader.release_shard(shard_name)
            gc.collect()
            continue

        keys = shard_to_tensors[shard_name]
        shard_cache_dir = cache_dir / shard_name.replace(".safetensors", "").replace(".", "_")
        shard_cache_dir.mkdir(exist_ok=True)

        # --- A. Compute & cache each tensor ---
        print(f"\n📦 Processing shard: {shard_name} ({len(keys)} tensors)")
        pbar = tqdm(keys, desc="Computing tensors", unit="tensor")
        for key in pbar:
            safe_key = key.replace(".", "_").replace("/", "_") + ".pt"
            cache_file = shard_cache_dir / safe_key

            if cache_file.exists():
                continue

            try:
                available_tensors = []
                sources_used = []

                for i, loader in enumerate(loaders):
                    try:
                        t = loader.get_tensor(key)
                        available_tensors.append(t)
                        sources_used.append(i)
                    except KeyError:
                        pass

                if not available_tensors:
                    raise KeyError(f"No model contains tensor {key}")

                ref_idx_in_list = sources_used.index(config_idx)
                ref_tensor = available_tensors[ref_idx_in_list]
                target_shape = ref_tensor.shape

                if len(available_tensors) == 1:
                    merged = ref_tensor.clone()
                else:
                    resized_tensors = _resize_tensors_to_common_shape(available_tensors, target_shape)
                    merged = merge_func(resized_tensors, key)

                # Save in float32 for precision (bfloat16 may lose info in robust merge)
                torch.save(merged.to(torch.float32), cache_file)

                del available_tensors, merged, resized_tensors
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"\n❌ Error during tensor {key}: {e}")
                raise

        pbar.close()

        # --- B. Assemble shard ---
        print(f"   💾 Assembling shard: {shard_name}")
        shard_state_dict = {}
        for key in tqdm(keys, desc="Loading cache", unit="tensor"):
            safe_key = key.replace(".", "_").replace("/", "_") + ".pt"
            cache_file = shard_cache_dir / safe_key
            if not cache_file.exists():
                raise FileNotFoundError(f"Cache missing for {key}")
            # Load as float32 (for precision during merge), then cast to bfloat16 for final output
            tensor_float32 = torch.load(cache_file, map_location="cpu")
            shard_state_dict[key] = tensor_float32.to(torch.bfloat16)

        save_file(shard_state_dict, str(final_shard_path), metadata={"format": "pt"})
        print(f"   ✅ Saved: {final_shard_path}")

        # Clean up cache
        shutil.rmtree(shard_cache_dir)
        del shard_state_dict, tensor_float32
        gc.collect()

        # ✅ CRITICAL: Release this shard's file handles from all loaders
        for loader in loaders:
            loader.release_shard(shard_name)
        gc.collect()

    # Final cleanup
    for loader in loaders:
        loader.close()

    if cache_dir.exists() and not any(cache_dir.iterdir()):
        cache_dir.rmdir()

    print("\n🎉 YOYO Fusion completed successfully!")


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    paths = [
        r"path/to/model_A",
        r"path/to/model_B",
        r"path/to/model_C"
    ]
    out_path = r"path/to/merged_model"

    run_merge(
        model_paths=paths,
        output_dir=out_path,
        anchor_index=1,
        config_dir=1,
        use_geometric_median=False,
        use_matrix_boost=False,
        sign_reference_mode=0,
        use_z_median=True,
    )

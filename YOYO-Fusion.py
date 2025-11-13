import json
import shutil
from pathlib import Path
from typing import List, Dict, Callable

import numpy as np
import torch
from safetensors import torch as st
from tqdm import tqdm


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

@torch.no_grad()
def subspace_robust_merge(
        tensors: List[torch.Tensor],
        eps: float = 1e-8,
        anchor_index: int = 0,
        use_k_minus_one_truncation: bool = True,
        use_geometric_median: bool = False,
) -> torch.Tensor:
    assert len(tensors) >= 2, "Need at least two tensors to merge."
    K = len(tensors)
    shape = tensors[0].shape

    xs = [t.detach().cpu().contiguous().view(-1).float().requires_grad_(False) for t in tensors]
    rms_vals = torch.stack([rms(x) for x in xs])
    rms_vals = torch.clamp(rms_vals, min=1e-6)
    us = [x / (r + eps) for x, r in zip(xs, rms_vals)]

    X = torch.stack(us, dim=0)

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
            if use_k_minus_one_truncation:
                target_rank = min(K - 1, len(S_np))
                use_energy_scaling = True
            else:
                target_rank = min(K, len(S_np))
                use_energy_scaling = False

            if target_rank <= 0:
                y_prime = M.clone()
            else:
                if use_energy_scaling:
                    retained_energy = np.sum(S_np[:target_rank] ** 2)
                    total_energy = np.sum(S_np ** 2)
                    p = retained_energy / (total_energy + 1e-16)
                    scale_factor = 1.0 / (p + 1e-16)
                    scale_factor = np.minimum(scale_factor, 10.0).item()
                else:
                    scale_factor = 1.0

                U_m_np = U_np[:, :target_rank]
                U_m = torch.from_numpy(U_m_np).to(torch.float32)
                Z = torch.matmul(R, U_m)

                s_coord_vals = []
                for j in range(Z.shape[1]):
                    s_j = mad_scale(Z[:, j])
                    s_coord_vals.append(max(s_j, 1e-12))
                s_coord = torch.tensor(s_coord_vals, dtype=torch.float32)

                z_row_norms = torch.linalg.vector_norm(Z, dim=1)
                s_global = mad_scale(z_row_norms)
                s_global = max(s_global, 1e-12)

                c = 4.685
                coord_ratio = torch.abs(Z) / (c * s_coord + eps)
                w_coord = torch.clamp(1.0 - coord_ratio ** 2, min=0.0) ** 2
                global_ratio = z_row_norms / (c * s_global + eps)
                w_global = torch.clamp(1.0 - global_ratio ** 2, min=0.0) ** 2
                w_global = w_global.view(-1, 1)
                W = w_coord * w_global
                numerator = torch.sum(W * Z, dim=0)
                denom = torch.sum(W, dim=0) + eps
                z_star = numerator / denom
                r_star = torch.matmul(U_m, z_star) * scale_factor
                y_prime = M + r_star

                del V64, U_np, S_np, VT_np, U_m, Z, W, numerator, denom, z_star, r_star

    avg_rms = torch.mean(rms_vals)
    y = y_prime * avg_rms
    orig_norms = torch.stack([torch.linalg.vector_norm(x) for x in xs])
    avg_orig_norm = torch.mean(orig_norms)
    y_norm = torch.linalg.vector_norm(y)
    alpha = (avg_orig_norm / (y_norm + eps)).item()
    y = y * alpha
    del xs, us, X, M, R, rms_vals, orig_norms, y_prime
    return y.view(*shape).contiguous().float()


# ---------------------------
# I/O and merge logic
# ---------------------------

def has_index_file(model_dir: Path) -> bool:
    return (model_dir / "model.safetensors.index.json").exists()


def read_index_json(model_dir: Path) -> Dict:
    with open(model_dir / "model.safetensors.index.json", "r", encoding="utf-8") as f:
        return json.load(f)


def collect_json_files(src_dir: Path) -> List[Path]:
    return [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]


def ensure_output_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def copy_json_files(src_dir: Path, out_dir: Path):
    for f in collect_json_files(src_dir):
        shutil.copy2(str(f), str(out_dir / f.name))


def build_weight_map(model_dir: Path) -> Dict[str, str]:
    index = read_index_json(model_dir)
    wm = index.get("weight_map", {})
    if not wm:
        raise ValueError(f"Empty weight_map in index for {model_dir}")
    return wm


def write_shard_tensors(shard_path: Path, tensors: Dict[str, torch.Tensor]):
    out_dict = {k: v.detach().cpu().to(torch.bfloat16).contiguous() for k, v in tensors.items()}
    st.save_file(out_dict, str(shard_path))


# ---------------------------
# Unified merge logic
# ---------------------------

def _validate_tensors(tensors: List[torch.Tensor], key: str):
    """Validate that all tensors have the same shape"""
    shape0 = tensors[0].shape
    for i, t in enumerate(tensors[1:], start=1):
        if t.shape != shape0:
            raise ValueError(f"Shape mismatch for {key}: model0 {shape0} vs model{i} {t.shape}")


def _resize_tensors_to_common_shape(tensors: List[torch.Tensor], key: str) -> List[torch.Tensor]:
    """
    Resize tensors to a common shape by truncating or padding.
    For embedding layers, we typically truncate to the smallest vocab size or pad to largest.
    """
    if len(tensors) < 2:
        return tensors

    # Get all shapes
    shapes = [t.shape for t in tensors]

    # Check if all shapes are the same
    if all(shape == shapes[0] for shape in shapes):
        return tensors

    # If this is an embedding layer (first dim is vocab size), use the config model's vocab size
    if key == "model.embed_tokens.weight":
        # Use the first tensor's shape as reference (config model)
        ref_shape = tensors[0].shape
        resized_tensors = []
        for i, t in enumerate(tensors):
            if t.shape == ref_shape:
                resized_tensors.append(t)
            else:
                # Resize to match reference shape
                new_tensor = torch.zeros(ref_shape, dtype=t.dtype, device=t.device)
                # Copy the common part
                min_rows = min(t.shape[0], ref_shape[0])
                min_cols = min(t.shape[1], ref_shape[1])
                new_tensor[:min_rows, :min_cols] = t[:min_rows, :min_cols]
                resized_tensors.append(new_tensor)
        return resized_tensors

    # For other tensors, find the minimum common dimensions
    min_shape = []
    for dim_idx in range(len(shapes[0])):
        min_size = min(shape[dim_idx] for shape in shapes)
        min_shape.append(min_size)

    min_shape = tuple(min_shape)
    resized_tensors = []
    for i, t in enumerate(tensors):
        if t.shape == min_shape:
            resized_tensors.append(t)
        else:
            # Create a new tensor with minimum shape
            new_tensor = torch.zeros(min_shape, dtype=t.dtype, device=t.device)
            # Copy the common part
            slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(t.shape, min_shape))
            new_tensor[slices] = t[slices]
            resized_tensors.append(new_tensor)

    return resized_tensors


def _merge_tensors(
        tensor_groups: List[Dict[str, torch.Tensor]],
        common_keys: List[str],
        merge_func: Callable,
        desc: str = "Merging tensors"
) -> Dict[str, torch.Tensor]:
    """Generic tensor merging function"""
    merged_tensors = {}

    with tqdm(total=len(common_keys), desc=desc, unit="tensor") as pbar:
        for key in common_keys:
            tensors = [d[key].detach().cpu().float().requires_grad_(False) for d in tensor_groups]

            # Resize tensors if they have different shapes
            try:
                tensors = _resize_tensors_to_common_shape(tensors, key)
            except Exception as e:
                print(f"Warning: Could not resize tensors for {key}: {e}")
                _validate_tensors(tensors, key)  # This will raise error if shapes still don't match

            merged = merge_func(tensors)
            merged_tensors[key] = merged
            del tensors
            pbar.update(1)

    return merged_tensors


def _load_single_file_tensors(model_dirs: List[Path]) -> List[Dict[str, torch.Tensor]]:
    """Load tensors from single-file models"""
    all_tensors_list = []
    for model_dir in model_dirs:
        safetensor_path = model_dir / "model.safetensors"
        if not safetensor_path.exists():
            raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
        data = st.load_file(str(safetensor_path), device="cpu")
        all_tensors_list.append(data)
    return all_tensors_list


def _find_common_keys(tensor_groups: List[Dict[str, torch.Tensor]]) -> List[str]:
    """Find common keys across all tensor groups"""
    common_keys = set(tensor_groups[0].keys())
    for d in tensor_groups[1:]:
        common_keys &= set(d.keys())
    return sorted(common_keys)


def run_single_file_merge(
        model_dirs: List[Path],
        out_dir: Path,
        config_idx: int,
        anchor_index: int,
        use_k_minus_one_truncation: bool,
        use_geometric_median: bool,
):
    ensure_output_dir(out_dir)
    copy_json_files(model_dirs[config_idx], out_dir)

    all_tensors_list = _load_single_file_tensors(model_dirs)
    common_keys = _find_common_keys(all_tensors_list)

    if not common_keys:
        raise ValueError("No common tensors across models.")

    def merge_func(tensors):
        return subspace_robust_merge(
            tensors,
            anchor_index=anchor_index,
            use_k_minus_one_truncation=use_k_minus_one_truncation,
            use_geometric_median=use_geometric_median,
        )

    merged_tensors = _merge_tensors(all_tensors_list, common_keys, merge_func)
    write_shard_tensors(out_dir / "model.safetensors", merged_tensors)
    print(f"Single-file merge complete. Output at: {out_dir}")


def run_sharded_merge(
        model_dirs: List[Path],
        out_dir: Path,
        config_idx: int,
        anchor_index: int,
        use_k_minus_one_truncation: bool,
        use_geometric_median: bool,
):
    ensure_output_dir(out_dir)
    copy_json_files(model_dirs[config_idx], out_dir)

    weight_maps = []
    for d in model_dirs:
        if not has_index_file(d):
            raise ValueError(f"Model {d} missing index file.")
        weight_maps.append(build_weight_map(d))

    common_tensors = sorted(set.intersection(*(set(wm.keys()) for wm in weight_maps)))
    if not common_tensors:
        raise ValueError("No common tensors.")

    config_index = read_index_json(model_dirs[config_idx])
    config_weight_map = config_index["weight_map"]
    shard_to_tensors: Dict[str, List[str]] = {}
    for tname in common_tensors:
        if tname in config_weight_map:
            shard_name = config_weight_map[tname]
            shard_to_tensors.setdefault(shard_name, []).append(tname)

    def merge_func(tensors):
        return subspace_robust_merge(
            tensors,
            anchor_index=anchor_index,
            use_k_minus_one_truncation=use_k_minus_one_truncation,
            use_geometric_median=use_geometric_median,
        )

    for shard_name in sorted(shard_to_tensors.keys()):
        tensor_names = shard_to_tensors[shard_name]

        with tqdm(total=len(tensor_names), desc=f"Shard {shard_name}", unit="tensor") as pbar:
            merged_buffer = {}
            for tname in tensor_names:
                per_model_tensors = []
                try:
                    for wm, mdir in zip(weight_maps, model_dirs):
                        src_shard = wm[tname]
                        data = st.load_file(str(mdir / src_shard), device="cpu")
                        if tname not in data:
                            raise KeyError(f"Tensor {tname} missing in {src_shard} of {mdir}")
                        tensor = data[tname].detach().cpu().float().requires_grad_(False)
                        per_model_tensors.append(tensor)
                        del data

                    # Resize tensors if they have different shapes
                    per_model_tensors = _resize_tensors_to_common_shape(per_model_tensors, tname)
                    merged_tensor = merge_func(per_model_tensors)
                    merged_buffer[tname] = merged_tensor
                    del per_model_tensors
                except Exception as e:
                    if 'per_model_tensors' in locals():
                        del per_model_tensors
                    raise e
                pbar.update(1)

            write_shard_tensors(out_dir / shard_name, merged_buffer)

    filtered_index = dict(config_index)
    filtered_index["weight_map"] = {t: config_weight_map[t] for t in common_tensors if t in config_weight_map}
    with open(out_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(filtered_index, f, ensure_ascii=False, indent=2)

    print(f"Sharded merge complete. Output at: {out_dir}")


def run_mixed_merge(
        model_dirs: List[Path],
        out_dir: Path,
        config_idx: int,
        anchor_index: int,
        use_k_minus_one_truncation: bool,
        use_geometric_median: bool,
):
    """
    Run merge with mixed sharding support. The config_idx model determines the output format.
    """
    ensure_output_dir(out_dir)
    copy_json_files(model_dirs[config_idx], out_dir)

    # Check if config model is sharded
    is_config_sharded = has_index_file(model_dirs[config_idx])

    if is_config_sharded:
        # Output will be sharded based on config model
        config_index = read_index_json(model_dirs[config_idx])
        config_weight_map = config_index["weight_map"]

        # Load all tensors for each model
        all_model_tensors = []
        for i, model_dir in enumerate(model_dirs):
            if has_index_file(model_dir):
                # This model is sharded
                wm = build_weight_map(model_dir)
                model_tensors = {}
                for tensor_name, shard_name in wm.items():
                    data = st.load_file(str(model_dir / shard_name), device="cpu")
                    if tensor_name in data:
                        model_tensors[tensor_name] = data[tensor_name]
            else:
                # This model is single file
                safetensor_path = model_dir / "model.safetensors"
                if not safetensor_path.exists():
                    raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
                model_tensors = st.load_file(str(safetensor_path), device="cpu")

            all_model_tensors.append(model_tensors)

        # Find common tensors across all models
        common_tensors = set(config_weight_map.keys())  # Only tensors from config model
        for model_tensors in all_model_tensors:
            common_tensors &= set(model_tensors.keys())

        common_tensors = sorted(common_tensors)
        if not common_tensors:
            raise ValueError("No common tensors found between models.")

        # Group tensors by shard based on config model
        shard_to_tensors: Dict[str, List[str]] = {}
        for tname in common_tensors:
            shard_name = config_weight_map[tname]
            shard_to_tensors.setdefault(shard_name, []).append(tname)

        def merge_func(tensors):
            return subspace_robust_merge(
                tensors,
                anchor_index=anchor_index,
                use_k_minus_one_truncation=use_k_minus_one_truncation,
                use_geometric_median=use_geometric_median,
            )

        # Process each shard
        for shard_name in sorted(shard_to_tensors.keys()):
            tensor_names = shard_to_tensors[shard_name]

            with tqdm(total=len(tensor_names), desc=f"Shard {shard_name}", unit="tensor") as pbar:
                merged_buffer = {}
                for tname in tensor_names:
                    per_model_tensors = []
                    try:
                        for model_tensors in all_model_tensors:
                            if tname not in model_tensors:
                                raise KeyError(f"Tensor {tname} missing in model")
                            tensor = model_tensors[tname].detach().cpu().float().requires_grad_(False)
                            per_model_tensors.append(tensor)

                        # Resize tensors if they have different shapes
                        per_model_tensors = _resize_tensors_to_common_shape(per_model_tensors, tname)
                        merged_tensor = merge_func(per_model_tensors)
                        merged_buffer[tname] = merged_tensor
                        del per_model_tensors
                    except Exception as e:
                        if 'per_model_tensors' in locals():
                            del per_model_tensors
                        raise e
                    pbar.update(1)

                write_shard_tensors(out_dir / shard_name, merged_buffer)

        # Write index file
        filtered_index = dict(config_index)
        filtered_index["weight_map"] = {t: config_weight_map[t] for t in common_tensors if t in config_weight_map}
        with open(out_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(filtered_index, f, ensure_ascii=False, indent=2)

    else:
        # Output will be single file based on config model
        all_model_tensors = []
        for i, model_dir in enumerate(model_dirs):
            if has_index_file(model_dir):
                # This model is sharded, need to load all shards
                wm = build_weight_map(model_dir)
                model_tensors = {}
                unique_shards = set(wm.values())
                for shard_name in unique_shards:
                    data = st.load_file(str(model_dir / shard_name), device="cpu")
                    model_tensors.update(data)
            else:
                # This model is single file
                safetensor_path = model_dir / "model.safetensors"
                if not safetensor_path.exists():
                    raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
                model_tensors = st.load_file(str(safetensor_path), device="cpu")

            all_model_tensors.append(model_tensors)

        common_keys = _find_common_keys(all_model_tensors)
        if not common_keys:
            raise ValueError("No common tensors across models.")

        def merge_func(tensors):
            return subspace_robust_merge(
                tensors,
                anchor_index=anchor_index,
                use_k_minus_one_truncation=use_k_minus_one_truncation,
                use_geometric_median=use_geometric_median,
            )

        merged_tensors = _merge_tensors(all_model_tensors, common_keys, merge_func)
        write_shard_tensors(out_dir / "model.safetensors", merged_tensors)

    print(f"Mixed merge complete. Output at: {out_dir}")


# ---------------------------
# Main entry
# ---------------------------

def run_merge(
        model_paths: List[str],
        output_dir: str,
        anchor_index: int = 0,
        config_dir: int = 1,
        use_k_minus_one_truncation: bool = True,
        use_geometric_median: bool = False,
):
    assert len(model_paths) >= 2
    model_dirs = [Path(p) for p in model_paths]
    for p in model_dirs:
        if not p.exists():
            raise FileNotFoundError(f"Model path not found: {p}")

    if not (1 <= config_dir <= len(model_dirs)):
        raise ValueError(f"config_dir must be 1..{len(model_dirs)}")
    config_idx = config_dir - 1
    out_dir = Path(output_dir)

    # Check sharding status for each model
    sharding_status = [has_index_file(d) for d in model_dirs]

    # Determine which type of merge to run
    config_is_sharded = sharding_status[config_idx]
    all_same_sharding = all(status == config_is_sharded for status in sharding_status)

    if all_same_sharding:
        # All models have the same sharding format
        if config_is_sharded:
            run_sharded_merge(model_dirs, out_dir,
                              config_idx=config_idx,
                              anchor_index=anchor_index,
                              use_k_minus_one_truncation=use_k_minus_one_truncation,
                              use_geometric_median=use_geometric_median)
        else:
            run_single_file_merge(model_dirs, out_dir,
                                  config_idx=config_idx,
                                  anchor_index=anchor_index,
                                  use_k_minus_one_truncation=use_k_minus_one_truncation,
                                  use_geometric_median=use_geometric_median)
    else:
        # Mixed sharding - use the config model's format
        run_mixed_merge(model_dirs, out_dir,
                        config_idx=config_idx,
                        anchor_index=anchor_index,
                        use_k_minus_one_truncation=use_k_minus_one_truncation,
                        use_geometric_median=use_geometric_median)


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
        anchor_index=0,  # n=0: no anchor; n>=1: use n-th model as anchor
        config_dir=1,    # m>=1: use m-th as config
        use_k_minus_one_truncation=True,   # True: truncation + energy scaling; False: full SVD
        use_geometric_median=True,         # True: use geometric median; False: use standard median
    )

    # The last switches only take effect when anchor_index=0


"""MRI dataset utilities and ImageFolder-like dataset for .pt files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


_COMMON_PT_KEYS: Tuple[str, ...] = (
    "slices",
    "slice",
    "image",
    "images",
    "data",
    "x",
    "pixel_values",
    "tensor",
)


@dataclass(frozen=True)
class PTFileInfo:
    """Metadata inferred from a .pt file without preloading all slices into RAM."""

    num_slices: int
    slice_first_dim: bool
    used_key: Optional[str]


def _is_tensor_like(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return hasattr(value, "shape")


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _select_container(obj: Any, pt_key: str) -> Tuple[Any, Optional[str]]:
    if isinstance(obj, torch.Tensor):
        return obj, None

    if isinstance(obj, dict):
        if pt_key in obj and _is_tensor_like(obj[pt_key]):
            return obj[pt_key], pt_key

        for key in _COMMON_PT_KEYS:
            if key in obj and _is_tensor_like(obj[key]):
                return obj[key], key

        for key, value in obj.items():
            if _is_tensor_like(value):
                return value, str(key)

        raise ValueError("No tensor-like field found in dict object.")

    if _is_tensor_like(obj):
        return obj, None

    raise TypeError(f"Unsupported .pt payload type: {type(obj)}")


def _analyze_container(container: Any) -> Tuple[int, bool]:
    if isinstance(container, (list, tuple)):
        if len(container) == 0:
            raise ValueError("Empty list/tuple container in .pt file.")
        return len(container), True

    tensor = _to_tensor(container)
    if tensor.ndim == 4:
        # N, C, H, W
        return int(tensor.shape[0]), True
    if tensor.ndim == 3:
        # Heuristic:
        # - (C, H, W): first dim small (<= 4), treat as single sample
        # - (N, H, W): first dim larger, treat as slice dimension
        c_or_n, h, w = tensor.shape
        looks_chw = c_or_n <= 4 and h > 8 and w > 8
        if looks_chw:
            return 1, False
        return int(c_or_n), True
    if tensor.ndim == 2:
        # H, W
        return 1, False

    raise ValueError(f"Unsupported tensor shape for MRI sample: {tuple(tensor.shape)}")


def inspect_pt_file(path: str | Path, pt_key: str = "slices") -> PTFileInfo:
    """Inspect one .pt file and infer how many slice-level samples it provides."""
    obj = torch.load(path, map_location="cpu")
    container, used_key = _select_container(obj, pt_key=pt_key)
    num_slices, slice_first_dim = _analyze_container(container)
    return PTFileInfo(num_slices=num_slices, slice_first_dim=slice_first_dim, used_key=used_key)


def _extract_slice(container: Any, slice_index: Optional[int], slice_first_dim: bool) -> torch.Tensor:
    if isinstance(container, (list, tuple)):
        if slice_index is None:
            raise ValueError("slice_index must be provided for sequence container.")
        image = container[slice_index]
    else:
        tensor = _to_tensor(container)
        if slice_first_dim:
            if slice_index is None:
                raise ValueError("slice_index must be provided when slice_first_dim=True.")
            image = tensor[slice_index]
        else:
            image = tensor

    image = _to_tensor(image)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    elif image.ndim == 3:
        # Convert HWC -> CHW if needed.
        if image.shape[-1] <= 4 and image.shape[0] > 8 and image.shape[1] > 8:
            image = image.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected 2D/3D slice tensor, got shape: {tuple(image.shape)}")

    return image.to(dtype=torch.float32).contiguous()


class MRIDataset(Dataset):
    """ImageFolder-like dataset for MRI .pt files.

    Expected root layout:
        root/
          class_a/
            *.pt
          class_b/
            *.pt
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        pt_key: str = "slices",
        extensions: Sequence[str] = (".pt",),
        recursive: bool = True,
        return_metadata: bool = False,
        max_samples: Optional[int] = None,
        target_mode: str = "class_index",
        skip_broken_files: bool = False,
        verbose: bool = False,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.transform = transform
        self.target_transform = target_transform
        self.pt_key = pt_key
        self.extensions = tuple(x.lower() for x in extensions)
        self.recursive = recursive
        self.return_metadata = return_metadata
        self.max_samples = max_samples
        self.target_mode = target_mode
        self.skip_broken_files = skip_broken_files
        self.verbose = verbose

        if self.target_mode not in {"class_index", "class_name"}:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")

        self.classes, self.class_to_idx = self.find_classes(self.root)
        self._records: List[Dict[str, Any]] = []
        self.samples: List[Tuple[str, int]] = []
        self.targets: List[int] = []
        self.skipped_files: List[str] = []

        self._index_dataset()
        self.imgs = self.samples  # torchvision ImageFolder compatibility

    @staticmethod
    def find_classes(directory: Path) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in directory.iterdir() if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Found no class folders in {directory}")
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        return classes, class_to_idx

    def _iter_pt_files(self, class_dir: Path) -> List[Path]:
        if self.recursive:
            files = sorted(path for path in class_dir.rglob("*.pt") if path.is_file())
        else:
            files = sorted(path for path in class_dir.glob("*.pt") if path.is_file())
        return files

    def _append_record(
        self,
        path: Path,
        class_name: str,
        class_index: int,
        slice_first_dim: bool,
        used_key: Optional[str],
        slice_index: Optional[int],
    ) -> bool:
        if self.max_samples is not None and len(self._records) >= int(self.max_samples):
            return False

        sample_id = f"{path}::slice={slice_index}" if slice_index is not None else str(path)
        self._records.append(
            {
                "path": str(path),
                "class_name": class_name,
                "class_index": class_index,
                "slice_first_dim": slice_first_dim,
                "slice_index": slice_index,
                "used_key": used_key,
            }
        )
        self.samples.append((sample_id, class_index))
        self.targets.append(class_index)
        return True

    def _index_dataset(self) -> None:
        for class_name in self.classes:
            class_index = self.class_to_idx[class_name]
            class_dir = self.root / class_name
            pt_files = self._iter_pt_files(class_dir)

            for path in pt_files:
                try:
                    info = inspect_pt_file(path, pt_key=self.pt_key)
                except Exception as exc:
                    if self.skip_broken_files:
                        self.skipped_files.append(str(path))
                        if self.verbose:
                            print(f"[MRIDataset] Skip broken file: {path} ({exc})")
                        continue
                    raise RuntimeError(f"Failed to index {path}: {exc}") from exc

                if info.num_slices == 1:
                    slice_index = 0 if info.slice_first_dim else None
                    did_append = self._append_record(
                        path=path,
                        class_name=class_name,
                        class_index=class_index,
                        slice_first_dim=info.slice_first_dim,
                        used_key=info.used_key,
                        slice_index=slice_index,
                    )
                    if not did_append:
                        return
                else:
                    for slice_index in range(info.num_slices):
                        did_append = self._append_record(
                            path=path,
                            class_name=class_name,
                            class_index=class_index,
                            slice_first_dim=info.slice_first_dim,
                            used_key=info.used_key,
                            slice_index=slice_index,
                        )
                        if not did_append:
                            return

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int):
        record = self._records[index]
        obj = torch.load(record["path"], map_location="cpu")
        container, used_key = _select_container(obj, pt_key=self.pt_key)
        image = _extract_slice(
            container=container,
            slice_index=record["slice_index"],
            slice_first_dim=record["slice_first_dim"],
        )

        if self.transform is not None:
            image = self.transform(image)

        if self.target_mode == "class_name":
            target: Any = record["class_name"]
        else:
            target = record["class_index"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_metadata:
            metadata = {
                "path": record["path"],
                "slice_index": record["slice_index"],
                "class_name": record["class_name"],
                "class_index": record["class_index"],
                "used_key": used_key if used_key is not None else record["used_key"],
            }
            return image, target, metadata

        return image, target

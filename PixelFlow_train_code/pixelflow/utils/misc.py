import os
import random
import numpy as np
import torch

def seed_everything(seed=0, deterministic_ops=True, allow_tf32=False):
    """
    Sets the seed for reproducibility across various libraries and frameworks, and configures PyTorch backend settings.

    Args:
        seed (int): The seed value for random number generation. Default is 0.
        deterministic_ops (bool): Whether to enable deterministic operations in PyTorch.
            Enabling this can make results reproducible at the cost of potential performance degradation. Default is True.
        allow_tf32 (bool): Whether to allow TensorFloat-32 (TF32) precision in PyTorch operations. TF32 can improve performance but may affect reproducibility. Default is False.

    Effects:
        - Seeds Python's random module, NumPy, and PyTorch (CPU and GPU).
        - Sets the environment variable `PYTHONHASHSEED` to the specified seed.
        - Configures PyTorch to use deterministic algorithms if `deterministic_ops` is True.
        - Configures TensorFloat-32 precision based on `allow_tf32`.
        - Issues warnings if configurations may impact reproducibility.

    Notes:
        - Setting `torch.backends.cudnn.deterministic` to False allows nondeterministic operations, which may introduce variability.
        - Allowing TF32 (`allow_tf32=True`) may lead to non-reproducible results, especially in matrix operations.
    """
    # Seed standard random number generators
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Seed PyTorch random number generators
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Configure deterministic operations
    if deterministic_ops:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        torch.backends.cudnn.deterministic = False
        print("WARNING: torch.backends.cudnn.deterministic is set to False, reproducibility is not guaranteed.")

    # Configure TensorFloat-32 precision
    if allow_tf32:
        print("WARNING: TensorFloat-32 (TF32) is enabled; reproducibility is not guaranteed.")

    torch.backends.cudnn.allow_tf32 = allow_tf32  # Default True in PyTorch 2.6.0
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Default False in PyTorch 2.6.0

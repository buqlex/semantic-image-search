import torch


def get_accelerator() -> torch.device:
    """Utility for automatically determining an available accelerator.
    Falls back to the CPU.

    Returns
    -------
    torch.device
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device() -> str:
    if torch.cuda.is_available():
        return "gpu"
    else:
        return "cpu"

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {get_accelerator()}")

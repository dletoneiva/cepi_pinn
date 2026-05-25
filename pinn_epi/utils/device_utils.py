import torch

def get_device():
    """Get the best available device (CUDA if available, otherwise CPU).
    
    Returns:
        torch.device: The appropriate device for tensor operations
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    return device

def move_to_device(obj, device):
    """Move a tensor or model to the specified device.
    
    Args:
        obj: Tensor or model to move
        device: Target device
        
    Returns:
        Object moved to the specified device
    """
    if hasattr(obj, 'to'):
        return obj.to(device)
    return obj

# Initialize device when module is imported
DEVICE = get_device()

import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Print the number of available GPUs
    print(torch.cuda.device_count(), "GPU(s) available")
    # Print the name of the current GPU
    print("Current GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Make sure you have a GPU and have installed the necessary drivers.")

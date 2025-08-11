import torch
import gc
import os

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Currently Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Currently Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
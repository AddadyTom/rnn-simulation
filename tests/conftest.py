import torch

# Small networks run fastest single-threaded and this avoids oversubscribing CPUs
# when tests run next to a training job.
torch.set_num_threads(1)

# upload_pretrained_model.py
from clearml import Task
import os
import torch # Assuming it's a PyTorch model

# Initialize a dummy task just for uploading the model
# Set the task type to 'optimizer' or 'data_processing' to indicate it's not a training run
task = Task.init(project_name='TP601375_DiffusionDenoiser',
                 task_name='Upload Pretrained 256x256_uncond Model (initial)',
                 task_type=Task.TaskTypes.optimizer # or data_processing, utility, etc.
                )

# --- Load your pre-trained model locally ---
local_model_path = './models/256x256_diffusion_uncond.pt' # <--- CHANGE THIS
"""
# For demonstration, create a dummy model file
if not os.path.exists(local_model_path):
    dummy_model = torch.nn.Linear(10, 1)
    torch.save(dummy_model.state_dict(), local_model_path)
    print(f"Dummy model saved to {local_model_path}")
"""
# --- Upload it to ClearML as an artifact ---
# Give it a meaningful artifact name, e.g., 'my_base_diffusion_model'
task.upload_artifact(name='pretrained_256x256_diffusion_uncond', artifact_object=local_model_path)

print(f"Pre-trained model '{local_model_path}' uploaded to ClearML as artifact 'my_base_diffusion_model'.")
print(f"Task ID for this upload: {task.id}")

# The task will automatically finalize when the script exits
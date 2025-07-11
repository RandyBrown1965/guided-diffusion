# upload_pretrained_model.py
from clearml import Task

# Initialize a dummy task just for uploading the model
# Set the task type to 'optimizer' or 'data_processing' to indicate it's not a training run
task = Task.init(project_name='TP601375_DiffusionDenoiser',
                 task_name='Upload Pretrained 256x256_uncond Model (initial)',
                 task_type=Task.TaskTypes.optimizer # or data_processing, utility, etc.
                )

# --- Load your pre-trained model locally ---
local_model_path = './models/256x256_diffusion_uncond.pt' # <--- CHANGE THIS
# --- Upload it to ClearML as an artifact ---
# Give it a meaningful artifact name, e.g., 'my_base_diffusion_model'
artifact_name = 'pretrained_256x256_diffusion_uncond'
task.upload_artifact(name=artifact_name, artifact_object=local_model_path)

print(f"Pre-trained model '{local_model_path}' uploaded to ClearML as artifact '{artifact_name}'.")
print(f"Task ID for this upload: {task.id}")

# The task will automatically finalize when the script exits
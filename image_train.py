"""
Train a diffusion model on images.
"""

import argparse
from clearml import Task, Dataset
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import torch.distributed as dist
import torch as th


def main():
    args = create_argparser().parse_args()
    print("\nARGS=", args,"\n")

    dist_util.setup_dist()
    logger.configure()

    task = Task.init(project_name='TP601375_DiffusionDenoiser', 
                 task_name='TP602603_ImageGeneratorTraining', 
                 output_uri='https://files.clearml.thefoundry.co.uk')
    task.upload_artifact('summaries', artifact_object='./clearml_summary') # Access to summary folder or .zip file
    task.set_packages('requirements.txt')
    #task.connect_configuration('./configs/config_train_generatore_size256_channels256.yaml')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
  
    # Load pre-trained model
    upload_task_id_of_pretrained_model = '1c7287c02b4344f08e15f32858d6a582' # e.g., 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6'
    artifact_name_of_pretrained_model = 'pretrained_256x256_diffusion_uncond' # The name you gave in task.upload_artifact()
    logger.log(f"Attempting to retrieve pre-trained model artifact '{artifact_name_of_pretrained_model}' from task {upload_task_id_of_pretrained_model}...")
    # Get the task that uploaded the artifact
    upload_task = Task.get_task(task_id=upload_task_id_of_pretrained_model)
    # Get the artifact object
    pretrained_model_artifact = upload_task.artifacts[artifact_name_of_pretrained_model]
    logger.log("PRETRAINED MODEL ARTIFACT=", pretrained_model_artifact)
    # Download the artifact file to a temporary local path
    local_pretrained_model_path = pretrained_model_artifact.get_local_copy()
    model_state_dict = th.load(local_pretrained_model_path, map_location='cpu') # Load state dict
    model.load_state_dict(model_state_dict)
    logger.log("Pre-trained model loaded into architecture successfully!")
    model.to(dist_util.dev())
    logger.log("Model loaded to", dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    dataset = Dataset.get(dataset_name = "Imagenet256_clean_for_TP602603")
    dataset_path = dataset.get_local_copy()
    dataset_meta = dataset.get_metadata()

    data = load_data(
        #data_dir=args.data_dir,
        data_dir = dataset_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()
    task.close()
    dist.destroy_process_group()

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

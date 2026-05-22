"""This script demonstrates how to train ACT Policy on a real-world dataset."""

import argparse
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_training_state,
    update_last_checkpoint,
)


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


def save_checkpoint(
    run_dir: Path,
    total_steps: int,
    step: int,
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    optimizer: torch.optim.Optimizer,
) -> Path:
    """与 `lerobot_train` 相同布局：`checkpoints/<step>/pretrained_model` + `training_state`。"""
    checkpoint_dir = get_step_checkpoint_dir(run_dir, total_steps, step)
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(pretrained_dir)
    preprocessor.save_pretrained(pretrained_dir)
    postprocessor.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, optimizer, scheduler=None)
    update_last_checkpoint(checkpoint_dir)
    return checkpoint_dir


def resolve_resume_paths(resume: Path) -> tuple[Path, Path]:
    """返回 (pretrained_model 目录, checkpoint 根目录含 training_state)。

    允许传入 `.../checkpoints/000600/pretrained_model` 或 `.../checkpoints/000600`。
    """
    resume = resume.expanduser().resolve()
    pm = PRETRAINED_MODEL_DIR
    if resume.name == pm and resume.is_dir():
        pretrained_dir = resume
        checkpoint_dir = resume.parent
    elif (resume / pm).is_dir():
        pretrained_dir = resume / pm
        checkpoint_dir = resume
    else:
        raise FileNotFoundError(
            f"断点路径无效: {resume}（需要存在 `{pm}` 子目录，或直接指向该目录）"
        )
    return pretrained_dir, checkpoint_dir


def main(resume_pretrained: Path | None = None):
    output_directory = Path("outputs/act_5_12")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device("cuda")  # or "cuda" or "cpu"

    # Local dataset root: <repo>/dataset/data_5_9 (must contain meta/, data/, etc.)
    repo_id = "data_5_9"
    dataset_root = Path(__file__).resolve().parents[3] / "dataset" / repo_id

    # This specifies the inputs the model will be expecting and the outputs it will produce
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    resume_checkpoint_dir: Path | None = None
    if resume_pretrained is not None:
        pretrained_dir, resume_checkpoint_dir = resolve_resume_paths(resume_pretrained)
        policy = ACTPolicy.from_pretrained(pretrained_dir, local_files_only=True)
        cfg = policy.config
        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=str(pretrained_dir),
        )
        print(f"Resuming policy & processors from {pretrained_dir}")
    else:
        # Align with `device` below so PreTrainedConfig does not warn about `device=None`.
        cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            device=str(device),
        )
        policy = ACTPolicy(cfg)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    # To perform action chunking, ACT expects a given number of actions as targets
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }

    # add image features if they are present
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    # Use pyav here: torchcodec needs matching FFmpeg libs on the system (often fails on conda).
    # Note: this script does not read CLI flags like `lerobot-train --dataset.video_backend=...`.
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    # Create the optimizer and dataloader for offline training
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    step = 0
    if resume_checkpoint_dir is not None:
        step, optimizer, _ = load_training_state(resume_checkpoint_dir, optimizer, scheduler=None)
        print(f"Resuming optimizer & RNG from {resume_checkpoint_dir} (training_step={step})")

    batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Number of training steps and logging frequency
    training_steps = 15000
    log_freq = 40
    saving_steps = 600

    # Run training loop
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if saving_steps > 0 and step % saving_steps == 0:
                ckpt = save_checkpoint(
                    output_directory,
                    training_steps,
                    step,
                    policy,
                    preprocessor,
                    postprocessor,
                    optimizer,
                )
                print(f"Saved checkpoint at step {step} -> {ckpt}")
            if step >= training_steps:
                done = True
                break

    # Save the policy checkpoint, alongside the pre/post processors
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    # Save all assets to the Hub
    # policy.push_to_hub("<user>/robot_learning_tutorial_act")
    # preprocessor.push_to_hub("<user>/robot_learning_tutorial_act")
    # postprocessor.push_to_hub("<user>/robot_learning_tutorial_act")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CHECKPOINT_OR_PRETRAINED",
        help=(
            "从断点继续：传 `outputs/.../checkpoints/000600/pretrained_model` "
            "或 `outputs/.../checkpoints/000600`（需含 training_state/ 与 pretrained_model/）"
        ),
    )
    args = parser.parse_args()
    main(resume_pretrained=args.resume)

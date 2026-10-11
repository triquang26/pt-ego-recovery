from __future__ import annotations

import dataclasses
import pathlib

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import weight_loaders
from typing_extensions import override

from ptego.policy.transforms import AdvantagePrompt, ManiSkillInputs, ManiSkillOutputs

REPO = pathlib.Path(__file__).resolve().parents[3]
PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"
ASSETS = "pi05_maniskill_acbc_lora"


@dataclasses.dataclass(frozen=True)
class LeRobotManiSkillDataConfig(_config.DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "advantage": "advantage",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data = _transforms.Group(
            inputs=[AdvantagePrompt(), ManiSkillInputs(model_type=model_config.model_type)],
            outputs=[ManiSkillOutputs()],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data,
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


def pi05_lora(action_horizon: int) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_horizon=action_horizon,
        discrete_state_input=True,
    )


def acbc_config(name: str, repo_id: str, steps: int, batch_size: int = 4, action_horizon: int = 20) -> _config.TrainConfig:
    model = pi05_lora(action_horizon)
    return _config.TrainConfig(
        name=name,
        project_name="pt-ego-recovery",
        model=model,
        data=LeRobotManiSkillDataConfig(
            repo_id=repo_id,
            assets=_config.AssetsConfig(assets_dir=str(REPO / "assets" / ASSETS)),
            base_config=_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(PI05_BASE),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        batch_size=batch_size,
        num_train_steps=steps,
        save_interval=1000,
        keep_period=5000,
        assets_base_dir=str(REPO / "assets"),
        checkpoint_base_dir=str(REPO / "checkpoints"),
        wandb_enabled=False,
    )


CONFIGS = [
    acbc_config("pi05_maniskill_acbc_lora", "local/stackcube_initial", steps=10_000),
    acbc_config("pi05_maniskill_smoke", "local/stackcube_initial", steps=20, batch_size=2),
]


def register() -> None:
    for config in CONFIGS:
        _config._CONFIGS_DICT[config.name] = config

from __future__ import annotations

from pathlib import Path

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.shared import nnx_utils
from openpi.shared import normalize as _normalize
from openpi.training import weight_loaders

from ptego.store import Store

PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"
HORIZON = 20
ACTION_DIM = 14
SLOTS = {"base_0_rgb": "cam_high", "left_wrist_0_rgb": "cam_left_wrist", "right_wrist_0_rgb": "cam_right_wrist"}


def model_config(horizon: int = HORIZON) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_horizon=horizon,
        discrete_state_input=True,
    )


def trainable(config: pi0_config.Pi0Config) -> nnx.filterlib.Filter:
    return nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))


def norm_stats(store: Store) -> dict[str, _normalize.NormStats]:
    path = store.root / "norm_stats.json"
    if path.exists():
        return _normalize.load(store.root)
    stats = {"state": _normalize.RunningStats(), "actions": _normalize.RunningStats()}
    for episode in store.select(split="train"):
        stats["state"].update(store.states(episode))
        stats["actions"].update(store.actions(episode))
    result = {k: v.get_statistics() for k, v in stats.items()}
    _normalize.save(store.root, result)
    return result


def prompt_with(prompt: str, advantage: bool | None) -> str:
    if advantage is None:
        return prompt
    return f"{prompt} Advantage: {'positive' if advantage else 'negative'}."


class Adapter:
    def __init__(self, stats: dict[str, _normalize.NormStats], config: pi0_config.Pi0Config):
        self.normalize = _transforms.Normalize(stats, use_quantiles=True)
        self.tokenize = _transforms.TokenizePrompt(
            _tokenizer.PaligemmaTokenizer(config.max_token_len), discrete_state_input=True
        )
        self.pad = _transforms.PadStatesAndActions(config.action_dim)

    def __call__(self, sample: dict, advantage: bool | None = None) -> dict:
        data = {"state": sample["state"], "prompt": prompt_with(sample["prompt"], advantage)}
        if "actions" in sample:
            data["actions"] = sample["actions"]
        data = self.pad(self.tokenize(self.normalize(data)))
        data["image"] = {slot: sample["images"][camera] for slot, camera in SLOTS.items()}
        data["image_mask"] = {slot: np.True_ for slot in SLOTS}
        return data

    @staticmethod
    def collate(items: list[dict]) -> tuple[_model.Observation, np.ndarray | None]:
        batch = jax.tree.map(lambda *x: np.stack(x), *items)
        actions = batch.pop("actions", None)
        return _model.Observation.from_dict(batch), actions


def base_params(shape: nnx.State) -> dict:
    loaded = weight_loaders.CheckpointWeightLoader(PI05_BASE).load(shape.to_pure_dict())
    return {k: v for k, v in traverse_util.flatten_dict(loaded).items() if not isinstance(v, jax.ShapeDtypeStruct)}


def save_lora(params: nnx.State, config: pi0_config.Pi0Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = traverse_util.flatten_dict(params.filter(trainable(config)).to_pure_dict(), sep="/")
    np.savez(path, **{k: np.asarray(jax.device_get(v)) for k, v in flat.items()})


def load_lora(path: Path) -> dict:
    with np.load(path) as data:
        return {tuple(k.split("/")): data[k] for k in data.files}


def create(config: pi0_config.Pi0Config, rng: jax.Array, lora: Path | None) -> nnx.State:
    def init(rng: jax.Array, partial: dict) -> nnx.State:
        model = config.create(rng)
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(partial)
        state = nnx.state(nnx.merge(graphdef, state))
        return nnx_utils.state_map(state, config.get_freeze_filter(), lambda p: p.replace(p.value.astype(jnp.bfloat16)))

    shape = jax.eval_shape(lambda r: nnx.state(config.create(r)), rng)
    flat = base_params(shape)
    if lora is not None:
        flat.update(load_lora(lora))
    return jax.jit(init, donate_argnums=1)(rng, traverse_util.unflatten_dict(flat))


def graphdef(config: pi0_config.Pi0Config) -> nnx.GraphDef:
    return nnx.graphdef(nnx.eval_shape(lambda: config.create(jax.random.key(0))))


class Sampler:
    def __init__(self, config: pi0_config.Pi0Config, params: nnx.State, steps: int = 10):
        model = nnx.merge(graphdef(config), params)
        model.eval()
        self.steps = steps
        self._sample = nnx_utils.module_jit(model.sample_actions)

    def __call__(self, rng: jax.Array, observation: _model.Observation, samples: int) -> np.ndarray:
        repeated = jax.tree.map(lambda x: jnp.repeat(x, samples, axis=0), observation)
        actions = self._sample(rng, repeated, num_steps=self.steps)
        actions = np.asarray(actions)[..., :ACTION_DIM]
        return actions.reshape(-1, samples, *actions.shape[1:])

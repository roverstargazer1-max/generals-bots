"""Shared helpers for experimental PPO training and evaluation."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents._heuristic_logic import HEURISTIC_NAMES, heuristic_action
from generals.agents.ppo_policy_agent import (
    greedy_policy_action,
    index_to_action,
    normalize_action,
    obs_to_array,
    sampled_policy_action,
)
from generals.core import game
from generals.core.action import compute_valid_move_mask
from generals.core.grid import generate_grid

TEACHER_NAMES = ("expander-soft",) + HEURISTIC_NAMES
TEACHER_NAME_TO_ID = {name: idx for idx, name in enumerate(TEACHER_NAMES)}
OPPONENT_NAMES = ("random",) + HEURISTIC_NAMES
OPPONENT_NAME_TO_ID = {name: idx for idx, name in enumerate(OPPONENT_NAMES)}


def initialize_policy_network(network_cls, key, grid_size, init_model_path=None):
    """Create a policy network, optionally loading weights from an .eqx checkpoint."""
    network = network_cls(key, grid_size=grid_size)
    if init_model_path is None:
        return network

    path = Path(init_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Initial model checkpoint not found: {path}")
    try:
        return eqx.tree_deserialise_leaves(path, network)
    except Exception as exc:
        raise ValueError(f"Failed to load initial checkpoint for grid_size={grid_size}: {path}") from exc


def make_simple_general_grid(key, grid_size):
    """Create an empty square grid with two random generals."""
    grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
    idx = jrandom.choice(key, grid_size * grid_size, shape=(2,), replace=False)
    pos_a = (idx[0] // grid_size, idx[0] % grid_size)
    pos_b = (idx[1] // grid_size, idx[1] % grid_size)
    return grid.at[pos_a].set(1).at[pos_b].set(2)


def make_grids(
    key,
    count,
    grid_size,
    map_generator,
    mountain_density_range,
    num_cities_range,
    min_generals_distance,
    max_generals_distance,
    castle_val_range,
):
    """Generate a batch of simple or generated grids."""
    keys = jrandom.split(key, count)

    if map_generator == "simple":
        return jax.vmap(lambda k: make_simple_general_grid(k, grid_size))(keys)

    return jax.vmap(
        lambda k: generate_grid(
            k,
            grid_dims=(grid_size, grid_size),
            pad_to=grid_size,
            mountain_density_range=mountain_density_range,
            num_cities_range=num_cities_range,
            min_generals_distance=min_generals_distance,
            max_generals_distance=max_generals_distance,
            castle_val_range=castle_val_range,
        )
    )(keys)


def make_state_pool(
    key,
    pool_size,
    grid_size,
    map_generator,
    mountain_density_range,
    num_cities_range,
    min_generals_distance,
    max_generals_distance,
    castle_val_range,
):
    """Generate a reusable pool of initial states for auto-reset."""
    grids = make_grids(
        key,
        pool_size,
        grid_size,
        map_generator,
        mountain_density_range,
        num_cities_range,
        min_generals_distance,
        max_generals_distance,
        castle_val_range,
    )
    return jax.vmap(game.create_initial_state)(grids)


def action_to_index(action, grid_size):
    """Encode an action as the flattened policy index used by PolicyValueNetwork."""
    action = normalize_action(action)
    is_pass, row, col, direction, is_half = action
    encoded_dir = jnp.where(is_pass > 0, 8, jnp.where(is_half > 0, direction + 4, direction))
    return encoded_dir * grid_size * grid_size + row * grid_size + col


def action_to_target_probs(action, grid_size):
    """Encode one teacher action as a one-hot policy target."""
    grid_cells = grid_size * grid_size
    index = action_to_index(action, grid_size)
    return jax.nn.one_hot(index, 9 * grid_cells, dtype=jnp.float32)


def opponent_action(opponent_id, key, obs, random_action_fn):
    """Dispatch a random or heuristic opponent action."""
    return jax.lax.cond(
        opponent_id == 0,
        lambda _: random_action_fn(key, obs),
        lambda _: heuristic_action(opponent_id - 1, key, obs),
        None,
    )


def expander_target_probs(obs):
    """Return the stochastic Expander target distribution over policy indices."""
    armies = obs.armies
    owned_cells = obs.owned_cells
    opponent_cells = obs.opponent_cells
    neutral_cells = obs.neutral_cells
    valid_mask = compute_valid_move_mask(armies, owned_cells, obs.mountains)
    grid_size = armies.shape[-1]
    grid_cells = grid_size * grid_size

    i_idx = jnp.arange(grid_size)[:, None, None]
    j_idx = jnp.arange(grid_size)[None, :, None]
    directions = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)
    dest_i = jnp.clip(i_idx + directions[None, None, :, 0], 0, grid_size - 1)
    dest_j = jnp.clip(j_idx + directions[None, None, :, 1], 0, grid_size - 1)

    source_armies = armies[:, :, None]
    dest_armies = armies[dest_i, dest_j]
    is_opponent = opponent_cells[dest_i, dest_j]
    is_neutral = neutral_cells[dest_i, dest_j]
    is_owned = owned_cells[dest_i, dest_j]

    can_capture = source_armies > dest_armies + 1
    is_expansion = ~is_owned & (is_opponent | is_neutral)
    opponent_multiplier = jnp.where(is_opponent, 2.0, 1.0)
    scores = source_armies.astype(jnp.float32)
    scores = jnp.where(is_expansion & can_capture, scores * 10.0 * opponent_multiplier, scores)
    scores = jnp.where(valid_mask & can_capture, scores, 0.0)

    score_sum = jnp.sum(scores)
    num_valid = jnp.sum(valid_mask)
    move_probs = jnp.where(
        score_sum > 0,
        scores / score_sum,
        valid_mask.astype(jnp.float32) / jnp.maximum(num_valid, 1),
    )
    move_probs = jnp.where(num_valid > 0, move_probs, jnp.zeros_like(move_probs))

    target = jnp.zeros(9 * grid_cells, dtype=jnp.float32)
    flat_move_probs = jnp.transpose(move_probs, (2, 0, 1)).reshape(4 * grid_cells)
    target = target.at[: 4 * grid_cells].set(flat_move_probs)
    target = target.at[8 * grid_cells].set(jnp.where(num_valid == 0, 1.0, 0.0))
    return target

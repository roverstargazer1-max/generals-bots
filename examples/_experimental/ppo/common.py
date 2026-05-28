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


def parse_name_pool(value, name_to_id, label):
    """Parse a comma-separated pool of known strategy names."""
    raw_names = value.split(",") if isinstance(value, str) else list(value)
    names = tuple(name.strip() for name in raw_names if name.strip())
    if not names:
        raise ValueError(f"{label} pool cannot be empty")

    unknown = [name for name in names if name not in name_to_id]
    if unknown:
        known = ", ".join(name_to_id)
        raise ValueError(f"Unknown {label} pool item(s): {', '.join(unknown)}. Available: {known}")

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label} pool item(s): {', '.join(duplicates)}")

    ids = jnp.array([name_to_id[name] for name in names], dtype=jnp.int32)
    return names, ids


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


def resolve_min_generals_distance(grid_size, min_generals_distance):
    """Return the explicit or default minimum distance between generals."""
    if min_generals_distance is None:
        return max(3, grid_size // 2)
    return min_generals_distance


def validate_training_args(parser, args, num_envs=None, require_pool_size=True):
    """Validate common grid, generated-map, and pool-size arguments."""
    env_count = args.num_envs if num_envs is None and hasattr(args, "num_envs") else num_envs
    if args.grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if require_pool_size and env_count is not None and args.pool_size < env_count:
        parser.error("--pool-size must be at least num_envs")
    if hasattr(args, "mountain_density_min") and not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if hasattr(args, "num_cities_min") and not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if hasattr(args, "city_army_min") and args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")


def random_action(key, obs):
    """Random valid action."""
    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    valid = jnp.argwhere(mask, size=mask.size, fill_value=-1)
    num_valid = jnp.sum(jnp.all(valid >= 0, axis=-1))

    k1, k2 = jrandom.split(key)
    idx = jrandom.randint(k1, (), 0, jnp.maximum(num_valid, 1))
    move = jnp.where(
        num_valid > 0,
        valid[idx],
        jnp.array([0, 0, 0], dtype=jnp.int32),
    )
    should_pass = num_valid == 0
    is_half = jrandom.randint(k2, (), 0, 2)

    return jnp.array([should_pass, move[0], move[1], move[2], is_half], dtype=jnp.int32)


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


def make_initial_states(pool, num_envs):
    """Take initial states from the pool and spread future reset indices."""
    states = jax.tree.map(lambda x: x[:num_envs], pool)
    pool_size = pool.armies.shape[0]
    pool_idx = (jnp.arange(num_envs, dtype=jnp.int32) + num_envs) % pool_size
    return states._replace(pool_idx=pool_idx)


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


def teacher_action_targets(teacher_id, key, obs):
    """Dispatch a behavior-cloning teacher and return action, target distribution, and sampled index."""
    grid_size = obs.armies.shape[-1]

    def collect_soft(_):
        target_probs = expander_target_probs(obs)
        sampled_index = jrandom.categorical(key, jnp.log(target_probs + 1e-8))
        return index_to_action(sampled_index, grid_size), target_probs, sampled_index

    def collect_hard(_):
        action = heuristic_action(teacher_id - 1, key, obs)
        index = action_to_index(action, grid_size)
        target_probs = action_to_target_probs(action, grid_size)
        return action, target_probs, index

    return jax.lax.cond(teacher_id == 0, collect_soft, collect_hard, None)


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

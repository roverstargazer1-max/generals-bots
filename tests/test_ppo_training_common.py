import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from argparse import ArgumentParser, Namespace

from examples._experimental.ppo.common import (
    OPPONENT_NAME_TO_ID,
    TEACHER_NAME_TO_ID,
    initialize_policy_network,
    make_initial_states,
    make_state_pool,
    parse_name_pool,
    resolve_min_generals_distance,
    validate_training_args,
)
from examples._experimental.ppo.behavior_clone import collect_teacher_batch
from examples._experimental.ppo.train import rollout_step
from generals.core import game
from generals.agents.ppo_policy_agent import PolicyValueNetwork


def test_initialize_policy_network_random_initializes_without_checkpoint():
    network = initialize_policy_network(PolicyValueNetwork, jrandom.PRNGKey(0), 4)

    assert isinstance(network, PolicyValueNetwork)


def test_initialize_policy_network_loads_checkpoint(tmp_path):
    model_path = tmp_path / "policy.eqx"
    expected = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4)
    eqx.tree_serialise_leaves(model_path, expected)

    loaded = initialize_policy_network(PolicyValueNetwork, jrandom.PRNGKey(1), 4, model_path)

    expected_params = eqx.filter(expected, eqx.is_inexact_array)
    loaded_params = eqx.filter(loaded, eqx.is_inexact_array)
    assert all((a == b).all() for a, b in zip(jax_leaves(expected_params), jax_leaves(loaded_params), strict=True))


def test_initialize_policy_network_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Initial model checkpoint"):
        initialize_policy_network(PolicyValueNetwork, jrandom.PRNGKey(0), 4, tmp_path / "missing.eqx")


def test_initialize_policy_network_rejects_grid_size_mismatch(tmp_path):
    model_path = tmp_path / "policy-4x4.eqx"
    network = PolicyValueNetwork(jrandom.PRNGKey(0), grid_size=4)
    eqx.tree_serialise_leaves(model_path, network)

    with pytest.raises(ValueError, match="grid_size=5"):
        initialize_policy_network(PolicyValueNetwork, jrandom.PRNGKey(1), 5, model_path)


def jax_leaves(tree):
    import jax

    return jax.tree.leaves(tree)


def test_resolve_min_generals_distance_defaults_from_grid_size():
    assert resolve_min_generals_distance(8, None) == 4
    assert resolve_min_generals_distance(4, None) == 3
    assert resolve_min_generals_distance(8, 5) == 5


def test_validate_training_args_rejects_invalid_pool_size():
    args = Namespace(
        grid_size=8,
        num_envs=16,
        pool_size=8,
        mountain_density_min=0.12,
        mountain_density_max=0.22,
        num_cities_min=4,
        num_cities_max=8,
        city_army_min=40,
        city_army_max=51,
    )

    with pytest.raises(SystemExit):
        validate_training_args(ArgumentParser(), args)


def test_make_initial_states_spreads_pool_indices():
    grid = game.create_initial_state(jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2))
    pool = jax.tree.map(lambda x: jnp.stack([x, x, x, x]), grid)

    states = make_initial_states(pool, 2)

    assert states.armies.shape == (2, 4, 4)
    assert states.pool_idx.tolist() == [2, 3]


def test_parse_name_pool_returns_unique_ids():
    names, ids = parse_name_pool("expander-soft, balanced", TEACHER_NAME_TO_ID, "teacher")

    assert names == ("expander-soft", "balanced")
    assert ids.tolist() == [TEACHER_NAME_TO_ID["expander-soft"], TEACHER_NAME_TO_ID["balanced"]]


def test_parse_name_pool_rejects_invalid_values():
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_name_pool("", TEACHER_NAME_TO_ID, "teacher")
    with pytest.raises(ValueError, match="Unknown teacher"):
        parse_name_pool("missing", TEACHER_NAME_TO_ID, "teacher")
    with pytest.raises(ValueError, match="Duplicate opponent"):
        parse_name_pool("random,random", OPPONENT_NAME_TO_ID, "opponent")


def test_collect_teacher_batch_accepts_teacher_and_opponent_pools():
    key = jrandom.PRNGKey(0)
    pool = make_state_pool(
        key,
        4,
        4,
        "simple",
        (0.0, 0.0),
        (2, 2),
        3,
        None,
        (40, 41),
    )
    states = make_initial_states(pool, 2)
    teacher_ids = jnp.array(
        [TEACHER_NAME_TO_ID["expander-soft"], TEACHER_NAME_TO_ID["expander"]],
        dtype=jnp.int32,
    )
    opponent_ids = jnp.array(
        [OPPONENT_NAME_TO_ID["random"], OPPONENT_NAME_TO_ID["balanced"]],
        dtype=jnp.int32,
    )

    next_states, (obs, masks, targets, indices, sampled_teacher_ids, dones, winners), _ = collect_teacher_batch(
        states,
        pool,
        key,
        2,
        20,
        teacher_ids,
        opponent_ids,
    )

    jax.block_until_ready(obs)
    assert next_states.armies.shape == (2, 4, 4)
    assert obs.shape[:2] == (2, 2)
    assert masks.shape[:2] == (2, 2)
    assert targets.shape[:2] == (2, 2)
    assert indices.shape == (2, 2)
    assert sampled_teacher_ids.shape == (2, 2)
    assert dones.shape == (2, 2)
    assert winners.shape == (2, 2)


def test_rollout_step_accepts_opponent_pool():
    key = jrandom.PRNGKey(1)
    network = PolicyValueNetwork(key, grid_size=4)
    pool = make_state_pool(
        key,
        4,
        4,
        "simple",
        (0.0, 0.0),
        (2, 2),
        3,
        None,
        (40, 41),
    )
    states = make_initial_states(pool, 2)
    opponent_ids = jnp.array(
        [OPPONENT_NAME_TO_ID["random"], OPPONENT_NAME_TO_ID["expander"]],
        dtype=jnp.int32,
    )

    next_states, data, _ = rollout_step(states, pool, network, key, 20, opponent_ids)
    obs, masks, actions, logprobs, values, rewards, dones, infos = data

    jax.block_until_ready(next_states.armies)
    assert next_states.armies.shape == (2, 4, 4)
    assert obs.shape[:2] == (2, 9)
    assert masks.shape == (2, 4, 4, 4)
    assert actions.shape == (2, 5)
    assert logprobs.shape == (2,)
    assert values.shape == (2,)
    assert rewards.shape == (2,)
    assert dones.shape == (2,)
    assert infos.winner.shape == (2,)

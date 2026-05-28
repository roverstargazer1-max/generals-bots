import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from argparse import ArgumentParser, Namespace

from examples._experimental.ppo.common import (
    initialize_policy_network,
    make_initial_states,
    resolve_min_generals_distance,
    validate_training_args,
)
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

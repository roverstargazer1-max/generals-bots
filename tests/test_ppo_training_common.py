import equinox as eqx
import jax.random as jrandom
import pytest

from examples._experimental.ppo.common import initialize_policy_network
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

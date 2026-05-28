"""JAX PPO using the GeneralsEnv wrapper.

The raw-game trainer in train.py remains the primary experimental PPO path.
This wrapper-based trainer is kept for API-oriented experiments that need
GeneralsEnv auto-reset behavior.
"""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from generals.core import game
from generals.core.env import GeneralsEnv
from generals.core.rewards import composite_reward_fn

from common import initialize_policy_network, random_action, resolve_min_generals_distance, validate_training_args,compute_valid_move_mask
from network import PolicyValueNetwork, obs_to_array


def _rollout_step_inner(states, pool, env, network, key):
    """Collect one vectorized environment step."""
    num_envs = states.armies.shape[0]

    obs_p0_prior = jax.vmap(lambda s: game.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game.get_observation(s, 1))(states)

    obs_arr = jax.vmap(obs_to_array)(obs_p0_prior)
    masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p0_prior)

    key, *keys = jrandom.split(key, num_envs + 1)
    actions_p0, values, logprobs, _ = jax.vmap(network, in_axes=(0, 0, 0, None))(
        obs_arr,
        masks,
        jnp.stack(keys),
        None,
    )

    key, *keys = jrandom.split(key, num_envs + 1)
    actions_p1 = jax.vmap(random_action)(jnp.stack(keys), obs_p1_prior)

    actions = jnp.stack([actions_p0, actions_p1], axis=1)
    timesteps, new_states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)

    terminal_states = timesteps.last_state
    obs_p0_new = jax.vmap(lambda s: game.get_observation(s, 0))(terminal_states)
    rewards = jax.vmap(composite_reward_fn)(obs_p0_prior, actions_p0, obs_p0_new)
    dones = timesteps.terminated | timesteps.truncated

    return new_states, (obs_arr, masks, actions_p0, logprobs, values, rewards, dones, timesteps.info), key


@eqx.filter_jit
def rollout_step(states, pool, env, network, key):
    """Vectorized rollout step using GeneralsEnv."""
    return _rollout_step_inner(states, pool, env, network, key)


def make_collect_rollout(env, num_steps):
    """Create a rollout collection function using lax.scan."""

    @eqx.filter_jit
    def collect_rollout(states, pool, network, key):
        def rollout_body(carry, _):
            carry_states, carry_key = carry
            new_states, data, new_key = _rollout_step_inner(carry_states, pool, env, network, carry_key)
            return (new_states, new_key), data

        (final_states, final_key), rollout_data = jax.lax.scan(
            rollout_body,
            (states, key),
            None,
            length=num_steps,
        )
        return final_states, rollout_data, final_key

    return collect_rollout


@jax.jit
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute GAE advantages and value returns."""
    num_steps, num_envs = rewards.shape
    values_with_bootstrap = jnp.concatenate([values, jnp.zeros((1, num_envs))], axis=0)

    def gae_step(carry, inputs):
        last_adv = carry
        reward, value, next_value, done = inputs
        nonterminal = 1.0 - done
        delta = reward + gamma * next_value * nonterminal - value
        advantage = delta + gamma * lam * nonterminal * last_adv
        return advantage, advantage

    inputs = (
        rewards[::-1],
        values[::-1],
        values_with_bootstrap[1:][::-1],
        dones[::-1],
    )
    _, advantages_rev = jax.lax.scan(gae_step, jnp.zeros(num_envs), inputs)
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


@jax.jit
def ppo_loss(network, obs, mask, action, old_logprob, advantage, ret, clip=0.2):
    """PPO loss for single sample."""
    _, value, logprob, entropy = network(obs, mask, None, action)

    ratio = jnp.exp(logprob - old_logprob)
    clipped = jnp.clip(ratio, 1 - clip, 1 + clip) * advantage
    policy_loss = -jnp.minimum(ratio * advantage, clipped)
    value_loss = 0.5 * (value - ret) ** 2
    entropy_loss = -0.01 * entropy

    return policy_loss + value_loss + entropy_loss


@eqx.filter_jit
def train_step(network, opt_state, minibatch, optimizer):
    """Single training step on a minibatch."""
    obs, masks, actions, old_logprobs, advantages, returns = minibatch

    def loss_fn(net):
        losses = jax.vmap(lambda o, m, a, olp, adv, r: ppo_loss(net, o, m, a, olp, adv, r))(
            obs,
            masks,
            actions,
            old_logprobs,
            advantages,
            returns,
        )
        return jnp.mean(losses)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    network = eqx.apply_updates(network, updates)

    return network, opt_state, loss


def train_epoch(network, opt_state, batch, optimizer, key, minibatch_size):
    """Train for one epoch with minibatching."""
    obs, masks, actions, old_logprobs, advantages, returns = batch

    batch_size = obs.shape[0] * obs.shape[1]
    obs_flat = obs.reshape(batch_size, *obs.shape[2:])
    masks_flat = masks.reshape(batch_size, *masks.shape[2:])
    actions_flat = actions.reshape(batch_size, -1)
    old_logprobs_flat = old_logprobs.reshape(-1)
    advantages_flat = advantages.reshape(-1)
    returns_flat = returns.reshape(-1)

    perm = jrandom.permutation(key, batch_size)
    obs_flat = obs_flat[perm]
    masks_flat = masks_flat[perm]
    actions_flat = actions_flat[perm]
    old_logprobs_flat = old_logprobs_flat[perm]
    advantages_flat = advantages_flat[perm]
    returns_flat = returns_flat[perm]

    num_complete_batches = batch_size // minibatch_size
    total_loss = 0.0

    for i in range(num_complete_batches):
        start_idx = i * minibatch_size
        end_idx = start_idx + minibatch_size
        minibatch = (
            obs_flat[start_idx:end_idx],
            masks_flat[start_idx:end_idx],
            actions_flat[start_idx:end_idx],
            old_logprobs_flat[start_idx:end_idx],
            advantages_flat[start_idx:end_idx],
            returns_flat[start_idx:end_idx],
        )
        network, opt_state, loss = train_step(network, opt_state, minibatch, optimizer)
        total_loss += loss

    avg_loss = total_loss / max(num_complete_batches, 1)
    return network, opt_state, avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train the experimental GeneralsEnv-based JAX PPO agent.")
    parser.add_argument("num_envs", nargs="?", type=int, default=128, help="Number of parallel environments.")
    parser.add_argument("--num-steps", type=int, default=128, help="Rollout steps per PPO iteration.")
    parser.add_argument("--num-iterations", type=int, default=50, help="Number of PPO iterations.")
    parser.add_argument("--num-epochs", type=int, default=1, help="PPO epochs per rollout batch.")
    parser.add_argument("--minibatch-size", type=int, default=256, help="Minibatch size for PPO updates.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--pool-size", type=int, default=10_000, help="Auto-reset state pool size.")
    parser.add_argument("--grid-size", type=int, default=4, help="Square map size used by the policy network.")
    parser.add_argument("--truncation", type=int, default=500, help="Maximum game steps before an auto-reset.")
    parser.add_argument("--mountain-density-min", type=float, default=0.18, help="Minimum mountain density.")
    parser.add_argument("--mountain-density-max", type=float, default=0.24, help="Maximum mountain density.")
    parser.add_argument("--num-cities-min", type=int, default=9, help="Minimum number of generated cities.")
    parser.add_argument("--num-cities-max", type=int, default=11, help="Maximum number of generated cities.")
    parser.add_argument("--min-generals-distance", type=int, default=None, help="Minimum distance between generals.")
    parser.add_argument("--max-generals-distance", type=int, default=None, help="Maximum distance between generals.")
    parser.add_argument("--city-army-min", type=int, default=40, help="Generated city minimum starting army.")
    parser.add_argument("--city-army-max", type=int, default=51, help="Generated city maximum starting army.")
    parser.add_argument(
        "--init-model-path",
        default=None,
        help="Optional .eqx checkpoint to finetune from. Optimizer state is not restored.",
    )
    parser.add_argument("--model-path", default="jax_ppo_model_env.eqx", help="Path where the trained model is saved.")
    args = parser.parse_args()

    grid_size = args.grid_size
    min_generals_distance = resolve_min_generals_distance(grid_size, args.min_generals_distance)

    validate_training_args(parser, args)

    print("JAX PPO (GeneralsEnv API)")
    print(f"Environments:  {args.num_envs}")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Grid:          {grid_size}x{grid_size} with composite rewards")
    print(f"Mountains:     {args.mountain_density_min:.2f}-{args.mountain_density_max:.2f}")
    print(f"Cities:        {args.num_cities_min}-{args.num_cities_max}")
    print(f"General dist:  min={min_generals_distance}, max={args.max_generals_distance}")
    print(f"Epochs:        {args.num_epochs}")
    print(f"Minibatch:     {args.minibatch_size}")
    if args.init_model_path:
        print(f"Init model:    {args.init_model_path}")
    print()

    key = jrandom.PRNGKey(42)
    key, net_key = jrandom.split(key)
    network = initialize_policy_network(PolicyValueNetwork, net_key, grid_size, args.init_model_path)
    optimizer = optax.adam(args.lr)
    params = eqx.filter(network, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    print(f"Parameters: {sum(x.size for x in jax.tree.leaves(params)):,}")

    env = GeneralsEnv(
        grid_dims=(grid_size, grid_size),
        truncation=args.truncation,
        pool_size=args.pool_size,
        mountain_density_range=(args.mountain_density_min, args.mountain_density_max),
        num_cities_range=(args.num_cities_min, args.num_cities_max),
        min_generals_distance=min_generals_distance,
        max_generals_distance=args.max_generals_distance,
        castle_val_range=(args.city_army_min, args.city_army_max),
    )
    collect_rollout = make_collect_rollout(env, args.num_steps)

    key, pool_key = jrandom.split(key)
    pool, _ = env.reset(pool_key)
    key, *init_keys = jrandom.split(key, args.num_envs + 1)
    states = jax.vmap(env.init_state)(jnp.stack(init_keys))

    print("\nWarming up...")
    for _ in range(3):
        states, _, key = rollout_step(states, pool, env, network, key)
    jax.block_until_ready(states)

    print("Training...\n")

    for iteration in range(args.num_iterations):
        t0 = time.time()

        states, rollout_data, key = collect_rollout(states, pool, network, key)
        jax.block_until_ready(states)

        obs, masks, actions, logprobs, values, rewards, dones, infos = rollout_data
        advantages, returns = compute_gae(rewards, values, dones)
        policy_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch = (obs, masks, actions, logprobs, policy_advantages, returns)
        epoch_losses = []
        for _ in range(args.num_epochs):
            key, epoch_key = jrandom.split(key)
            network, opt_state, loss = train_epoch(
                network,
                opt_state,
                batch,
                optimizer,
                epoch_key,
                args.minibatch_size,
            )
            epoch_losses.append(loss)
        jax.block_until_ready(network)

        avg_loss = jnp.mean(jnp.array(epoch_losses))
        elapsed = time.time() - t0

        if iteration % 10 == 0:
            avg_reward = rewards.mean()
            num_episodes = int(dones.sum())
            wins = int(jnp.sum(dones & (infos.winner == 0)))
            losses = int(jnp.sum(dones & (infos.winner == 1)))
            win_rate = wins / max(num_episodes, 1) * 100
            sps = (args.num_envs * args.num_steps) / elapsed
            print(
                f"Iter {iteration:4d} | Loss: {float(avg_loss):.4f} | "
                f"Reward: {float(avg_reward):+.4f} | Episodes: {num_episodes:3d} | "
                f"W/L: {wins:2d}/{losses:2d} ({win_rate:.0f}%) | "
                f"SPS: {sps:7.0f} | Time: {elapsed:.2f}s"
            )

    print("\nTraining complete!")
    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()

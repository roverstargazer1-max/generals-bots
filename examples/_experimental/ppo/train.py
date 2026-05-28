"""Clean JAX PPO using the raw game API for maximum performance."""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax

from generals.core.action import compute_valid_move_mask
from generals.core import game
from generals.core.grid import generate_grid
from generals.core.rewards import composite_reward_fn

from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, initialize_policy_network, opponent_action
from network import PolicyValueNetwork, obs_to_array


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
    keys = jrandom.split(key, pool_size)

    if map_generator == "simple":
        grids = jax.vmap(lambda k: make_simple_general_grid(k, grid_size))(keys)
    else:
        grids = jax.vmap(
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

    return jax.vmap(game.create_initial_state)(grids)


def make_initial_states(pool, num_envs):
    """Take initial states from the pool and spread future reset indices."""
    states = jax.tree.map(lambda x: x[:num_envs], pool)
    pool_size = pool.armies.shape[0]
    pool_idx = (jnp.arange(num_envs, dtype=jnp.int32) + num_envs) % pool_size
    return states._replace(pool_idx=pool_idx)


@eqx.filter_jit
def rollout_step(states, pool, network, key, truncation, opponent_id):
    """Vectorized rollout step for all environments."""
    num_envs = states.armies.shape[0]
    
    # Observations (BEFORE step for reward calculation)
    obs_p0_prior = jax.vmap(lambda s: game.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game.get_observation(s, 1))(states)
    
    # Actions from network
    obs_arr = jax.vmap(obs_to_array)(obs_p0_prior)
    masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p0_prior)
    
    key, *keys = jrandom.split(key, num_envs + 1)
    actions_p0, values, logprobs, entropies = jax.vmap(network, in_axes=(0, 0, 0, None))(
        obs_arr, masks, jnp.stack(keys), None
    )
    
    # Opponent actions for p1
    key, *keys = jrandom.split(key, num_envs + 1)
    actions_p1 = jax.vmap(lambda k, o: opponent_action(opponent_id, k, o, random_action))(jnp.stack(keys), obs_p1_prior)
    
    # Step game
    actions = jnp.stack([actions_p0, actions_p1], axis=1)
    new_states, infos = jax.vmap(game.step)(states, actions)
    
    # Get new observations (AFTER step)
    obs_p0_new = jax.vmap(lambda s: game.get_observation(s, 0))(new_states)
    
    # Compute rewards using composite reward function
    rewards = jax.vmap(composite_reward_fn)(
        obs_p0_prior, actions_p0, obs_p0_new
    )
    
    # Terminated/truncated
    terminated = infos.is_done
    truncated = (new_states.time >= truncation) & ~terminated
    dones = terminated | truncated

    # Auto-reset from a pre-generated pool. This keeps rollout_step fast and
    # lets the raw trainer use complex generated maps without regenerating them
    # inside every environment step.
    pool_size = pool.armies.shape[0]
    reset_indices = new_states.pool_idx % pool_size
    reset_states = jax.tree.map(lambda x: x[reset_indices], pool)
    next_pool_idx = jnp.where(dones, new_states.pool_idx + num_envs, new_states.pool_idx)
    reset_states = reset_states._replace(pool_idx=next_pool_idx)
    current_states = new_states._replace(pool_idx=next_pool_idx)

    final_states = jax.tree.map(
        lambda reset, current: jnp.where(dones.reshape(num_envs, *([1] * (reset.ndim - 1))), reset, current),
        reset_states,
        current_states,
    )
    
    return final_states, (obs_arr, masks, actions_p0, logprobs, values, rewards, dones, infos), key


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
def train_step(network, opt_state, batch, optimizer):
    """Single training step."""
    obs, masks, actions, old_logprobs, advantages, returns = batch
    
    # Flatten batch
    bs = obs.shape[0] * obs.shape[1]
    obs_flat = obs.reshape(bs, *obs.shape[2:])
    masks_flat = masks.reshape(bs, *masks.shape[2:])
    actions_flat = actions.reshape(bs, -1)
    old_logprobs_flat = old_logprobs.reshape(-1)
    advantages_flat = advantages.reshape(-1)
    returns_flat = returns.reshape(-1)
    
    def loss_fn(net):
        losses = jax.vmap(lambda o, m, a, olp, adv, r: ppo_loss(net, o, m, a, olp, adv, r))(
            obs_flat, masks_flat, actions_flat, old_logprobs_flat, advantages_flat, returns_flat
        )
        return jnp.mean(losses)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    network = eqx.apply_updates(network, updates)

    return network, opt_state, loss


def main():
    parser = argparse.ArgumentParser(description="Train the experimental raw-game JAX PPO agent.")
    parser.add_argument("num_envs", nargs="?", type=int, default=128, help="Number of parallel environments.")
    parser.add_argument("--num-steps", type=int, default=128, help="Rollout steps per PPO iteration.")
    parser.add_argument("--num-iterations", type=int, default=50, help="Number of PPO iterations.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--grid-size", type=int, default=4, help="Square map size used by the policy network.")
    parser.add_argument("--truncation", type=int, default=500, help="Maximum game steps before an auto-reset.")
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="random", help="Player-1 training opponent.")
    parser.add_argument("--pool-size", type=int, default=2048, help="Number of pre-generated reset states.")
    parser.add_argument(
        "--map-generator",
        choices=("simple", "generated"),
        default="simple",
        help="Use simple empty maps or generated maps with mountains/cities.",
    )
    parser.add_argument("--mountain-density-min", type=float, default=0.18, help="Generated-map minimum mountain density.")
    parser.add_argument("--mountain-density-max", type=float, default=0.24, help="Generated-map maximum mountain density.")
    parser.add_argument("--num-cities-min", type=int, default=9, help="Generated-map minimum number of cities.")
    parser.add_argument("--num-cities-max", type=int, default=11, help="Generated-map maximum number of cities.")
    parser.add_argument(
        "--min-generals-distance",
        type=int,
        default=None,
        help="Generated-map minimum Manhattan distance between generals.",
    )
    parser.add_argument(
        "--max-generals-distance",
        type=int,
        default=None,
        help="Generated-map maximum Manhattan distance between generals.",
    )
    parser.add_argument("--city-army-min", type=int, default=40, help="Generated city minimum starting army.")
    parser.add_argument("--city-army-max", type=int, default=51, help="Generated city maximum starting army.")
    parser.add_argument(
        "--init-model-path",
        default=None,
        help="Optional .eqx checkpoint to finetune from. Optimizer state is not restored.",
    )
    parser.add_argument("--model-path", default="jax_ppo_model.eqx", help="Path where the trained model is saved.")
    args = parser.parse_args()

    num_envs = args.num_envs
    num_steps = args.num_steps
    num_iterations = args.num_iterations
    lr = args.lr
    grid_size = args.grid_size
    min_generals_distance = args.min_generals_distance
    if min_generals_distance is None:
        min_generals_distance = max(3, grid_size // 2)

    if grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if args.pool_size < num_envs:
        parser.error("--pool-size must be at least num_envs")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if not (args.city_army_min < args.city_army_max):
        parser.error("city army range must satisfy min < max")
    
    print("JAX PPO (Raw Game API - Max Performance)")
    print(f"Environments:  {num_envs}")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Opponent:      {args.opponent}")
    print(f"Grid:          {grid_size}x{grid_size} ({args.map_generator}, truncation={args.truncation})")
    if args.map_generator == "generated":
        print(f"Mountains:     {args.mountain_density_min:.2f}-{args.mountain_density_max:.2f}")
        print(f"Cities:        {args.num_cities_min}-{args.num_cities_max}")
        print(f"General dist:  min={min_generals_distance}, max={args.max_generals_distance}")
    print(f"Reset pool:    {args.pool_size}")
    if args.init_model_path:
        print(f"Init model:    {args.init_model_path}")
    print()
    
    # Initialize
    key = jrandom.PRNGKey(42)
    key, net_key = jrandom.split(key)
    network = initialize_policy_network(PolicyValueNetwork, net_key, grid_size, args.init_model_path)
    optimizer = optax.adam(lr)
    params = eqx.filter(network, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    opponent_id = OPPONENT_NAME_TO_ID[args.opponent]

    print(f"Parameters: {sum(x.size for x in jax.tree.leaves(params)):,}")
    
    key, pool_key = jrandom.split(key)
    pool = make_state_pool(
        pool_key,
        args.pool_size,
        grid_size,
        args.map_generator,
        (args.mountain_density_min, args.mountain_density_max),
        (args.num_cities_min, args.num_cities_max),
        min_generals_distance,
        args.max_generals_distance,
        (args.city_army_min, args.city_army_max),
    )
    jax.block_until_ready(pool.armies)
    states = make_initial_states(pool, num_envs)
    
    print("\nWarming up...")
    for _ in range(3):
        states, _, key = rollout_step(states, pool, network, key, args.truncation, opponent_id)
    jax.block_until_ready(states)
    
    print("Training...\n")
    
    for iteration in range(num_iterations):
        t0 = time.time()
        
        # Collect rollout
        rollout_data = []
        for _ in range(num_steps):
            states, data, key = rollout_step(states, pool, network, key, args.truncation, opponent_id)
            rollout_data.append(data)
        jax.block_until_ready(states)
        
        # Stack data
        obs = jnp.stack([d[0] for d in rollout_data])
        masks = jnp.stack([d[1] for d in rollout_data])
        actions = jnp.stack([d[2] for d in rollout_data])
        logprobs = jnp.stack([d[3] for d in rollout_data])
        values = jnp.stack([d[4] for d in rollout_data])
        rewards = jnp.stack([d[5] for d in rollout_data])
        dones = jnp.stack([d[6] for d in rollout_data])
        infos_list = [d[7] for d in rollout_data]
        infos = jax.tree.map(lambda *xs: jnp.stack(xs), *infos_list)
        
        # Compute advantages
        advantages, returns = compute_gae(rewards, values, dones)
        policy_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Train
        batch = (obs, masks, actions, logprobs, policy_advantages, returns)
        network, opt_state, loss = train_step(network, opt_state, batch, optimizer)
        jax.block_until_ready(network)
        
        elapsed = time.time() - t0
        
        if iteration % 10 == 0:
            avg_reward = rewards.mean()
            num_episodes = int(dones.sum())
            wins = int(jnp.sum((dones) & (infos.winner == 0)))
            losses = int(jnp.sum((dones) & (infos.winner == 1)))
            win_rate = wins / max(num_episodes, 1) * 100
            sps = (num_envs * num_steps) / elapsed
            print(f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                  f"Reward: {float(avg_reward):+.4f} | Episodes: {num_episodes:3d} | "
                  f"Wins: {wins:2d}/{num_episodes} ({win_rate:.0f}%) | "
                  f"SPS: {sps:7.0f} | Time: {elapsed:.2f}s")
    
    print("\nTraining complete!")
    
    # Save model
    model_path = args.model_path
    eqx.tree_serialise_leaves(model_path, network)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()

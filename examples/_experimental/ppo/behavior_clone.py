"""Behavior cloning warm-start for the experimental PPO policy."""

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
from generals.core.action import compute_valid_move_mask

from common import (
    TEACHER_NAME_TO_ID,
    TEACHER_NAMES,
    action_to_index,
    action_to_target_probs,
    expander_target_probs,
    heuristic_action,
    initialize_policy_network,
    make_initial_states,
    make_state_pool,
    random_action,
    resolve_min_generals_distance,
    validate_training_args,
)
from network import PolicyValueNetwork, obs_to_array


@eqx.filter_jit
def collect_teacher_batch(states, pool, key, steps, truncation, teacher_id):
    """Roll out teacher-vs-Random games and collect labels for player 0."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, key = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)

        key, teacher_key, random_key = jrandom.split(key, 3)
        teacher_keys = jrandom.split(teacher_key, num_envs)
        random_keys = jrandom.split(random_key, num_envs)
        grid_size = obs_p0.armies.shape[-1]

        def collect_soft(_):
            target_probs = jax.vmap(expander_target_probs)(obs_p0)
            sampled_indices = jax.vmap(lambda k, p: jrandom.categorical(k, jnp.log(p + 1e-8)))(
                teacher_keys, target_probs
            )
            grid_cells = grid_size * grid_size
            teacher_dirs = sampled_indices // grid_cells
            teacher_positions = sampled_indices % grid_cells
            teacher_rows = teacher_positions // grid_size
            teacher_cols = teacher_positions % grid_size
            teacher_is_pass = teacher_dirs == 8
            teacher_is_half = (teacher_dirs >= 4) & (teacher_dirs < 8)
            teacher_actual_dirs = jnp.where(
                teacher_is_pass, 0, jnp.where(teacher_is_half, teacher_dirs - 4, teacher_dirs)
            )
            teacher_actions = jnp.stack(
                [teacher_is_pass, teacher_rows, teacher_cols, teacher_actual_dirs, teacher_is_half], axis=1
            ).astype(jnp.int32)
            return teacher_actions, target_probs, sampled_indices

        def collect_hard(_):
            teacher_actions = jax.vmap(lambda k, o: heuristic_action(teacher_id - 1, k, o))(teacher_keys, obs_p0)
            teacher_indices = jax.vmap(lambda a: action_to_index(a, grid_size))(teacher_actions)
            target_probs = jax.vmap(lambda a: action_to_target_probs(a, grid_size))(teacher_actions)
            return teacher_actions, target_probs, teacher_indices

        actions_p0, targets, teacher_indices = jax.lax.cond(teacher_id == 0, collect_soft, collect_hard, None)
        actions_p1 = jax.vmap(random_action)(random_keys, obs_p1)

        new_states, infos = jax.vmap(game.step)(states, jnp.stack([actions_p0, actions_p1], axis=1))
        terminated = infos.is_done
        truncated = (new_states.time >= truncation) & ~terminated
        dones = terminated | truncated

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

        obs_arr = jax.vmap(obs_to_array)(obs_p0)
        masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p0)
        return (final_states, key), (obs_arr, masks, targets, teacher_indices, dones, infos.winner)

    (states, key), batch = jax.lax.scan(body, (states, key), None, length=steps)
    return states, batch, key


@eqx.filter_jit
def train_bc_step(network, opt_state, obs, masks, targets, teacher_indices, optimizer):
    """Train one behavior-cloning batch."""
    batch_size = obs.shape[0] * obs.shape[1]
    obs_flat = obs.reshape(batch_size, *obs.shape[2:])
    masks_flat = masks.reshape(batch_size, *masks.shape[2:])
    targets_flat = targets.reshape(batch_size, targets.shape[-1])
    teacher_indices_flat = teacher_indices.reshape(batch_size)

    def loss_fn(net):
        def sample_logits(o, mask):
            logits, _ = net.logits_value(o, mask)
            return logits

        logits = jax.vmap(sample_logits)(obs_flat, masks_flat)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        losses = -jnp.sum(targets_flat * log_probs, axis=-1)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == teacher_indices_flat)
        return jnp.mean(losses), accuracy

    (loss, accuracy), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Behavior-clone the experimental PPO policy from heuristic teachers.")
    parser.add_argument("num_envs", nargs="?", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--teacher", choices=TEACHER_NAMES, default="expander-soft")
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--truncation", type=int, default=250)
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--min-generals-distance", type=int, default=None)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument(
        "--init-model-path",
        default=None,
        help="Optional .eqx checkpoint to finetune from. Optimizer state is not restored.",
    )
    parser.add_argument("--model-path", default="/tmp/generals-bc-8x8.eqx")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    validate_training_args(parser, args)
    return args


def main():
    args = parse_args()
    min_generals_distance = resolve_min_generals_distance(args.grid_size, args.min_generals_distance)

    print("Behavior cloning from heuristic teacher")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Teacher:       {args.teacher}")
    print(f"Environments:  {args.num_envs}")
    print(f"Grid:          {args.grid_size}x{args.grid_size} ({args.map_generator})")
    print(f"Iterations:    {args.num_iterations} x {args.num_steps} steps")
    print(f"Reset pool:    {args.pool_size}")
    if args.init_model_path:
        print(f"Init model:    {args.init_model_path}")
    print()

    key = jrandom.PRNGKey(args.seed)
    key, net_key, pool_key = jrandom.split(key, 3)
    network = initialize_policy_network(PolicyValueNetwork, net_key, args.grid_size, args.init_model_path)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))

    pool = make_state_pool(
        pool_key,
        args.pool_size,
        args.grid_size,
        args.map_generator,
        (args.mountain_density_min, args.mountain_density_max),
        (args.num_cities_min, args.num_cities_max),
        min_generals_distance,
        args.max_generals_distance,
        (args.city_army_min, args.city_army_max),
    )
    jax.block_until_ready(pool.armies)
    states = make_initial_states(pool, args.num_envs)

    for iteration in range(args.num_iterations):
        t0 = time.time()
        states, (obs, masks, targets, teacher_indices, dones, winners), key = collect_teacher_batch(
            states,
            pool,
            key,
            args.num_steps,
            args.truncation,
            TEACHER_NAME_TO_ID[args.teacher],
        )
        network, opt_state, loss, accuracy = train_bc_step(
            network,
            opt_state,
            obs,
            masks,
            targets,
            teacher_indices,
            optimizer,
        )
        jax.block_until_ready(network)

        if iteration % 10 == 0 or iteration == args.num_iterations - 1:
            episodes = int(dones.sum())
            wins = int(jnp.sum(dones & (winners == 0)))
            elapsed = time.time() - t0
            samples = args.num_envs * args.num_steps
            print(
                f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                f"Acc: {float(accuracy) * 100:5.1f}% | "
                f"Episodes: {episodes:4d} | Teacher wins: {wins:4d} | "
                f"SPS: {samples / elapsed:8.0f} | Time: {elapsed:.2f}s"
            )

    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()

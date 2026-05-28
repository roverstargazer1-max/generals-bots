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
    OPPONENT_NAME_TO_ID,
    OPPONENT_NAMES,
    TEACHER_NAME_TO_ID,
    TEACHER_NAMES,
    initialize_policy_network,
    make_initial_states,
    make_state_pool,
    opponent_action,
    parse_name_pool,
    random_action,
    resolve_min_generals_distance,
    teacher_action_targets,
    validate_training_args,
)
from network import PolicyValueNetwork, obs_to_array


@eqx.filter_jit
def collect_teacher_batch(states, pool, key, steps, truncation, teacher_pool_ids, opponent_pool_ids):
    """Roll out teacher-vs-opponent games and collect labels for player 0."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, key = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)

        key, teacher_choice_key, opponent_choice_key, teacher_key, opponent_key = jrandom.split(key, 5)
        teacher_slots = jrandom.randint(teacher_choice_key, (num_envs,), 0, teacher_pool_ids.shape[0])
        opponent_slots = jrandom.randint(opponent_choice_key, (num_envs,), 0, opponent_pool_ids.shape[0])
        teacher_ids = teacher_pool_ids[teacher_slots]
        opponent_ids = opponent_pool_ids[opponent_slots]
        teacher_keys = jrandom.split(teacher_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)

        actions_p0, targets, teacher_indices = jax.vmap(teacher_action_targets)(teacher_ids, teacher_keys, obs_p0)
        actions_p1 = jax.vmap(lambda i, k, o: opponent_action(i, k, o, random_action))(
            opponent_ids, opponent_keys, obs_p1
        )

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
        return (final_states, key), (obs_arr, masks, targets, teacher_indices, teacher_ids, dones, infos.winner)

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
        top3_indices = jax.lax.top_k(logits, 3)[1]
        top1_accuracy = jnp.mean(top3_indices[:, 0] == teacher_indices_flat)
        top3_accuracy = jnp.mean(jnp.any(top3_indices == teacher_indices_flat[:, None], axis=-1))
        target_log_probs = jnp.where(targets_flat > 0.0, jnp.log(targets_flat + 1e-8), 0.0)
        kl = jnp.mean(
            jnp.sum(
                jnp.where(targets_flat > 0.0, targets_flat * (target_log_probs - log_probs), 0.0),
                axis=-1,
            )
        )
        return jnp.mean(losses), (top1_accuracy, top3_accuracy, kl)

    (loss, (top1_accuracy, top3_accuracy, kl)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
    params = eqx.filter(network, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, loss, top1_accuracy, top3_accuracy, kl


def parse_args():
    parser = argparse.ArgumentParser(description="Behavior-clone the experimental PPO policy from heuristic teachers.")
    parser.add_argument("num_envs", nargs="?", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--teacher", choices=TEACHER_NAMES, default="expander-soft")
    parser.add_argument(
        "--teacher-pool",
        default=None,
        help="Comma-separated teacher pool. Overrides --teacher when provided.",
    )
    parser.add_argument(
        "--opponent-pool",
        default="random",
        help="Comma-separated opponent pool drawn per environment.",
    )
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
    try:
        args.teacher_pool_names, args.teacher_pool_ids = parse_name_pool(
            args.teacher_pool if args.teacher_pool is not None else args.teacher,
            TEACHER_NAME_TO_ID,
            "teacher",
        )
        args.opponent_pool_names, args.opponent_pool_ids = parse_name_pool(
            args.opponent_pool,
            OPPONENT_NAME_TO_ID,
            "opponent",
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    min_generals_distance = resolve_min_generals_distance(args.grid_size, args.min_generals_distance)

    print("Behavior cloning from heuristic teacher")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Teachers:      {', '.join(args.teacher_pool_names)}")
    print(f"Opponents:     {', '.join(args.opponent_pool_names)}")
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
        states, (obs, masks, targets, teacher_indices, teacher_ids, dones, winners), key = collect_teacher_batch(
            states,
            pool,
            key,
            args.num_steps,
            args.truncation,
            args.teacher_pool_ids,
            args.opponent_pool_ids,
        )
        network, opt_state, loss, top1_accuracy, top3_accuracy, kl = train_bc_step(
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
            teacher_counts = jnp.bincount(teacher_ids.reshape(-1), length=len(TEACHER_NAMES))
            teacher_samples = ", ".join(
                f"{name}:{int(teacher_counts[idx])}"
                for name, idx in TEACHER_NAME_TO_ID.items()
                if int(teacher_counts[idx]) > 0
            )
            elapsed = time.time() - t0
            samples = args.num_envs * args.num_steps
            print(
                f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                f"Top1: {float(top1_accuracy) * 100:5.1f}% | "
                f"Top3: {float(top3_accuracy) * 100:5.1f}% | "
                f"KL: {float(kl):.4f} | "
                f"Episodes: {episodes:4d} | Teacher wins: {wins:4d} | "
                f"SPS: {samples / elapsed:8.0f} | Time: {elapsed:.2f}s | "
                f"Teachers: {teacher_samples}"
            )

    eqx.tree_serialise_leaves(args.model_path, network)
    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    main()

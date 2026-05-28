"""Batch evaluation for experimental PPO policy checkpoints."""

import argparse
import csv
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

from generals.core import game

from common import (
    OPPONENT_NAME_TO_ID,
    OPPONENT_NAMES,
    greedy_policy_action,
    make_grids,
    opponent_action,
    parse_name_pool,
    random_action,
    resolve_min_generals_distance,
    sampled_policy_action,
)
from network import PolicyValueNetwork


@eqx.filter_jit
def evaluate_batch(network, states, key, max_steps, opponent, policy_mode):
    """Evaluate a network against Random or a heuristic on a batch of states."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, key = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)

        key, k0, k1 = jrandom.split(key, 3)
        policy_keys = jrandom.split(k0, num_envs)
        actions_p0 = jax.lax.cond(
            policy_mode == 0,
            lambda _: jax.vmap(lambda o: greedy_policy_action(network, o))(obs_p0),
            lambda _: jax.vmap(lambda o, k: sampled_policy_action(network, o, k))(obs_p0, policy_keys),
            None,
        )
        opponent_keys = jrandom.split(k1, num_envs)
        actions_p1 = jax.vmap(lambda k, o: opponent_action(opponent, k, o, random_action))(opponent_keys, obs_p1)

        new_states, infos = jax.vmap(game.step)(states, jnp.stack([actions_p0, actions_p1], axis=1))
        keep_old = jax.vmap(game.get_info)(states).is_done
        final_states = jax.tree.map(lambda old, new: jnp.where(keep_old.reshape(num_envs, *([1] * (old.ndim - 1))), old, new), states, new_states)
        return (final_states, key), infos

    (states, key), _ = jax.lax.scan(body, (states, key), None, length=max_steps)
    info = jax.vmap(game.get_info)(states)
    return info


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an experimental PPO policy checkpoint.")
    parser.add_argument("model_path")
    parser.add_argument("--num-games", type=int, default=1024)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="random")
    parser.add_argument(
        "--opponent-pool",
        default=None,
        help="Comma-separated opponent suite. Overrides --opponent when provided.",
    )
    parser.add_argument("--policy-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--mountain-density-min", type=float, default=0.12)
    parser.add_argument("--mountain-density-max", type=float, default=0.22)
    parser.add_argument("--num-cities-min", type=int, default=4)
    parser.add_argument("--num-cities-max", type=int, default=8)
    parser.add_argument("--min-generals-distance", type=int, default=None)
    parser.add_argument("--max-generals-distance", type=int, default=None)
    parser.add_argument("--city-army-min", type=int, default=40)
    parser.add_argument("--city-army-max", type=int, default=51)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--csv-path", default=None, help="Optional path for per-opponent CSV metrics.")
    args = parser.parse_args()

    if args.grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if args.num_games <= 0:
        parser.error("--num-games must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    try:
        args.opponent_pool_names, args.opponent_pool_ids = parse_name_pool(
            args.opponent_pool if args.opponent_pool is not None else args.opponent,
            OPPONENT_NAME_TO_ID,
            "opponent",
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def summarize_eval_info(opponent, info, num_games, elapsed):
    """Compute report-friendly aggregate evaluation metrics."""
    wins = int(jnp.sum(info.winner == 0))
    losses = int(jnp.sum(info.winner == 1))
    draws = int(jnp.sum(info.winner < 0))
    decisive = wins + losses
    return {
        "opponent": opponent,
        "games": num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / num_games,
        "decisive_win_rate": wins / max(decisive, 1),
        "draw_rate": draws / num_games,
        "mean_time": float(jnp.mean(info.time)),
        "eval_seconds": elapsed,
    }


def write_eval_csv(path, rows):
    """Write per-opponent evaluation rows in a stable CSV schema."""
    fieldnames = (
        "opponent",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "decisive_win_rate",
        "draw_rate",
        "mean_time",
        "eval_seconds",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    min_generals_distance = resolve_min_generals_distance(args.grid_size, args.min_generals_distance)

    key = jrandom.PRNGKey(args.seed)
    key, net_key, map_key, eval_key = jrandom.split(key, 4)
    network = PolicyValueNetwork(net_key, grid_size=args.grid_size)
    network = eqx.tree_deserialise_leaves(args.model_path, network)

    grids = make_grids(
        map_key,
        args.num_games,
        args.grid_size,
        args.map_generator,
        (args.mountain_density_min, args.mountain_density_max),
        (args.num_cities_min, args.num_cities_max),
        min_generals_distance,
        args.max_generals_distance,
        (args.city_army_min, args.city_army_max),
    )
    states = jax.vmap(game.create_initial_state)(grids)

    policy_mode = 0 if args.policy_mode == "greedy" else 1
    eval_keys = jrandom.split(eval_key, len(args.opponent_pool_names))

    print("Policy evaluation")
    print(f"Model:              {args.model_path}")
    print(f"Device:             {jax.devices()[0]}")
    print(f"Grid:               {args.grid_size}x{args.grid_size} ({args.map_generator})")
    print(f"Opponents:          {', '.join(args.opponent_pool_names)}")
    print(f"Policy mode:        {args.policy_mode}")
    print(f"Games:              {args.num_games}")
    print(f"Max steps:          {args.max_steps}")
    print()
    print("Opponent            Wins/Losses/Draws  WinRate  Decisive  DrawRate  MeanTime  Seconds")

    rows = []
    for opponent, opponent_code, key in zip(
        args.opponent_pool_names,
        args.opponent_pool_ids.tolist(),
        eval_keys,
        strict=True,
    ):
        t0 = time.time()
        info = evaluate_batch(network, states, key, args.max_steps, opponent_code, policy_mode)
        jax.block_until_ready(info.winner)
        elapsed = time.time() - t0
        row = summarize_eval_info(opponent, info, args.num_games, elapsed)
        rows.append(row)
        print(
            f"{opponent:<19} "
            f"{row['wins']:4d}/{row['losses']:<4d}/{row['draws']:<4d} "
            f"{row['win_rate']:.4f}   "
            f"{row['decisive_win_rate']:.4f}    "
            f"{row['draw_rate']:.4f}    "
            f"{row['mean_time']:7.1f}  "
            f"{row['eval_seconds']:7.2f}"
        )

    if args.csv_path:
        write_eval_csv(args.csv_path, rows)
        print(f"\nCSV saved to:        {args.csv_path}")


if __name__ == "__main__":
    main()

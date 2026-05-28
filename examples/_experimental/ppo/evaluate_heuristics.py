"""Batch evaluation for built-in heuristic agents."""

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

from generals.agents._heuristic_logic import HEURISTIC_NAME_TO_ID, HEURISTIC_NAMES, heuristic_action
from generals.core import game

from common import OPPONENT_NAME_TO_ID, OPPONENT_NAMES, make_grids, opponent_action, random_action, resolve_min_generals_distance


@eqx.filter_jit
def evaluate_heuristic_batch(states, key, max_steps, agent_id, opponent_id):
    """Evaluate one heuristic as player 0 against a random or heuristic player 1."""
    num_envs = states.armies.shape[0]

    def body(carry, _):
        states, key = carry
        obs_p0 = jax.vmap(lambda s: game.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game.get_observation(s, 1))(states)

        key, k0, k1 = jrandom.split(key, 3)
        agent_keys = jrandom.split(k0, num_envs)
        opponent_keys = jrandom.split(k1, num_envs)
        actions_p0 = jax.vmap(lambda k, o: heuristic_action(agent_id, k, o))(agent_keys, obs_p0)
        actions_p1 = jax.vmap(lambda k, o: opponent_action(opponent_id, k, o, random_action))(opponent_keys, obs_p1)

        new_states, infos = jax.vmap(game.step)(states, jnp.stack([actions_p0, actions_p1], axis=1))
        keep_old = jax.vmap(game.get_info)(states).is_done
        final_states = jax.tree.map(
            lambda old, new: jnp.where(keep_old.reshape(num_envs, *([1] * (old.ndim - 1))), old, new),
            states,
            new_states,
        )
        return (final_states, key), infos

    (states, key), _ = jax.lax.scan(body, (states, key), None, length=max_steps)
    return jax.vmap(game.get_info)(states)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate built-in heuristic agents.")
    parser.add_argument("--agent", choices=HEURISTIC_NAMES, default="balanced")
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="expander")
    parser.add_argument("--num-games", type=int, default=1024)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--map-generator", choices=("simple", "generated"), default="generated")
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
    args = parser.parse_args()

    if args.grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if args.num_games <= 0:
        parser.error("--num-games must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not (0.0 <= args.mountain_density_min <= args.mountain_density_max <= 1.0):
        parser.error("mountain density must satisfy 0 <= min <= max <= 1")
    if not (2 <= args.num_cities_min <= args.num_cities_max):
        parser.error("city count must satisfy 2 <= min <= max")
    if args.city_army_min >= args.city_army_max:
        parser.error("city army range must satisfy min < max")
    return args


def main():
    args = parse_args()
    min_generals_distance = resolve_min_generals_distance(args.grid_size, args.min_generals_distance)

    key = jrandom.PRNGKey(args.seed)
    key, map_key, eval_key = jrandom.split(key, 3)
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

    t0 = time.time()
    info = evaluate_heuristic_batch(
        states,
        eval_key,
        args.max_steps,
        HEURISTIC_NAME_TO_ID[args.agent],
        OPPONENT_NAME_TO_ID[args.opponent],
    )
    jax.block_until_ready(info.winner)
    elapsed = time.time() - t0

    wins = int(jnp.sum(info.winner == 0))
    losses = int(jnp.sum(info.winner == 1))
    draws = int(jnp.sum(info.winner < 0))
    decisive = wins + losses
    win_rate = wins / args.num_games
    decisive_win_rate = wins / max(decisive, 1)
    draw_rate = draws / args.num_games
    mean_time = float(jnp.mean(info.time))

    print("Heuristic evaluation")
    print(f"Device:             {jax.devices()[0]}")
    print(f"Grid:               {args.grid_size}x{args.grid_size} ({args.map_generator})")
    print(f"Agent:              {args.agent}")
    print(f"Opponent:           {args.opponent}")
    print(f"Games:              {args.num_games}")
    print(f"Max steps:          {args.max_steps}")
    print(f"Wins/Losses/Draws:  {wins}/{losses}/{draws}")
    print(f"Win rate:           {win_rate:.4f}")
    print(f"Decisive win rate:  {decisive_win_rate:.4f}")
    print(f"Draw rate:          {draw_rate:.4f}")
    print(f"Mean final time:    {mean_time:.1f}")
    print(f"Eval seconds:       {elapsed:.2f}")


if __name__ == "__main__":
    main()

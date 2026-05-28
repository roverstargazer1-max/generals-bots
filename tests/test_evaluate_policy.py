from types import SimpleNamespace

import jax.numpy as jnp

from examples._experimental.ppo.evaluate_policy import summarize_eval_info, write_eval_csv


def test_summarize_eval_info_reports_rates():
    info = SimpleNamespace(
        winner=jnp.array([0, 1, -1, 0]),
        time=jnp.array([10, 12, 20, 14]),
    )

    row = summarize_eval_info("random", info, num_games=4, elapsed=1.25)

    assert row["opponent"] == "random"
    assert row["wins"] == 2
    assert row["losses"] == 1
    assert row["draws"] == 1
    assert row["win_rate"] == 0.5
    assert row["decisive_win_rate"] == 2 / 3
    assert row["draw_rate"] == 0.25
    assert row["mean_time"] == 14.0
    assert row["eval_seconds"] == 1.25


def test_write_eval_csv_uses_stable_schema(tmp_path):
    csv_path = tmp_path / "eval.csv"
    rows = [
        {
            "opponent": "random",
            "games": 4,
            "wins": 2,
            "losses": 1,
            "draws": 1,
            "win_rate": 0.5,
            "decisive_win_rate": 2 / 3,
            "draw_rate": 0.25,
            "mean_time": 14.0,
            "eval_seconds": 1.25,
        }
    ]

    write_eval_csv(csv_path, rows)

    assert csv_path.read_text().splitlines()[0] == (
        "opponent,games,wins,losses,draws,win_rate,decisive_win_rate,draw_rate,mean_time,eval_seconds"
    )
    assert "random,4,2,1,1,0.5," in csv_path.read_text()

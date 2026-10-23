"""Cfg invariants for the headstand-exit task (CPU, no sim)."""

from mjlab_microduck.tasks.microduck_headstand_exit_env_cfg import (
    EPISODE_LENGTH_S,
    MicroduckHeadstandExitRlCfg,
    make_microduck_headstand_exit_env_cfg,
)

WALKING_REWARDS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "pose",
    "upright",
)


def test_env_builds_train_and_play():
    assert make_microduck_headstand_exit_env_cfg() is not None
    assert make_microduck_headstand_exit_env_cfg(play=True) is not None


def test_episode_length():
    cfg = make_microduck_headstand_exit_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 8.0


def test_no_walking_rewards_survive():
    cfg = make_microduck_headstand_exit_env_cfg()
    for name in WALKING_REWARDS:
        assert name not in cfg.rewards, f"walking reward survived: {name}"


def test_exit_rewards_present():
    cfg = make_microduck_headstand_exit_env_cfg()
    for name in (
        "exit_progress",
        "upright_at_height",
        "upright_linear",
        "pose_stand_legs",
        "gentle_motion",
        "no_flight",
    ):
        assert name in cfg.rewards, f"missing reward: {name}"


def test_penalty_weight_signs():
    # Self-negating penalties (return <= 0) must carry POSITIVE weights;
    # a negative weight double-negates into a reward for the violation.
    cfg = make_microduck_headstand_exit_env_cfg()
    assert cfg.rewards["gentle_motion"].weight > 0
    assert cfg.rewards["no_flight"].weight > 0
    # Task objectives are positive rewards.
    assert cfg.rewards["exit_progress"].weight > 0
    assert cfg.rewards["upright_at_height"].weight > 0


def test_spawns_inverted():
    cfg = make_microduck_headstand_exit_env_cfg()
    assert "headstand_spawn" in cfg.events
    assert cfg.events["headstand_spawn"].params["near_inverted_prob"] == 1.0
    assert "reset_headstand_exit_state" in cfg.events


def test_rl_cfg():
    assert MicroduckHeadstandExitRlCfg.experiment_name == "microduck_headstand_exit"

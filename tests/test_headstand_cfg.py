"""Cfg invariants for the headstand task (CPU, no sim)."""

from mjlab_microduck.tasks.microduck_headstand_env_cfg import (
    EPISODE_LENGTH_S,
    MicroduckHeadStandRlCfg,
    make_microduck_headstand_env_cfg,
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
    assert make_microduck_headstand_env_cfg() is not None
    assert make_microduck_headstand_env_cfg(play=True) is not None


def test_episode_length():
    cfg = make_microduck_headstand_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 8.0


def test_no_walking_rewards_survive():
    cfg = make_microduck_headstand_env_cfg()
    for name in WALKING_REWARDS:
        assert name not in cfg.rewards, f"walking reward survived: {name}"


def test_headstand_rewards_present():
    cfg = make_microduck_headstand_env_cfg()
    for name in (
        "headstand_progress",
        "headstand_hold",
        "feet_contact_inverted",
        "pose_legs_home",
        "gentle_motion",
        "no_flight",
        "feet_height",
    ):
        assert name in cfg.rewards, f"missing reward: {name}"


def test_penalty_weight_signs():
    # Self-negating penalties (return ≤ 0) must carry POSITIVE weights;
    # a negative weight double-negates into a reward for the violation.
    cfg = make_microduck_headstand_env_cfg()
    assert cfg.rewards["feet_contact_inverted"].weight > 0
    assert cfg.rewards["gentle_motion"].weight > 0
    assert cfg.rewards["no_flight"].weight > 0
    assert cfg.rewards["root_z_ceiling"].weight > 0
    assert cfg.rewards["roll_rate_hinge"].weight > 0
    # Task objectives are positive rewards.
    assert cfg.rewards["headstand_progress"].weight > 0
    assert cfg.rewards["headstand_hold"].weight > 0
    assert cfg.rewards["pose_legs_home"].weight > 0


def test_flight_termination_present():
    cfg = make_microduck_headstand_env_cfg()
    assert "flight" in cfg.terminations
    assert cfg.terminations["flight"].params["sensor_name"] == "body_ground_contact"
    assert "reset_flight_run" in cfg.events


def test_sim2real_regularisers_kept():
    cfg = make_microduck_headstand_env_cfg()
    for name in ("action_rate_l2", "dof_pos_limits", "self_collisions"):
        assert name in cfg.rewards, f"regulariser lost: {name}"


def test_fall_termination_removed_nan_kept():
    cfg = make_microduck_headstand_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations
    assert "head_ground_contact" in cfg.terminations["nan_state"].params["sensor_names"]


def test_command_slots_present():
    cfg = make_microduck_headstand_env_cfg()
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        assert "command" in terms
        assert "head_command" in terms
        assert "body_command" in terms


def test_rl_cfg_identity():
    assert MicroduckHeadStandRlCfg.experiment_name == "microduck_headstand"
    assert MicroduckHeadStandRlCfg.actor.obs_normalization is True

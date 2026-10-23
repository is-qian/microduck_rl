"""Microduck *headstand exit* task (v1) — smooth return from headstand to stand.

Episodic policy: spawn inverted in the headstand (head top on the floor,
legs up), then SMOOTHLY come back down and finish standing (HOME legs +
STAND_Z + upright). The yoga-style exit: fold the legs down with control,
land the feet softly in a crouch, then rise.

Reward design:
  - ``exit_progress``: mirror of headstand_progress — potential-based frontier
    on the UPRIGHT cosine, integrated only while ground-supported, per-step
    payment capped. Camping inverted or bobbing pays zero; a full smooth
    invert→stand exit pays 2×weight in total.
  - Standing annuity: upright_gaussian_at_height + HOME leg pose pay only at
    the destination, so arriving early and holding pays the most.
  - Gentleness is weighted HIGHER than in the entry task (the ask is a
    *smooth* exit): |a_z|, joint_torque_rate from step 0, no-flight.
  - Warm-start from the trained headstand policy (knows the inverted
    balance); roulade works too (knows rolling).
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
    MICRODUCK_ROUGH_TERRAINS_CFG,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

EPISODE_LENGTH_S = 8.0

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# Standing destination (measured standing equilibrium, same as standup).
STAND_Z = 0.115
HEADSTAND_MIN_FORCE = 1.0


def make_microduck_headstand_exit_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck headstand-exit environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern=rf"^({'|'.join(HEAD_BODY_NAMES)})$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Whole-robot ground contact: the true support signal (a chest-supported
    # fold phase is NOT flight).
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, head_ground_cfg, self_collision_cfg, body_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # Exit progress: pays only supported rotation toward upright, capped per
    # step — a smooth exit is the argmax, collapsing fast forfeits payment.
    cfg.rewards["exit_progress"] = RewardTermCfg(
        func=microduck_mdp.headstand_exit_progress,
        weight=4.0,
        params={
            # Fastest fully-paid exit = 2.0 / 0.04 steps = 1.0 s; rotating
            # faster forfeits the excess, so a hop pays less than a fold.
            "max_paid_per_step": 0.04,
            "feet_sensor_name": "feet_ground_contact",
            "head_sensor_name": "head_ground_contact",
            "support_sensor_name": "body_ground_contact",
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Standing annuity: sharp upright Gaussian paying only at standing height.
    cfg.rewards["upright_at_height"] = RewardTermCfg(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=3.0,
        params={
            "std": 0.1,
            "height_low": 0.10,
            "height_high": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=2.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,  # HOME = standing
        },
    )

    # Smoothness: heavier than the entry task — the ask is a SMOOTH exit.
    # POSITIVE weights: these are self-negating penalties.
    cfg.rewards["gentle_motion"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["no_flight"] = RewardTermCfg(
        func=microduck_mdp.headstand_flight_penalty,
        weight=5.0,
        params={
            "feet_sensor_name": "feet_ground_contact",
            "head_sensor_name": "head_ground_contact",
            "support_sensor_name": "body_ground_contact",
        },
    )

    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=-0.01
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.01
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards.pop("soft_landing", None)
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to the headstand/standup policies) ────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    av = cfg.observations["actor"].terms["base_ang_vel"]
    av.func = microduck_mdp.base_ang_vel_imu_misaligned
    av.params = {"max_angle_deg": 6.0}
    g = cfg.observations["actor"].terms[gravity_term_name]
    g.func = microduck_mdp.projected_gravity_imu_misaligned
    g.params = {"max_angle_deg": 6.0}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    cfg.events["encoder_bias"].params["bias_range"] = (-0.015, 0.015)
    cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
    cfg.observations["critic"].terms["joint_pos"].params["biased"] = False

    # ── Commands (zero-padded, layout parity) ────────────────────────────────
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact", "head_ground_contact")},
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events.pop("push_robot", None)

    # Every episode starts in the headstand (the deployment hand-off).
    # Wide tilt + initial tip velocity: at policy switch-over the real robot
    # may already be falling with some angular rate, and the exit policy must
    # catch that, not just the ideal vertical headstand.
    cfg.events["headstand_spawn"] = EventTermCfg(
        func=microduck_mdp.set_headstand_spawn_state,
        mode="reset",
        params={
            "near_inverted_prob": 1.0,
            "tilt_min_deg": 150.0,
            "tilt_max_deg": 180.0,
            "root_z": 0.145,
            "root_z_noise": 0.01,
            "joint_noise_std": 0.05,
            "roll_vel_range": (-0.5, 0.5),
        },
    )
    cfg.events["reset_headstand_exit_state"] = EventTermCfg(
        func=microduck_mdp.reset_headstand_exit_state,
        mode="reset",
    )

    # DR matched to the velocity env.
    cfg.events["randomize_com"] = EventTermCfg(
        func=dr.body_ipos,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "operation": "add",
            "ranges": (-0.003, 0.003),
        },
    )
    cfg.events["randomize_head_com"] = EventTermCfg(
        func=dr.body_ipos,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
            "operation": "add",
            "ranges": (-0.003, 0.003),
        },
    )
    cfg.events["randomize_armature"] = EventTermCfg(
        func=dr.joint_armature,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )
    cfg.events["randomize_mass_inertia"] = EventTermCfg(
        func=dr.pseudo_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "alpha_range": (math.log(0.95) / 2.0, math.log(1.05) / 2.0),
        },
    )
    cfg.events["randomize_joint_friction"] = EventTermCfg(
        func=microduck_mdp.randomize_bam_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale_range": (0.9, 1.1),
        },
    )

    # ── Terrain ───────────────────────────────────────────────────────────────
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG
        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    return cfg


MicroduckHeadstandExitRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_headstand_exit",
    run_name="microduck_headstand_exit",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)

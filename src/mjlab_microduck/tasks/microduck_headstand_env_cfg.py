"""Microduck *headstand* task (v7.1) — inverted balance on the head shell.

V3 (2026-10-22): v1's ungated rewards were farmed lying down; v2's
orientation-gated leg pose was farmed flopped with feet tucked (headstand_
progress frozen at ~0.02 — no supported tipping gradient got discovered).
Now: potential-based progress frontier (camping/bobbing/flailing pay zero)
and leg pose as a multiplicative composite with the hold gate. V4: the
progress reward is now a SUPPORT-GATED INTEGRATOR with a per-step payment
cap — v3's state-based frontier let ballistic hops bank mid-air rotation at
touchdown (seen on video: bouncing, not entering). body_ang_vel near-zero
during discovery. V7 (entry needs SPEED — a fast roll, not a static
push-off): progress payment cap raised 0.05→0.2/step, body_ang_vel off, and
mid-roll spawns (100–180° tilt) carry 1–3 rad/s of forward angular momentum
so the policy learns to catch a fast roll into the hold.
V7.1: slower roll — spawn ω halved to 0.5–1.5 rad/s, payment cap
tightened to 0.1/step (entry in ≥0.4 s): roll in, don't whip in.

Episodic policy: start standing at HOME, enter a headstand (crouch → plant
head → swing legs up), then hold inverted with feet in the air until timeout.
The duck has no arms; support is the head shell alone (head is ~38% of body
mass and sits LOW when inverted, so the hold itself is a stable equilibrium —
the hard part is the entry maneuver).

Reward design (per AGENTS.md playbook):
  - ``inverted_linear`` (cos of deviation from inverted) gives a bootstrap
    gradient from any orientation — a Gaussian-only stack has no gradient far
    from the target.
  - ``headstand_hold`` is the hard gate defining the maneuver: pays only while
    head-supported AND feet airborne, scaled by a sharp inversion Gaussian.
    Ungated inversion rewards would be farmed by mid-air flailing.
  - ``feet_contact_inverted`` kills the "inverted but feet still down" cheat.
  - ``gentle_motion`` (|a_z|) prices entry violence; kept small — it is
    unavoidable during the entry and a big weight is a motion-blocker.
  - Legs track HOME (extended, symmetric) to keep the CoM over the support.

Pushes are OFF: any disturbance tax while the entry is undiscovered makes
"do nothing" win. V5: 40% of episodes spawn near-inverted (reverse
curriculum) so the hold is learned before the entry exists — standing-only
spawns never discovered the supported tip-over in v2–v4. Re-add pushes with
a curriculum once the hold exists.

Commands: twist zero-ish, body_pose zero-padded, head_pose sampled tiny-alive
but NOT tracked — the neck must stay free for balance corrections.
"""

import math
from copy import deepcopy

# DR toggles/ranges matched to the velocity env for sim2real parity.
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

COM_RANDOMIZATION_RANGE = 0.003
HEAD_COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

EPISODE_LENGTH_S = 8.0

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
# Fully straight legs during the hold: hip_roll/hip_pitch/knee/ankle → 0
# (HOME has a slight zig-zag bend; 0 extends the leg along the trunk axis).
_STRAIGHT_LEG_OVERRIDES = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0}

# Head-ground contact force (N) above which the hold gate opens. Well below
# weight-at-rest (~8 N) so noisy balance contact counts.
HEADSTAND_MIN_FORCE = 1.0
# Sharp Gaussian std on deviation-from-inverted for the hold reward (~8.6°).
INVERTED_STD = 0.15

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


def make_microduck_headstand_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck headstand environment configuration."""

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

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, head_ground_cfg, self_collision_cfg)
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
        # Fights the inversion objective directly.
        "upright",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # Task mass ~9.5 to keep velocity's regularisers at their transferred
    # relative strength (standup's 2026-07 lesson).
    # Full entry pays 2×weight in total; weight set so the entry payoff
    # dwarfs the attempt taxes.
    cfg.rewards["headstand_progress"] = RewardTermCfg(
        func=microduck_mdp.headstand_progress,
        weight=5.0,
        params={
            "feet_sensor_name": "feet_ground_contact",
            "head_sensor_name": "head_ground_contact",
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    cfg.rewards["headstand_hold"] = RewardTermCfg(
        func=microduck_mdp.headstand_hold,
        weight=6.0,
        params={
            "head_sensor_name": "head_ground_contact",
            "feet_sensor_name": "feet_ground_contact",
            "std": INVERTED_STD,
            "min_force": HEADSTAND_MIN_FORCE,
            # Chin-on-ground also passes head-contact + inversion (v5);
            # require the flat head top like roulade's over-the-head latch.
            "require_head_top": True,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # No hopping: the entry must stay supported (smooth tip-over, no airborne
    # phase). Self-negating penalty → POSITIVE weight.
    cfg.rewards["no_flight"] = RewardTermCfg(
        func=microduck_mdp.headstand_flight_penalty,
        weight=0.5,
        params={
            "feet_sensor_name": "feet_ground_contact",
            "head_sensor_name": "head_ground_contact",
        },
    )

    cfg.rewards["feet_contact_inverted"] = RewardTermCfg(
        func=microduck_mdp.feet_contact_inverted_penalty,
        weight=1.0,
        params={
            "feet_sensor_name": "feet_ground_contact",
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    cfg.rewards["pose_legs_home"] = RewardTermCfg(
        func=microduck_mdp.headstand_leg_pose_gated,
        weight=2.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": _STRAIGHT_LEG_OVERRIDES,
            "feet_sensor_name": "feet_ground_contact",
            "head_sensor_name": "head_ground_contact",
            "min_force": HEADSTAND_MIN_FORCE,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Feet as high as possible during the hold (legs extended upward).
    cfg.rewards["feet_height"] = RewardTermCfg(
        func=microduck_mdp.headstand_feet_height,
        weight=4.0,
        params={
            "feet_sensor_name": "feet_ground_contact",
            "head_sensor_name": "head_ground_contact",
            "min_force": HEADSTAND_MIN_FORCE,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # POSITIVE weight: trunk_vertical_accel_penalty already returns -|a_z|.
    cfg.rewards["gentle_motion"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.005,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Regulariser tuning matched to standup's transferred values.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0  # curriculum later
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = 0.0  # off: the entry is a FAST roll, any ω tax blocks it
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards.pop("soft_landing", None)
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (identical layout to walking / standup policies) ─────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    # No terrain-height sensor in this scene (walking foot rewards dropped too).
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    # NaN-safe wrappers on sensor-derived critic terms — one non-finite contact
    # force kills a run via rsl_rl's check_nan.
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

    # IMU obs delay + noise matched to velocity's audited values.
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

    # IMU mounting-misalignment DR at the observation level (match velocity).
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

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

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── Commands ──────────────────────────────────────────────────────────────
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),
            (-0.05, 0.05),
            (-0.07, 0.07),
            (-0.015, 0.015),
        ),
    )

    # Layout parity: [twist(3), head_pose(4), body_pose(6)]. body_pose stays
    # zero-padded; head_pose carries the alive command but is untracked.
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
    # Falling over is the learning signal here, not a failure — no tilt term.
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
    cfg.events["reset_headstand_state"] = EventTermCfg(
        func=microduck_mdp.reset_headstand_state,
        mode="reset",
    )
    cfg.events["headstand_spawn"] = EventTermCfg(
        func=microduck_mdp.set_headstand_spawn_state,
        mode="reset",
        params={
            "near_inverted_prob": 0.4,
            "tilt_min_deg": 100.0,
            "tilt_max_deg": 180.0,
            "root_z": 0.145,
            "root_z_noise": 0.01,
            "joint_noise_std": 0.1,
            # mid-roll states carry forward angular momentum (1–3 rad/s)
            "roll_vel_range": (0.5, 1.5),
        },
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Pushes off in v1 — see module docstring.
    cfg.events.pop("push_robot", None)

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
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


MicroduckHeadStandRlCfg = RslRlOnPolicyRunnerCfg(
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
        symmetry_cfg=None,  # inverted pose is symmetric, but entry need not be
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_headstand",
    run_name="microduck_headstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)

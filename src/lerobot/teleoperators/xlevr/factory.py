from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.teleoperators.xlevr.config_xlevr import XLeVRTeleopConfig
from lerobot.teleoperators.xlevr.xlevr_processor import XLeVRDeltaEEMapper


def make_xlevr_a10_processors(
    teleop_config: XLeVRTeleopConfig,
) -> tuple[
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    RobotProcessorPipeline[RobotObservation, RobotObservation],
]:
    teleop_action_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            XLeVRDeltaEEMapper(
                pos_scale=teleop_config.pos_scale,
                angle_scale=teleop_config.angle_scale,
                fine_trigger_threshold=teleop_config.fine_trigger_threshold,
                fine_scale_factor=teleop_config.fine_scale_factor,
                vr_to_robot_scale=teleop_config.vr_to_robot_scale,
                pos_deadzone_m=teleop_config.pos_deadzone_m,
                angle_deadzone_deg=teleop_config.angle_deadzone_deg,
                max_delta_pos_m=teleop_config.max_delta_pos_m,
                max_delta_angle_deg=teleop_config.max_delta_angle_deg,
                position_control_mode=teleop_config.position_control_mode,
                stale_timeout_s=teleop_config.stale_timeout_s,
                max_vr_speed_m_s=teleop_config.max_vr_speed_m_s,
                spike_recovery_frames=teleop_config.spike_recovery_frames,
                compute_anchor_shadow=teleop_config.compute_anchor_shadow,
                motion_shaping_mode=teleop_config.motion_shaping_mode,
                position_cutoff_hz=teleop_config.position_cutoff_hz,
                max_linear_speed_m_s=teleop_config.max_linear_speed_m_s,
                max_linear_accel_m_s2=teleop_config.max_linear_accel_m_s2,
                engage_ramp_s=teleop_config.engage_ramp_s,
                require_squeeze_to_move=teleop_config.require_squeeze_to_move,
                gripper_thumbstick_axis=teleop_config.gripper_thumbstick_axis,
                gripper_thumbstick_deadzone=teleop_config.gripper_thumbstick_deadzone,
                axis_remap=teleop_config.axis_remap,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return (
        teleop_action_processor,
        make_default_robot_action_processor(),
        make_default_robot_observation_processor(),
    )

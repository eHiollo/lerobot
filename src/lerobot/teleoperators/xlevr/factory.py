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
                vr_to_robot_scale=teleop_config.vr_to_robot_scale,
                pos_deadzone_m=teleop_config.pos_deadzone_m,
                angle_deadzone_deg=teleop_config.angle_deadzone_deg,
                max_delta_pos_m=teleop_config.max_delta_pos_m,
                max_delta_angle_deg=teleop_config.max_delta_angle_deg,
                require_squeeze_to_move=teleop_config.require_squeeze_to_move,
                gripper_thumbstick_axis=teleop_config.gripper_thumbstick_axis,
                gripper_thumbstick_deadzone=teleop_config.gripper_thumbstick_deadzone,
                axis_remap=teleop_config.axis_remap,
                delta_roll_sign=teleop_config.delta_roll_sign,
                delta_pitch_sign=teleop_config.delta_pitch_sign,
                delta_yaw_sign=teleop_config.delta_yaw_sign,
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

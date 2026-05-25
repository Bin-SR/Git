"""
Launch file: Panda Robot - MuJoCo + MoveIt2 + RViz.

Brings up:
  1. MuJoCo physics simulation (panda2.xml from mujoco_menagerie)
  2. Planning bridge (FollowJointTrajectory -> MuJoCo)
  3. robot_state_publisher (TF from URDF)
  4. MoveIt2 move_group (motion planning)
  5. RViz2 with MotionPlanning panel

Usage:
  ros2 launch panda_mujoco_bringup panda_mujoco.launch.py
  ros2 launch panda_mujoco_bringup panda_mujoco.launch.py headless:=true
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # =========== Resolve paths (evaluated at launch time) ===========
    pkg_share = get_package_share_directory("panda_mujoco_bringup")
    panda_desc_share = get_package_share_directory(
        "moveit_resources_panda_description"
    )
    panda_moveit_share = get_package_share_directory(
        "moveit_resources_panda_moveit_config"
    )

    # URDF
    urdf_path = os.path.join(panda_desc_share, "urdf", "panda.urdf.xacro")
    robot_description_content = Command([
        FindExecutable(name="xacro"), " ", urdf_path, " hand:=true",
    ])
    robot_description = {
        "robot_description": ParameterValue(
            robot_description_content, value_type=str
        )
    }

    # SRDF
    srdf_path = os.path.join(panda_moveit_share, "config", "panda.srdf")

    # RViz config
    rviz_config = os.path.join(panda_moveit_share, "config", "moveit.rviz")

    # Default MJCF: prefer mujoco_menagerie (has mesh assets)
    try:
        import mujoco_menagerie  # type: ignore
        menagerie_dir = Path(mujoco_menagerie.__file__).parent
        default_mjcf = str(menagerie_dir / "franka_emika_panda" / "panda2.xml")
    except ImportError:
        default_mjcf = os.path.join(pkg_share, "description", "panda2.xml")

    # Panda arm joint names (matching URDF)
    PANDA_ARM_JOINTS = [
        "panda_joint1", "panda_joint2", "panda_joint3",
        "panda_joint4", "panda_joint5", "panda_joint6",
        "panda_joint7",
    ]

    # =========== Launch arguments ===========
    headless_arg = DeclareLaunchArgument(
        "headless", default_value="false",
        description="Run MuJoCo without viewer window"
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz", default_value="true",
        description="Launch RViz2"
    )
    mjcf_path_arg = DeclareLaunchArgument(
        "mjcf_path",
        default_value=default_mjcf,
        description="Path to MuJoCo scene XML (panda2.xml)"
    )

    # =========== Nodes ===========

    # 1. MuJoCo simulation
    mujoco_node = Node(
        package="panda_mujoco_bringup",
        executable="mujoco_sim_node",
        name="mujoco_sim_node",
        output="screen",
        parameters=[{
            "mjcf_path": LaunchConfiguration("mjcf_path"),
            "sim_dt": 0.002,
            "publish_rate": 50.0,
            "headless": LaunchConfiguration("headless"),
        }],
    )

    # 2. Planning bridge
    planning_bridge = Node(
        package="panda_mujoco_bringup",
        executable="planning_bridge",
        name="planning_bridge",
        output="screen",
    )

    # 3. robot_state_publisher
    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    # 4. MoveIt2 move_group (controllers INLINED, not loaded from YAML)
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            robot_description,
            {"robot_description_semantic": Command([
                FindExecutable(name="xacro"), " ", srdf_path,
            ])},
            {"robot_description_kinematics": {
                "panda_arm": {
                    "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
                    "kinematics_solver_search_resolution": 0.005,
                    "kinematics_solver_timeout": 0.005,
                    "kinematics_solver_attempts": 3,
                },
            }},
            {"use_sim_time": False},
            # ---- Controller config (inlined to bypass file-loading issues) ----
            {"moveit_controller_manager":
                "moveit_simple_controller_manager/MoveItSimpleControllerManager"},
            {"moveit_manage_controllers": False},
            {"controller_names": ["panda_arm_controller"]},
            {"panda_arm_controller.type": "FollowJointTrajectory"},
            {"panda_arm_controller.action_ns": "follow_joint_trajectory"},
            {"panda_arm_controller.default": True},
            {"panda_arm_controller.joints": PANDA_ARM_JOINTS},
            # ---- Trajectory execution ----
            {"trajectory_execution.allowed_execution_duration_scaling": 2.0},
            {"trajectory_execution.allowed_goal_duration_margin": 0.5},
            {"trajectory_execution.allowed_start_tolerance": 0.01},
        ],
    )

    # 5. RViz2
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    # =========== Launch sequence ===========
    ld = LaunchDescription([
        headless_arg,
        use_rviz_arg,
        mjcf_path_arg,
        mujoco_node,
        planning_bridge,
        robot_state_pub,
        TimerAction(
            period=3.0,
            actions=[move_group_node, rviz_node],
        ),
    ])

    return ld

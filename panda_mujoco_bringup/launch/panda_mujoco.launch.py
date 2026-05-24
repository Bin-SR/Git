"""
Launch file: Panda Robot - MuJoCo + MoveIt2 + RViz.

Brings up the full stack:
  1. MuJoCo physics simulation
  2. Planning bridge (FollowJointTrajectory -> MuJoCo)
  3. robot_state_publisher (TF from URDF)
  4. MoveIt2 move_group (motion planning server)
  5. RViz2 with MotionPlanning panel

Usage:
  ros2 launch panda_mujoco_bringup panda_mujoco.launch.py
  ros2 launch panda_mujoco_bringup panda_mujoco.launch.py headless:=true  # no GUI
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # =========== Paths ===========
    pkg_share = FindPackageShare("panda_mujoco_bringup")
    panda_desc_share = FindPackageShare("moveit_resources_panda_description")

    # Load the Panda URDF
    urdf_path = PathJoinSubstitution([
        panda_desc_share, "urdf", "panda.urdf.xacro"
    ])
    robot_description_content = Command([
        FindExecutable(name="xacro"), " ", urdf_path,
        " hand:=true",
    ])
    robot_description = {"robot_description": ParameterValue(
        robot_description_content, value_type=str
    )}

    # SRDF path
    srdf_path = PathJoinSubstitution([
        FindPackageShare("moveit_resources_panda_moveit_config"),
        "config", "panda.srdf",
    ])

    # Controller config
    controllers_yaml = PathJoinSubstitution([
        pkg_share, "config", "mujoco_controllers.yaml",
    ])

    # RViz config
    rviz_config = PathJoinSubstitution([
        FindPackageShare("moveit_resources_panda_moveit_config"),
        "config", "moveit.rviz",
    ])

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
        default_value=PathJoinSubstitution([pkg_share, "description", "panda_scene.xml"]),
        description="Path to MuJoCo scene XML"
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

    # 2. Planning bridge (FollowJointTrajectory action server)
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

    # 4. MoveIt2 move_group
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
            {"robot_description_kinematics": {}},
            {"use_sim_time": False},
            {"moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager"},
            {"moveit_manage_controllers": False},
            {"trajectory_execution.allowed_execution_duration_scaling": 2.0},
            {"trajectory_execution.allowed_goal_duration_margin": 0.5},
            {"trajectory_execution.allowed_start_tolerance": 0.01},
            controllers_yaml,
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
        # Start MoveIt and RViz after a short delay
        TimerAction(
            period=3.0,
            actions=[move_group_node, rviz_node],
        ),
    ])

    return ld

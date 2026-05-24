"""
Planning Bridge: FollowJointTrajectory action server.

Bridges MoveIt2 motion planning with MuJoCo simulation.
Receives planned trajectories from MoveIt2, forwards joint position
commands to the MuJoCo simulation node, and provides execution feedback.

Architecture:
  MoveIt2 (move_group) ---FollowJointTrajectory action---> planning_bridge
                                                              |
                                                /panda_joint_commands (Float64MultiArray)
                                                              |
                                                       mujoco_sim_node
"""

import time
import threading
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


# Panda arm joint names (must match the MuJoCo model)
PANDA_ARM_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6",
    "panda_joint7",
]


class PlanningBridge(Node):
    """Action server that forwards MoveIt2 trajectories to MuJoCo."""

    def __init__(self):
        super().__init__("planning_bridge")

        # Reentrant callback group for action server
        self._cb_group = ReentrantCallbackGroup()

        # --- Action server ---
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/follow_joint_trajectory",
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )

        # --- Publisher to MuJoCo ---
        self._cmd_pub = self.create_publisher(
            Float64MultiArray, "/panda_joint_commands", 10
        )

        # --- Joint state subscriber (for feedback) ---
        self._latest_joint_state: JointState | None = None
        self._js_lock = threading.Lock()
        self._js_sub = self.create_subscription(
            JointState, "/joint_states", self._js_callback, 10
        )

        # --- Execution state ---
        self._cancel_requested = False

        self.get_logger().info("Planning bridge ready (FollowJointTrajectory action).")

    # ==================================================================
    #  Action callbacks
    # ==================================================================

    def _goal_callback(self, goal_request) -> GoalResponse:
        self.get_logger().info(
            f"Received trajectory goal: {len(goal_request.trajectory.points)} waypoints"
        )
        joint_names = goal_request.trajectory.joint_names
        self.get_logger().info(f"  Joints: {joint_names}")
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        self.get_logger().info("Cancel requested")
        self._cancel_requested = True
        return CancelResponse.ACCEPT

    async def _execute_callback(self, goal_handle):
        """Execute a FollowJointTrajectory goal."""
        self._cancel_requested = False

        trajectory = goal_handle.request.trajectory
        joint_names = trajectory.joint_names
        points: List[JointTrajectoryPoint] = trajectory.points

        feedback_msg = FollowJointTrajectory.Feedback()
        result = FollowJointTrajectory.Result()

        self.get_logger().info(
            f"Executing trajectory: {len(points)} points, "
            f"joints={joint_names}"
        )

        # Build joint name -> index mapping
        name_to_idx = {n: i for i, n in enumerate(joint_names)}

        current_point = 0
        while current_point < len(points) and rclpy.ok():
            if self._cancel_requested:
                result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                result.error_string = "Execution cancelled"
                goal_handle.abort()
                self.get_logger().info("Trajectory execution aborted.")
                return result

            point = points[current_point]

            # Build command array for Panda arm joints
            cmd = self._build_command(point, joint_names, name_to_idx)

            # Send to MuJoCo
            cmd_msg = Float64MultiArray()
            cmd_msg.data = cmd
            self._cmd_pub.publish(cmd_msg)

            # Update feedback
            feedback_msg.joint_names = PANDA_ARM_JOINTS
            with self._js_lock:
                if self._latest_joint_state is not None:
                    feedback_msg.actual.positions = self._extract_arm_positions(
                        self._latest_joint_state
                    )
            feedback_msg.desired = point
            goal_handle.publish_feedback(feedback_msg)

            # Wait for the waypoint duration
            time_from_start = point.time_from_start
            dt = time_from_start.sec + time_from_start.nanosec * 1e-9
            if current_point > 0:
                prev_time = points[current_point - 1].time_from_start
                prev_dt = prev_time.sec + prev_time.nanosec * 1e-9
                wait_time = dt - prev_dt
            else:
                wait_time = dt

            if wait_time > 0:
                # Sleep in small chunks to remain responsive
                slept = 0.0
                while slept < wait_time and not self._cancel_requested:
                    time.sleep(min(0.01, wait_time - slept))
                    slept += 0.01

            current_point += 1

        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = ""
        goal_handle.succeed()
        self.get_logger().info("Trajectory execution completed successfully.")
        return result

    # ==================================================================
    #  Helpers
    # ==================================================================

    def _build_command(
        self,
        point: JointTrajectoryPoint,
        joint_names: List[str],
        name_to_idx: dict,
    ) -> List[float]:
        """Build a 7-element command array from a trajectory point."""
        cmd = [0.0] * len(PANDA_ARM_JOINTS)
        for i, arm_joint in enumerate(PANDA_ARM_JOINTS):
            if arm_joint in name_to_idx:
                idx = name_to_idx[arm_joint]
                cmd[i] = point.positions[idx] if idx < len(point.positions) else 0.0
        return cmd

    def _extract_arm_positions(self, js: JointState) -> List[float]:
        """Extract the 7 arm joint positions from a JointState message."""
        positions = []
        for arm_joint in PANDA_ARM_JOINTS:
            if arm_joint in js.name:
                idx = js.name.index(arm_joint)
                positions.append(js.position[idx] if idx < len(js.position) else 0.0)
            else:
                positions.append(0.0)
        return positions

    def _js_callback(self, msg: JointState) -> None:
        with self._js_lock:
            self._latest_joint_state = msg


def main(args=None):
    rclpy.init(args=args)
    node = PlanningBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""
MuJoCo simulation node for Franka Emika Panda robot arm.

Runs physics simulation, publishes joint states and TF transforms,
and accepts joint position commands to drive the robot.

MuJoCo model reference: mujoco_menagerie / franka_emika_panda
"""

import os
import time
import math
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

import mujoco
import numpy as np


# Panda joint names (7 arm joints + 2 finger joints)
PANDA_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2",
]

# Number of arm joints (for command interface)
NUM_ARM_JOINTS = 7


class MujocoSimNode(Node):
    """MuJoCo physics simulation node with joint state publishing."""

    def __init__(self):
        super().__init__("mujoco_sim_node")

        # --- Parameters ---
        self.declare_parameter("mjcf_path", "")
        self.declare_parameter("sim_dt", 0.002)          # physics timestep
        self.declare_parameter("publish_rate", 50.0)      # Hz for joint_states
        self.declare_parameter("headless", False)
        self.declare_parameter("render_every", 1)         # render every N steps

        self._sim_dt = self.get_parameter("sim_dt").value
        self._publish_rate = self.get_parameter("publish_rate").value
        self._headless = self.get_parameter("headless").value
        self._render_every = self.get_parameter("render_every").value

        # --- Load MuJoCo model ---
        mjcf_path = self.get_parameter("mjcf_path").value
        if not mjcf_path:
            # Default: look in the package''s description folder
            pkg_dir = Path(__file__).parent.parent
            mjcf_path = str(pkg_dir / "description" / "panda_scene.xml")

        if not os.path.exists(mjcf_path):
            self.get_logger().error(f"MJCF not found: {mjcf_path}")
            raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

        self.get_logger().info(f"Loading MuJoCo model: {mjcf_path}")
        self._model = mujoco.MjModel.from_xml_path(mjcf_path)
        self._data = mujoco.MjData(self._model)

        self._n_actuators = self._model.nu
        self.get_logger().info(
            f"Model loaded: {self._model.nq} DoF, {self._n_actuators} actuators"
        )

        # --- Joint mapping ---
        # Map ROS joint names -> MuJoCo actuator indices
        self._joint_name_to_actuator: dict[str, int] = {}
        for i in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and name in PANDA_JOINT_NAMES:
                self._joint_name_to_actuator[name] = i

        self._controlled_joints = list(self._joint_name_to_actuator.keys())
        self.get_logger().info(f"Controlled joints: {self._controlled_joints}")

        # Joint limits for safety
        self._joint_lower = np.zeros(self._n_actuators)
        self._joint_upper = np.zeros(self._n_actuators)
        for name, idx in self._joint_name_to_actuator.items():
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                qpos_idx = self._model.jnt_qposadr[jid]
                self._joint_lower[idx] = self._model.jnt_range[jid, 0]
                self._joint_upper[idx] = self._model.jnt_range[jid, 1]

        # Command buffer: target position for each actuator
        self._target_positions = np.zeros(self._n_actuators)

        # --- Subscribers ---
        self._cmd_sub = self.create_subscription(
            Float64MultiArray,
            "/panda_joint_commands",
            self._command_callback,
            10,
        )

        # --- Publishers ---
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._joint_state_pub = self.create_publisher(
            JointState, "/joint_states", reliable_qos
        )

        # --- TF broadcaster ---
        self._tf_broadcaster = TransformBroadcaster(self)

        # --- Timers ---
        self._physics_timer = self.create_timer(self._sim_dt, self._physics_step)
        self._publish_timer = self.create_timer(
            1.0 / self._publish_rate, self._publish_joint_states
        )

        # --- Rendering ---
        self._viewer: Optional[mujoco.MjViewer] = None
        self._renderer: Optional[mujoco.Renderer] = None
        self._step_count = 0

        if not self._headless:
            self._viewer = mujoco.MjViewer(self._model, self._data)
            self._viewer._render_every = self._render_every

        self.get_logger().info("MuJoCo simulation node ready.")

    # ------------------------------------------------------------------
    def _command_callback(self, msg: Float64MultiArray) -> None:
        """Receive joint position commands (arm joints only, 7 values)."""
        cmd = np.array(msg.data, dtype=np.float64)
        if len(cmd) < NUM_ARM_JOINTS:
            self.get_logger().warn(
                f"Expected {NUM_ARM_JOINTS} arm joints, got {len(cmd)}"
            )
            return

        # Apply commands to the first 7 actuators (arm joints)
        for i, name in enumerate(PANDA_JOINT_NAMES[:NUM_ARM_JOINTS]):
            if name in self._joint_name_to_actuator:
                idx = self._joint_name_to_actuator[name]
                clamped = np.clip(cmd[i], self._joint_lower[idx], self._joint_upper[idx])
                self._target_positions[idx] = clamped

    # ------------------------------------------------------------------
    def _physics_step(self) -> None:
        """Run one physics step with position control."""
        # Apply position servo (simple P-controller) for controlled joints
        kp = 200.0  # position gain
        kd = 10.0   # velocity damping

        for name, act_idx in self._joint_name_to_actuator.items():
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            qpos_idx = self._model.jnt_qposadr[jid]
            qvel_idx = self._model.jnt_dofadr[jid]

            pos_error = self._target_positions[act_idx] - self._data.qpos[qpos_idx]
            vel = self._data.qvel[qvel_idx]
            self._data.ctrl[act_idx] = kp * pos_error - kd * vel

        mujoco.mj_step(self._model, self._data)

        # Render if viewer exists
        if self._viewer is not None and self._step_count % self._render_every == 0:
            self._viewer.render()

        self._step_count += 1

    # ------------------------------------------------------------------
    def _publish_joint_states(self) -> None:
        """Publish current joint positions and velocities."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = PANDA_JOINT_NAMES
        msg.position = []
        msg.velocity = []
        msg.effort = []

        for name in PANDA_JOINT_NAMES:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                msg.position.append(0.0)
                msg.velocity.append(0.0)
                msg.effort.append(0.0)
            else:
                qpos_idx = self._model.jnt_qposadr[jid]
                qvel_idx = self._model.jnt_dofadr[jid]
                msg.position.append(float(self._data.qpos[qpos_idx]))
                msg.velocity.append(float(self._data.qvel[qvel_idx]))
                # effort = actuator force
                if name in self._joint_name_to_actuator:
                    act_idx = self._joint_name_to_actuator[name]
                    msg.effort.append(float(self._data.actuator_force[act_idx]))
                else:
                    msg.effort.append(0.0)

        self._joint_state_pub.publish(msg)
        self._broadcast_base_tf()

    # ------------------------------------------------------------------
    def _broadcast_base_tf(self) -> None:
        """Broadcast world -> panda_link0 TF."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = "panda_link0"
        # Panda base is typically at origin; adjust if your scene moves it
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(t)

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if self._viewer:
            self._viewer.close()
        self.get_logger().info("MuJoCo simulation node stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

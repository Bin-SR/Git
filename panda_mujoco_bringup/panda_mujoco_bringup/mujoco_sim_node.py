"""MuJoCo simulation node for Franka Emika Panda robot arm.

Runs physics simulation, publishes joint states and TF transforms,
and accepts joint position commands to drive the robot.
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


# ---- ROS / URDF joint names (must match moveit_resources_panda_description) ----
PANDA_ARM_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6",
    "panda_joint7",
]

PANDA_FINGER_JOINTS = [
    "panda_finger_joint1",
    "panda_finger_joint2",
]

PANDA_JOINT_NAMES = PANDA_ARM_JOINTS + PANDA_FINGER_JOINTS

# MuJoCo joint names may differ from ROS names; map ROS -> MuJoCo where needed
ROS_TO_MUJOCO_JOINT = {
    "panda_finger_joint1": "finger_joint1",
    "panda_finger_joint2": "finger_joint2",
}

NUM_ARM_JOINTS = 7


class MujocoSimNode(Node):
    """MuJoCo physics simulation node with joint state publishing."""

    def __init__(self):
        super().__init__("mujoco_sim_node")

        # --- Parameters ---
        self.declare_parameter("mjcf_path", "")
        self.declare_parameter("sim_dt", 0.002)
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("headless", False)
        self.declare_parameter("render_every", 1)

        self._sim_dt = self.get_parameter("sim_dt").value
        self._publish_rate = self.get_parameter("publish_rate").value
        self._headless = self.get_parameter("headless").value
        self._render_every = self.get_parameter("render_every").value

        # --- Load MuJoCo model ---
        mjcf_path = self.get_parameter("mjcf_path").value
        if not mjcf_path:
            pkg_dir = Path(__file__).parent.parent
            mjcf_path = str(pkg_dir / "description" / "panda2.xml")

        if not os.path.exists(mjcf_path):
            self.get_logger().error(f"MJCF not found: {mjcf_path}")
            raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

        self.get_logger().info(f"Loading MuJoCo model: {mjcf_path}")
        self._model = mujoco.MjModel.from_xml_path(mjcf_path)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        self._n_actuators = self._model.nu
        self.get_logger().info(
            f"Model loaded: {self._model.nq} DoF, {self._n_actuators} actuators"
        )

        # --- Joint -> actuator mapping via transmission targets ---
        self._ros_joint_to_actuator: dict[str, int] = {}
        for act_idx in range(self._model.nu):
            trn_type = self._model.actuator_trntype[act_idx]
            trn_id = self._model.actuator_trnid[act_idx, 0]

            # Direct joint transmission (mjTRN_JOINT = 0, mjTRN_JOINTINPARENT = 1)
            if trn_type in (mujoco.mjtTrn.mjTRN_JOINT,
                            mujoco.mjtTrn.mjTRN_JOINTINPARENT):
                jnt_name = mujoco.mj_id2name(
                    self._model, mujoco.mjtObj.mjOBJ_JOINT, trn_id
                )
                if jnt_name is None:
                    continue
                # Reverse-map: MuJoCo joint name -> ROS joint name
                ros_name = jnt_name
                for k, v in ROS_TO_MUJOCO_JOINT.items():
                    if v == jnt_name:
                        ros_name = k
                        break
                if ros_name in PANDA_JOINT_NAMES:
                    self._ros_joint_to_actuator[ros_name] = act_idx

        # Fallback: if finger joints are tendon-driven, map last actuator to fingers
        for finger in PANDA_FINGER_JOINTS:
            if finger not in self._ros_joint_to_actuator:
                self._ros_joint_to_actuator[finger] = self._n_actuators - 1

        self._controlled_joints = list(self._ros_joint_to_actuator.keys())
        self.get_logger().info(f"Controlled joints: {self._controlled_joints}")

        # --- Joint limits ---
        self._joint_lower = np.zeros(self._n_actuators)
        self._joint_upper = np.zeros(self._n_actuators)
        for ros_name, act_idx in self._ros_joint_to_actuator.items():
            mj_name = ROS_TO_MUJOCO_JOINT.get(ros_name, ros_name)
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
            if jid >= 0:
                qpos_idx = self._model.jnt_qposadr[jid]
                self._joint_lower[act_idx] = self._model.jnt_range[jid, 0]
                self._joint_upper[act_idx] = self._model.jnt_range[jid, 1]

        # Command buffer: initialise to current positions so controller starts at rest
        self._target_positions = np.zeros(self._n_actuators)
        for ros_name, act_idx in self._ros_joint_to_actuator.items():
            mj_name = ROS_TO_MUJOCO_JOINT.get(ros_name, ros_name)
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
            if jid >= 0:
                qpos_idx = self._model.jnt_qposadr[jid]
                self._target_positions[act_idx] = float(self._data.qpos[qpos_idx])

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
        self._step_count = 0

        if not self._headless:
            self._viewer = mujoco.MjViewer(self._model, self._data)
            self._viewer._render_every = self._render_every

        self.get_logger().info("MuJoCo simulation node ready.")

    # ------------------------------------------------------------------
    def _command_callback(self, msg: Float64MultiArray) -> None:
        """Receive 7-element arm joint position commands."""
        cmd = np.array(msg.data, dtype=np.float64)
        if len(cmd) < NUM_ARM_JOINTS:
            self.get_logger().warn(
                f"Expected {NUM_ARM_JOINTS} arm joints, got {len(cmd)}"
            )
            return

        for i, ros_name in enumerate(PANDA_ARM_JOINTS):
            if ros_name in self._ros_joint_to_actuator:
                act_idx = self._ros_joint_to_actuator[ros_name]
                clamped = np.clip(
                    cmd[i], self._joint_lower[act_idx], self._joint_upper[act_idx]
                )
                self._target_positions[act_idx] = clamped

    # ------------------------------------------------------------------
    def _physics_step(self) -> None:
        """Run one physics step with P-controller position servo."""
        kp = 200.0
        kd = 10.0

        for ros_name, act_idx in self._ros_joint_to_actuator.items():
            mj_name = ROS_TO_MUJOCO_JOINT.get(ros_name, ros_name)
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
            if jid < 0:
                continue
            qpos_idx = self._model.jnt_qposadr[jid]
            qvel_idx = self._model.jnt_dofadr[jid]

            pos_error = self._target_positions[act_idx] - self._data.qpos[qpos_idx]
            vel = self._data.qvel[qvel_idx]
            self._data.ctrl[act_idx] = kp * pos_error - kd * vel

        mujoco.mj_step(self._model, self._data)

        if self._viewer is not None and self._step_count % self._render_every == 0:
            self._viewer.render()

        self._step_count += 1

    # ------------------------------------------------------------------
    def _publish_joint_states(self) -> None:
        """Publish current joint positions using ROS joint names."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = PANDA_JOINT_NAMES
        msg.position = []
        msg.velocity = []
        msg.effort = []

        for ros_name in PANDA_JOINT_NAMES:
            mj_name = ROS_TO_MUJOCO_JOINT.get(ros_name, ros_name)
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
            if jid < 0:
                msg.position.append(0.0)
                msg.velocity.append(0.0)
                msg.effort.append(0.0)
            else:
                qpos_idx = self._model.jnt_qposadr[jid]
                qvel_idx = self._model.jnt_dofadr[jid]
                msg.position.append(float(self._data.qpos[qpos_idx]))
                msg.velocity.append(float(self._data.qvel[qvel_idx]))
                if ros_name in self._ros_joint_to_actuator:
                    act_idx = self._ros_joint_to_actuator[ros_name]
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

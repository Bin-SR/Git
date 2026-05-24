# Panda MuJoCo Bringup

Panda 机械臂：**RViz 运动规划 + MuJoCo 物理仿真同步**。

在 RViz 中使用 MoveIt2 MotionPlanning 面板规划轨迹，执行时 MuJoCo 仿真环境中的 Panda 机械臂同步运动。可作为具身智能案例的基础框架。

## 架构

```
┌──────────┐    FollowJointTrajectory    ┌──────────────────┐
│  MoveIt2 │ ──────────────────────────> │ planning_bridge  │
│ move_group│ <─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │ (action server)   │
└──────────┘          feedback           └────────┬─────────┘
                                                  │ /panda_joint_commands
┌──────────┐    /joint_states + TF       ┌───────▼──────────┐
│   RViz2  │ <────────────────────────── │ mujoco_sim_node  │
│ MotionPl │                             │ (physics + render)│
└──────────┘                             └──────────────────┘
```

- **mujoco_sim_node** — 运行 MuJoCo 物理引擎，发布 `/joint_states` 和 TF，接收关节位置指令
- **planning_bridge** — `FollowJointTrajectory` action server，将 MoveIt2 规划的轨迹转发给 MuJoCo
- **MoveIt2** — 运动规划引擎（OMPL），碰撞检测，轨迹生成
- **RViz2** — 可视化 + MotionPlanning 交互面板

## 依赖

```bash
# ROS2 Humble
sudo apt install ros-humble-moveit ros-humble-moveit-resources
sudo apt install ros-humble-joint-state-publisher-gui ros-humble-robot-state-publisher
sudo apt install ros-humble-rviz2 ros-humble-xacro

# Python
pip install mujoco mujoco-menagerie
```

## 安装

```bash
cd ~/ros2_ws/src
ln -s /path/to/panda_mujoco_bringup .
cd ~/ros2_ws
colcon build --packages-select panda_mujoco_bringup --symlink-install
source install/setup.bash
```

## 使用

### 1. 启动完整系统

```bash
ros2 launch panda_mujoco_bringup panda_mujoco.launch.py
```

不带 GUI：
```bash
ros2 launch panda_mujoco_bringup panda_mujoco.launch.py headless:=true
```

### 2. 在 RViz 中进行运动规划

1. RViz 打开后会显示 Panda 机械臂和 MuJoCo 场景中的物体
2. 在 **MotionPlanning** 面板中：
   - 选择 **Planning Group**: `panda_arm`
   - 设置 **Start State**: `<current>`
   - 设置 **Goal State**: 拖拽交互标记到目标位置，或手动输入关节角度
   - 点击 **Plan** 生成轨迹
   - 点击 **Execute** 执行 — MuJoCo 中的机器人会同步运动

### 3. 编程方式规划

```python
from moveit_py import MoveItPy, PlanningComponent

panda = MoveItPy(node_name="my_planner")
panda_arm = panda.get_planning_component("panda_arm")

# 设置目标关节角度
panda_arm.set_start_state_to_current_state()
panda_arm.set_goal_state(configuration=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# 规划并执行
plan_result = panda_arm.plan()
if plan_result:
    panda_arm.execute(plan_result.trajectory)
```

## 自定义仿真环境（具身智能基础）

### 修改 MuJoCo 场景

编辑 `description/panda_scene.xml`，添加自定义物体、传感器等：

```xml
<!-- 添加可交互物体 -->
<body name="target_object" pos="0.5 0.2 0.43">
  <freejoint/>
  <geom type="cylinder" size="0.03 0.08" rgba="1.0 0.8 0.0 1.0" mass="0.1"/>
</body>

<!-- 添加相机传感器 -->
<camera name="wrist_cam" pos="0 0 0" ... />
```

### 添加感知能力

在 `mujoco_sim_node.py` 的 `_publish_joint_states()` 中发布额外的传感器数据：

```python
# 发布 RGB 图像
from sensor_msgs.msg import Image
self._image_pub.publish(rgb_image)

# 发布物体位姿
from geometry_msgs.msg import PoseStamped
self._pose_pub.publish(object_pose)
```

### 典型具身智能扩展方向

| 方向 | 描述 |
|------|------|
| 视觉抓取 | 添加 RGB-D 相机，结合目标检测进行抓取规划 |
| 强化学习 | 通过 Gymnasium + MuJoCo 接口训练策略，ROS2 作为控制中间件 |
| 模仿学习 | 录制 joint_states 和传感器数据，训练行为克隆模型 |
| 任务规划 | 结合 LLM 进行高层任务分解，MoveIt2 执行底层运动 |
| 多机器人协作 | 在同一 MuJoCo 场景中添加多个机械臂 |

## 文件结构

```
panda_mujoco_bringup/
├── package.xml
├── setup.py / setup.cfg
├── launch/
│   └── panda_mujoco.launch.py          # 主启动文件
├── config/
│   └── mujoco_controllers.yaml          # MoveIt2 控制器配置
├── description/
│   └── panda_scene.xml                 # MuJoCo 场景 (Panda + 桌子 + 物体)
└── panda_mujoco_bringup/
    ├── __init__.py
    ├── mujoco_sim_node.py              # MuJoCo 物理仿真节点
    └── planning_bridge.py              # MoveIt2 <-> MuJoCo 桥接
```

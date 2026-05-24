from setuptools import find_packages, setup
import os
from glob import glob

package_name = "panda_mujoco_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.launch.py")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
        ),
        (
            os.path.join("share", package_name, "description"),
            glob(os.path.join("description", "*.xml")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bin_SR",
    maintainer_email="1072235132@qq.com",
    description="Panda robot: RViz + MoveIt2 + MuJoCo sync",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mujoco_sim_node = panda_mujoco_bringup.mujoco_sim_node:main",
            "planning_bridge = panda_mujoco_bringup.planning_bridge:main",
        ],
    },
)

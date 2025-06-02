from typing import Any, Optional

import jax
import jax.numpy as jnp
from brax import base

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.locomotion.walk_env import WalkEnv
from toddlerbot.reference.walk_simple_ref import WalkSimpleReference
from toddlerbot.reference.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import quat2euler, quat_inv, rotate_vec


class WalkRoughEnv(WalkEnv, env_name="walk_rough"):
    """Walk environment with ToddlerBot."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        ref_motion_type: str = "zmp",
        fixed_base: bool = False,
        add_noise: bool = True,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        """Initializes the walking controller with specified configuration and motion reference type.

        Args:
            name (str): The name of the controller.
            robot (Robot): The robot instance to be controlled.
            cfg (MJXConfig): Configuration settings for the controller.
            ref_motion_type (str, optional): Type of motion reference to use, either 'simple' or 'zmp'. Defaults to 'zmp'.
            fixed_base (bool, optional): Indicates if the robot has a fixed base. Defaults to False.
            add_noise (bool, optional): Whether to add noise to the simulation. Defaults to True.
            add_domain_rand (bool, optional): Whether to add domain randomization. Defaults to True.
            **kwargs (Any): Additional keyword arguments for the superclass initializer.

        Raises:
            ValueError: If an unknown `ref_motion_type` is provided.
        """
        super.__init__(name,robot,cfg,ref_motion_type,fixed_base,add_noise,add_domain_rand,**kwargs)

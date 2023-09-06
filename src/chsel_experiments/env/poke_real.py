import copy
import os
import rospy
import ros_numpy
from threading import Lock

from scipy.ndimage import uniform_filter
import numpy as np
# for converting strings to dictionary

from arc_utilities.listener import Listener

from arm_robots.victor import Victor
from pytorch_kinematics import transforms as tf

from geometry_msgs.msg import Pose, WrenchStamped

import logging
import torch

from arm_pytorch_utilities import tensor_utils
from sensor_msgs.msg import Image
from base_experiments import cfg
from base_experiments.env.env import TrajectoryLoader, handle_data_format_for_state_diff, EnvDataSource
from stucco.detection import ContactDetector
from stucco_experiments.env.arm_real import RealArmEnv, pose_msg_to_pos_quaternion, KukaResidualContactSensor
from chsel_experiments.env.poke_real_nonros import Levels

from victor_hardware_interface_msgs.msg import MotionStatus, ControlMode

import gym.spaces

logger = logging.getLogger(__name__)

DIR = "poke_real"


def xy_pose_distance(a: Pose, b: Pose):
    pos_distance = np.linalg.norm(ros_numpy.numpify(a.position) - ros_numpy.numpify(b.position))
    return pos_distance


class RealPokeLoader(TrajectoryLoader):
    @staticmethod
    def _info_names():
        return []

    def _process_file_raw_data(self, d):
        x = d['X']
        if self.config.predict_difference:
            y = RealPokeEnv.state_difference(x[1:], x[:-1])
        else:
            raise RuntimeError("Too hard to predict discontinuous normalized angles; use predict difference")

        xu, y, cc = self._apply_masks(d, x, y)

        return xu, y, cc


class RealPokeEnv(RealArmEnv):
    nu = 3
    nx = 3
    MAX_FORCE = 30
    MAX_GRIPPER_FORCE = 30
    MAX_PUSH_DIST = 0.05
    OPEN_ANGLE = 0.055
    CLOSE_ANGLE = 0.0

    # REST_JOINTS = [0.142, 0.453, -0.133, -1.985, 0.027, -0.875, 0.041]
    # REST_JOINTS = [0.1141799424293767, 0.45077912971064055, -0.1378142121392566, -1.9944268727534853,
    #                -0.013151296594681122, -0.8713510018630727, 0.06839882517621013]
    # base aligned - maximum workspace
    # REST_JOINTS = [0.14082464951729448,
    #                -0.3989144003142722,
    #                0.6129646414136141,
    #                -1.8655067531081384,
    #                -1.4373121948920622,
    #                0.7413360632488053,
    #                0.919484143142184]
    # world frame rectilinearly aligned
    # REST_JOINTS = [0.2828029603611603,
    #                -0.03556846047071725,
    #                0.46793766349977467,
    #                -1.9773933343184746,
    #                -2.6146387104445354,
    #                1.3326174068600685,
    #                1.6537109844559428]
    REST_JOINTS = [1.6062333802263173,
                   1.5544142508291496,
                   -2.337494964736803,
                   -1.9804216281600897,
                   0.046002837423621684,
                   -0.40735081533659745,
                   0.6873008110878134]
    # backup for elbow down configuration (not actually at rest POS, but close to it)
    # REST_JOINTS = [0.7451858757153524,
    #                1.6693149845690027,
    #                0.41778811650139713,
    #                1.6096283913057212,
    #                -0.28469437066085357,
    #                0.5413406216581451,
    #                -1.3222414050401743]

    # REST_POS = [0.530, 0, 1.011]
    # for the palm as the EE
    REST_POS = [0.77, 0.38, 1.0]
    # REST_ORIENTATION = [0, -np.pi / 2, 0]
    # for the palm as the EE
    REST_ORIENTATION = [0, 0, -np.pi / 2]
    # REST_ORIENTATION = [0.6230904, -0.3204842, 0.6724652, -0.2384087]

    # EE_LINK_NAME = "victor_left_arm_link_7"
    EE_LINK_NAME = "l_palm"
    WORLD_FRAME = "world"
    ACTIVE_MOVING_ARM = 0  # left arm

    # --- BubbleBaseEnv overrides
    @classmethod
    def get_name(cls):
        return 'real_poke_env'

    def _is_done(self, observation, a):
        return a is None or a['dxyz'] is None

    def _get_action_space(self):
        u_min, u_max = self.get_control_bounds()
        actions = gym.spaces.Box(u_min, u_max)
        return gym.spaces.Dict({'dxyz': actions})
        # u_names = self.control_names()
        # action_dict = {name: gym.spaces.Box(low=u_min[i], high=u_max[i]) for i, name in enumerate(u_names)}
        # return gym.spaces.Dict(action_dict)

    def _get_observation(self):
        obs = {}
        obs['xyz'] = self._observe_ee(return_z=True)
        obs['seed'] = self.seed
        return obs

    @staticmethod
    def state_names():
        return ['x ee (m)', 'y ee (m)', 'z ee (m)']

    @staticmethod
    def get_ee_pos(state):
        return state[:3]

    @staticmethod
    @tensor_utils.ensure_2d_input
    def get_ee_pos_states(states):
        return states[:, :3]

    @classmethod
    @tensor_utils.ensure_2d_input
    def get_state_ee_pos(cls, pos):
        raise NotImplementedError()

    @classmethod
    @handle_data_format_for_state_diff
    def state_difference(cls, state, other_state):
        """Get state - other_state in state space"""
        dpos = state[:, :3] - other_state[:, :3]
        return dpos

    @classmethod
    def state_cost(cls):
        return np.diag([1, 1, 1])

    @classmethod
    def state_distance(cls, state_difference):
        return state_difference[:, :3].norm(dim=1)

    @staticmethod
    def control_names():
        return ['d$x_r$', 'd$y_r$', 'd$z_r$']

    @staticmethod
    def get_control_bounds():
        u_min = np.array([-1, -1, -1])
        u_max = np.array([1, 1, 1])
        return u_min, u_max

    @classmethod
    @handle_data_format_for_state_diff
    def control_similarity(cls, u1, u2):
        return torch.cosine_similarity(u1, u2, dim=-1).clamp(0, 1)

    @classmethod
    def control_cost(cls):
        return np.diag([1 for _ in range(cls.nu)])

    def __init__(self, *args, seed=0, environment_level=0, vel=0.4, downsample_info=50, use_cameras=True, **kwargs):
        self.seed = seed
        level = Levels(environment_level)
        save_path = os.path.join(cfg.DATA_DIR, DIR, level.name)
        self.vel = vel
        RealArmEnv.__init__(self, *args, environment_level=level, vel=vel, **kwargs)
        # force an orientation; if None, uses the rest orientation
        self.orientation = None
        # prevent saving gigantic arrays to csv
        self.downsample_info = downsample_info
        # listen for imprints (easier than working with the data directly...)

    # def enter_cartesian_mode(self):
    #     self.robot.set_left_arm_control_mode(ControlMode.JOINT_IMPEDANCE, vel=self.vel)
    #     # , x_stiffness=1000, y_stiffness=5000, z_stiffnes=5000)

    def reset(self):
        # self.return_to_rest(self.robot.arm_group)
        self.state = self._obs()
        self.contact_detector.clear()
        return np.copy(self.state), None

    def _setup_robot_ros(self, residual_threshold=15., residual_precision=None):
        victor = Victor(force_trigger=5.0)
        self.robot = victor
        # adjust timeout according to velocity (at vel = 0.1 we expect 400s per 1m)
        self.robot.cartesian._timeout_per_m = 40 / self.vel
        victor.connect()
        self.vis.init_ros()

        self.motion_status_input_lock = Lock()
        self._temp_wrenches = []
        # subscribe to status messages
        self.right_arm_contact_listener = rospy.Subscriber(victor.ns("left_arm/motion_status"), MotionStatus,
                                                           self.contact_listener)
        self.cleaned_wrench_publisher = rospy.Publisher(victor.ns("left_arm/cleaned_wrench"), WrenchStamped,
                                                        queue_size=10)
        self.large_wrench_publisher = rospy.Publisher(victor.ns("left_arm/large_wrench"), WrenchStamped,
                                                      queue_size=10)
        self.large_wrench_world_publisher = rospy.Publisher(victor.ns("left_arm/large_wrench_world"),
                                                            WrenchStamped, queue_size=10)

        # to reset the rest pose, manually jog it there then read the values
        # rest_pose = pose_msg_to_pos_quaternion(victor.get_link_pose(self.EE_LINK_NAME))
        base_pose = pose_msg_to_pos_quaternion(victor.get_link_pose('victor_left_arm_mount'))

        # reset to rest position
        self.return_to_rest(self.robot.left_arm_group)
        status = victor.get_left_arm_status()
        canonical_joints = [status.measured_joint_position.joint_1, status.measured_joint_position.joint_2,
                            status.measured_joint_position.joint_3, status.measured_joint_position.joint_4,
                            status.measured_joint_position.joint_5, status.measured_joint_position.joint_6,
                            status.measured_joint_position.joint_7]

        self.last_ee_pos = self._observe_ee(return_z=True)
        self.REST_POS[2] = self.last_ee_pos[-1]
        self.state = self._obs()

        if residual_precision is None:
            residual_precision = np.diag([1, 1, 0, 1, 1, 1])
        # parallel visualizer for ROS and pybullet
        self._contact_detector = ContactDetector(residual_precision, window_size=50)
        self._contact_detector.register_contact_sensor(KukaResidualContactSensor("victor", residual_threshold,
                                                                                 # base_pose=base_pose,
                                                                                 default_joint_config=canonical_joints,
                                                                                 canonical_pos=self.REST_POS,
                                                                                 canonical_orientation=self.REST_ORIENTATION,
                                                                                 visualizer=self.vis))
        self.vis.sim = None

    # --- observing state from simulation
    def _obs(self):
        """Observe current state from simulator"""
        state = self._observe_ee(return_z=True)
        return state

    # --- control (commonly overridden)
    def _unpack_action(self, action):
        dx = action[0] * self.MAX_PUSH_DIST
        dy = action[1] * self.MAX_PUSH_DIST
        dz = action[2] * self.MAX_PUSH_DIST
        return dx, dy, dz

    def step(self, action, dz=0., orientation=None):
        info = self._do_action(action)
        cost, done = self.evaluate_cost(self.state, action)
        return np.copy(self.state), -cost, done, info

    def _do_action(self, action):
        self._clear_state_before_step()
        success = True
        if action is not None:
            action = np.clip(action, *self.get_control_bounds())
            # normalize action such that the input can be within a fixed range
            self.last_ee_pos = self._observe_ee(return_z=True)
            dx, dy, dz = self._unpack_action(action)

            self._single_step_contact_info = {}
            # orientation = self.orientation
            # if orientation is None:
            #     orientation = copy.deepcopy(self.REST_ORIENTATION)

            self.robot.move_delta_cartesian_impedance(self.ACTIVE_MOVING_ARM, dx=dx, dy=dy,
                                                      target_z=self.last_ee_pos[2] + dz,
                                                      stop_on_force_threshold=10.,
                                                      # stop_callback=self._stop_push,
                                                      blocking=True, step_size=0.025,
                                                      # target_orientation=orientation
                                                      )
            s = self.robot.cartesian.status
            if s.reached_joint_limit or s.reached_force_threshold or s.callback_stopped:
                success = False

        full_info = self.aggregate_info()
        info = {}
        for key, value in full_info.items():
            if not isinstance(value, np.ndarray) or len(value.shape) == 1:
                info[key] = value
            else:
                info[key] = value[::self.downsample_info]
        info['success'] = success

        return info


def filter_depth_map(depth_map, size=10):
    depth_map = depth_map.squeeze(-1)
    filtered_depth_map = uniform_filter(depth_map, size=size, mode='reflect')
    return filtered_depth_map


class RealPokeDataSource(EnvDataSource):

    @staticmethod
    def _default_data_dir():
        return DIR

    @staticmethod
    def _loader_map(env_type):
        loader_map = {RealPokeEnv: RealPokeLoader, }
        return loader_map.get(env_type, None)

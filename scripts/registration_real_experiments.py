#! /usr/bin/env python
import argparse
import time

import numpy as np

np.float = np.float64
import logging

from chsel import initialization
from chsel_experiments.env.poke_real import RealPokeEnv
from chsel_experiments.poking_controller import PokingController

# from stucco.env.real_env import VideoLogger
from base_experiments.env_getters.getter import EnvGetter
import os
from datetime import datetime

from base_experiments import cfg
from base_experiments.env.env import draw_AABB
from chsel_experiments.env import poke_real
from chsel_experiments.env import poke_real_nonros
from stucco import tracking
from stucco.tracking import ContactSet
from victor_hardware_interface_msgs.msg import ControlMode
from base_experiments.util import MakedirsFileHandler

import pytorch_seed as rand

import pytorch_volumetric as pv
from pytorch_kinematics import transforms as tf
import torch
import chsel

try:
    import rospy

    rospy.init_node("victor_retrieval", log_level=rospy.INFO)
    # without this we get not logging from the library
    import importlib

    importlib.reload(logging)
except RuntimeError as e:
    print("Proceeding without ROS: {}".format(e))

ask_before_moving = True
CONTACT_POINT_SIZE = (0.01, 0.01, 0.3)

ch = logging.StreamHandler()
fh = MakedirsFileHandler(os.path.join(cfg.LOG_DIR, "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)


class StubContactSet(ContactSet):
    def __init__(self):
        pass

    def get_batch_data_for_dynamics(self, total_num):
        pass

    def dynamics(self, x, u, contact_data):
        pass

    def update(self, x, dx, p=None, info=None):
        pass


def predetermined_poke_range():
    # y,z order of poking
    return {
        poke_real_nonros.Levels.DRILL: ((0.0, 0.05, -0.06), (0.0, 0.05, 0.08)),
        poke_real_nonros.Levels.DRILL_OPPOSITE: ((0.07, -0.11), (-0.03, 0.05, 0.13)),
        poke_real_nonros.Levels.MUSTARD: ((0.0, 0.05), (-0.02, 0.04, 0.13)),
        poke_real_nonros.Levels.MUSTARD_SIDEWAYS: ((0.05, 0.12), (-0.02, 0.04, 0.13)),
    }


class RealPokeGetter(EnvGetter):
    @staticmethod
    def dynamics_prefix() -> str:
        return "poke_real"

    @classmethod
    def env(cls, level=poke_real_nonros.Levels.MUSTARD, **kwargs):
        level = poke_real_nonros.Levels(level)
        env = poke_real.RealPokeEnv(environment_level=level, **kwargs)
        return env

    @staticmethod
    def ds(env, data_dir, **kwargs):
        return None

    @staticmethod
    def controller_options(env):
        return None

    @staticmethod
    def contact_parameters(env: poke_real.RealArmEnv, **kwargs) -> tracking.ContactParameters:
        # know we are poking a single object so can be more conservative in overestimating object length scale

        params = tracking.ContactParameters(length=1.0,  # use large length scale initially to ensure everyone is 1 body
                                            penetration_length=0.002,
                                            hard_assignment_threshold=0.4,
                                            # need higher for deformable bubble
                                            intersection_tolerance=0.010)
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(params, k, v)
        return params


class PokingControllerWrapper:
    def get_controller_params(self):
        return {'kp': self.ctrl.kp, 'x_rest': self.ctrl.x_rest}

    def __init__(self, env, *args, force_threshold=10, **kwargs):
        self.env = env
        self.force_threshold = force_threshold
        self.ctrl = PokingController(*args, **kwargs)

    @classmethod
    def get_name(cls):
        return "predefined_poking_ctrl"

    def head_to_next_poke_yz(self):
        target_pos = [self.ctrl.x_rest] + list(self.ctrl.target_yz[0])
        logger.info("Moving to next target: {}".format(target_pos))
        self.env.robot.set_control_mode(control_mode=ControlMode.JOINT_POSITION, vel=self.env.vel / 2)
        self.env.robot.plan_to_pose(self.env.robot.left_arm_group, self.env.EE_LINK_NAME,
                                    target_pos + self.env.REST_ORIENTATION)
        self.env.enter_cartesian_mode()
        self.ctrl.push_i = 0

        # wait a little bit
        rospy.sleep(0.3)
        self.env.recalibrate_static_wrench()
        self.env.reset()

    def control(self, obs, info=None):
        with self.env.motion_status_input_lock:
            if isinstance(obs, dict):
                obs = obs['xyz']

        u = [0 for _ in range(self.ctrl.nu)]
        if info is None:
            return {'dxyz': u}

        # first target
        if self.ctrl.i == 0 and not self.ctrl.done():
            self.head_to_next_poke_yz()

        self.ctrl.x_history.append(obs)

        # if len(self.ctrl.x_history) > 1:
        #     self.ctrl.update(obs, info)

        if self.ctrl.done():
            self.ctrl.mode = self.ctrl.Mode.DONE
            return {'dxyz': None}
        else:
            # push along +x
            u[0] = 1.
            self.ctrl.push_i += 1
            # ensure we do at least some probes
            if self.ctrl.push_i <= 3:
                pass
            elif not info['success'] or self.ctrl.push_i >= self.ctrl.push_forward_count or np.linalg.norm(
                    info['reaction']) > self.force_threshold:
                # directly go to next target rather than return backward
                assert isinstance(self.env, RealPokeEnv)

                logger.info(f"Finished probe at x={obs[0]} , moving back")
                # move back so planning is allowed (don't start in collision)
                u[0] = -0.7
                # max 5 steps
                for j in range(5):
                    if obs[0] > self.ctrl.x_rest:
                        break
                    obs, _, _, _ = self.env.step(u)

                self.ctrl.target_yz = self.ctrl.target_yz[1:]
                if not self.ctrl.done():
                    logger.info("Finished moving back, moving to start of next yz")
                    self.head_to_next_poke_yz()
                return {'dxyz': None}

            self.ctrl.i += 1

        if u is not None:
            self.ctrl.u_history.append(u)

        return {'dxyz': u}


def randomly_downsample(seq, seed, num=10):
    with rand.SavedRNG():
        rand.seed(seed)
        selected = np.random.permutation(range(len(seq)))
        seq = seq[selected[:num]]
    return seq


def plot_estimate_set(env: poke_real_nonros.PokeRealNoRosEnv, vis, estimate_set, ns="estimate_set", downsample=10):
    # estimate_set = estimate_set.inverse()
    vis.clear_visualization_after(ns, 0)
    estimate_set = randomly_downsample(estimate_set, 64, num=downsample)
    for i, T in enumerate(estimate_set):
        pose = tf.rotation_conversions.matrix_to_pos_rot(T)
        env.obj_factory.draw_mesh(vis, ns, pose, (0., .0, 0.8, 0.1), object_id=i)


def run_poke(env: poke_real.RealPokeEnv, env_process: poke_real_nonros.PokeRealNoRosEnv, seed=0, control_wait=0.):
    rand.seed(seed)
    batch = 30
    registration_iterations = 5

    # input("enter to start execution")
    y_order, z_order = predetermined_poke_range()[env.level]
    # these are specified relative to the rest position
    y_order = list(y + env.REST_POS[1] for y in y_order)
    z_order = list(z + env.REST_POS[2] for z in z_order)
    ctrl = PokingControllerWrapper(env, env.contact_detector, StubContactSet(), y_order=y_order, z_order=z_order,
                                   x_rest=env.REST_POS[0], push_forward_count=5)
    num_probes = len(y_order) * len(z_order)

    env.recalibrate_static_wrench()
    obs, info = env.reset()

    known_sdf_voxels = pv.VoxelSet(torch.empty(0, 3), torch.empty(0))

    previous_solutions = None
    # visualize workspace
    draw_AABB(env.vis, env_process.freespace_ranges)

    # TODO initialize transforms - maybe do so only at first contact since that's when we can narrow down its translation
    world_to_link = initialization.initialize_transform_estimates(batch, env_process.freespace_ranges,
                                                                  chsel.initialization.InitMethod.RANDOM,
                                                                  None,
                                                                  device=env_process.device,
                                                                  dtype=env_process.dtype)
    # plot these transforms
    plot_estimate_set(env_process, env.vis, world_to_link)

    while not ctrl.ctrl.done():
        action = ctrl.control(obs, info)
        action = action['dxyz']
        # finished a probe, do registration
        if action is None:
            logger.info("Finished a probe, potentially doing registration now")
            known_pos, known_sdf = known_sdf_voxels.get_known_pos_and_values()
            logger.info("Number of contact points: {}".format(len(known_pos)))
            # skip if we haven't made any contact yet
            # TODO reduce required number of points before we start registration
            if len(known_pos) > 20:
                # TODO do registration
                free_pos, _ = env_process.free_voxels.get_known_pos_and_values()
                free_sem = [chsel.SemanticsClass.FREE] * free_pos.shape[0]
                positions = torch.cat((known_pos, free_pos), dim=0)
                semantics = np.concatenate((known_sdf.cpu().numpy(), np.asarray(free_sem, dtype=object)), axis=0)
                registration = chsel.CHSEL(env_process.target_sdf, positions, semantics,
                                           free_voxels=env_process.free_voxels,
                                           known_sdf_voxels=known_sdf_voxels,
                                           qd_iterations=100, free_voxels_resolution=0.005)
                res, previous_solutions = registration.register(initial_tsf=world_to_link, batch=batch,
                                                                iterations=registration_iterations,
                                                                low_cost_transform_set=previous_solutions, )
                world_to_link = chsel.solution_to_world_to_link_matrix(res)

                # plot latest registration
                plot_estimate_set(env_process, env.vis, world_to_link)
        else:
            observation, reward, done, info = env.step(action)
            # TODO process info (query contact detector, and feed transformed known robot points as free points)
            # TODO observe contact point from detector and add it to known_sdf_voxels
            pt, _ = env.contact_detector.get_last_contact_location(visualizer=env.vis)
            if pt is not None:
                known_sdf_voxels[pt] = torch.zeros(pt.shape[0])
            # TODO observe pose and add points transformed by it to free_voxels


def main(args):
    level = task_map[args.task]
    # obj_name = poke_real.level_to_obj_map[level]

    np.set_printoptions(suppress=True, precision=2, linewidth=200)
    env = RealPokeGetter.env(level, seed=args.seed)
    env_process = poke_real_nonros.PokeRealNoRosEnv(level, device="cuda")

    # move to the actual left side
    # env.vis.clear_visualizations(["0", "0a", "1", "1a", "c", "reaction", "tmptbest", "residualmag"])
    if args.home:
        return

    run_poke(env, env_process, args.seed)
    env.vis.clear_visualizations()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run pokes with a real robot')
    parser.add_argument('--seed', type=int,
                        default=0,
                        help='random seed to run')
    # run parameters
    parser.add_argument('--name', default="", help='additional name for the experiment (concatenated with method)')
    task_map = {level.name.lower(): level for level in poke_real_nonros.Levels}
    parser.add_argument('--task', default="mustard", choices=task_map.keys(), help='what task to run')
    parser.add_argument('--home', action='store_true', help="take robot to home position instead of running a poke")
    main(parser.parse_args())

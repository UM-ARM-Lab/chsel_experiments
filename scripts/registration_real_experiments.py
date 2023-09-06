#! /usr/bin/env python
import argparse
import numpy as np
import logging

from chsel import initialization
from chsel_experiments.env.poke_real import RealPokeEnv
from chsel_experiments.poking_controller import PokingController

# from stucco.env.real_env import VideoLogger
from base_experiments.env_getters.getter import EnvGetter
import os
from datetime import datetime

from base_experiments import cfg
from chsel_experiments.env import poke_real
from chsel_experiments.env import poke_real_nonros
from stucco import tracking
from stucco.tracking import ContactSet
from victor_hardware_interface_msgs.msg import ControlMode
from base_experiments.util import MakedirsFileHandler

import pytorch_volumetric as pv
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
        poke_real_nonros.Levels.DRILL: ((0.05, -0.08), (0.0, 0.05, 0.12)),
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
        self.env.robot.set_control_mode(control_mode=ControlMode.JOINT_POSITION, vel=self.env.vel)
        self.env.robot.plan_to_pose(self.env.robot.arm_group, self.env.EE_LINK_NAME,
                                    target_pos + self.env.REST_ORIENTATION)
        self.env.enter_cartesian_mode()
        self.ctrl.push_i = 0

    def control(self, obs, info=None):
        with self.env.motion_status_input_lock:
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
            if not info['success'] or self.ctrl.push_i >= self.ctrl.push_forward_count or np.linalg.norm(
                    info['reaction']) > self.force_threshold:
                # directly go to next target rather than return backward
                assert isinstance(self.env, RealPokeEnv)

                # move back so planning is allowed (don't start in collision)
                target_pos = [self.ctrl.x_rest, self.ctrl.target_yz[0][0], obs[2]]
                self.env.robot.cartesian_impedance_raw_motion(target_pos, frame_id=self.env.EE_LINK_NAME,
                                                              ref_frame=self.env.WORLD_FRAME, blocking=True,
                                                              goal_tolerance=np.array([0.001, 0.001, 0.01]))
                self.ctrl.target_yz = self.ctrl.target_yz[1:]
                if not self.ctrl.done():
                    self.head_to_next_poke_yz()
                return {'dxyz': None}

            self.ctrl.i += 1

        if u is not None:
            self.ctrl.u_history.append(u)

        return {'dxyz': u}


def run_poke(env: poke_real.RealPokeEnv, env_process: poke_real_nonros.PokeRealNoRosEnv, seed=0, control_wait=0.):
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

    obs, info = env.reset()
    env.recalibrate_static_wrench()

    known_sdf_voxels = pv.VoxelSet(torch.empty(0), torch.empty(0))

    previous_solutions = None
    # TODO initialize transforms - maybe do so only at first contact since that's when we can narrow down its translation
    world_to_link = initialization.initialize_transform_estimates(batch, env_process.freespace_ranges,
                                                                  chsel.initialization.InitMethod.RANDOM,
                                                                  None,
                                                                  device=env_process.device,
                                                                  dtype=env_process.dtype)

    while not ctrl.ctrl.done():
        action = ctrl.control(obs, info)
        action = action['dxyz']
        if action is None:
            break
        observation, reward, done, info = env.step(action)
        # TODO process info (query contact detector, and feed transformed known robot points as free points)
        # TODO do registration
        known_pos, known_sdf = known_sdf_voxels.get_known_pos_and_values()
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

        # TODO broadcast registration


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

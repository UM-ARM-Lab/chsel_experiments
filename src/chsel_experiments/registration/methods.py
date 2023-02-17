import math

import pytorch_kinematics.transforms.rotation_conversions

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

from pytorch3d.ops import iterative_closest_point
from pytorch3d.ops.points_alignment import SimilarityTransform
import pytorch3d.transforms as tf

from chsel import quality_diversity

from base_experiments.env.env import draw_AABB
from chsel_experiments.registration.sgd import iterative_closest_point_sgd
from chsel_experiments.registration import volumetric
from chsel_experiments.registration.medial_constraints import MedialConstraintCost
from base_experiments.sdf import draw_pose_distribution
from base_experiments import cfg
from pytorch_volumetric.sdf import ObjectFrameSDF
import logging

from stucco_experiments.registration import init_random_transform_with_given_init

logger = logging.getLogger(__file__)


# from https://github.com/richardos/icp


def point_based_matching(point_pairs):
    """
    This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
    by F. Lu and E. Milios.
    :param point_pairs: the matched point pairs [((x1, y1), (x1', y1')), ..., ((xi, yi), (xi', yi')), ...]
    :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points
    """

    x_mean = 0
    y_mean = 0
    xp_mean = 0
    yp_mean = 0
    n = len(point_pairs)

    if n == 0:
        return None, None, None

    for pair in point_pairs:
        (x, y), (xp, yp) = pair

        x_mean += x
        y_mean += y
        xp_mean += xp
        yp_mean += yp

    x_mean /= n
    y_mean /= n
    xp_mean /= n
    yp_mean /= n

    s_x_xp = 0
    s_y_yp = 0
    s_x_yp = 0
    s_y_xp = 0
    for pair in point_pairs:
        (x, y), (xp, yp) = pair

        s_x_xp += (x - x_mean) * (xp - xp_mean)
        s_y_yp += (y - y_mean) * (yp - yp_mean)
        s_x_yp += (x - x_mean) * (yp - yp_mean)
        s_y_xp += (y - y_mean) * (xp - xp_mean)

    rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    translation_x = xp_mean - (x_mean * math.cos(rot_angle) - y_mean * math.sin(rot_angle))
    translation_y = yp_mean - (x_mean * math.sin(rot_angle) + y_mean * math.cos(rot_angle))

    return rot_angle, translation_x, translation_y


def icp(reference_points, points, max_iterations=100, distance_threshold=0.3, convergence_translation_threshold=1e-3,
        convergence_rotation_threshold=1e-4, point_pairs_threshold=10, verbose=False):
    """
    An implementation of the Iterative Closest Point algorithm that matches a set of M 2D points to another set
    of N 2D (reference) points.
    :param reference_points: the reference point set as a numpy array (N x 2)
    :param points: the point that should be aligned to the reference_points set as a numpy array (M x 2)
    :param max_iterations: the maximum number of iteration to be executed
    :param distance_threshold: the distance threshold between two points in order to be considered as a pair
    :param convergence_translation_threshold: the threshold for the translation parameters (x and y) for the
                                              transformation to be considered converged
    :param convergence_rotation_threshold: the threshold for the rotation angle (in rad) for the transformation
                                               to be considered converged
    :param point_pairs_threshold: the minimum number of point pairs the should exist
    :param verbose: whether to print informative messages about the process (default: False)
    :return: the transformation history as a list of numpy arrays containing the rotation (R) and translation (T)
             transformation in each iteration in the format [R | T] and the aligned points as a numpy array M x 2
    """

    transformation_history = []

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)

    for iter_num in range(max_iterations):
        if verbose:
            print('------ iteration', iter_num, '------')

        closest_point_pairs = []  # list of point correspondences for closest point rule

        distances, indices = nbrs.kneighbors(points)
        for nn_index in range(len(distances)):
            if distances[nn_index][0] < distance_threshold:
                closest_point_pairs.append((points[nn_index], reference_points[indices[nn_index][0]]))

        # if only few point pairs, stop process
        if verbose:
            print('number of pairs found:', len(closest_point_pairs))
        if len(closest_point_pairs) < point_pairs_threshold:
            if verbose:
                print('No better solution can be found (very few point pairs)!')
            break

        # compute translation and rotation using point correspondences
        closest_rot_angle, closest_translation_x, closest_translation_y = point_based_matching(closest_point_pairs)
        if closest_rot_angle is not None:
            if verbose:
                print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                print('Translation:', closest_translation_x, closest_translation_y)
        if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
            if verbose:
                print('No better solution can be found!')
            break

        # transform 'points' (using the calculated rotation and translation)
        c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
        rot = np.array([[c, -s],
                        [s, c]])
        aligned_points = np.dot(points, rot.T)
        aligned_points[:, 0] += closest_translation_x
        aligned_points[:, 1] += closest_translation_y

        # update 'points' for the next iteration
        points = aligned_points

        # update transformation history
        transformation_history.append(np.hstack((rot, np.array([[closest_translation_x], [closest_translation_y]]))))

        # check convergence
        if (abs(closest_rot_angle) < convergence_rotation_threshold) \
                and (abs(closest_translation_x) < convergence_translation_threshold) \
                and (abs(closest_translation_y) < convergence_translation_threshold):
            if verbose:
                print('Converged!')
            break

    return transformation_history, points


# from https://github.com/ClayFlannigan/icp
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Mxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp_2(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Mxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


def icp_pytorch3d(A, B, given_init_pose=None, batch=30):
    given_init_pose = init_random_transform_with_given_init(A.shape[1], batch, A.dtype, A.device,
                                                            given_init_pose=given_init_pose)
    given_init_pose = SimilarityTransform(given_init_pose[:, :3, :3],
                                          given_init_pose[:, :3, 3],
                                          torch.ones(batch, device=A.device, dtype=A.dtype))

    res = iterative_closest_point(A.repeat(batch, 1, 1), B.repeat(batch, 1, 1), init_transform=given_init_pose,
                                  allow_reflection=False)
    T = torch.eye(4, device=A.device).repeat(batch, 1, 1)
    T[:, :3, :3] = res.RTs.R.transpose(-1, -2)
    T[:, :3, 3] = res.RTs.T
    distances = res.rmse
    return T, distances


def _our_res_to_world_to_link_matrix(res):
    batch = res.RTs.T.shape[0]
    device = res.RTs.T.device
    dtype = res.RTs.T.dtype
    T = torch.eye(4, device=device, dtype=dtype).repeat(batch, 1, 1)
    T[:, :3, :3] = res.RTs.R
    T[:, :3, 3] = res.RTs.T
    return T


def icp_pytorch3d_sgd(A, B, given_init_pose=None, batch=30, **kwargs):
    # initialize transform with closed form solution
    # T, distances = icp_pytorch3d(A, B, given_init_pose=given_init_pose, batch=batch)
    # given_init_pose = T.inverse()

    given_init_pose = init_random_transform_with_given_init(A.shape[1], batch, A.dtype, A.device,
                                                            given_init_pose=given_init_pose)
    given_init_pose = SimilarityTransform(given_init_pose[:, :3, :3],
                                          given_init_pose[:, :3, 3],
                                          torch.ones(batch, device=A.device, dtype=A.dtype))

    res = iterative_closest_point_sgd(A.repeat(batch, 1, 1), B.repeat(batch, 1, 1), init_transform=given_init_pose,
                                      allow_reflection=True, **kwargs)
    T = _our_res_to_world_to_link_matrix(res)
    distances = res.rmse
    return T, distances


obj_id_map = {}


def initialize_qd_archive(T, rmse, range_pos_sigma=3):
    TT = T[:, :3, 3]
    filt = rmse < torch.quantile(rmse, 0.8)
    pos = TT[filt]
    pos_std = pos.std(dim=-2).cpu().numpy()
    centroid = pos.mean(dim=-2).cpu().numpy()

    # diff = pos - pos.mean(dim=-2)
    # s = diff.square().sum()
    # pos_total_std = (s / len(pos)).sqrt().cpu().numpy()

    # extract translation measure
    centroid = centroid
    pos_std = pos_std
    ranges = np.array((centroid - pos_std * range_pos_sigma, centroid + pos_std * range_pos_sigma)).T
    return ranges


def icp_volumetric(volumetric_cost, A, given_init_pose=None, batch=30, optimization=volumetric.Optimization.SGD,
                   debug=False, range_pos_sigma=3, **kwargs):
    given_init_pose = init_random_transform_with_given_init(A.shape[1], batch, A.dtype, A.device,
                                                            given_init_pose=given_init_pose)
    given_init_pose = SimilarityTransform(given_init_pose[:, :3, :3],
                                          given_init_pose[:, :3, 3],
                                          torch.ones(batch, device=A.device, dtype=A.dtype))

    if optimization == volumetric.Optimization.SGD:
        res = volumetric.iterative_closest_point_volumetric(volumetric_cost, A.repeat(batch, 1, 1),
                                                            init_transform=given_init_pose,
                                                            **kwargs)
    elif optimization == volumetric.Optimization.CMAES:
        op = quality_diversity.CMAES(volumetric_cost, A.repeat(batch, 1, 1), init_transform=given_init_pose,
                                     savedir=cfg.DATA_DIR, **kwargs)
        res = op.run()
    elif optimization in [volumetric.Optimization.CMAME, volumetric.Optimization.CMAMEGA]:
        # feed it the result of SGD optimization
        res_init = volumetric.iterative_closest_point_volumetric(volumetric_cost, A.repeat(batch, 1, 1),
                                                                 init_transform=given_init_pose,
                                                                 **kwargs)

        # create range based on SGD results (where are good fits)
        # filter outliers out based on RMSE
        T = _our_res_to_world_to_link_matrix(res_init)
        archive_range = initialize_qd_archive(T, res_init.rmse, range_pos_sigma=range_pos_sigma)
        # bins_per_std = 40
        # bins = pos_std / pos_total_std * bins_per_std
        # bins = np.round(bins).astype(int)
        bins = 40
        logger.info("QD position bins %s %s", bins, archive_range)

        method_specific_kwargs = {}
        if optimization == volumetric.Optimization.CMAME:
            QD = quality_diversity.CMAME
        else:
            QD = quality_diversity.CMAMEGA
        op = QD(volumetric_cost, A.repeat(batch, 1, 1), init_transform=given_init_pose,
                iterations=500, num_emitters=1, bins=bins,
                ranges=archive_range, savedir=cfg.DATA_DIR, **method_specific_kwargs, **kwargs)

        if debug:
            # visualize archive
            z = 0.1
            aabb = np.concatenate((op.ranges, np.array((z, z)).reshape(1, -1)), axis=0)
            draw_AABB(volumetric_cost.vis, aabb)
            T = _our_res_to_world_to_link_matrix(res_init)
            draw_pose_distribution(T.inverse(), obj_id_map, volumetric_cost.vis, volumetric_cost.obj_factory,
                                   sequential_delay=None, show_only_latest=False)

        x = op.get_numpy_x(res_init.RTs.R, res_init.RTs.T)
        op.add_solutions(x)
        res = op.run()

        if debug:
            print(res_init.rmse)
            print(res.rmse)
            T = _our_res_to_world_to_link_matrix(res)
            draw_pose_distribution(T.inverse(), obj_id_map, volumetric_cost.vis, volumetric_cost.obj_factory,
                                   sequential_delay=None, show_only_latest=False)

    elif optimization == volumetric.Optimization.SVGD:
        res = volumetric.iterative_closest_point_volumetric_svgd(volumetric_cost, A.repeat(batch, 1, 1),
                                                                 init_transform=given_init_pose,
                                                                 **kwargs)
    else:
        raise RuntimeError(f"Unsupported optimization method {optimization}")

    T = _our_res_to_world_to_link_matrix(res)
    distances = res.rmse
    return T, distances


def icp_medial_constraints(obj_sdf: ObjectFrameSDF, medial_balls, A, given_init_pose=None, batch=30, vis=None,
                           obj_factory=None, cmame=False, **kwargs):
    given_init_pose = init_random_transform_with_given_init(A.shape[1], batch, A.dtype, A.device,
                                                            given_init_pose=given_init_pose)
    given_init_pose = SimilarityTransform(given_init_pose[:, :3, :3],
                                          given_init_pose[:, :3, 3],
                                          torch.ones(batch, device=A.device, dtype=A.dtype))

    medial_constraint_cost = MedialConstraintCost(medial_balls, obj_sdf, A, vis=vis, obj_factory=obj_factory,
                                                  debug=False)
    op = quality_diversity.CMAES(medial_constraint_cost, A.repeat(batch, 1, 1), init_transform=given_init_pose,
                                 savedir=cfg.DATA_DIR, **kwargs)
    res = op.run()

    # do extra QD optimization
    if cmame:
        T = _our_res_to_world_to_link_matrix(res)
        archive_range = initialize_qd_archive(T, res.rmse, range_pos_sigma=3)
        bins = 40
        logger.info("QD position bins %s %s", bins, archive_range)
        op = quality_diversity.CMAME(medial_constraint_cost, A.repeat(batch, 1, 1), init_transform=given_init_pose,
                                     iterations=500, num_emitters=1, bins=bins,
                                     ranges=archive_range, **kwargs)

        x = op.get_numpy_x(res.RTs.R, res.RTs.T)
        op.add_solutions(x)
        res = op.run()

    T = _our_res_to_world_to_link_matrix(res)
    distances = res.rmse
    return T, distances


def icp_mpc(A, B, cost_func, given_init_pose=None, batch=30, horizon=10, num_samples=100,
            max_rot_mag=0.05, max_trans_mag=0.1, steps=30,
            rot_sigma=0.01, trans_sigma=0.03, draw_mesh=None):
    from pytorch_mppi import mppi
    # use the given cost function and run MPC with deltas on the transforms to optimize it
    given_init_pose = init_random_transform_with_given_init(A.shape[1], batch, A.dtype, A.device,
                                                            given_init_pose=given_init_pose)

    d = A.device
    nx = 16  # 4 x 4 transformation matrix
    # TODO use axis angle representation for delta rotation
    nu_r = 3
    nu_t = 3
    noise_sigma = torch.diag(torch.tensor([rot_sigma] * nu_r + [trans_sigma] * nu_t, device=d))

    def dynamics(transform, delta_transform):
        # convert flattened delta to its separate 6D rotations and 3D translations
        N = transform.shape[0]
        dH = torch.eye(4, dtype=transform.dtype, device=d).repeat(N, 1, 1)
        # dH[:, :3, :3] = rotation_6d_to_matrix(delta_transform[:, :6])
        # dH[:, :3, 3] = delta_transform[:, 6:]
        dH[:, :3, :3] = tf.euler_angles_to_matrix(delta_transform[:, :3], "XYZ")
        dH[:, :3, 3] = delta_transform[:, 3:]
        return (dH @ transform.view(dH.shape)).view(N, nx)
        # return (transform.view(dH.shape) @ dH).view(N, nx)

    ctrl = mppi.MPPI(dynamics, cost_func, nx, noise_sigma, num_samples=num_samples, horizon=horizon,
                     device=d,
                     u_min=torch.tensor([-max_rot_mag] * nu_r + [-max_trans_mag] * nu_t, device=d),
                     u_max=torch.tensor([max_rot_mag] * nu_r + [max_trans_mag] * nu_t, device=d))

    visual_obj_id_map = {}
    obs = given_init_pose[0].inverse().contiguous()
    for i in range(steps):
        delta_H = ctrl.command(obs.view(-1))
        rollout = ctrl.get_rollouts(obs.view(-1)).squeeze()
        # visualize current state after each MPC step
        if draw_mesh is not None:
            pos, rot = pytorch_kinematics.transforms.rotation_conversions.matrix_to_pos_rot(obs.view(4, 4).inverse())
            id = visual_obj_id_map.get(i, None)
            visual_obj_id_map[i] = draw_mesh("state", (pos, rot), (0., i / steps, i / steps, 0.5),
                                             object_id=id)
            # visualize rollout
            # for j, x in enumerate(rollout):
            #     pos, rot = util.matrix_to_pos_rot(x.view(4, 4).inverse())
            #     internal_id = (j + 1) * steps
            #     id = visual_obj_id_map.get(internal_id, None)
            #     visual_obj_id_map[internal_id] = draw_mesh("rollout", (pos, rot), (
            #     (j + 1) / (len(rollout) + 1), i / steps, i / steps, 0.5),
            #                                                object_id=id)

        obs = dynamics(obs.unsqueeze(0), delta_H.unsqueeze(0))

    # TODO can't really do anything with the batch size?
    obs = obs.view(4, 4).repeat(batch, 1, 1)
    return obs, cost_func(obs)

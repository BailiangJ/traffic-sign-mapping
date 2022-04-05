import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from helper import load_camera, load_poses, get_extrinsic_params, get_intrinsic_params, backward_matrix
from typing import Tuple
import matplotlib.pyplot as plt


def load_initialization(init_path: str, frame_dict_path: str, camera_set_file: str, pose_file: str):
    """
    Load bundle adjustment initialization data
    Args:
        init_path: str, json file saving triangluation result
        frame_dict_path: str, json file saving absoute index of every frame
        camera_set_file: str, csv file of camera parameters, 'camera_set.csv'
        pose_file: str, '*.poses'
    Returns:
        points_3d: np.ndarray (n_traffic_sign, 3)
        camera_params: np.ndarray (n_camera, 24)
        extrinsic_params: np.ndarray (n_frame, 12)
        point_indices: np.ndarray (n_observations, )
        camera_indices: np.ndarray (n_observations, )
        frame_indices: np.ndarray (n_observations, )
        points_2d: np.ndarray(n_observations, )
        gt_3d: np.ndarray (n_traffic_sign, 3), ground truth 3d coordinate of traffic sign
    """
    init_data = json.load(open(init_path, 'r'))
    init_data = init_data['init_data']
    frame_dict = json.load(open(frame_dict_path, 'r'))
    pose = load_poses(pose_file)
    camera_params = load_camera(camera_set_file)

    frame_ids = list(map(int, frame_dict.keys()))
    extrinsic_params = pose.loc[frame_ids].values
    point_ind = 0
    points_3d = []
    point_indices = []
    camera_indices = []
    frame_indices = []
    points_2d = []
    gt_3d = []
    for sign in init_data:
        if not np.isnan(sign["init_3d"]).any():
            points_3d.append(sign["init_3d"])
            gt_3d.append(sign["gt_3d"])
            for observation in sign["bounding_boxes"]:
                bbox = observation["bounding_box"]
                points_2d.append([(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2])
                camera_indices.append(observation["camera_id"])
                frame_indices.append(frame_dict[str(observation["frame_id"])])
                point_indices.append(point_ind)
            point_ind += 1
    points_3d = np.array(points_3d)

    return points_3d, camera_params, extrinsic_params, \
           point_indices, camera_indices, frame_indices, np.array(points_2d), np.array(gt_3d)


def project(point_3d: np.ndarray, camera_params: np.ndarray, extrinsic_params: np.ndarray, point_2d: np.ndarray):
    """
    Project a 3D point onto image plane and compute its residual with ground truth
    Args:
        point_3d: np.ndarray, (3,)
        camera_params: np.ndarray, (24,), K, d, Rc, tc
        extrinsic_params: np.ndarray, (12,), Rw, tw
        point_2d: np.ndarray, (2,), ground truth 2D coordinate on image plane

    Returns:
        residual: np.ndarray, (2,), residual between projection and ground truth
    """
    K, d, Rc, tc = get_intrinsic_params(camera_params)
    Rw, tw = get_extrinsic_params(extrinsic_params)
    Pc = backward_matrix(Rc, tc)
    Pw = backward_matrix(Rw, tw)

    point_3d = np.append(point_3d, 1)
    U = Pc @ Pw @ point_3d.T

    # camera distortion
    r_square = (U[0] / U[2]) ** 2 + (U[1] / U[2]) ** 2
    print(f"r_square:{r_square}")
    dis_coeff = 1 + d[0] * r_square + d[1] * (r_square ** 2) + d[2] * (r_square ** 3)
    Dx, Dy = dis_coeff * U[:2]
    U = np.array([Dx, Dy, U[2]], dtype=np.float64)

    point_proj = K @ U
    point_proj = point_proj[:2] / point_proj[2]
    return point_proj - point_2d


def project_batch(params: Tuple[np.ndarray, np.ndarray],
                  n_points: int,
                  camera_params: np.ndarray,
                  extrinsic_params: np.ndarray,
                  point_indices: np.ndarray,
                  camera_indices: np.ndarray,
                  frame_indices: np.ndarray,
                  points_2d: np.ndarray,
                  mode: str = "both"):
    """
    Project a batch of 3D points onto image plane and compute the residuals
    Args:
        params:
            points_3d: np.ndarray, (n_points,3), 3D coordinates of input points
            extrinsic_params: np.ndarray, (n_frame, 12), pose parameters of each frame
        n_points: int, number of input points
        camera_params: np.ndarray, (n_camera, 24), intrinsic parameters of each camera
        extrinsic_params: np.ndarray, (n_frame, 12), pose parameters of each frame
        point_indices: np.ndarray, (n_observations,), index of each observed points
        camera_indices: np.ndarray, (n_observations,), camera index of each observation
        frame_indices: np.ndarray, (n_observations,), frame index of each observation
        points_2d: np.ndarray, (n_observation, 2)
        mode: str,
              "both": params contain both the 3d points and extrinsic parameters
              "point": params only contain the 3d points
              "translation": params including 3d points and translation of extrinsic parameters

    Returns:

    """
    if mode == "both":
        points_3d = params[:n_points * 3].reshape(-1, 3)
        extrinsic_params = params[n_points * 3:].reshape(-1, 12)
    elif mode == "point":
        points_3d = params.reshape(-1, 3)
    elif mode == "translation":
        points_3d = params[:n_points * 3].reshape(-1, 3)
        tw = params[n_points * 3:].reshape(-1, 3)
        extrinsic_params = np.hstack((extrinsic_params[:, :9], tw))
    else:
        raise ValueError
    # intrinsic params
    K = camera_params[camera_indices, :9].reshape(-1, 3, 3)
    d = camera_params[camera_indices, 9:12].reshape(-1, 3)
    Rc = camera_params[camera_indices, 12:21].reshape(-1, 3, 3)
    tc = camera_params[camera_indices, 21:].reshape(-1, 3)
    # extrinsic params
    Rw = extrinsic_params[frame_indices, :9].reshape(-1, 3, 3)
    tw = extrinsic_params[frame_indices, 9:].reshape(-1, 3)
    # projection
    points_3d = points_3d[point_indices]
    points_3d = np.einsum('ijk,ij->ik', Rw, points_3d) - np.einsum('ijk,ij->ik', Rw, tw)
    U = np.einsum('ijk,ij->ik', Rc, points_3d) - np.einsum('ijk,ij->ik', Rc, tc)
    assert (U.shape[1] == 3)
    # distortion
    r_square = np.linalg.norm(U[:, :2] / U[:, 2][:, np.newaxis], axis=1) ** 2
    assert (r_square.shape[0] == d.shape[0])
    dis_coeffs = 1 + d[:, 0] * r_square + d[:, 1] * (r_square ** 2) + d[:, 2] * (r_square ** 3)
    assert (dis_coeffs.shape[0] == U.shape[0])
    Dx = dis_coeffs * U[:, 0]
    Dy = dis_coeffs * U[:, 1]
    D = np.stack((Dx, Dy, U[:, 2]), axis=-1)
    # camera model
    points_proj = np.einsum('ijk,ik->ij', K, D)
    points_proj = points_proj[:, :2] / points_proj[:, 2][:, np.newaxis]
    return (points_proj - points_2d).ravel()


def jac_sparsity(n_points: int, point_indices: np.ndarray, n_frames: int, frame_indices: np.ndarray,
                 mode: str = "both"):
    """
    Generate the sparsity matrix A for Jacobian calculation,
    reference: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.htm
    Args:
        n_points: int, number of input points
        point_indices: np.ndarray, (n_observations,), index of each observed points
        n_frames: int, number of input frames
        frame_indices: np.ndarray, (n_observations,), frame index of each observation
        mode: str,
                  "both": optimize both the 3d points and extrinsic parameters
                  "point": only optimize the 3d points
                  "translation": optimize the 3d points and translation vectors of extrinsic parameters

    Returns:
        A: scipy.sparse.lil_matrix, sparse matrix, 1 entries define useful jacobian, 0 entries define unuseful jacobian
    """
    n_observation = len(point_indices)
    m = n_observation * 2
    point_indices = np.array(point_indices)
    i = np.arange(n_observation)
    if mode == "point":
        n = n_points * 3
        A = lil_matrix((m, n), dtype=int)
    elif mode == "both":
        n = n_points * 3 + n_frames * 12
        A = lil_matrix((m, n), dtype=int)

        frame_indices = np.array(frame_indices)
        for s in range(12):
            A[2 * i, n_points * 3 + frame_indices * 12 + s] = 1
            A[2 * i + 1, n_points * 3 + frame_indices * 12 + s] = 1
    elif mode == "translation":
        n = n_points * 3 + n_frames * 3
        A = lil_matrix((m, n), dtype=int)

        frame_indices = np.array(frame_indices)
        for s in range(3):
            A[2 * i, n_points * 3 + frame_indices * 3 + s] = 1
            A[2 * i + 1, n_points * 3 + frame_indices * 3 + s] = 1

    else:
        raise ValueError

    for s in range(3):
        A[2 * i, point_indices * 3 + s] = 1
        A[2 * i + 1, point_indices * 3 + s] = 1

    return A


def run_bundle_adjustment(init_path: str, frame_dict_path: str,
                          camera_set_file: str, pose_file: str,
                          mode: str = "both"):
    """
    Run large-scale bundle adjustment
    Args:
        init_path:
        frame_dict_path:
        camera_set_file:
        pose_file:
        mode:

    Returns:

    """
    points_3d, camera_params, extrinsic_params, \
    point_indices, camera_indices, frame_indices, \
    points_2d, gt_3d = load_initialization(init_path, frame_dict_path, camera_set_file, pose_file)

    n_points = points_3d.shape[0]
    print("number of traffic sign:", n_points)
    n_frames = extrinsic_params.shape[0]
    print("number of frames:", n_frames)

    if mode == "both":
        params = np.hstack((points_3d.ravel(), extrinsic_params.ravel()))
    elif mode == "point":
        params = points_3d.ravel()
    elif mode == "translation":
        params = np.hstack((points_3d.ravel(), extrinsic_params[:, 9:].ravel()))
    else:
        raise ValueError

    res_init = project_batch(params, n_points,
                             camera_params, extrinsic_params,
                             point_indices, camera_indices, frame_indices,
                             points_2d, mode)

    A = jac_sparsity(n_points, point_indices, n_frames, frame_indices, mode)
    res = least_squares(project_batch, params, jac_sparsity=A,
                        verbose=2, x_scale='jac',
                        xtol=1e-8, ftol=1e-8, method='trf', max_nfev=1000,
                        args=(n_points, camera_params, extrinsic_params, point_indices,
                              camera_indices, frame_indices, points_2d, mode)
                        )
    return res_init, res, gt_3d, points_3d

def save_ba_estimation(estimated_3d, init_path:str, save_path:str):
    """
    Save bundle adjustment into json file
    Args:
        estimated_3d: np.ndarray, (_, 3)
        init_path:
        save_path:

    Returns:

    """
    init_data = json.load(open(init_path, "r"))
    init_data = init_data["init_data"]
    point_ind = 0
    for sign in init_data:
        if not np.isnan(sign["init_3d"]).any():
            sign["ba_3d"] = estimated_3d[point_ind].tolist()
            point_ind += 1
    assert point_ind == estimated_3d.shape[0]
    ba_data = dict()
    ba_data["ba_data"] = init_data
    with open(save_path, 'w') as outfile:
        json.dump(ba_data, outfile)

def relative_error(gt_3d, point_3d_init, point_3d_ba):
    """
    Compute the relative error of the estimated 3d coordinate
    Args:
        gt_3d: np.ndarray, (n_observation, 3), ground truth 3d coordinates
        point_3d_init: np.ndarray, (n_observation, 3), estimated 3d coordinate from triangulation
        point_3d_ba: np.ndarray, (n_observation, 3), estimated 3d coordinate from bundle adjustment

    Returns:

    """
    error_init = np.abs(point_3d_init - gt_3d)
    error_ba = np.abs(point_3d_ba - gt_3d)
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("absolute error of triangulation and ba")
    axes[0].boxplot(error_init)
    axes[1].boxplot(error_ba)
    for ax in axes:
        ax.set_xlabel("coordinate axis")
        ax.set_ylabel("absoulte error (dm)")
        ax.set_xticklabels(['x', 'y', 'z'])
        ax.set_xticklabels(['x', 'y', 'z'])
    plt.show()

if __name__ == "__main__":
    seq_id = 4
    gt_seq_path = f"/mnt/bailiang/traffic-sign-mapping/sequence_GT/sequence{seq_id}_GT.json"
    pose_path = f"/mnt/bailiang/traffic-sign-mapping/poses/Seq0{seq_id}.poses"
    camera_set_path = "/mnt/bailiang/traffic-sign-mapping/camera_set.csv"
    init_dict_path = f"/mnt/bailiang/traffic-sign-mapping/init_files/initialization0{seq_id}.json"
    frame_dict_path = f"/mnt/bailiang/traffic-sign-mapping/frame_files/frame_dict0{seq_id}.json"
    res_init, res, gt_3d, points_3d_init = run_bundle_adjustment(init_dict_path,frame_dict_path, camera_set_path,pose_path)
    n_points = points_3d_init.shape[0]
    points_3d_ba = res.x[:n_points * 3].reshape(-1, 3)
    relative_error(gt_3d, points_3d_init, points_3d_ba)
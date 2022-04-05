import os
import sys
import pandas as pd
import json
import numpy as np


def load_gt_sequence(seq_file: str):
    """
    Load sequence json file.
    Args:
        seq_file: str, file name, 'sequence*_GT.json'

    Returns:
        list of dictionary, poles and traffic signs annotation in sequence
    """
    gt_sequence = json.load((open(seq_file, "r")))
    return gt_sequence["GT_data"]


def load_camera(camera_set_file: str):
    """
    Load parameters of cameras
    Args:
        camera_set_file: str, camera set file name, 'camera_set.csv'

    Returns:
        numpy.ndarray, shape (10, 25)
        parameters: k11,k12,k13,k21,k22,k23,k31,k32,k33,d1,d2,d3,
                    r11,r12,r13,r21,r22,r23,r31,r32,r33,t1,t2,t3
    """
    camera_set = pd.read_csv(camera_set_file, index_col=0)
    return camera_set.values


def load_poses(pose_file: str):
    """
    Load camera poses (extrinsic parameters) of each sequence
    Args:
        pose_file: str, camera poses file name, '*.poses'

    Returns:
        pose: pd.DataFrame, (#frame, 12)
        Rw: world rotation (9,)
        tw: world translation (3,)
    """
    pose = pd.read_csv(pose_file, sep=' ', header=4, index_col=False)
    pose.set_index("frame_number", inplace=True)
    return pose


def get_intrinsic_params(camera_params: np.ndarray):
    """
    Transfer camera parameters into matrix form
    Args:
        camera_params: np.ndarray, (24,), intrinsic parameters

    Returns:
        K: np.ndarray, (3,3), intrinsic matrix
        d: np.ndarray, (3,), distortion params
        Rc: np.ndarray, (3,3), rotation matrix with respect to the center of car (car coordinate)
        tc: np.ndarray, (3,), translation params with repect to the center of car (car coordinate)
    """
    # intrinsic matrix K
    K = camera_params[:9].reshape(3, 3)
    # distortion parameters
    d = camera_params[9:12]
    # rotation matrix R
    Rc = camera_params[12:21].reshape(3, 3)
    # translation vector t
    tc = camera_params[21:]
    return K, d, Rc, tc


def get_extrinsic_params(extrinsic_params: np.ndarray):
    """
    Transfer extrinsic parameters into matrix form
    Args:
        extrinsic_params: np.ndarray, (12,), poses

    Returns:
        Rw: np.ndarray, (3,3), rotation matrix in world coordinate
        tw: np.ndarray, (3,), translation vector in world coordinate
    """
    Rw = extrinsic_params[:9].reshape(3, 3)
    tw = extrinsic_params[9:]
    return Rw, tw


# def get_intrinsic_params(camera_id: int, camera_params: np.ndarray):
#     """
#     Get intrinsic parameters of ith camera
#     Args:
#         camera_id: int, id of camera
#         camera_params: np.ndarray, (10,24), intrinsic parameters of 10 cameras
#
#     Returns:
#         K: np.ndarray, (3,3), intrinsic matrix
#         d: np.ndarray, (3,), distortion params
#         Rc: np.ndarray, (3,3), rotation matrix with respect to the center of car (car coordinate)
#         tc: np.ndarray, (3,), translation params with repect to the center of car (car coordinate)
#     """
#     camera_params = camera_params[camera_id]
#
#     # intrinsic matrix K
#     K = camera_params[:9].reshape(3, 3)
#     # distortion parameters
#     d = camera_params[9:12]
#     # rotation matrix R
#     Rc = camera_params[12:21].reshape(3, 3)
#     # translation vector t
#     tc = camera_params[21:]
#     return K, d, Rc, tc
#
#
# def get_extrinsic_params(frame_id: int, poses: pd.DataFrame):
#     """
#     Get extrinsic parameters of ith frame
#     Args:
#         frame_id: int, id of frame
#         poses: np.ndarray, (#frame, 12), extrinsic parameters of each frame
#
#     Returns:
#         Rw: np.ndarray, (3,3), rotation matrix in world coordinate
#         tw: np.ndarray, (3,), translation vector in world coordinate
#     """
#     extrinsic_params = poses.loc[frame_id].values
#     Rw = extrinsic_params[:9].reshape(3, 3)
#     tw = extrinsic_params[9:]
#     return Rw, tw


def forward_matrix(R, t):
    """
    Generate the corrsponding forward projection matrix of R and t
    Args:
        R: np.ndarray, (3,3), rotation matrix
        t: np.ndarray, (3,), translation vector

    Returns:
        P: np.ndarray, (4,4), homogeneous projection matrix

    """
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = t.T
    return P


def backward_matrix(R, t):
    """
    Generate the corrsponding backward projection matrix of R and t
    Args:
        R: np.ndarray, (3,3), rotation matrix
        t: np.ndarray, (3,), translation vector

    Returns:
        P: np.ndarray, (4,4), homogeneous projection matrix

    """
    P = np.eye(4)
    P[:3, :3] = R.T
    p[:3, 3] = -R.T @ t
    return P

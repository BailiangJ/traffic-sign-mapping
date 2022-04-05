import numpy as np
from helper import get_intrinsic_params, get_extrinsic_params, backward_matrix, forward_matrix
from typing import List, Union


def back_project(bbox: Union[List[int], np.ndarray], camera_params: np.ndarray, poses: np.ndarray,
                 camera_id: int, frame_id: int):
    """
    Shoot rays from bounding box center to the corresponding 3D point
    Args:
        bbox: list[int] or np.ndarray, (4,), 2D coordinate of the bounding box detection
        camera_params: np.ndarray, (10,25), intrinsic parameters of 10 cameras
        poses: np.ndarray, (#frame, 12), extrinsic parameters of each frame
        camera_id: int, id of frame
        frame_id: int, id of camera

    Returns:

    """
    K, d, Rc, tc = get_intrinsic_params(camera_params[camera_id])
    Rw, tw = get_extrinsic_params(poses.loc[frame_id].values)

    m = np.array([(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2, 1], dtype=float)
    # reverse projection
    D = np.linalg.pinv(K) @ m
    # solve distortion equation
    # check camera.pdf page3
    r = D[0] ** 2 + D[1] ** 2
    poly = np.poly1d([d[1], d[0], 1, -r], r=False)
    roots = np.roots(poly)
    roots = roots[np.isreal(roots)]
    # print("real roots of distortion polynomial:", roots)
    root = roots[0]
    coeff = d[1] * root ** 2 + d[0] * root + 1
    Ux = D[0] / coeff
    Uy = D[1] / coeff
    # estimation of U in camera coordinate
    U_est = np.array([Ux, Uy, D[2], 1], dtype=np.float)

    # # solution 1
    # Pc = backward_matrix(Rc, tc)
    # Pw = backward_matrix(Rw, tw)
    #
    # # map estimated 3D coordinates from camera coordinate to car coordinate then to world coordinate
    # M_est = np.linalg.pinv(Pw) @ np.linalg.pinv(Pc) @ U_est
    # # map camera center from car coordinate to world coordinate
    # cam_pos = forward_matrix(Rw, tw) @ np.append(tc, 1)

    # solution 2
    Pc = forward_matrix(Rc, tc)
    Pw = forward_matrix(Rw, tw)

    # map estimated 3D coordinates from camera coordinate to car coordinate then to world coordinate
    M_est = Pw @ Pc @ U_est
    # map camera center from car coordinate to world coordinate
    cam_pos = Pw @ np.append(tc, 1)

    # ray is from camera center to estimated point in world coordinate
    ray = np.vstack((cam_pos[:3], M_est[:3]))
    return ray


def line_2d(ray: np.ndarray):
    """
    Compute the 2D line Ax+By+C=0 with z fixed.
    Args:
        ray: np.ndarray, (2,3), coordinates of two separate points on a line

    Returns:
        parameters of the 2d line
        A:
        B:
        C:

    """
    A = ray[1, 1] - ray[0, 1]
    B = ray[0, 0] - ray[1, 0]
    C = ray[1, 1] * ray[0, 0] - ray[1, 0] * ray[0, 1]
    return A, B, C


def intersection_2d(ray_1: np.ndarray, ray_2: np.ndarray):
    """
    Check whether two line intersect,
    if yes, return the intersection coordinates,
    if no, return False
    Args:
        ray_1: np.ndarray, (2,3)
        ray_2: np.ndarray, (2,3)

    Returns:


    """
    A1, B1, C1 = line_2d(ray_1)
    A2, B2, C2 = line_2d(ray_2)

    D = A1 * B2 - B1 * A2
    if abs(D) < 1e-5:
        # check whether two lines are parallel
        # print(D)
        return False
    else:
        Dx = B2 * C1 - B1 * C2
        Dy = -A2 * C1 + A1 * C2
        x = Dx / D
        y = Dy / D
        return x, y


def calculate_z(ray: np.ndarray, x: float, y: float):
    """
    Calculate z coordinate given a line and (x,y)
    Args:
        ray: np.ndarray, (2,3)
        x:
        y:

    Returns:
        z: float
    """
    tx = (x - ray[0, 0]) / (ray[1, 0] - ray[0, 0])
    ty = (y - ray[0, 1]) / (ray[1, 1] - ray[0, 1])
    if abs(tx - ty) < 1e-5:
        t = tx
    else:
        t = (tx + ty) / 2
    z = ray[0, 2] + t * (ray[1, 2] - ray[0, 2])
    return z

def intersection_3d(ray_1:np.ndarray, ray_2:np.ndarray)->np.ndarray:
    """
    Compute the 3d intersection coordinates of two line
    Args:
        ray_1: np.ndarray, (2,3)
        ray_2: np.ndarray, (2,3)

    Returns:
        np.ndarray, (3,)

    """
    x,y = intersection_2d(ray_1, ray_2)
    z_1 = calculate_z(ray_1, x, y)
    z_2 = calculate_z(ray_2, x, y)
    z = (z_1 + z_2) / 2
    return np.array((x, y, z))
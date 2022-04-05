import sys
import os
import json
import pandas as pd
import numpy as np


def parse_seq_to_json(seq_file: str):
    """
    Parse text file 'sequence*_GT.txt' and save it as json file.
    Args:
        seq_file: str, pole annotations files of each sequence

    Returns:
        json file keys
        pold_id: id of pole
        pole_3D: 3D coordinate of pole
        num_bdbox: number of bounding box of traffic signs on the pole
        bounding_boxes: list of bounding box information
        camera_id: id of camera
        frame_id: id of frame
        bounding_box: 2D coordinate of rectangle bounding box
        TS_3D: 3D coordinate of traffic sign
        sign_type: the type of traffic sign
    """
    filename = seq_file.split('.')[0]
    with open(seq_file, "r") as f:
        line = f.readlines()
    num_poles = int(lines[0][:-1])
    current_pole = 0
    current_line = 0
    dict_list = []
    while current_pole < num_poles:
        pole_dict = {}
        pold_id = int(lines[current_line + 1][:-1].split(' ')[1])

        coord_3D = np.array(lines[current_line + 2][:-1].split(' '), dtype=np.float).tolist()

        num_center = int(lines[current_line + 4])

        num_bdbox = int(lines[current_line + 4 + num_center + 1])

        current_line = current_line + 4 + num_center + 1 + 1

        pole_dict = dict(pole_id=pole_id, pole_3D=coord_3D, num_bdbox=num_bdbox)

        frame_dict_list = []

        for i in range(num_bdbox):
            bd_box = np.array(lines[current_line + 4 * i][:-1].split(' '), dtype=np.float).tolist()

            camera_id, frame_id, _ = np.array(lines[current_line + 4 * i + 1][:-1].split(' '), dtype=np.int).tolist()

            sign_type = lines[current_line + 4 * i + 2].split(';')[0]

            TS_3D = np.array(lines[current_line + 4 * i + 3][:-1].split(' '), dtype=np.float).tolist()

            frame_dict = dict(camera_id=camera_id, frame_id=frame_id, bounding_box=bd_box, TS_3D=TS_3D,
                              sign_type=sign_type)

            frame_dict_list.append(frame_dict)

        # each bounding box has 4 lines
        current_line = current_line + num_bdbox * 4

        current_pole += 1

        pole_dict["bounding_boxes"] = frame_dict_list
        dict_list.append(pole_dict)

    data = {}
    data["GT_data"] = dict_list
    with open('.'.join([filename, 'json']), "w") as outfile:
        json.dump(data, outfile)

def parse_camera_to_csv(camera_set_file:str):
    """
    Parse text file 'camera_set.txt' and save it as csv file.
    Args:
        camera_set_file: calibration file containing intrinsic parameters of 10 camera

    Returns:

    """
    with open(camera_set_file, "r") as f:
        lines = f.readlines()

    num_camera = int(lines[0][:-1])
    current_line = 0
    # intrinsic matrices K
    k_matrices = []
    # distortion parameters kappa
    kappa_params = []
    # rotation matrices R
    r_matrices = []
    # translation parameters t
    t_params = []
    for i in range(num_camera):
        # parse intrinsic matrix K
        k_rows = []
        for j in range(3):
            kj_ = np.array(lines[current_line + i * 16 + 2 + j][:-1].split(' '), dtype=np.float64)
            k_rows.append(kj_)
        k_rows = np.hstack(k_rows)
        k_matrices.append(k_rows)
        # parse distortion parameters kappa
        kappa = np.array(lines[current_line + i * 16 + 6][:-1].split(' '), dtype=np.float64)
        kappa_params.append(kappa)
        # parse rotation matrix R
        r_rows = []
        for j in range(3):
            rj_ = np.array(lines[current_line + i * 16 + 8 + j][:-1].split(' '), dtype=np.float64)
            r_rows.append(rj_)
        r_rows = np.hstack(r_rows)
        r_matrices.append(r_rows)
        # parse translation t
        t = np.array(lines[current_line + i * 16 + 12][:-1].split(' '), dtype=np.float32)
        t_params.append(t)
    k_matrices = np.vstack(k_matrices)
    kappa_params = np.vstack(kappa_params)
    r_matrices = np.vstack(r_matrices)
    t_params = np.vstack(t_params)

    camera_params = np.hstack([k_matrices, kappa_params, r_matrices, t_params])
    columns = ['k11','k12','k13','k21','k22','k23','k31','k32','k33',
               'd1','d2','d3',
               'r11','r12','r13','r21','r22','r23','r31','r32','r33',
               't1','t2','t3']
    camera_params = pd.DataFrame(camera_params, columns=columns)
    camera_params.to_csv("camera_set1.csv", sep='\t')




if __name__ == "__main__":
    parse_camera_to_csv('camera_set.txt')
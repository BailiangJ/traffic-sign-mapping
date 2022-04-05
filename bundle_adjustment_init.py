import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from itertools import combinations
from triangluation import back_project, line_2d, intersection_2d, intersection_3d
from helper import load_gt_sequence, load_poses, load_camera


def initialize_ba(seq_file: str, camera_set_file: str, pose_file: str, output_path: str, frame_dict_path: str,
                  mu:int=0, sigma:int=0):
    """
    Using Triangulation to initialize bundle adjustment. Save initialization in json file.
    Args:
        seq_file: str, file path of sequence, 'sequence*_GT.json'
        camera_set_file: str, file path of camera parameters, 'camera_set.csv'
        pose_file: str, file path of poses, '*.poses'
        output_path: str, bundle adjustment initialization save path
        frame_dict_path: str, frame dict save path, record the absolute index of each frames in poses array
        mu: int, mean of Gaussian noise added to initialization
        sigma: int, standard deviation of Gaussian noise added to initialization
    Returns:

    """
    if mu != 0 and sigma != 0:
        perturb_flag = True
        mu = [mu] * 3
        cov = np.diag([sigma, sigma, sigma])
    else:
        perturb_flag = False

    gt_sequence = load_gt_sequence(seq_file)
    camera_params = load_camera(camera_set_file)
    poses = load_poses(pose_file)

    ba_init = dict()
    initial_estimation = []
    frame_dict = dict()
    f_index = 0

    for i, pole in enumerate(gt_sequence):
        signs = pole['bounding_boxes']
        while len(signs) > 0:
            init_dict = dict()
            rays = []
            sign_type = signs[0]['sign_type']
            gt_pnt_3d = np.array(signs[0]['TS_3D'])
            index_to_pop = []
            bounding_boxes = []
            for j, sign in enumerate(signs):
                if sign['sign_type'] == sign_type and abs(gt_pnt_3d[2] - sign["TS_3D"][2]) < 1e-1:
                    camera_id = sign['camera_id']
                    frame_id = sign['frame_id']

                    if frame_id not in frame_dict.keys():
                        frame_dict[frame_id] = f_index
                        f_index += 1

                    bbox = sign['bounding_box']
                    ray = back_project(bbox, camera_params, poses, camera_id, frame_id)
                    rays.append(ray)
                    index_to_pop.append(j)
                    sign.pop('sign_type')
                    bounding_boxes.append(sign)
                for k in sorted(index_to_pop, reverse=True):
                    signs.pop(k)

                pnts_3d = []
                for com in combinations(rays, 2):
                    ray_1, ray_2 = com
                    pnt_3d = intersection_3d(ray_1, ray_2)
                    pnts_3d.append(pnt_3d)

                pnts_3d = np.array(pnts_3d)
                init_est = np.mean(pnts_3d, axis=0)
                if not np.isnan(init_est).any() and perturb_flag:
                    gauss_noise = np.random.multivariate_normal(mu, cov)
                    init_est += gauss_noise

                init_dict['sign_type'] = sign_type
                init_dict['gt_3d'] = gt_pnt_3d.tolist()
                init_dict['init_3d'] = init_est.tolist()
                init_dict['bounding_boxes'] = bounding_boxes
                initial_estimation.append(init_dict)
    ba_init['init_data'] = initial_estimation
    # print(ba_init)
    with open(output_path, 'w') as outfile:
        json.dump(ba_init, outfile)
    with open(frame_dict_path, 'w') as outfile:
        json.dump(frame_dict, outfile)


if __name__ == "__main__":
    seq_id = 3
    gt_seq_path = f"/mnt/bailiang/traffic-sign-mapping/sequence_GT/sequence{seq_id}_GT.json"
    pose_path = f"/mnt/bailiang/traffic-sign-mapping/poses/Seq0{seq_id}.poses"
    camera_set_path = "/mnt/bailiang/traffic-sign-mapping/camera_set.csv"
    output_path = f"/mnt/bailiang/traffic-sign-mapping/init_files/initialization0{seq_id}.json"
    frame_dict_path = f"/mnt/bailiang/traffic-sign-mapping/frame_files/frame_dict0{seq_id}.json"
    initialize_ba(gt_seq_path, camera_set_path, pose_path, output_path, frame_dict_path)

from Triangulation import Triangulation
import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

font = {'family': 'monospace',
        'color': 'blue',
        'style': 'normal',
        'weight': 'bold',
        'size': 10}


def select_frame(camera_set_path, pose_path, sequence_path, sel_frames):
    mu = 0
    sigma = 10
    camera_set = pd.read_csv(camera_set_path, index_col=0)
    camera_params = camera_set.values

    pose = pd.read_csv(pose_path, sep=" ", header=4, index_col=False)
    pose.set_index("frame_number", inplace=True)

    gt_sequence = json.load(open(sequence_path, "r"))
    gt_sequence = gt_sequence["GT_data"]

    triangulation = Triangulation()

    rays_list = []
    gt_3d_list = []

    for i, pole in enumerate(gt_sequence):
        signs = pole['bounding_boxes']
        while len(signs) > 0:
            rays = []
            sign_type = signs[0]['sign_type']
            gt_pnt_3d = np.array(signs[0]['TS_3D'])
            index_to_pop = []
            for j, sign in enumerate(signs):
                if sign['sign_type'] == sign_type:
                    camera_id = sign['camera_id']
                    frame_id = sign['frame_id']
                    if frame_id in sel_frames:
                        bbox = np.array(sign['bounding_box'])

                        # noise = np.random.normal(loc=mu, scale=sigma, size=bbox.shape)
                        # bbox = bbox + noise

                        ray = triangulation.back_project(bbox, camera_params[camera_id],
                                                         pose.loc[frame_id].values)
                        rays.append(ray)
                    index_to_pop.append(j)
            if len(rays) > 0:
                print(sign_type)
                rays_list.append(rays)
                gt_3d_list.append(gt_pnt_3d)
                pnts_3d = []
                for com in combinations(rays, 2):
                    ray_1, ray_2 = com
                    pnt_3d = triangulation.intersection_3d(ray_1, ray_2)
                    pnts_3d.append(pnt_3d)

                pnts_3d = np.array(pnts_3d)
                init_est = np.mean(pnts_3d, axis=0)
                print(init_est)
            for k in sorted(index_to_pop, reverse=True):
                signs.pop(k)

    return rays_list, gt_3d_list


def plot_rays_2d(rays, gt_3d):
    rays = np.array(rays)[[0,2,1],:]
    plt.figure()
    for i, ray in enumerate(rays):
        cam_pos = ray[0, :]
        direction = ray[1, :] - ray[0, :]
        end_pnt = cam_pos + 9 * direction
        ray = np.vstack((cam_pos, end_pnt))
        plt.plot(cam_pos[0], cam_pos[1], '*')
        plt.text(x=cam_pos[0], y=cam_pos[1], s=f"frame_{i}", color='g', fontdict=font)
        plt.plot(ray[:, 0], ray[:, 1], '-')
    plt.plot(gt_3d[0], gt_3d[1], 'b+')
    plt.show()


def plot_rays_3d(rays, gt_3d):
    rays = np.array(rays)[[0, 2, 1], :]
    gt_3d = np.array(gt_3d)
    print(gt_3d.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.split(gt_3d, 3, axis=0)
    ax.plot(x, y, z, '+')
    for i, ray in enumerate(rays):
        cam_pos = ray[0, :]
        direction = ray[1, :] - ray[0, :]
        end_pnt = cam_pos + 9 * direction
        ray = np.vstack((cam_pos, end_pnt))

        cx, cy, cz = np.split(cam_pos, 3, axis=0)
        ax.plot(cx, cy, cz, '*')
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f"frame_{i}", color='black')
        ax.plot(ray[:, 0], ray[:, 1], ray[:, 2])
    plt.show()


if __name__ == "__main__":
    seq_id = 2
    gt_seq_path = f"/mnt/bailiang/traffic-sign-mapping/sequence_GT/sequence{seq_id}_GT.json"
    pose_path = f"/mnt/bailiang/traffic-sign-mapping/poses/Seq0{seq_id}.poses"
    camera_set_path = "/mnt/bailiang/traffic-sign-mapping/camera_set.csv"
    init_dict_path = f"/mnt/bailiang/traffic-sign-mapping/initialization0{seq_id}.json"
    frame_dict_path = f"/mnt/bailiang/traffic-sign-mapping/frame_dict0{seq_id}.json"

    sel_frames = [3867,3868,3869]
    rays_list, gt_3d_list = select_frame(camera_set_path, pose_path, gt_seq_path, sel_frames)
    plot_rays_2d(rays_list[1], gt_3d_list[1])
    plot_rays_3d(rays_list[1], gt_3d_list[1])

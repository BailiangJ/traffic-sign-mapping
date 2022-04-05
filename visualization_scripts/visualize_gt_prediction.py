import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import cv2 as cv
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from images.process_ts import load_parameters, backward_matrix, distortion, project


def load_ba_result(ba_result_path):
    ba_data = json.load(open(ba_result_path, "r"))
    ba_data = ba_data["ba_data"]
    return ba_data


def visualization(image_path, save_path, bboxes, gt_2d, init_2d, ba_2d):
    """

    Args:
        image_path:
        save_path:
        bboxes:
        gt_2d: ground truth 2d image point
        init_2d: initialized 2d image point
        ba_2d: bundle adjustment 2d image point

    Returns:

    """
    img = cv.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    for (y1, x1, y2, x2) in bboxes:
        rectangle = Rectangle(xy=(x1, y1), width=x2 - x1, height=y2 - y1, linewidth=2, edgecolor='r', alpha=0.5)
        ax.add_patch(rectangle)
    for (x, y) in gt_2d:
        ax.plot(x, y, "rx")
    for (x, y) in ba_2d:
        ax.plot(x, y, "co")
    for (x, y) in init_2d:
        ax.plot(x, y, "b+")

    # plt.show()
    plt.savefig(save_path, dpi=300)
    # plt.clf()
    # plt.close()


def select_frame(ba_data, frame_id):
    flag = False
    bboxes = []
    gt_3d = []
    init_3d = []
    ba_3d = []
    camera_id = -1
    for sign in ba_data:
        for observation in sign["bounding_boxes"]:
            if observation["frame_id"] == frame_id:
                bboxes.append(observation["bounding_box"])
                if camera_id == -1:
                    camera_id = observation["camera_id"]
                else:
                    assert camera_id == observation["camera_id"]
                flag = True
        if flag:
            print(sign)
            gt_3d.append(sign["gt_3d"])
            init_3d.append(sign["init_3d"])
            ba_3d.append(sign["ba_3d"])
            flag = False
    return camera_id, bboxes, gt_3d, init_3d, ba_3d


if __name__ == "__main__":
    seq_id = 1
    camera_id = 5
    pose_path = f"/mnt/bailiang/traffic-sign-mapping/poses/Seq0{seq_id}.poses"
    camera_set_path = "/mnt/bailiang/traffic-sign-mapping/camera_set.csv"
    ba_result_path = f"/mnt/bailiang/traffic-sign-mapping/ba_result0{seq_id}.json"
    image_dir = f"/mnt/bailiang/traffic-sign-mapping/images/camera0{camera_id}/Seq0{seq_id}"
    frame_id = 34905
    image_name = ".".join(['image', f"{frame_id:06d}", 'jp2'])
    image_path = os.path.join(image_dir, image_name)
    pose, camera_params = load_parameters(pose_path, camera_set_path)
    ba_data = load_ba_result(ba_result_path)
    camera_id, bboxes, gt_3d, init_3d, ba_3d = select_frame(ba_data, frame_id)
    gt_2d = project(gt_3d, camera_params[camera_id, :], pose.loc[frame_id].values)
    init_2d = project(init_3d, camera_params[camera_id, :], pose.loc[frame_id].values)
    ba_2d = project(ba_3d, camera_params[camera_id, :], pose.loc[frame_id].values)
    save_dir = "/mnt/bailiang/traffic-sign-mapping/images"
    save_path = os.path.join(save_dir, f"{camera_id:02d}" + "_" + f"{frame_id:06d}" + ".png")
    visualization(image_path, save_path, bboxes, gt_2d, init_2d, ba_2d)

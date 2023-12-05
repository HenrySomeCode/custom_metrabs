# import sys

# def trace(frame, event, arg):
#   print(frame.filename, frame.lineno)

# sys.settrace(trace)

import argparse
import urllib.request

import cameralib
import numpy as np
import posepile.joint_info
import poseviz
import simplepyutils as spu
import torch
import torchvision.io

import sys
sys.path.append("/mnt/datadrive/annh/metrabs/metrabs_pytorch")
sys.path.append("/mnt/datadrive/annh/metrabs")


from metrabs_pytorch.backbones import efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.util import get_config

import cv2
import json 

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import torch.nn.functional as F
import os

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--image-path', type=str)
    spu.argparse.initialize(parser)
    get_config(f'{spu.FLAGS.model_dir}/config.yaml')
    # print("from demo_image.py: spu.FLAGS.model_dir", spu.FLAGS.model_dir) # metrabs_eff2l_384px_800k_28ds_pytorch
    # get_config("/mnt/datadrive/annh/metrabs/metrabs_eff2l_384px_800k_28ds_pytorch/config.yaml")

    # skeleton = 'smpl+head_30'
    skeleton = 'coco_19'

    multiperson_model_pt = load_multiperson_model().cuda()
    
    joint_names = multiperson_model_pt.per_skeleton_joint_names[skeleton]
    joint_edges = multiperson_model_pt.per_skeleton_joint_edges[skeleton].cpu().numpy()

    print("from demo_image.py: ", joint_names)
    print("from demo_image.py: ", joint_edges)

    with torch.inference_mode(), torch.device('cuda'):
        # with poseviz.PoseViz(joint_names, joint_edges, paused=True) as viz:
            # print("from demo_image.py: here")
            # print(xxx)
        image_filepath = get_image(spu.argparse.FLAGS.image_path)
        image_name = image_filepath.split('/')[-1].split('.')[0]

        image = torchvision.io.read_image(image_filepath).cuda()
        camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=image.shape[1:])
        
        intrinsic_matrix = torch.tensor([[5000.0, 0.0, 288.0], [0.0, 5000.0, 512.0], [0.0, 0.0, 1.0]])
        
        for num_aug in range(1,2):
            print("from demo_image.py: ", num_aug)

            pred = multiperson_model_pt.detect_poses(
                image, detector_threshold=0.01, suppress_implausible_poses=False,
                max_detections=1, intrinsic_matrix=camera.intrinsic_matrix,
                skeleton='coco_19', num_aug=num_aug)

            # print("from demo_image.py: ", pred.keys()) #dict_keys(['boxes', 'poses3d', 'poses2d'])

            print("from demo_image.py: pred['poses3d']: ", pred['poses3d']) 
            print(pred['poses3d'].max())
            print(pred['poses3d'].min()) 

            

            # min_value = pred['poses3d'].min()
            # max_value = pred['poses3d'].max()
            # scaled_tensor = (pred['poses3d'] - min_value) / (max_value - min_value) * 2 - 1

            # pred['poses3d'] = scaled_tensor

            # visualize(
            #         tf.image.decode_jpeg(tf.io.read_file(image_filepath)).cpu().numpy(), 
            #         pred['boxes'].cpu().numpy(),
            #         pred['poses3d'].cpu().numpy(),
            #         pred['poses2d'].cpu().numpy(),
            #         multiperson_model_pt.per_skeleton_joint_edges['coco_19'].cpu().numpy())

            # write pkl
            # pred['boxes'] = pred['boxes'].cpu().tolist()
            # pred['poses3d'] = pred['poses3d'].cpu().tolist()
            # pred['poses2d'] = pred['poses2d'].cpu().tolist()

            # with open(f"{image_name}_pred.json", "w") as f:
            #     json.dump(pred, f)

            
                # with open("/mnt/datadrive/annh/metrabs/k7_model.json", "w") as f:
                #     json.dump(multiperson_model_pt.per_skeleton_joint_edges['smpl_24'].cpu().tolist(), f)

                # print("here")
                #----

                # viz.update(
                #     frame=image.cpu().numpy().transpose(1, 2, 0),
                #     boxes=pred['boxes'].cpu().numpy(),
                #     poses=pred['poses3d'].cpu().numpy(), camera=camera)
                # cv2.imwrite("viz.jpg",image)
    

def visualize(image, detections, poses3d, poses2d, edges):
    fig = plt.figure(figsize=(10, 5.2))
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.imshow(image)
    print("from demo_image.py: here: 0")
    for x, y, w, h in detections[:, :4]:
        print("from demo_image.py:", x, y, w, h)
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))
    print("from demo_image.py: here: 1")

    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.view_init(5, -85)
    pose_ax.set_xlim3d(-1, 0)
    pose_ax.set_zlim3d(-1, 1)
    pose_ax.set_ylim3d(0, 2)

    # Matplotlib plots the Z axis as vertical, but our poses have Y as the vertical axis.
    # Therefore, we do a 90° rotation around the X axis:
    poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
    for pose3d, pose2d in zip(poses3d, poses2d):
        for i_start, i_end in edges:
            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
            pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
        image_ax.scatter(*pose2d.T, s=2)
        pose_ax.scatter(*pose3d.T, s=2)

    fig.tight_layout()
    # plt.show()
    print("from demo_image.py: drawing")
    plt.savefig("k7_kps/result.png")


def load_multiperson_model():
    model_pytorch = load_crop_model()
    skeleton_infos = spu.load_pickle(f'{spu.FLAGS.model_dir}/skeleton_infos.pkl')
    joint_transform_matrix = np.load(f'{spu.FLAGS.model_dir}/joint_transform_matrix.npy')

    with torch.device('cuda'):
        return multiperson_model.Pose3dEstimator(
            model_pytorch.cuda(), skeleton_infos, joint_transform_matrix)
            
    # import tensorflow as tf
    # import tensorflow_hub as hub

    # model = hub.load('https://bit.ly/metrabs_xl')

    # return model 


def load_crop_model():
    cfg = get_config()
    ji_np = np.load(f'{spu.FLAGS.model_dir}/joint_info.npz')
    ji = posepile.joint_info.JointInfo(ji_np['joint_names'], ji_np['joint_edges'])
    backbone_raw = getattr(effnet_pt, f'efficientnet_v2_{cfg.efficientnet_size}')()
    preproc_layer = effnet_pt.PreprocLayer()
    backbone = torch.nn.Sequential(preproc_layer, backbone_raw.features)
    model = metrabs_pt.Metrabs(backbone, ji)
    model.eval()

    inp = torch.zeros((1, 3, cfg.proc_side, cfg.proc_side), dtype=torch.float32)
    intr = torch.eye(3, dtype=torch.float32)[np.newaxis]

    model((inp, intr))
    model.load_state_dict(torch.load(f'{spu.FLAGS.model_dir}/ckpt.pt'))
    return model


def get_image(source, temppath='/tmp/image.jpg'):
    if not source.startswith('http'):
        return source

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(source, temppath)
    return temppath


if __name__ == '__main__':
    try:
        # Your code goes here
        main()
    except Exception as e:
        print(e, file=sys.stderr)
    

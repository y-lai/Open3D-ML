#!/usr/bin/env python3
import numpy as np
import math
import os, sys
from os.path import exists, join, isfile, dirname, abspath, split
from glob import glob
from pathlib import Path

import open3d as o3d
from open3d_ros_helper import open3d_ros_helper as orh

import rospy
import ros_numpy
import torch

import json
import signal

class JRDBPreprocessing():
    ''''Pre-processing class to merge pointcloud from both velodynes on the JackRabbot.'''
    def __init__(self,
                 dataset_path,
                 no_labels=False):
        rospy.init_node('jrdb_preprocessing_node',anonymous=True)
        self.dataset_path_full = dataset_path
        self.no_labels = no_labels
        
        # JRDB dataset has identical folder names between upper and lower velodyne.
        self.upper_folders = glob(join(self.dataset_path_full, 'upper_velodyne','*'))
        self.upper_folders.sort()

        # check if joined folder exists - mkdir if so
        self.joined_dataset_path_full = join(self.dataset_path_full,'both_velodyne')
        #  check if folders already exist - basic indication that preprocessing has been done
        if os.path.exists(self.joined_dataset_path_full):
            print('Pre-processing has been done. both_velodyne folder exists.')
            sys.exit(0)
        else:
            os.mkdir(self.joined_dataset_path_full)
            os.chmod(self.joined_dataset_path_full,0o755)
            os.chown(self.joined_dataset_path_full,1000,1000)

        if self.no_labels == False:
            self.joined_labels = join(self.dataset_path_full,'labels')
            if not os.path.exists(self.joined_labels):
                os.mkdir(self.joined_labels)
                os.chmod(self.joined_labels,0o755)
                os.chown(self.joined_labels,1000,1000)

        print('Merging pointclouds and generating labels')
        self.merge_pointclouds(self.upper_folders)

        if self.no_labels == False:
            for root, dirs, files in os.walk(self.joined_labels):
                for momo in dirs:
                    os.chmod(join(root,momo),0o755)
                    os.chown(join(root, momo), 1000, 1000)
                for momo in files:
                    os.chmod(join(root,momo),0o766)
                    os.chown(join(root, momo), 1000, 1000)


    def merge_pointclouds(self,upper_folders):
        # index to shift all different scenes into the same index count
        idx = 0
        # for each pair, update calibration to same frame, merge pointclouds, and rename the pcd
        for folder in upper_folders:
            folderidx = 0
            data = []
            if self.no_labels==False:
                currpath = Path(self.dataset_path_full)
                parentpath = str(currpath.parent.resolve())
                jsonpath = join(parentpath,'labels','labels_3d',os.path.basename(folder)+'.json')
                f = open(jsonpath)
                print('Opening label file for ',f)
                data = json.load(f)

            print('Folder path: ',folder)
            upper_folder_glob = glob(join(self.dataset_path_full,'upper_velodyne',folder, '*.pcd'))
            upper_folder_glob.sort()
            lower_folder_glob = glob(join(self.dataset_path_full,'lower_velodyne',folder, '*.pcd'))
            lower_folder_glob.sort()

            for (upper_cloud_file,lower_cloud_file) in zip(upper_folder_glob,lower_folder_glob):
                tempup = o3d.io.read_point_cloud(upper_cloud_file)
                templow = o3d.io.read_point_cloud(lower_cloud_file)
                upper_ros = orh.o3dpc_to_rospc(tempup)
                lower_ros = orh.o3dpc_to_rospc(templow)

                upper_pc = ros_numpy.numpify(upper_ros).astype({'names':['x','y','z'], 'formats':['f4','f4','f4'], 'offsets':[0,4,8], 'itemsize':32})
                lower_pc = ros_numpy.numpify(lower_ros).astype({'names':['x','y','z'], 'formats':['f4','f4','f4'], 'offsets':[0,4,8], 'itemsize':32})

                upper_torch = torch.from_numpy(upper_pc.view(np.float32).reshape(upper_pc.shape + (-1,)))[:, [0,1,2]]
                lower_torch = torch.from_numpy(lower_pc.view(np.float32).reshape(lower_pc.shape + (-1,)))[:, [0,1,2]]
                upper_pc = self.convert_to_base_chassis_tf(upper_torch, upper=True)
                lower_pc = self.convert_to_base_chassis_tf(lower_torch, upper=False)
                joined_pc = torch.cat([upper_pc, lower_pc], dim=0)
                joined_np = joined_pc.numpy()

                jointpc = o3d.geometry.PointCloud()
                jointpc.points = o3d.utility.Vector3dVector(joined_np)
                # convert idx into 6 digit number
                index_str = str(idx).zfill(6)
                o3d.io.write_point_cloud(join(self.joined_dataset_path_full,index_str+'.pcd'),jointpc)

                # check if label exists. Load JSON label data and save in text file
                if self.no_labels==False:
                    labelpath = join(self.joined_labels,index_str+'.txt')
                    # open file for writing
                    label_file = open(labelpath,'w',encoding='utf-8')

                    people = data['labels'][str(folderidx).zfill(6)+'.pcd']
                    # iterate people
                    for person in people:
                        towrite = []
                        box = person['box']
                        for attr in box:
                            towrite.append(box[attr])
                        # change order to fit BEVBox3D
                        # x, y, z, width, height, depth, yaw
                        towrite[3], towrite[4:] = towrite[-1], towrite[3:-1]
                        # BEVBox3D yaw is 0 yaw angle at -y direction
                        if float(towrite[-1])+(math.pi/2) > 2*math.pi:
                            val = float(towrite[-1])
                            # Add pi/2 since yaw starts at +x
                            val -= 3*math.pi/2
                            towrite[-1] = str(val)
                        # labels are in the rotational reference for upper_velodyne, translational reference for rgb camera.
                        # changing z value to match height
                        # towrite[2] = str((float(towrite[2])-0.4704))
                        towrite[2] = str((float(towrite[2])))
                        label_file.write(' '.join(str(item) for item in towrite))
                        # label and confidence for BEVBox3D
                        label_file.write(' Pedestrian -1.0')
                        label_file.write('\r\n')
                    label_file.close()
                idx+=1
                folderidx+=1

    def convert_to_base_chassis_tf(self, pointcloud, upper=True):
        if upper:
            pointcloud[:,:3] += torch.Tensor([-0.019685, 0, 0]).type(pointcloud.type())
            # pointcloud[:,:3] += torch.Tensor([-0.019685, 0, -1.077382]).type(pointcloud.type())
            # theta = 0.085
            theta = 0
        else:
            pointcloud[:,:3] += torch.tensor([-0.019685, 0, 0]).type(pointcloud.type())
            # pointcloud[:,:3] += torch.tensor([-0.019685, 0, -0.606982]).type(pointcloud.type())
            theta = 0

        rot_mat = torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta),np.cos(theta)]]).type(pointcloud.type())
        pointcloud[:, :2] = torch.matmul(rot_mat,pointcloud[:,:2].unsqueeze(2)).squeeze()
        return pointcloud

def signal_handler(sig, frame):
    print('SIGINT obtained from Ctrl+C')
    sys.exit(0)

if __name__ == '__main__':
    print('Note: roscore required for numpy_ros functionalities.')
    signal.signal(signal.SIGINT, signal_handler)
    if len(sys.argv) == 2:
        print('Input argument for dataset_path: ',sys.argv[1])
        proc = JRDBPreprocessing(dataset_path=sys.argv[1])
    elif len(sys.argv) == 3:
        print('Input argument for dataset_path: ',sys.argv[1],' No labels to process ')
        proc = JRDBPreprocessing(dataset_path=sys.argv[1],no_labels=True)
    else:
        print('No input argument given. Must use at least argument: jrdb dataset path.')




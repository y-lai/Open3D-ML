#!/usr/bin/env python3
import numpy as np
import os, sys
from os.path import exists, join, isfile, dirname, abspath, split
from glob import glob

import open3d as o3d
from open3d_ros_helper import open3d_ros_helper as orh

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
import torch

from .calibration_imported import OmniCalibration

class JRDBPreprocessing():
    ''''Pre-processing class to merge pointcloud from both velodynes on the JackRabbot.'''
    def __init__(self,
                 calib_folder,
                 dataset_path):
        rospy.init_node('jrdb_preprocessing_node',anonymous=True)
        self.calib_folder = calib_folder
        self.calib = OmniCalibration(calib_folder=self.calib_folder)
        self.dataset_path_full = dataset_path
        
        # JRDB dataset has identical folder names between upper and lower velodyne.
        self.upper_folders = glob(join(self.dataset_path_full, 'upper_velodyne','*'))
        self.upper_folders.sort()
        
        # check if joined folder exists - mkdir if so
        self.joined_dataset_path_full = join(self.dataset_path_full,'both_velodyne')
        if not os.path.exists(self.joined_dataset_path_full):
            os.mkdir(self.joined_dataset_path_full)
            os.chmod(self.joined_dataset_path_full,0o775)
            os.chown(self.joined_dataset_path_full,1000,1000)
        self.all_files = self.merge_pointclouds(self.upper_folders)


    def merge_pointclouds(self,upper_folders):
        # index to shift all different scenes into the same index count
        idx = 0
        # for each pair, update calibration to same frame, merge pointclouds, and rename the pcd
        for folder in upper_folders:
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
                upper_pc = self.calib.move_lidar_to_camera_frame(upper_torch, upper=True)
                lower_pc = self.calib.move_lidar_to_camera_frame(lower_torch, upper=False)
                joined_pc = torch.cat([upper_pc, lower_pc], dim=0)
                joined_np = joined_pc.numpy()

                jointpc = o3d.geometry.PointCloud()
                jointpc.points = o3d.utility.Vector3dVector(joined_np)
                # convert idx into 6 digit number
                index_str = str(idx).zfill(6)
                o3d.io.write_point_cloud(join(self.joined_dataset_path_full,index_str+'.pcd'),jointpc)
                idx+=1


if __name__ == '__main__':
    if len(sys.argv) == 3:
        print('Input argument for dataset_path: ',sys.argv[1],' Input calib_folder: ',sys.argv[2])
        proc = JRDBPreprocessing(dataset_path=sys.argv[1],calib_folder=sys.argv[2])
    else:
        print('No input argument given. Must use two arguments: jrdb dataset path, folder where the calibration yaml files exist.')




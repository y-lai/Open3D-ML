import numpy as np
import sys
import signal
from os.path import join
from glob import glob
from pathlib import Path

import ml3d
from ml3d.datasets.utils import BEVBox3D
import open3d as o3d

def signal_handler(sig, frame):
    print('SIGINT obtained from Ctrl+C')
    sys.exit(0)

def viewTrainingSet():
    for (label_path,pcd_path) in zip(all_labels,all_pointcloud):
        assert Path(pcd_path).exists()
        assert Path(label_path).exists()
        # print('Showing pointcloud: ',pcd_path)
        
        pointcloud = o3d.io.read_point_cloud(pcd_path)
        
        # extract bounding boxes of labels
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        bboxes = []
        for line in lines:
            label = line.strip().split(' ')
            center = [float(label[0]),float(label[1]),float(label[2])]
            size = [float(label[3]),float(label[4]),float(label[5])]
            yaw = float(label[6])
            label_class = label[7]
            confidence = float(label[8])
            bboxes.append(BEVBox3D(center, size, yaw, label_class, confidence))    
        
        data = [{'name': 'pointcloud', 'points': np.asarray(pointcloud.points)}]
        vis = ml3d.vis.Visualizer()
        vis.visualize(data,lut=None,bounding_boxes=bboxes)

def viewTestSet():
    for pcd_path in all_pointcloud:
        assert Path(pcd_path).exists()
        
        pointcloud = o3d.io.read_point_cloud(pcd_path)
                
        data = [{'name': 'pointcloud', 'points': np.asarray(pointcloud.points)}]
        vis = ml3d.vis.Visualizer()
        vis.visualize(data,lut=None,bounding_boxes=None)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('One argument required: <Path-to-directory-with-joined_velodyne-and-labels>')
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
        
    print('Dataset visualisation directory: ',sys.argv[1])
    path_to_data = sys.argv[1]
    
    path_pointcloud = join(path_to_data,'both_velodyne')
    path_labels = join(path_to_data,'labels')
    
    all_pointcloud = glob(join(path_pointcloud,'*.pcd'))
    all_pointcloud.sort()
    
    all_labels = glob(join(path_labels,'*.txt'))
    all_labels.sort()

    if Path(path_labels).exists():
        viewTrainingSet()
    else:
        viewTestSet()
    
    
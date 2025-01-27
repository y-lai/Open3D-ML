from multiprocessing.sharedctypes import Value
import numpy as np
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
from glob import glob
import logging
import codecs
# from .utils.jrdb_preprocessing import JRDBPreprocessing
from .base_dataset import BaseDataset
from ..utils import make_dir, DATASET
from .utils import BEVBox3D
import open3d as o3d

log = logging.getLogger(__name__)

class JRDB(BaseDataset):
    """This class is used to create a dataset based on the JRDB dataset. Dataset details can be found at: https://jrdb.erc.monash.edu/"""

    def __init__(self,
                 dataset_path,
                 name='JRDB',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 test_result_folder='./test',
                 **kwargs):

        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.name=cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 1
        self.label_to_names = dict({0: 'Pedestrian'})
        self.calib_folder = join(self.dataset_path,'train_dataset_with_activity','calibration')

        self.train_dataset_path = join(cfg.dataset_path, 'train_dataset_with_activity', 'pointclouds')
        self.test_dataset_path = join(cfg.dataset_path, 'test_dataset_without_labels', 'pointclouds')
        
        # assume that JRDB dataset is pre-processed. Pointclouds are merged using JRDBPreprocessing class
        if not Path(join(self.train_dataset_path,'both_velodyne')).exists():
            print('Training dataset has not been pre-processed. Assertion will fail.')
            # JRDBPreprocessing(calib_folder=self.calib_folder,dataset_path=self.train_dataset_path)
        assert(Path(join(self.train_dataset_path,'both_velodyne')).exists())

        if not Path(join(self.test_dataset_path,'both_velodyne')).exists():
            print('Testing dataset has not been pre-processed. Assertion will fail.')
            # JRDBPreprocessing(calib_folder=self.calib_folder,dataset_path=self.test_dataset_path,no_labels=True)
        assert(Path(join(self.test_dataset_path,'both_velodyne')).exists())

        self.all_label_files = glob(join(self.train_dataset_path,'labels','*.txt'))
        self.all_label_files.sort()
        self.all_train_files = glob(join(self.train_dataset_path, 'both_velodyne', '*.pcd'))
        self.all_train_files.sort()
        print('Length of all files: ',len(self.all_train_files),'. Validation split at 70\% - no. of files: ',int(len(self.all_train_files)*0.7))
        self.train_files = self.all_train_files[:int(len(self.all_train_files)*0.7)]
        self.train_label_files = self.all_label_files[:int(len(self.all_train_files)*0.7)]
        self.val_files = self.all_train_files[int(len(self.all_train_files)*0.7)+1:]
        self.val_label_files = self.all_label_files[int(len(self.all_train_files)*0.7)+1:]

        self.test_files = glob(join(self.test_dataset_path, 'both_velodyne', '*.pcd'))
        self.test_files.sort()

    def get_split_list(self, split):
        if split in ['train','training']:
            return self.train_files
        elif split in ['test', 'testing']:
            return self.test_files
        elif split in ['val', 'validation']:
            return self.val_files
        elif split in ['all']:
            return self.train_files + self.val_files + self.train_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def get_label_list(self, split):
        if split in ['train','training']:
            return self.train_label_files
        elif split in ['val','validation']:
            return self.val_label_files
        elif split in ['all']:
            return self.train_label_files + self.val_label_files
        elif split in ['test','testing']:
            return []
        else:
            raise ValueError("Invalid split {} for label files".format(split))

    def get_split(self, split):
        return JRDBSplit(self, split=split)

    @staticmethod
    def get_label_to_names():
        return {0: 'Pedestrian'}

    def is_tested(self):
        pass

    def read_pointcloud(self, path):
        # open path of pointcloud
        assert Path(path).exists()
        pointcloud = o3d.io.read_point_cloud(path)
        # numpify o3d pointcloud
        return np.asarray(pointcloud.points)

    def read_label(self, path):
        # read txt file for labels
        label = []
        if not Path(path).exists():
            return []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        objects = []
        for line in lines:
            label = line.strip().split(' ')

            center = [float(label[0]),float(label[1]),float(label[2])]
            size = [float(label[3]),float(label[4]),float(label[5])]
            yaw = float(label[6])
            label_class = label[7]
            confidence = float(label[8])
            objects.append(BEVBox3D(center,size,yaw,label_class,confidence))
        return objects

    def save_test_result(self, results, attrs):
        make_dir(self.cfg.test_result_folder)
        for attr, res in zip(attrs, results):
            name =attr['name']
            path = join(self.cfg.test_result_folder, name + '.txt')
            f = open(path, 'w')
            # update BEVBox3D to save results (potentially can remove this since not visualising trained data)
            for box in res:
                f.write(box.to_kitti_format(box.confidence))
                f.write('\n')


class JRDBSplit():
    def __init__(self, dataset, split='train'):
        self.split = split
        self.dataset = dataset
        self.path_list = dataset.get_split_list(split)
        self.label_path_list = dataset.get_label_list(split)

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        path  = self.path_list[idx]
        pointcloud = self.dataset.read_pointcloud(path)
        if self.label_path_list == []:
            label = []
        else:
            labelpath = self.label_path_list[idx]
            label = self.dataset.read_label(labelpath)

        # convert to KITTI data style from KITTISplit
        return {
            "point": pointcloud,
            "full_point": pointcloud,
            "feat": None,
            "calib": None,
            "bounding_boxes": label,
        }

    def get_attr(self,idx):
        path = self.path_list[idx]
        name = Path(path).name.split('.')[0]
        return {'name': name, 'path': path, 'split': self.split}


DATASET._register_module(JRDB)
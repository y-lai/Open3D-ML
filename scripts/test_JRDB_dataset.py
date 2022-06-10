import numpy as np
import sys
import signal
from os.path import join
from glob import glob
from pathlib import Path

# import ml3d
import ml3d.torch as _ml3d
from ml3d.utils import Config
from ml3d.datasets import JRDB
from ml3d.vis import Visualizer

def signal_handler(sig,frame):
    print('SIGINT obtained from Ctrl+C')
    sys.exit(0)
    
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('One argument required: <Path-to-config-file>')
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    cfgpath = sys.argv[1]
    
    cfg = Config.load_from_file(cfgpath)
    model = _ml3d.models.PointPillars(**cfg.model)
    
    dataset = JRDB(cfg.dataset.pop('dataset_path', None),**cfg.dataset)    
    pipeline = _ml3d.ObjectDetection(model,dataset=dataset, device="gpu", **cfg.pipeline)
    pipeline.load_ckpt(ckpt_path=model.cfg['ckpt_path'])
    
    testsplit = dataset.get_split('test')
    
    for idx in range(len(testsplit)):
        data = testsplit.get_data(idx)
        result = pipeline.run_inference(data)
        
        boxes = data['bounding_boxes']
        boxes.extend(result[0])
        
        visdata = [{'name': 'pointcloud', 'points': data['point']}]
        vis = Visualizer()
        vis.visualize(visdata,lut=None,bounding_boxes=boxes)
        
        
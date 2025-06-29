import os
import numpy as np
import torch
from tqdm import tqdm 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mobileposer.models import *
from mobileposer.config import *
from mobileposer.utils.model_utils import *
from mobileposer.viewers import SMPLViewer
from mobileposer.loader import DataLoader
import mobileposer.articulate as art


class Viewer:
    def __init__(self, dataset: str='imuposer', seq_num: int=0, combo: str='lw_rp'):
        """Viewer class for visualizing pose."""
        # load models 
        self.device = model_config.device
        self.model = load_model(paths.weights_file).to(self.device).eval()
        self.headio = HeadIO.load_from_checkpoint("checkpoints/headio.ckpt")

        # setup dataloader
        self.dataloader = DataLoader(dataset, combo=combo, device=self.device)
        self.data = self.dataloader.load_data(seq_num)
    
    def _evaluate_model(self):
        """Evaluate the model."""
        data = self.data['imu']
        if getenv('ONLINE'):
            # online model evaluation (slower)
            pose, joints, tran, contact = [torch.stack(_) for _ in zip(*[self.model.forward_online(f) for f in tqdm(data)])]
        else:
            # offline model evaluation  
            with torch.no_grad():
                pose, joints, tran, contact = self.model.forward_offline(data.unsqueeze(0), [data.shape[0]]) 
        return pose, tran, joints, contact

    def view(self, with_tran: bool=False):
        """View the pose."""
        pose_t, tran_t = self.data['pose'], self.data['tran']
        pose_p, tran_p, _, _ = self._evaluate_model()
        viewer = SMPLViewer()
        
        imu = self.data['imu']
        head_tran = self.headio.forward_offline(imu.unsqueeze(0), joints=pose_t, input_lengths=[imu.shape[0]]).squeeze(0).detach()
        viewer.view(pose_p, head_tran, pose_t, tran_t, with_tran=with_tran)

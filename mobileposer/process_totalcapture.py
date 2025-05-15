import numpy as np
import argparse 
import glob
import os
import torch
import math
import cv2
import scipy.io as sio
import pickle as pkl
import quaternion as pip_quaternion # The quaternion installed via pip

from mobileposer.articulate.math import *
from mobileposer.config import paths


# MobilePoser joint indices
TC_2_MP = [5, 6, 7, 8, 0, 2] # [LeftForeArm, RightForeArm, LeftUpLeg, RightUpLeg, Head, Hips]

# Whether to save as Pickle file
SAVE_PKL = True


# Read IMU
def read_imu(imu_file_path):
    with open(imu_file_path) as fin:
        all_lines = fin.readlines()
    del all_lines[0]
    del all_lines[::14]

    def f1(line):
        str_list = line.split()[1:]
        num_list = [float(str_num) for str_num in str_list]
        return num_list
    data = map(f1, all_lines)
    data = np.array(list(data))
    imu_ori = data[:, :4].reshape(-1, 13, 4)
    imu_acc = data[:, 4:].reshape(-1, 13, 3)
    return imu_ori, imu_acc

# Convert quaternion to rotation matrix
def ori_to_rotmat(imu_ori_quat):
    imu_ori_tensor = torch.from_numpy(imu_ori_quat)
    imu_ori_tensor = imu_ori_tensor.reshape(-1, 4) # (N, 13, 4) -> (N*13, 4)
    imu_ori_tensor = quaternion_to_rotation_matrix(imu_ori_tensor)
    imu_ori_tensor = imu_ori_tensor.reshape(-1, 13, 3, 3) # (N, 13, 4) -> (N*13, 4)
    return imu_ori_tensor.clone().detach().cpu().numpy()

# Read calibration file
def read_calib(calib_file_path):
    with open(calib_file_path) as fin:
        all_lines = fin.readlines()
    del all_lines[0]

    def f1(line):
        str_list = line.split()[1:]
        num_list = [float(str_num) for str_num in str_list]
        return num_list
    
    data = list(map(f1, all_lines))
    data = np.array(data)    

    data = data[:, [3, 0, 1, 2]] # quaternions are in the form (x, y, z, w). So we need to reorder them to (w, x, y, z)
    data_tensor = torch.from_numpy(data)
    data_tensor = quaternion_to_rotation_matrix(data_tensor)
    return data_tensor.clone().detach().cpu().numpy()

def process_imu(imu_file_path):
    imu_file_path = str(imu_file_path)

    # Read IMU data
    imu_ori_quat, imu_acc = read_imu(imu_file_path)
    imu_ori_mat = ori_to_rotmat(imu_ori_quat)

    # Read reference data
    calib_file_path = imu_file_path.replace('_Xsens.sensors', '_calib_imu_ref.txt')
    calib_ref_mat = read_calib(calib_file_path) # [13, 3, 3]

    # Read bone data
    calib_file_path = imu_file_path.replace('_Xsens.sensors', '_calib_imu_bone.txt')
    calib_bone_matrix = read_calib(calib_file_path)

    save_ext = '.pkl' if SAVE_PKL else '.pt' 
    save_path = calib_file_path.replace(str(folder_path), str(folder_path)).replace('_calib_imu_bone.txt', save_ext).replace('raw', 'processed')

    if SAVE_PKL:
        save_path = os.path.join(paths.calibrated_totalcapture, save_path.split('/')[-1]) 

    rot_y = cv2.Rodrigues(np.array([0, np.pi, 0]))[0] # first elem is rot. matrix, second elem is jacob. matrix
    calib_ref_rotated = rot_y@calib_ref_mat

    total = len(imu_ori_mat)
    ori_global, acc_global = [], []
    for i in range(total):
        for j in range(13):
            ori_tmp = imu_ori_mat[i][j]
            ori_calib = calib_ref_rotated[j]
            new_ori = ori_calib@ori_tmp
            ori_global.append(new_ori)			

            acc_tmp = imu_acc[i][j]
            new_acc = new_ori@acc_tmp.flatten() - np.array([0, 9.8707, 0])
            acc_global.append(new_acc)
        
    ori_global = np.array(ori_global)
    ori_global = ori_global.reshape([-1, 13, 3, 3])
    acc_global = np.array(acc_global)
    acc_global = acc_global.reshape([-1, 13, 3])

    for i in range(1, ori_global.shape[0]):
        for j in range(0, ori_global.shape[1]):
            ori_global[i, j, :, :] = np.dot(ori_global[i, j, :, :], ori_global[0, j, :, :].T)
    
    res = {'ori': ori_global[1:, np.array(TC_2_MP), :, :], 'acc': acc_global[1:, np.array(TC_2_MP), :]}
    if SAVE_PKL:
        # Save as .pkl
        with open(save_path, 'wb') as fout:
            pkl.dump(res, fout)
    else:
        # Save as .pt
        torch.save(res, save_path)
    
    print(f'Finish {save_path}')

if __name__ == '__main__':
    folder_path = paths.calibrated_totalcapture
    
    all_imu_files = []
    for s in os.listdir(folder_path):
        user_path = os.path.join(folder_path, s)
        if os.path.isdir(user_path):
            all_imu_files.extend(glob.glob(os.path.join(user_path, '*Xsens.sensors')))

    for imu_file in all_imu_files:
        process_imu(imu_file)
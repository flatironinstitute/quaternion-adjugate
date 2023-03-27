import torch
import numpy as np
from AdjQuat import utils
from AdjQuat import solutions
from AdjQuat import LHM

### 2D SIMULATORS

def generate_data(replicates,size,error):

    U_set = []
    xy_set = []
    R_set = []

    for i in range(replicates):
        xy = torch.randn(size, 2, dtype=torch.double)
        quat2 = np.random.uniform(-1,1,2) 
        quat2 = quat2/np.sqrt(quat2.dot(quat2))#has to be normalized
        
        rot2 = utils.quat_to_rot2(quat2)
        my_U = utils.make_U(xy,quat2,error)
        
        U_set.append(my_U)
        xy_set.append(xy)
        R_set.append(rot2)
    
    return np.array([ R_set, xy_set, U_set ])

def get_lsq_from_datasets(datasets):
    R_set = []
    lsq_set = []
    lsq_mean_set = []
    lsq_rot_set = []
    lsq_mean_rot_set = []
    for key in datasets.keys():
        print(key)
        R_estimated = [solutions.R_2D_pose_function(data[1], data[2][0]) for data in datasets[key].T]
        lsq = [utils.least_squares_4_2D(R_estimated [data[0]],data[1][1], data[1][2][0]) for data in enumerate(datasets[key].T)]
        lsq_rot = [utils.least_squares_4_2D(data[0],data[1], data[2][0]) for data in datasets[key].T]
        
        R_set.append(R_estimated)
        lsq_set.append(lsq)
        lsq_mean_set.append(np.mean(lsq))
        lsq_rot_set.append(lsq_rot)
        lsq_mean_rot_set.append(np.mean(lsq_rot))
    
    return R_set, lsq_set, lsq_mean_set, lsq_rot_set, lsq_mean_rot_set

### 3D SIMULATORS
    
def generate_data_3D(replicates,size,error):

    U_set = []
    xyz_set = []
    R_set = []

    for i in range(replicates):
        xyz = torch.randn(size, 3, dtype=torch.double)
        quat = np.random.uniform(-1,1,4) 
        quat = quat/np.sqrt(quat.dot(quat))#has to be normalized
        
        rot = utils.quat_to_rot(quat)
        U =utils. make_U3(xyz,quat,error)
        
        U_set.append(U)
        xyz_set.append(xyz)
        R_set.append(rot)
    
    return np.array([ R_set, xyz_set, U_set ])

def get_lsq_from_3Ddatasets(datasets):
    R_set = []
    lsq_set = []
    lsq_mean_set = []
    lsq_rot_set = []
    lsq_mean_rot_set = []
    lsq_M_set = []
    lsq_mean_M_set = []
    for key in datasets.keys():
        print(key)
        Rt_estimated = [solutions.make_R_tilde(data[1], data[2][0]) for data in datasets[key].T]
        lsq = [utils.least_squares_4_3D(Rt_estimated[data[0]],data[1][1], data[1][2][0]) for data in enumerate(datasets[key].T)]
        lsq_rot = [utils.least_squares_4_3D(data[0],data[1], data[2][0]) for data in datasets[key].T]
        Mq_estimated = [solutions.make_M_opt_rot(data[1], data[2][0]) for data in datasets[key].T]
        lsq_M = [utils.least_squares_4_3D(Mq_estimated[data[0]],data[1][1], data[1][2][0]) for data in enumerate(datasets[key].T)]

        R_set.append(Rt_estimated)
        lsq_set.append(lsq)
        lsq_mean_set.append(np.mean(lsq))
        lsq_rot_set.append(lsq_rot)
        lsq_mean_rot_set.append(np.mean(lsq_rot))
        lsq_M_set.append(lsq_M)
        lsq_mean_M_set.append(np.mean(lsq_M))
    
    return R_set, lsq_set, lsq_mean_set, lsq_rot_set, lsq_mean_rot_set, lsq_M_set, lsq_mean_M_set
    

### SIMPLE 3D PERSPECTIVE SIMULATORS

def generate_perspective_data_3D(replicates,size,error,f):

    U_set = []
    xyz_set = []
    R_set = []

    for i in range(replicates):
        xyz = torch.randn(size, 3, dtype=torch.double)
        quat = np.random.uniform(-1,1,4) 
        quat = quat/np.sqrt(quat.dot(quat))#has to be normalized
        
        rot = utils.quat_to_rot(quat)
        U = utils.quat_to_persp_proj_sigma(xyz, quat, error, f)
        
        U_set.append(U)
        xyz_set.append(xyz)
        R_set.append(rot)
    
    return np.array([ R_set, xyz_set, U_set ])


def get_lsq_from_3Ddatasets_persp(datasets):
    R_set = []
    lsq_set = []
    lsq_mean_set = []
    lsq_rot_set = []
    lsq_mean_rot_set = []
    lsq_M_set = []
    lsq_mean_M_set = []
    lsq_LHM_set = []
    lsq_mean_LHM_set = []
    for key in datasets.keys():
        print(key)
        Rt_estimated = [solutions.make_R_tilde(data[1], data[2]) for data in datasets[key].T]
        lsq = [utils.least_squares_4_3D_persp(Rt_estimated[data[0]],data[1][1], key, data[1][2]) for data in enumerate(datasets[key].T)]
        lsq_rot = [utils.least_squares_4_3D_persp(data[0],data[1], key, data[2]) for data in datasets[key].T]
        Mq_estimated = [solutions.make_M_opt_rot(data[1], data[2]) for data in datasets[key].T]
        lsq_M = [utils.least_squares_4_3D_persp(Mq_estimated[data[0]],data[1][1], key, data[1][2]) for data in enumerate(datasets[key].T)]
        R_iter_estimated = [LHM.solve_LHM(data[1], data[2]) for data in datasets[key].T]
        lsq_LHM = [utils.least_squares_4_3D_persp(R_iter_estimated[data[0]],data[1][1], key, data[1][2]) for data in enumerate(datasets[key].T)]
        
        R_set.append(Rt_estimated)
        lsq_set.append(lsq)
        lsq_mean_set.append(np.mean(lsq))
        lsq_rot_set.append(lsq_rot)
        lsq_mean_rot_set.append(np.mean(lsq_rot))
        lsq_M_set.append(lsq_M)
        lsq_mean_M_set.append(np.mean(lsq_M))
        lsq_LHM_set.append(lsq_LHM)
        lsq_mean_LHM_set.append(np.mean(lsq_LHM))
    
    return lsq_mean_set, lsq_mean_rot_set, lsq_mean_M_set, lsq_mean_LHM_set


# Not that this data generation procedure was chosen to mirror that of this paper/code repository: XXXX

def generate_perspective_data_up(npt,f,sigma):
    z_min = f - 2;
    z_max = f + 2;
    
    Xc = [np.random.uniform(-2, 2, size=npt), 
      np.random.uniform(-2, 2, size=npt), 
      np.random.uniform(z_min, z_max, size=npt)];
    Xc = np.asarray(Xc)
    t = [np.mean(Xc[0]),np.mean(Xc[1]),np.mean(Xc[2])]
    f0 = t[2]
    gt_q = utils.gen_quat(1)
    gt_q = torch.from_numpy(gt_q[0]).float()
    gt_r = torch.from_numpy(utils.quat_to_rot(gt_q)).float()
    
    Xc = torch.from_numpy(Xc).float()
    T = torch.from_numpy(np.resize(t,[npt,3])).T.float()
    
    Xc_orig = Xc-T
    XXw = torch.mm(gt_r,(Xc_orig))
    error = np.random.normal(0,sigma,[3,len(XXw[0])])
    XXe = torch.stack([XXw[0]+T[0]+error[0],  XXw[1]+T[1]+error[1],  XXw[2]+T[2]+error[2]])
    xx = torch.stack([XXe[0]/XXe[2],  XXe[1]/XXe[2]])
    xx = xx
    
    return Xc, Xc_orig, XXw, XXe, xx, gt_q, gt_r

def generate_perspective_data_3D_up(replicates,npt,sigma,f):

    U_set = []
    xyz_set = []
    R_set = []

    for i in range(replicates):
        Xc, Xc_orig, XXw, XXe, xx, gt_q, gt_r = generate_perspective_data_up(npt,f,sigma)
        p3d = Xc_orig
        p2d = xx
        
        U_set.append(p2d.T.float())
        xyz_set.append(p3d.T.float())
        R_set.append(gt_r)
    
    return np.array([ R_set, xyz_set, U_set ])




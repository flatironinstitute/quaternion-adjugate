import numpy as np
from scipy.optimize import minimize

def quat_dist(q1, q2):
    error_p = 2*np.arccos(q1.dot(q2.T))*180/np.pi
    error_m = 2*np.arccos(q1.dot(-q2.T))*180/np.pi
    return min(error_p,error_m)

### 2D UTILS

def quat_to_rot2(quat2):
    a = quat2[0]
    b = quat2[1]
    
    matrix = np.zeros([2,2])
    
    matrix[0,0] = a**2 - b**2
    matrix[0,1] = -2*a*b
    matrix[1,0] = 2*a*b
    matrix[1,1] = a**2 - b**2
    
    return matrix

def quat_to_proj2(quat2):
    
    a = quat2[0]
    b = quat2[1]
    
    matrix = np.zeros([2])
    
    matrix[0] = a**2 - b**2
    matrix[1] = -2*a*b

    return matrix

def make_U(xy,quat,sigma):
    
    error = np.random.normal(0,sigma,[2,len(xy)])
    xy_error =  np.asarray([ np.asarray(xy.T)[0] + error[0], 
                             np.asarray(xy.T)[1] + error[1] ]).T
    
    proj = quat_to_proj2(quat)
    U = proj.dot(np.asarray(xy_error).T).T
    
    return U,xy_error


def least_squares_4_2D(R, xy,U): #eq 66
    value = 0
    for i in range(len(xy)):
        value += (R[0].dot(xy[i]) - U[i])**2
    return value
	
### 3D UTILS

def gen_quat(size):
    #this code generates a random quaternion
    #NOTE: this is _actually_ the _correct_ way to do a uniform random rotation in SO3
    quats = np.empty((size, 4))

    count = 0
    while count < size:

        quat = np.random.uniform(
            -1, 1, 4
        )  # note this is a half-open interval, so 1 is not included but -1 is
        norm = np.sqrt(np.sum(quat**2))

        if 0.2 <= norm <= 1.0:
            quats[count] = quat / norm
            count += 1

    return quats


def rotation2quaternion(M):
    tr = np.trace(M)
    m = M.reshape(-1)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[7] - m[5]) / s
        y = (m[2] - m[6]) / s
        z = (m[3] - m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        s = np.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        w = (m[7] - m[5]) / s
        x = 0.25 * s
        y = (m[1] + m[3]) / s
        z = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = np.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        w = (m[2] - m[6]) / s
        x = (m[1] + m[3]) / s
        y = 0.25 * s
        z = (m[5] + m[7]) / s
    else:
        s = np.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
        w = (m[3] - m[1]) / s
        x = (m[2] + m[6]) / s
        y = (m[5] + m[7]) / s
        z = 0.25 * s
    Q = np.array([w, x, y, z]).reshape(-1)
    return Q



def quat_to_rot(quat):
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
    
    matrix = np.zeros([3,3])
    
    matrix[0,0] = q0**2 + q1**2 - q2**2 - q3**2
    matrix[0,1] = 2*q1*q2 - 2*q0*q3
    matrix[0,2] = 2*q1*q3 + 2*q0*q2
    
    matrix[1,0] = 2*q1*q2 + 2*q0*q3
    matrix[1,1] = q0**2 - q1**2 + q2**2 - q3**2
    matrix[1,2] = 2*q2*q3 - 2*q0*q1

    matrix[2,0] = 2*q1*q3 - 2*q0*q2
    matrix[2,1] = 2*q2*q3 + 2*q0*q1
    matrix[2,2] = q0**2 - q1**2 - q2**2 + q3**2
    
    return matrix

def quat_to_proj(quat):
    
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
    
    matrix = np.zeros([2,3])
    
    # as with 2D, this just lops of the bottom row of the rotation matrix
    matrix[0,0] = q0**2 + q1**2 - q2**2 - q3**2
    matrix[0,1] = 2*q1*q2 - 2*q0*q3
    matrix[0,2] = 2*q1*q3 + 2*q0*q2
    
    matrix[1,0] = 2*q1*q2 + 2*q0*q3
    matrix[1,1] = q0**2 - q1**2 + q2**2 - q3**2
    matrix[1,2] = 2*q2*q3 - 2*q0*q1

    #matrix[2,0] = 2*q1*q3 - 2*q0*q2
    #matrix[2,1] = 2*q2*q3 + 2*q0*q1
    #matrix[2,2] = q0**2 - q1**2 - q2**2 + q3**2
    
    return matrix
    
def make_U3(xyz,quat,sigma):
    
    error = np.random.normal(0,sigma,[3,len(xyz)])
    xyz_error =  np.asarray([ np.asarray(xyz.T)[0].T+error[0],
                         np.asarray(xyz.T)[1].T+error[1],
                         np.asarray(xyz.T)[2].T+error[2] ]).T
    proj = quat_to_proj(quat)
    U = proj.dot(np.asarray(xyz_error).T).T
    
    return U, xyz_error


def least_squares_4_3D(R, xyz, U): #eq 66
    value = 0
    for i in range(len(xyz)):
        intermediate = R[0:2].dot(xyz[i]) - U[i]
        value += intermediate.dot(intermediate.T).T
    return value


def argmin_f(point_cloud_3D, projection_2D):
    
    def function(quat):
        q0 = quat[0]
        q1 = quat[1]
        q2 = quat[2]
        q3 = quat[3]
    
        qproj = [  [q0**2 + q1**2 - q2**2 - q3**2, 2*q1*q2 - 2*q0*q3,  2*q0*q2 + 2*q1*q3], 
               [2*q1*q2 + 2*q0*q3, q0**2 - q1**2 + q2**2 - q3**2, -2*q0*q1 + 2*q2*q3]  ]
        qproj = np.asarray(qproj)

        summation = 0
        for k in range(len(point_cloud_3D)):
            difference = qproj.dot(point_cloud_3D[k]) - projection_2D[k]
            summation += difference.dot(difference)
     
        return summation
    
    def constraints_function(quat):
    
        return quat[0]**2 + quat[1]**2 + quat[2]**2 + quat[3]**2 - 1
    
    point_cloud_3D = np.asarray(point_cloud_3D)
    constraints = {'type': 'eq', 'fun': constraints_function}
    initial_guess = [1,0,0,0]
    result = minimize(function, initial_guess, method='SLSQP', constraints=constraints)
    return result['x']



### PERSPECTIVE UTILS

#here we make a function to get a perspective projection matrix from our quaternion
def quat_to_persp_proj_sigma(xyz, quat, sigma, f):
    
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
    
    matrix = np.zeros([2,3])
    
    last_line = np.zeros([3])
    
    # as with 2D, this just lops of the bottom row of the rotation matrix
    matrix[0,0] = q0**2 + q1**2 - q2**2 - q3**2
    matrix[0,1] = 2*q1*q2 - 2*q0*q3
    matrix[0,2] = 2*q1*q3 + 2*q0*q2
    
    matrix[1,0] = 2*q1*q2 + 2*q0*q3
    matrix[1,1] = q0**2 - q1**2 + q2**2 - q3**2
    matrix[1,2] = 2*q2*q3 - 2*q0*q1

    last_line[0] = 2*q1*q3 - 2*q0*q2
    last_line[1] = 2*q2*q3 + 2*q0*q1
    last_line[2] = q0**2 - q1**2 - q2**2 + q3**2
    
    
    proj_mat = matrix
    
    error = np.random.normal(0,sigma,[3,len(xyz)])
    xyz_error =  np.asarray([ np.asarray(xyz.T)[0].T+error[0],
                         np.asarray(xyz.T)[1].T+error[1],
                         np.asarray(xyz.T)[2].T+error[2] ]).T
    
    proj_xyz = proj_mat.dot(np.asarray(xyz_error).T).T
    
    lastline_xyz = last_line.dot(np.asarray(xyz_error).T).T 

    proj_xyz_persp = (proj_xyz.T / ( 1 - lastline_xyz/ f) ).T
    
    return proj_xyz_persp


def least_squares_4_3D_persp(R, xyz, f, U):
    value = 0
    for i in range(len(xyz)):
        intermediate = R[0:2].dot(xyz[i])/(1 - R[2].dot(xyz[i])/f) - U[i]
        value += intermediate.dot(intermediate.T).T
    return value

def least_squares_4_3D_persp_up(R, xyz, f, U):
    value = 0
    for i in range(len(xyz)):
        intermediate = R[0:2].dot(xyz[i])/(R[2].dot(xyz[i]) + f) - U[i]
        value += intermediate.dot(intermediate.T).T
    return value

def perspective_argmin_f(point_cloud_3D, projection_2D,f):
    
    def function(quat):
        q0 = quat[0]
        q1 = quat[1]
        q2 = quat[2]
        q3 = quat[3]
    
        qproj = [  [q0**2 + q1**2 - q2**2 - q3**2, 2*q1*q2 - 2*q0*q3,  2*q0*q2 + 2*q1*q3], 
               [2*q1*q2 + 2*q0*q3, q0**2 - q1**2 + q2**2 - q3**2, -2*q0*q1 + 2*q2*q3]  ]
        qproj = np.asarray(qproj)
        last_line = [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, q0**2 - q1**2 - q2**2 + q3**2]
        last_line = np.asarray(last_line)
        
        summation = 0
        for k in range(len(point_cloud_3D)):
            #difference = qproj.dot(point_cloud_3D[k]) - projection_2D[k]
            difference = qproj.dot(point_cloud_3D[k])/(last_line.dot(point_cloud_3D[k]) + f) - projection_2D[k]
            summation += difference.dot(difference)
     
        return summation
    
    def constraints_function(quat):
    
        return quat[0]**2 + quat[1]**2 + quat[2]**2 + quat[3]**2 - 1
    
    point_cloud_3D = np.asarray(point_cloud_3D)
    constraints = {'type': 'eq', 'fun': constraints_function}
    initial_guess = [1,0,0,0]
    result = minimize(function, initial_guess, method='SLSQP', constraints=constraints)
    return result['x']





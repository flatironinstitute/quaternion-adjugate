import numpy as np
from AdjQuat import utils

def R_2D_pose_function(xy,U):
    XX = xy.T[0].dot(xy.T[0])
    XY = xy.T[0].dot(xy.T[1])
    YY = xy.T[1].dot(xy.T[1])
    UX = U.dot(xy.T[0])
    UY = U.dot(xy.T[1])
    
    d1 = np.linalg.det(np.array([[XX,XY],[XY,YY]]))
    d2 = np.linalg.det(np.array([[UX,XY],[UY,YY]]))
    d3 = np.linalg.det(np.array([[UX,XX],[UY,XY]]))


    D = np.array([[d2,-d3],[d3,d2]])
    R = 1/(np.sqrt(d2**2 + d3**2)) * D
    
    return R
    
def make_dets(xyz,U):
    
    XX = xyz.T[0].dot(xyz.T[0])
    XY = xyz.T[0].dot(xyz.T[1])
    XZ = xyz.T[0].dot(xyz.T[2])

    YY = xyz.T[1].dot(xyz.T[1])
    YZ = xyz.T[1].dot(xyz.T[2])
    ZZ = xyz.T[2].dot(xyz.T[2])

    UX = U.T[0].dot(xyz.T[0])
    UY = U.T[0].dot(xyz.T[1])
    UZ = U.T[0].dot(xyz.T[2])
    
    VX = U.T[1].dot(xyz.T[0])
    VY = U.T[1].dot(xyz.T[1])
    VZ = U.T[1].dot(xyz.T[2])
    
    d1 = np.linalg.det(np.array([[XX,XY,XZ],[XY,YY,YZ],[XZ,YZ,ZZ]])) 
    d2 = np.linalg.det(np.array([[XX,XY,UX],[XY,YY,UY],[XZ,YZ,UZ]]))
    d3 = np.linalg.det(np.array([[XX,XY,VX],[XY,YY,VY],[XZ,YZ,VZ]]))

    d4 = np.linalg.det(np.array([[XX,XZ,UX],[XY,YZ,UY],[XZ,ZZ,UZ]]))
    d5 = np.linalg.det(np.array([[XX,XZ,VX],[XY,YZ,VY],[XZ,ZZ,VZ]]))
    d6 = np.linalg.det(np.array([[XX,UX,VX],[XY,UY,VY],[XZ,UZ,VZ]]))
    
    d7 = np.linalg.det(np.array([[XY,XZ,UX],[YY,YZ,UY],[YZ,ZZ,UZ]]))
    d8 = np.linalg.det(np.array([[XY,XZ,VX],[YY,YZ,VY],[YZ,ZZ,VZ]]))
    d9 = np.linalg.det(np.array([[XY,UX,VX],[YY,UY,VY],[YZ,UZ,VZ]]))
    d10 = np.linalg.det(np.array([[XZ,UX,VX],[YZ,UY,VY],[ZZ,UZ,VZ]]))

    return d1, d2, d3, d4, d5, d6, d7, d8, d9, d10

def make_R_tilde(xyz,U):
    
    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 = make_dets(xyz,U)

    R_tilde = np.array([[d7/d1,  -d4/d1,     d2/d1],
                    [d8/d1,  -d5/d1,     d3/d1],
                    [d6/d1,   d9/d1,     d10/d1]])
    
    return R_tilde
    
    
def M_3D_pose_function(xyz,U):
    
    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 = make_dets(xyz,U)

    M = np.array([[d7-d5+d10, -(d3-d9),  -(d6-d2),    -(-d4-d8)],
                  [-(d3-d9),     d7+d5-d10, -d4+d8,     d6+d2] ,
                  [-(d6-d2),     -d4+d8,    -d7-d5-d10, d3+d9] ,
                  [-(-d4-d8),    d6+d2,     d3+d9,      -d7+d5+d10]])
    #we could keep this is as is, but if we divide by d1, we will have a matrix whose determinant is 1 instead d1^4
    #if we divide each element by d1, numerical properties are much bette
    
    return M/d1
    
def make_adjugate(matrix): #this function specifically does not depend on the determinant
    C = np.zeros(matrix.shape) #IE this function works even if det=0
    nrows, ncols = C.shape
    for row in range(nrows):
        for col in range(ncols):
            minor_int = np.delete(matrix,row,axis=0)
            minor = np.delete(minor_int,col,axis=1)
            C[row, col] = (-1)**(row+col) * np.linalg.det(minor) # this is the cofactor
    return C.T #the adjugate is the transpose of the cofactor

def make_M_opt_rot(xyz,U):
    M = M_3D_pose_function(xyz, U)
    l,u = np.linalg.eigh(M)
    chi = M - np.identity(4)*l[3]
    adjugate = make_adjugate(chi)
    diags = [abs(adjugate[diag[0],diag[0]]) for diag in enumerate(adjugate)]
    quat_estimate = adjugate[np.argmax(diags)]/np.linalg.norm(adjugate[np.argmax(diags)])
    
    #return [quat_estimate, utils.quat_to_rot(quat_estimate)]
    return utils.quat_to_rot(quat_estimate)
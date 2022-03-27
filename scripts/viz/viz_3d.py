import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)




def draw_coordinates(view_mat,ax,length = 1):

    pos = view_mat @ np.array([0,0,0,1],dtype=np.float64)
    
    v = view_mat @ np.array([length,0,0,0],dtype=np.float64)
    x = Arrow3D(*[[pos[i],pos[i]+v[i]] for i in range(3)], mutation_scale=20, 
            lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(x)

    v = view_mat @ np.array([0,length,0,0],dtype=np.float64)
    y = Arrow3D(*[[pos[i],pos[i]+v[i]] for i in range(3)], mutation_scale=20, 
            lw=3, arrowstyle="-|>", color="g")
    ax.add_artist(y)
    
    v = view_mat @ np.array([0,0,length,0],dtype=np.float64)
    print(pos,v/np.linalg.norm(v),sep="\n")
    z = Arrow3D(*[[pos[i],pos[i]+v[i]] for i in range(3)], mutation_scale=20, 
            lw=3, arrowstyle="-|>", color="b")
    ax.add_artist(z)
    
if __name__ == '__main__':
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    draw_coordinates(np.eye(4,dtype=np.float64),ax,0.5)
    plt.draw()
    plt.show()
    
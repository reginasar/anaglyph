import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def twisty(x_, y_, z_, camera_):
    Ve = np.array([x_, y_, z_])    
    A1 = np.arctan2(camera_[1], camera_[2])
    A2 = np.arcsin(camera_[0]/np.linalg.norm(camera_))
    R1 = np.array([[1,0,0],[0,np.cos(A1),-np.sin(A1)],[0,np.sin(A1),np.cos(A1)]])
    R2 = np.array([[np.cos(A2),0,-np.sin(A2)],[0,1,0],[np.sin(A2),0,np.cos(A2)]])    
    Vf = np.dot(np.dot(R2, R1), Ve)
    
    return Vf[0], Vf[1], Vf[2]

class particles:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def view(self, distance, camera=[1, 0, 0], plot_op='hist', save_name='view.jpg'):
        vx = [1,0,0]
        vy = [0,1,0]
        vz = [0,0,1]
        Ev = np.transpose(np.array([vx,vy,vz]))
        R3 = np.array([[1,0,0],[0,1,0],[0,0,1]])
        obs = np.dot(np.dot(Ev, R3), camera) * distance
        A1 = np.arctan2(obs[1], obs[2])
        A2 = np.arcsin(obs[0]/distance)
        R1 = np.array([[1,0,0],[0,np.cos(A1),-np.sin(A1)],[0,np.sin(A1),np.cos(A1)]])
        R2 = np.array([[np.cos(A2),0,-np.sin(A2)],[0,1,0],[np.sin(A2),0,np.cos(A2)]])
        R3 = np.array([[np.cos(A2),-np.sin(A2),0],[np.sin(A2),np.cos(A2),0],[0,0,1]])
        Ve = np.array([self.x, self.y, self.z])
        Vf = np.dot(np.dot(R2, R1), Ve).T

        if plot_op=='hist':
            crispy, _,_,_ = stats.binned_statistic_2d(self.x, self.y, \
                self.x, bins=30, statistic='count')
            plt.imshow(crispy.T)
            #plt.savefig(save_name)
            plt.show()
        elif plot_op=='point':
            plt.scatter(self.x, self.y, alpha=0.5)
            plt.show()
            #plt.savefig(save_name)
        else:
            raise ValueError('wrong plot_op, must be hist or point.')

    def rotate_camera(self, camera):
        self.x, self.y, self.z = twisty(Ve = np.array([self.x, self.y, self.z])    
        A1 = np.arctan2(camera[1], camera[2])
        A2 = np.arcsin(camera[0]/np.linalg.norm(camera))
        R1 = np.array([[1,0,0],[0,np.cos(A1),-np.sin(A1)],[0,np.sin(A1),np.cos(A1)]])
        R2 = np.array([[np.cos(A2),0,-np.sin(A2)],[0,1,0],[np.sin(A2),0,np.cos(A2)]])    
        Vf = np.dot(np.dot(R2, R1), Ve)
        self.x = Vf[0]
        self.y = Vf[1]
        self.z = Vf[2]
    
        return 

    def view_3D(self, camera=[1., 0., 0.], distance=60., popout=10., out_size=[40., 25.], ps=10, save_name='view.jpg'):
        """left-eye is covered with blue, while right goes with red"""
        camera = np.array(camera) / np.linalg.norm(camera)
        eye_sep = 4. #half separation between eyes [cm]
        
        camera_r = distance * np.array([0, 0, 1]) + eye_sep * np.array([1, 0, 0])
        camera_r = twisty(camera_r[0], camera_r[1], camera_r[2], camera)
        camera_l = distance * np.array([0, 0, 1]) - eye_sep * np.array([1, 0, 0])
        camera_l = twisty(camera_l[0], camera_l[1], camera_l[2], camera)
        
        margin = 1. # minimum empty space at edge [cm]
        xy_max = np.max(np.abs(np.stack((self.x, self.y))))
        if xy_max > np.max(np.abs(self.x)):
            scale = (out_size[1] - 2. * margin) / xy_max
        else:
            scale = (out_size[0] - 2. * margin) / xy_max

        x_s = self.x * scale # coordinates in cm
        y_s = self.y * scale
        z_s = self.z * scale - popout
        plt.scatter(x_s, y_s)
        
        xc, yc, zc = twisty(x_s, y_s, z_s, camera*distance)
        xr, yr, zr = twisty(x_s, y_s, z_s, camera_r)
        xl, yl, zl = twisty(x_s, y_s, z_s, camera_l)
        
        cm = 1/2.54  # centimeters in inches
        fig, ax = plt.subplots(figsize=(out_size[0]*cm, out_size[1]*cm))

        plt.scatter(xl, yl, c='cyan',alpha=0.5, s=ps)
        plt.scatter(xr, yr, c='red',alpha=0.5, s=ps)
        ax.set_facecolor('k')
        #plt.scatter(xc, yc, c='w', s=4)
        plt.show()
        plt.savefig(save_name)

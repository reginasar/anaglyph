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



    def view_3D(self, camera=[1., 0., 0], distance=60., popout=10., out_size=[40., 25.], save_name='view.jpg'):
    """

    camera=[1, 0, 0] : unit vector indicating the direction of the camera
    distance=60. : distance in cm from the viewer to display
    popout=10. : distance in cm that the image will pop out from the display 
    out_size=[40., 25.] : display size in cm [x, y]
    """
        camera = np.array(camera) / np.linalg.norm(camera)
        
        x_rot, y_rot, z_rot= twisty(self.x, self.y, self.z, camera*real_distance)
        
        margin = 1. # minimum empty space at edge [cm]
        xy_max = np.max(np.abs(np.stack((x_rot, y_rot))))
        if xy_max > np.max(np.abs(x_rot)):
            scale = (out_size[1] - 2. * margin) / xy_max
        else:
            scale = (out_size[0] - 2. * margin) / xy_max  
            
        x_s = x_rot * scale # coordinates in cm
        y_s = y_rot * scale
        z_s = z_rot * scale - popout
        plt.scatter(x_s, y_s)     
        sitdown_distance = real_distance*scale
            
        eye_sep = 4. #half separation between eyes [cm]
        
        camera_r = sitdown_distance * np.array([0, 0, 1]) + eye_sep * np.array([1, 0, 0])
        #camera_r = twisty(camera_r[0], camera_r[1], camera_r[2], camera)
        camera_l = sitdown_distance * np.array([0, 0, 1]) - eye_sep * np.array([1, 0, 0])
        #camera_l = twisty(camera_l[0], camera_l[1], camera_l[2], camera)
        
        xr, yr, zr = twisty(x_s, y_s, z_s, camera_r)
        xl, yl, zl = twisty(x_s, y_s, z_s, camera_l)
        
        cm_in = 1/2.54  # centimeters in inches
        fig, ax = plt.subplots(figsize=(outsize[0]*cm_in, outsize[1]*cm_in))

        plt.scatter(xl, yl, c='cyan',alpha=0.5, s=ps)
        plt.scatter(xr, yr, c='red',alpha=0.5, s=ps)
        ax.set_facecolor('k')
        #plt.scatter(xc, yc, c='w', s=4)
        plt.show()
        #plt.savefig(save_name)

class video_3D:
    
    def __init__(self, data=None, trajectory=None, sitdown_distance=None, popout=None, outsize=None, scale=None):
        self.fig, self.ax = plt.subplots()
        self.scatr = self.ax.scatter([], [], c='red', alpha=0.7, s=8)
        self.scatl = self.ax.scatter([], [], c='cyan', alpha=0.7, s=8)
        self.ax.set_facecolor('k')
        
        if data!=None:
            self.data = data
            
        if trajectory!=None:
            self.trajectory = trajectory
        
        if sitdown_distance==None:
            sitdown_distance = 150.
            
        self.sitdown_distance = sitdown_distance
        
        if popout==None:
            popout = 0.
            
        self.popout = popout
        
        if outsize==None:
            outsize = [32., 18.]
        
        self.outsize = outsize
        self.ax.set(xlim=(-outsize[0]/2., outsize[0]/2.), ylim=(-outsize[1]/2., outsize[1]/2.))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        
        if scale==None:
            scale = 1.
            
        self.scale = scale
        


    def set_data(self, data):
        self.data = data
        
    def set_trajectory(self, trajectory):
        self.trajectory = trajectory
        
    def set_sitdown_distance(self, sitdown_distance):
        self.sitdown_distance = sitdown_distance
        
    def set_popout(self, popout):
        self.popout = popout
        
    def set_outsize(self, outsize):
        self.outsize = outsize
        
    def set_scale(self, scale):
        self.scale = scale

    def ani_init(self):
        self.scatr.set_offsets(np.array([[0],[0]]).T)
        self.scatl.set_offsets(np.array([[0],[0]]).T)
        return self.scatr, self.scatl,

    def ani_update(self, i):
        x = self.data[0]
        y = self.data[1]
        z = self.data[2]
        x_vid, y_vid, z_vid = twisty(x, y, z, self.trajectory[i]) 
        
        x_s = x_vid * self.scale # coordinates in cm
        y_s = y_vid * self.scale
        z_s = z_vid * self.scale - self.popout
                
        eye_sep = 4. #half separation between eyes [cm]
        camera_r = self.sitdown_distance * np.array([0, 0, 1]) + eye_sep * np.array([1, 0, 0])
        camera_l = self.sitdown_distance * np.array([0, 0, 1]) - eye_sep * np.array([1, 0, 0])
        
        xr, yr, zr = twisty(x_s, y_s, z_s, camera_r)
        xl, yl, zl = twisty(x_s, y_s, z_s, camera_l)
        
        plot_data_r = np.stack([xr, yr]).T
        self.scatr.set_offsets(plot_data_r)
        plot_data_l = np.stack([xl, yl]).T
        self.scatl.set_offsets(plot_data_l)
        return self.scatr, self.scatl, 

    def animate(self):
        self.anim = animation.FuncAnimation(self.fig, self.ani_update, init_func=self.ani_init, frames=self.trajectory.shape[0], interval=50, blit=True)
        #plt.show()
        writer = animation.PillowWriter(fps=15,
                                        metadata=dict(artist='RS'),
                                        bitrate=1800)
        self.anim.save('3Danim.gif', writer=writer)






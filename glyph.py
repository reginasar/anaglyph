import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import sqeuclidean

def twisty(x_, y_, z_, camera_):
    Ve = np.array([x_, y_, z_])    
    A1 = np.arctan2(camera_[1], camera_[2])
    A2 = np.arcsin(camera_[0]/np.linalg.norm(camera_))
    R1 = np.array([[1,0,0],[0,np.cos(A1),-np.sin(A1)],[0,np.sin(A1),np.cos(A1)]])
    R2 = np.array([[np.cos(A2),0,-np.sin(A2)],[0,1,0],[np.sin(A2),0,np.cos(A2)]])    
    Vf = np.dot(np.dot(R2, R1), Ve)
    
    return Vf[0], Vf[1], Vf[2]
    

class particles:
    def __init__(self, data=None):
        if data is not None:
            try:
                data = np.array(data, dtype=np.float64)
            except ValueError:
                "data cannot be converted to ndarray type."
            
            if len(data.shape)==2:
                if data.shape[1]==3:
                    self.data = data 
                else:
                    raise ValueError("second dimension of data must be 3.")
            else:
                raise ValueError("data must have two dimensions.")
        else:
            raise ValueError("data must be provided.")

    def view(self, distance, camera=[1, 0, 0], plot_op="hist", save_name="view.jpg"):
        Vf = twisty(self.data[:, 0], self.data[:, 1], self.data[:, 2], camera)
        if plot_op=="hist":
            crispy, _,_,_ = stats.binned_statistic_2d(Vf[:, 0], Vf[:, 1], \
                Vf[:, 0], bins=30, statistic='count')
            plt.imshow(crispy.T)
            plt.savefig(save_name)
            #plt.show()
        elif plot_op=="scatter":
            plt.scatter(Vf[:, 0], Vf[:, 1], alpha=0.5)
            #plt.show()
            plt.savefig(save_name)
        else:
            raise ValueError('wrong plot_op, must be hist or scatter.')


    def view_3D(self, camera=[1., 0., 0.], distance=60., popout=10., out_size=[40., 25.], save_name='view.jpg'):
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

    def video_3D(self, trajectory=[0.,0.,0.], sitdown_distance=150.,\
                 popout=0., outsize=[32., 18.], scale=1., perspective=0.):
        if not isinstance(sitdown_distance, (int, float)):    
            raise TypeError("sitdown_distance type must be a scalar.")
        else:
            self.sitdown_distance = sitdown_distance

        if not isinstance(popout, (int, float)):    
            raise TypeError("popout type must be a scalar.")
        else:
            self.popout = popout

        if not isinstance(scale, (int, float)):    
            raise TypeError("scale type must be a scalar.") 
        else:
            self.scale = scale           

        if not isinstance(perspective, (int, float)):    
            raise TypeError("perspective type must be a scalar.") 
        else:
            self.perspective = perspective

        if len(outsize)==2:
            if not isinstance(outsize[0], (int, float)) or not isinstance(outsize[1], (int, float)):
                raise TypeError("outsize must be a list or array with two scalars.") 
        else: 
            raise TypeError("outsize must be a list or array with two scalars.") 

        try:
            trajectory = np.array(trajectory, dtype=np.float64)
        except ValueError:
            "trajectory cannot be converted to numpy array."

        if len(trajectory.shape)==2 and trajectory.shape[1]==3:
                self.trajectory = trajectory
        else:
            raise ValueError("trajectory must have dimensions [n, 3]")

        fig, ax = plt.subplots()
        self.scatw = ax.scatter([], [], c='white', alpha=0.7, s=1)
        self.scatr = ax.scatter([], [], c='red', alpha=0.7, s=1)
        self.scatl = ax.scatter([], [], c='cyan', alpha=0.7, s=1)
        ax.set_facecolor('k')
        ax.set(xlim=(-outsize[0]/2., outsize[0]/2.), ylim=(-outsize[1]/2., outsize[1]/2.))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

        self.anim = animation.FuncAnimation(fig, self.ani_update, init_func=self.ani_init, frames=self.trajectory.shape[0], interval=50, blit=True)
        #plt.show()
        writer = animation.PillowWriter(fps=15,
                                        metadata=dict(artist='RS'),
                                        bitrate=1800)
        self.anim.save('3Danim.gif', writer=writer)        


    def ani_init(self):
        self.scatw.set_offsets(np.array([[0],[0]]).T)
        self.scatr.set_offsets(np.array([[0],[0]]).T)
        self.scatl.set_offsets(np.array([[0],[0]]).T)
        return self.scatr, self.scatl,

    def ani_update(self, ii):
        x = self.data[:, 0]
        y = self.data[:, 1]
        z = self.data[:, 2]
        x_vid, y_vid, z_vid = twisty(x, y, z, self.trajectory[ii]) 
        
        x_s = x_vid * self.scale # coordinates in cm
        y_s = y_vid * self.scale
        z_s = z_vid * self.scale - self.popout
                
        eye_sep = 4. #half separation between eyes [cm]
        camera_r = self.sitdown_distance * np.array([0, 0, 1]) + eye_sep * np.array([1, 0, 0])
        camera_l = self.sitdown_distance * np.array([0, 0, 1]) - eye_sep * np.array([1, 0, 0])
        
        xr, yr, zr = twisty(x_s, y_s, z_s, camera_r)
        xl, yl, zl = twisty(x_s, y_s, z_s, camera_l)
        
        pp = self.perspective + self.popout
        xxr = xr * (pp - zr / pp)
        yyr = yr * (pp - zr / pp)
        xxl = xl * (pp - zl / pp)
        yyl = yl * (pp - zl / pp)

        dist_lr = np.array([sqeuclidean([xxr[jj], yyr[jj]], [xxl[jj], yyl[jj]]) for jj in range(xxr.size)])
        w_index = np.nonzero(dist_lr<0.1)[0]
        lr_index = np.nonzero(dist_lr>=0.1)[0]

        plot_data_w = np.stack([xxl[w_index], yyl[w_index]]).T
        self.scatw.set_offsets(plot_data_w)
        plot_data_r = np.stack([xxr[lr_index], yyr[lr_index]]).T
        self.scatr.set_offsets(plot_data_r)
        plot_data_l = np.stack([xxl[lr_index], yyl[lr_index]]).T
        self.scatl.set_offsets(plot_data_l)

        return self.scatr, self.scatl, self.scatw,




class course:
    def __init__(self, initial_pos=None):
        if initial_pos is not None:
            try:
                initial_pos = np.array(initial_pos, dtype=np.float64)
            except TypeError:
                "data cannot be converted to ndarray type."
            
            if len(initial_pos.shape)==1:
                if initial_pos.size==3:
                    self.initial_pos = initial_pos 
                else:
                    raise ValueError("initial position must be 1D with size 3.")
            elif all(initial_pos.shape==[1, 3]):
                self.initial_pos = initial_pos[0, :]
            else:
                raise ValueError("initial position must be 1D with size 3.")
        else:
            raise ValueError("initial position must be provided.")
            
        self.trajectory = np.array([np.copy(initial_pos)])

    def line(self, final_pos, steps=None):
        try:
            final_pos = np.array(final_pos, dtype=np.float64)
        except TypeError:
            "data cannot be converted to ndarray type."
            
        if len(final_pos.shape)==1:
            if final_pos.size!=3:
                raise ValueError("final position must be 1D with size 3.")
        else:
            raise ValueError("final position must be 1D with size 3.")

        if all(final_pos == self.initial_pos):
            raise ValueError("Initial and final positions are the same,"+\
                             "cannot define a line trajectory.")
            
        if steps is not None:
            if not isinstance(steps, int):    
                raise TypeError("steps type must be int.")
        else:
            steps = 30
            
        trajectory = (final_pos - self.initial_pos)
        trajectory = np.repeat([trajectory], steps, axis=0) * np.tile(np.linspace(0, 1, steps), (3, 1)).T
        trajectory += self.initial_pos
        self.trajectory = np.concatenate([self.trajectory, trajectory[1:,:]], axis=0)
        self.initial_pos = np.copy(final_pos)
            

    def circle(self, final_pos, steps=None):
        try:
            final_pos = np.array(final_pos, dtype=np.float64)
        except TypeError:
            "data cannot be converted to ndarray type."
            
        if len(final_pos.shape)==1:
            if final_pos.size!=3:
                raise ValueError("final position must be 1D with size 3.")
                
        if steps is not None:
            if not isinstance(steps, int):    
                raise TypeError("steps type must be int.")
        else:
            steps = 30
        
        if np.dot(self.initial_pos, final_pos)==1.:
            raise ValueError("Initial and final positions are in the same direction"+\
                             "and cannot define a plane for a circular trajectory.")
            
        theeta = np.arccos(np.clip(np.dot(self.initial_pos/np.linalg.norm(self.initial_pos),\
                                final_pos/np.linalg.norm(final_pos)), -1.0, 1.0))
        unit_traj = np.array((np.cos(np.linspace(0, theeta, steps)), \
                                  np.sin(np.linspace(0, theeta, steps)), np.zeros(steps)))
            
        r = np.linalg.norm(self.initial_pos)
        vf = r * final_pos / np.linalg.norm(final_pos)
        v090 = -np.dot(self.initial_pos, vf) / r**2 * self.initial_pos + vf
        v090 = v090 / np.linalg.norm(v090)
        vz = np.cross(self.initial_pos, v090)
        vz = vz / np.linalg.norm(vz)
        tbase = np.stack((self.initial_pos/r, v090, vz)).T
            
        trajectory = np.matmul(tbase, unit_traj).T * r
        self.trajectory = np.concatenate([self.trajectory, trajectory[1:,:]], axis=0)
        self.initial_pos = np.copy(final_pos)
        self.final_pos = None
    
    #def helicopter(self):



from pylab import *
import pandas as pd
import skimage
from skimage import io
from skimage import draw as dr
import os.path as osp
import os
import scipy.interpolate as sci
import sys
import pickle
plt.style.use('ggplot') # Make the graphs a bit prettier

color_list=[c['color'] for c in list(plt.rcParams['axes.prop_cycle'])]

def scale_dim(df,timescale,lengthscale):
    #time
    df['t']=df['frame']*timescale
    #length 
    for dim in ['x','y','z']:
        df[dim+'_scaled']=df[dim]/lengthscale
        
def compute_parameters(df):
    """This function computes different parameters: velocity, ... """
    r,c=df.shape
    
    #velocity components
    for dim in ['x','y','z']:
        a=np.empty(r)
        a[:]=np.nan
        df['v'+dim]=a
        groups=df.groupby(['traj'])
        for traj in df['traj'].unique():
            traj_group = groups.get_group(traj)
            components=(traj_group[dim+'_scaled'].shift(-1)-traj_group[dim+'_scaled'])/(traj_group['t'].shift(-1)-traj_group['t'])
            ind=traj_group.index.values
            df.loc[ind[1:],'v'+dim]=components[:-1].values
    #velocity modulus
    df['v']=sqrt(df['vx']**2+df['vy']**2+df['vz']**2)
    
def get_info(dirdata):
    filename=osp.join(dirdata,"info.txt")
    if osp.exists(filename):
        with open(filename) as f:
            info={'lengthscale':-1,'delta_t':-1,'columns':-1}
            for line in f:        
                if ('lengthscale' in line)==True:
                    if len(line.split())==3:
                        info['lengthscale']= float(line.split()[2])
                elif ('delta_t' in line)==True:
                    if len(line.split())==3:
                        info['delta_t'] = float(line.split()[2])
                elif ('columns' in line)==True:
                    if len(line.split())==3:
                        info['columns'] = line.split()[2].split(',')
    else: 
        print "ERROR: info.txt doesn't exist or is not at the right place"
    return info

def plot_frame_cells(df,groups,frame,data_dir,plot_traj=False,z_labeling=False,filtered_tracks=None,hide_labels=False):
    """ Print the tracked pictures with updated (=relinked) tracks"""
    print '\rplotting frame '+str(frame),
    sys.stdout.flush()
    if filtered_tracks is None:
        track_dir=osp.join(data_dir,'traj')
    else:
        track_dir=osp.join(data_dir,'traj_subset')
    if osp.isdir(track_dir)==False:
        os.mkdir(track_dir)
    group=groups.get_group(frame).reset_index(drop=True)
    r,c=group.shape
    if plot_traj:
        track_groups=df.groupby(['traj'])
    #import image
    filename=osp.join(data_dir,'raw/max_intensity_%04d.png'%int(frame))
    im = io.imread(filename)
    n,m,d = im.shape
    fig=figure(frameon=False)
    fig.set_size_inches(m/300,n/300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im,aspect='auto',origin='lower')
    xmin, ymin, xmax, ymax=ax.axis('off')
    if filtered_tracks is None:
        for i in range(0,r):
            #write label
            x=group.loc[i,'x']
            y=group.loc[i,'y']
            track=int(group.loc[i,'traj'])
            s='%d'%(track)
            if hide_labels is False:
                ax.text(x,y,s,fontsize=5,color='w')
            if plot_traj:
                #plot trajectory
                traj=get_obj_traj(track_groups,track,max_frame=frame)
                traj_length,c=traj.shape
                if traj_length>1:
                    if z_labeling:
                        #create z color map
                    else:
                        ax.plot(traj['x'],traj['y'],ls='-',color=color_list[track%7])
                        ax.axis([xmin, ymin, xmax, ymax])

    else:
        for i,track in enumerate(filtered_tracks):
            if (group['traj']==track).any():
                #write label
                x=group[group['traj']==track]['x'].values[0]
                y=group[group['traj']==track]['y'].values[0]
                s='%d'%track
                if hide_labels is False:
                    ax.text(x,y,s,fontsize=5,color='w')
                if plot_traj:
                    #plot trajectory
                    traj=get_obj_traj(track_groups,track,max_frame=frame)
                    traj_length,c=traj.shape
                    if traj_length>1:
                        ax.plot(traj['x'],traj['y'],ls='-',color=color_list[i%7])
                        ax.axis([xmin, ymin, xmax, ymax])
    filename=osp.join(track_dir,'traj_%04d.png'%int(frame))
    fig.savefig(filename, dpi=300)
    close('all')
                      
def get_obj_traj(track_groups,track,max_frame=None):
    '''gets the trajectory of an object. track_groups is the output of a groupby(['relabel'])'''
    group=track_groups.get_group(track)
    trajectory=group[['frame','t','x','y','z','v']].copy()
    if max_frame is not None:
        trajectory=trajectory[trajectory['frame']<=max_frame]
    return trajectory.reset_index(drop=True)

def plot_frame_vfield(df,groups,frame,data_dir,interpolate=False,filtered_tracks=False,grid_step=10):
    """ Print the tracked pictures with updated (=relinked) tracks"""
    print '\rplotting frame '+str(frame),
    sys.stdout.flush()
    if filtered_tracks:
        track_dir=osp.join(data_dir,'vfield_subset')
    else:
        track_dir=osp.join(data_dir,'vfield')
        
    if osp.isdir(track_dir)==False:
        os.mkdir(track_dir)
    group=groups.get_group(frame).reset_index(drop=True)
    r,c=group.shape
    #import image
    filename=osp.join(data_dir,'raw/max_intensity_%04d.png'%int(frame))
    im = io.imread(filename)
    n,m,d = im.shape
    fig=figure(frameon=False)
    fig.set_size_inches(m/300,n/300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im,aspect='auto',origin='lower')
    xmin, ymin, xmax, ymax=ax.axis('off')
    if interpolate:
        grid_x, grid_y = np.mgrid[0:m:m/grid_step, 0:n:n/grid_step]
        grid=[]
        for p in ['vx','vy','vz']:
            points=df[['x','y']].values
            values=df[p].values
            grid.append(sci.griddata(points, values, (grid_x, grid_y), method='linear'))
        quiver(grid_x,grid_y,grid[0],grid[1],grid[2])
        ax.axis([xmin, ymin, xmax, ymax])
    else:
        Q=quiver(group['x'],group['y'],group['vx'],group['vy'],group['vz'],units='x',cmap='plasma')
        cbaxes = fig.add_axes([0.4, 0.935, 0.025, 0.05])
        cbar = fig.colorbar(Q,cax = cbaxes,label='$v_z\ (\mu m.min^{-1})$')
        cbaxes.tick_params(labelsize=5,color='w')
        cbaxes.yaxis.label.set_color('white')
        ax.axis([xmin, ymin, xmax, ymax])
    filename=osp.join(track_dir,'vfield_%04d.png'%int(frame))
    fig.savefig(filename, dpi=600)
    close('all')


def filter_by_traj_len(df,min_traj_len=1,max_traj_len=None):
    df2=pd.DataFrame()
    if max_traj_len is None: #assign the longest possible track
        max_traj_len=df['frame'].max()-df['frame'].min()+1
    tracks=df.groupby('traj')
    for t in df['traj'].unique():
        track=tracks.get_group(t)
        if track.shape[0]>=min_traj_len and track.shape[0]<=max_traj_len:
            df2=pd.concat([df2,track])
    return df2


def run(data_dir,refresh=False,plot_frame=True,plot_data=True,plot_modified_tracks=False,plot_all_traj=False):
    #import
    pickle_fn=osp.join(data_dir,"data_base.p")
    if osp.exists(pickle_fn)==False or refresh:
#         data=loadtxt(osp.join(data_dir,'test_data.txt'))
        data=loadtxt(osp.join(data_dir,'table.txt'))
        info=get_info(data_dir)
        for inf in ['lengthscale','delta_t','columns']:
            if info[inf]==-1:
                print "WARNING: "+inf+" not provided in info.txt"
        lengthscale=info["lengthscale"];timescale=info["delta_t"];columns=info["columns"]
        df=pd.DataFrame(data[:,1:],columns=columns) 
        #scale data
        scale_dim(df,timescale,lengthscale)
        compute_parameters(df)
        #update pickle
        pickle.dump(df, open( osp.join(data_dir,"data_base.p"), "wb" ) )
    else:
        df=pickle.load( open( pickle_fn, "rb" ))
    
    return df


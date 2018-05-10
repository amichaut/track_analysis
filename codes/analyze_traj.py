from pylab import *
import pandas as pd
import skimage
from skimage import io
from skimage import draw as dr
from skimage.viewer.canvastools import RectangleTool, LineTool
from skimage.viewer import ImageViewer
import os.path as osp
import os
import scipy.interpolate as sci
import sys
import pickle
import multiprocessing
from joblib import Parallel, delayed
import seaborn as sns
import datetime
from mpl_toolkits.mplot3d import axes3d
from lmfit import Parameters, Model


color_list=[c['color'] for c in list(plt.rcParams['axes.prop_cycle'])]

welcome_message="""\n\n WELCOME TO TRACK_ANALYSIS \n Developped and maintained by Arthur Michaut: arthur.michaut@gmail.com \n Last release: 05-09-2018\n\n\n     _''_\n    / o  \\\n  <       |\n    \\    /__\n    /       \\-----\n    /    \\    \\   \\__\n    |     \\_____\\  __>\n     \\--       ___/  \n        \\     /\n         || ||\n         /\\ /\\\n\n"""
usage_message="""Usage: \n- plot cells analysis using cell_analysis(data_dir,refresh,parallelize,plot_traj,hide_labels,no_bkg,linewidth,plot3D,min_traj_len,frame_subset) \t data_dir: data directory, refresh (default False) to refresh the table values, parallelize (default False) to run analyses in parallel, 
plot_traj (default true) to print the cell trajectories, hide_labels (default True) to hide the cell label, no_bkg (default False) to remove the image background, linewidth being the trajectories width (default=1.0), plot3D (default:False), frame_subset: to plot only on a subset of frames [first,last]\n
- plot maps using map_analysis(data_dir,refresh,parallelize,x_grid_size,no_bkg,z0,dimensions,axis_on,plot_on_mean,black_arrows) \t data_dir: data directory, refresh (default False) to refresh the table values, parallelize (default False) to run analyses in parallel, 
x_grid_size: number of columns in the grid (default 10), no_bkg (default False) to remove the image background, z0: altitude of the z_flow surface (default None => center of z axis), dimensions ([row,column] default None) to give the image dimension in case of no_bkg, axis_on: display axes along maps (default False),plot_on_mean: plot vfield on mean_vel map (default=True),black_arrows: don't use vz to color code vfield arrows (default=True) \n
- plot average ROIs using avg_ROIs(data_dir,frame_subset=None,selection_frame=None,ROI_list=None,plot_on_map=True,plot_section=True,cumulative_plot=True,avg_plot=True) \t data_dir: data directory, frame_subset is a list [first,last], default None: open interactive choice \n
- plot XY flow through a vertical surface defined by a XY line using XY_flow(data_dir,window_size=None,refresh=False,line=None,orientation=None,frame_subset=None,selection_frame=None,z_depth=None) \t data_dir: data directory, frame_subset is a list [first,last], default None: open interactive choice, window_size = rolling average window in um, default None => interactive choice"""


print welcome_message
print usage_message
print 'WARNING parallelize is not available!'

global map_dic

#################################################################
###########   PREPARE METHODS   #################################
#################################################################

def get_cmap_color(value, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(value))

def scale_dim(df,dimensions=['x','y','z'],timescale=1.,lengthscale=1.):
    #time
    df['t']=df['frame']*timescale
    #length lengthscale px/um
    for dim in dimensions:
        df[dim+'_scaled']=df[dim]/lengthscale
        
def compute_parameters(df,dimensions=['x','y','z']):
    """This function computes different parameters: velocity, ... """
    r,c=df.shape
    
    #velocity components
    for dim in dimensions:
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
    sum_=0
    for dim in dimensions:
        sum_+=df['v'+dim]**2
    df['v']=sqrt(sum_)

    if 'z' in dimensions:
        #relative z: centered around mean
        df['z_rel']=df['z_scaled']-df['z_scaled'].mean()
    
def get_info(data_dir):
    """info.txt gives the lengthscale in um/px, the frame intervalle delta_t in min and the column names of the table"""
    filename=osp.join(data_dir,"info.txt")
    if osp.exists(filename):
        with open(filename) as f:
            info={'lengthscale':-1,'delta_t':-1,'columns':-1,'dimensions':-1}
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
                elif ('dimensions' in line)==True:
                    if len(line.split())==3:
                        dimensions = line.split()[2].split(',')
                        info['dimensions'] = [int(d) for d in dimensions]

    else: 
        print "ERROR: info.txt doesn't exist or is not at the right place"
    return info

def get_vlim(data_dir):
    """info.txt gives the lengthscale in um/px, the frame intervalle delta_t in min and the column names of the table"""
    filename=osp.join(data_dir,"vlim.txt")
    if osp.exists(filename):
        with open(filename) as f:
            vlim_dict={'vfield':None,'div':None,'mean_vel':None,'vx':None,'vy':None,'vz':None,'z_flow':None}
            for line in f:
                for key in vlim_dict.keys():
                    if (key in line)==True:
                        if len(line.split())==3:
                            vlim = line.split()[2].split(',')
                            vlim_dict[key] = [float(d) for d in vlim]
    else: 
        vlim_dict=None
    return vlim_dict

def get_data(data_dir,refresh=False,correct_shift=False):
    #import
    pickle_fn=osp.join(data_dir,"data_base.p")
    if osp.exists(pickle_fn)==False or refresh:
        #data=loadtxt(osp.join(data_dir,'test_data.txt'))
        data=loadtxt(osp.join(data_dir,'table.txt'))
        info=get_info(data_dir)
        for inf in ['lengthscale','delta_t','columns']:
            if info[inf]==-1:
                print "WARNING: "+inf+" not provided in info.txt"
        ## WARNING lengthscale is defined in the whole code in px/um so lengthscale=1/lengthscale
        lengthscale=1./info["lengthscale"];timescale=info["delta_t"];columns=info["columns"]
        df=pd.DataFrame(data[:,1:],columns=columns)
        df=df.loc[:, (df != 0).any(axis=0)] #remove columns filled with zeros
        #scale data
        dimensions=['x','y','z'] if 'z' in df.columns else ['x','y']
        dim=len(dimensions)
        scale_dim(df,dimensions,timescale,lengthscale)
        compute_parameters(df,dimensions)
        if correct_shift:
            shift=get_shift(data_dir,lengthscale,timescale)
        #update pickle
        pickle.dump([df,lengthscale,timescale,columns,dim], open( osp.join(data_dir,"data_base.p"), "wb" ) )
    else:
        [df,lengthscale,timescale,columns,dim]=pickle.load( open( pickle_fn, "rb" ))
    
    return df,lengthscale,timescale,columns,dim

def get_obj_traj(track_groups,track,max_frame=None,dim=3,shift=None,lengthscale=1.):
    '''gets the trajectory of an object. track_groups is the output of a groupby(['relabel'])'''
    group=track_groups.get_group(track)
    cols=['frame','t','x','y','z','z_scaled','z_rel','v'] if dim==3 else ['frame','t','x','y','v']
    trajectory=group[cols].copy()
    if max_frame is not None:
        trajectory=trajectory[trajectory['frame']<=max_frame]
    if shift is not None:
        df_comp=pd.merge(trajectory,shift,on=['t','frame'],how='inner') #shift only on subset of common frames
        r_orig = ['x','y','z'] if dim==3 else ['x','y']
        r_shift = ['x0','y0','z0'] if dim==3 else ['x0','y0']
        r_shifted = ['x_shifted','y_shifted','z_shifted'] if dim==3 else ['x_shifted','y_shifted']
        for i,r in enumerate(r_shift):
            if r in shift.columns:
                df_comp[r_shifted[i]]=df_comp[r_orig[i]]-df_comp[r]+df_comp.loc[0,r] #shifted(t)=original(t)-shift(t)+shift(t=0) so the first point of the trajectory starts at the same point
            else:
                df_comp[r_shifted[i]]=df_comp[r_orig[i]]
        trajectory=df_comp[['frame','t']+r_shifted+cols[2+len(r_shifted):]].copy()
        trajectory.columns=cols
        # #recalculate v TO BE CHECKED
        # dimensions=['x','y','z'] if dim==3 else ['x','y']
        # for dimension in ['x','y']:
        #     df_comp[dimension+'_scaled']=df_comp[dimension+'_shifted']/lengthscale
        #     df_comp.loc[df_comp.index[1:],'v'+dimension]=(df_comp[dimension+'_scaled'].shift(-1)-df_comp[dimension+'_scaled'])/(df_comp['t'].shift(-1)-df_comp['t'])
        # sum_=0
        # for dimension in dimensions:
        #     sum_+=df_comp['v'+dim]**2
        # trajectory['v']=sqrt(sum_)
    return trajectory.reset_index(drop=True)

def get_shift(data_dir,timescale,lengthscale):
    """Gets the trajectory of the shifting reference. Should be a cvs table with coordinates in px and frame definition: first=0 (can start at any moment though) """
    filename=osp.join(data_dir,"shift.csv")
    if osp.exists(filename):
        df=pd.read_csv(filename)
        df['t']=df['frame']*timescale
        for r in ['x0','y0','z0']:
            if r in df.columns:
                df[r]=df[r]*lengthscale
        return df

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

def get_background(df,data_dir,frame,no_bkg=False,image_dir=None,orig=None,axis_on=False):
    """Get image background or create white backgound if no_bkg"""
    if orig is None:
        orig = 'lower' if image_dir is None else 'upper' #trick to plot for the first time only inverting Yaxis: not very elegant...
    if image_dir is None:
        image_dir = osp.join(data_dir,'raw')
    if not osp.exists(image_dir): #check if images exist 
       no_bkg=True
    if no_bkg:
        #get approximative image size
        m=int(df['x'].max());n=int(df['y'].max())
        im = ones((n,m,3)) #create white background ==> not ideal, it would be better not to use imshow and to modify axes rendering
    else:
        filename=osp.join(image_dir,'%04d.png'%int(frame))
        im = io.imread(filename)
        n=im.shape[0]; m=im.shape[1]
    fig=figure(frameon=False)
    fig.set_size_inches(m/300.,n/300.)
    if axis_on:
        ax = gca()
    else: 
        ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im,aspect='auto',origin=orig)
    if axis_on:
        xmin,xmax,ymin,ymax=ax.axis('on')
    else: 
        xmin,xmax,ymin,ymax=ax.axis('off')
    return fig,ax,xmin,ymin,xmax,ymax,no_bkg

def make_grid(x_grid_size,data_dir,dimensions=None):
    """make a meshgrid. The boundaries can be passed by dimensions as [xmin,xmax,ymin,ymax] or using the raw image dimensions. x_grid_size is the number of cells in the grid along the x axis.
    It returns two grids: the node_grid with the positions of the nodes of each cells, and the center_grid with the position of the center of each cell"""
    if dimensions is None:
        raw_dir = osp.join(data_dir,'raw')
        if not osp.exists(raw_dir):
            print """ERROR: the grid can't be created, no dimensions are available"""
            return
        elif len(os.listdir(raw_dir))==0:
            print """ERROR: the grid can't be created, no dimensions are available"""
            return
        else:
            im = io.imread(osp.join(raw_dir,os.listdir(raw_dir)[0]))
            ymax = im.shape[0]; xmax = im.shape[1]
            xmin=0;ymin=0
    else:
        [xmin,xmax,ymin,ymax] = dimensions

    step=float(xmax-xmin)/x_grid_size
    y_node_array=arange(ymin,ymax+step,step)
    if y_node_array[-1]>ymax: #to ensure grid isn't longer than the actual map
        y_node_array=y_node_array[:-1]
    y_center_array=arange(ymin+step/2.,ymax,step)
    if y_center_array[-1]>y_node_array[-1]: #to ensure grid isn't longer than the actual map
        y_center_array=y_center_array[:-1]
    node_grid=meshgrid(arange(xmin,xmax+step,step),y_node_array)
    center_grid=meshgrid(arange(xmin+step/2.,xmax,step),y_center_array)
    return node_grid,center_grid

def compute_vfield(df,frame,groups,data_dir,grids=None,dim=3):
    print '\rcomputing velocity field '+str(frame),
    sys.stdout.flush()

    plot_dir=osp.join(data_dir,'vfield')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    coord_list=['vx','vy','vz']
        
    group=groups.get_group(frame).reset_index(drop=True)

    if grids is not None:
        node_grid,center_grid=grids
        X,Y=node_grid
        x,y=center_grid
        v_field=[zeros((x.shape[0],x.shape[1])) for _ in range(dim)]
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                ind=((group['x']>=X[i,j]) & (group['x']<X[i,j+1]) & (group['y']>=Y[i,j]) & (group['y']<Y[i+1,j]))
                for k in range(dim):
                    v_field[k][i,j]=group[ind][coord_list[k]].mean()
    else:
        v_field=[group[coord_list[k]].values for k in range(dim)]
        x=group['x'].values;y=group['y'].values

    #save data in pickle
    data=[x,y]+v_field
    save_map_data(plot_dir,data,frame)

    return data

def compute_div(df,frame,groups,data_dir,grids,dim=3,lengthscale=1):
    print '\rcomputing divergence field '+str(frame),
    sys.stdout.flush()

    plot_dir=osp.join(data_dir,'div')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    #get avg_vfield
    data=get_map_data(osp.join(data_dir,'vfield'),frame)
    avg_vfield=data[2:]

    #compute div
    node_grid,center_grid=grids
    x,y=center_grid
    div = zeros((x.shape[0],x.shape[1]))
    vx=avg_vfield[0];vy=avg_vfield[1]
    for i in range(1,x.shape[0]-1):
        for j in range(1,x.shape[1]-1):
            dy=y[i,j]-y[i-1,j]/lengthscale; dx=x[i,j]-x[i,j-1]/lengthscale
            Dvx=(vx[i,j+1]-vx[i,j-1])/(2*dx);Dvy=(vy[i+1,j]-vy[i-1,j])/(2*dy)
            div[i,j]=Dvx+Dvy

    #save data in pickle
    data=(x,y,div)
    save_map_data(plot_dir,data,frame)

    return data

def compute_mean_vel(df,frame,groups,data_dir,grids,dim=3):
    """Uses the vfield data to compute the modulus of the vfield on center_grid (x,y)"""
    print '\rcomputing mean velocity field '+str(frame),
    sys.stdout.flush()

    plot_dir=osp.join(data_dir,'mean_vel')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    #get avg_vfield
    node_grid,center_grid=grids
    X,Y=node_grid
    data=get_map_data(osp.join(data_dir,'vfield'),frame)
    avg_vfield=data[2:]

    #compute avg
    V=0
    for k in range(dim):
        V+=avg_vfield[k]**2
    mean_vel=sqrt(V)

    #save data in pickle
    data=(X,Y,mean_vel)
    save_map_data(plot_dir,data,frame)

    return data

def compute_z_flow(df,frame,groups,data_dir,grids,z0,timescale):
    print '\rcomputing z_flow field '+str(frame),
    sys.stdout.flush()
    #Make sure these are 3D data
    if 'z' not in df.columns:
        print "Not a 3D set of data"
        return

    plot_dir=osp.join(data_dir,'z_flow')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    group=groups.get_group(frame).reset_index(drop=True)

    df_layer=group[abs(group['vz']*timescale)>=abs(z0-group['z_rel'])] #layer of cells crossing the surface
    df_ascending=df_layer[((df_layer['vz']>=0) & (df_layer['z_rel']<=z0))] #ascending cells below the surface
    df_descending=df_layer[((df_layer['vz']<=0) & (df_layer['z_rel']>=z0))] #descending cells above the surface
    
    #calculate the intersection coordinates (x0,y0) of the vector and the surface (calculate homothety coefficient alpha)
    for df_ in [df_ascending,df_descending]:
        df_.loc[:,'alpha']=abs(z0-df_['z_rel'])/(df_['vz']*timescale)
        df_.loc[:,'x0']=df_['x']+df_['alpha']*df_['vx']*timescale
        df_.loc[:,'y0']=df_['y']+df_['alpha']*df_['vy']*timescale
    
    node_grid,center_grid=grids   
    X,Y=node_grid
    x,y=center_grid
    flow = zeros((x.shape[0],x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ind_asc=((df_ascending['x0']>=X[i,j]) & (df_ascending['x0']<X[i,j+1]) & (df_ascending['y0']>=Y[i,j]) & (df_ascending['y0']<Y[i+1,j]))
            ind_des=((df_descending['x0']>=X[i,j]) & (df_descending['x0']<X[i,j+1]) & (df_descending['y0']>=Y[i,j]) & (df_descending['y0']<Y[i+1,j]))
            flow[i,j]=(df_ascending[ind_asc].shape[0]-df_descending[ind_des].shape[0])/timescale

    #save data in pickle
    data=(x,y,flow)
    save_map_data(plot_dir,data,frame)

    return data

def get_coord(extents):
    """Small function used by skimage viewer"""
    global viewer,coord_list
    coord_list.append(extents)

def get_ROI(image_dir,frame,tool=RectangleTool):
    """Interactive function used to get ROIs coordinates of a given image"""
    global viewer,coord_list

    if type(frame) not in [int,float]:
        print "ERROR the given frame is not a number"
        return -1

    filename=osp.join(image_dir,'%04d.png'%int(frame))
    if osp.exists(filename) is False:
        print "ERROR the image does not exist for the given frame"
        return -1
    im = io.imread(filename)

    selecting=True
    while selecting:
        viewer = ImageViewer(im)
        coord_list = []
        rect_tool = tool(viewer, on_enter=get_coord) 
        print "Draw your selections, press ENTER to validate one and close the window when you are finished"
        viewer.show()
        print 'You have selected %d ROIs'%len(coord_list)
        finished=raw_input('Is the selection correct? [y]/n: ')
        if finished!='n':
            selecting=False
    return coord_list

def filter_by_ROI(df,data_dir):
    '''Function used to choose subsets of cells given there position at a certain frame'''
    tracks=df.groupby('traj')
    subdf_list=[]
    frame=input('Give the frame number at which you want to make your selection: ')
    image_dir=osp.join(data_dir,'raw')
    ROI_list=get_ROI(image_dir,frame,tool=RectangleTool)
    for ROI in ROI_list:
        xmin,xmax,ymin,ymax=ROI
        ind=((df['frame']==frame) & (df['x']>=xmin) & (df['x']<=xmax) & (df['y']>=ymin) & (df['y']<=ymax))
        subdf_frame=df[ind] #the subset at the given frame

        #get the subset in the whole dataset
        subdf=pd.DataFrame()
        for t in subdf_frame['traj'].unique():
            track=tracks.get_group(t)
            subdf=pd.concat([subdf,track])
        subdf_list.append(subdf)

    return subdf_list

def save_map_data(plot_dir,data,frame):
    datab_dir=osp.join(plot_dir,'data')
    if osp.isdir(datab_dir)==False:
        os.mkdir(datab_dir)
    pickle_fn=osp.join(datab_dir,'%04d.p'%frame)
    pickle.dump(data,open(pickle_fn,"wb"))

def get_map_data(plot_dir,frame):
    pickle_fn=osp.join(plot_dir,'data','%04d.p'%frame)
    if osp.exists(pickle_fn):
        data=pickle.load( open( pickle_fn, "rb" ))
    else:
        print 'ERROR: database does not exist'
    return data

def compute_vlim(df,compute_func,groups,data_dir,grids,data_coord,dim=3,show_hist=False,get_former_data=False,plot_dir=None,**kwargs):
    # compute the max and min over all frames of a map. Compute maps for all frames
    vmin=np.nan;vmax=np.nan #boudaries of colorbar
    for i,frame in enumerate(df['frame'].unique()):
        data=get_map_data(plot_dir,frame) if get_former_data else compute_func(df,frame,groups,data_dir,grids,dim,**kwargs)
        data=data[data_coord]
        if show_hist:
            if i==0:
                r,c=data.shape
                data_hist=data.reshape(r*c,1)
            else:
                r,c=data.shape
                data_hist=vstack((data_hist,data.reshape(r*c,1)))
        if isnan(nanmin(data))==False:
            if isnan(vmin): #if no value computed yet
                vmin=nanmin(data)
            else:
                vmin=nanmin(data) if nanmin(data)<vmin else vmin
        if isnan(nanmax(data))==False:
            if isnan(vmax): #if no value computed yet
                vmax=nanmax(data)
            else:
                vmax=nanmax(data) if nanmax(data)>vmax else vmax

    if show_hist:
        close('all')
        # ion()
        s=pd.Series(data_hist[:,0])
        s.plot.hist()
        show()
        vlim=raw_input('If you want to manually set the colorbar boundaries, enter the values (separated by a coma). Otherwise, press Enter: ')
        vlim=vlim.split(',')
        if len(vlim)==2:
            vmin=float(vlim[0]); vmax=float(vlim[1])
        close()
        # ioff()
    return [vmin,vmax]

def get_subblock_data(X,Y,data,ROI):
    square_ROI=False
    xmin,xmax,ymin,ymax=ROI
    ind=((X>=xmin) & (X<=xmax) & (Y>=ymin) & (Y<=ymax))
    x,y=meshgrid(np.unique(X[ind]),np.unique(Y[ind])) #sublock (x,y)
    if x.shape[0]==x.shape[1]:
        square_ROI=True
    dat=data[ind].reshape(*x.shape)
    return [x,y,dat,square_ROI]

def select_map_ROI(data_dir,map_kind,frame,ROI_list=None):
    """Get data from a map in rectangular ROIs. If ROIs not given, manually drawn with get_ROI."""
    image_dir1=osp.join(data_dir,'raw')

    map_kind_ = 'vfield' if map_kind in ['vx','vy','vz'] else map_kind
    if osp.isdir(image_dir1)==False:
        image_dir1=osp.join(data_dir,map_kind_)

    image_dir=osp.join(data_dir,map_kind_)
    not_found=True
    while not_found:
        if ROI_list is None:
            ROI_list=get_ROI(image_dir1,frame,tool=RectangleTool)
        data=get_map_data(image_dir,frame)
        grids=pickle.load( open(osp.join(image_dir,'data','grids.p'), "rb" ))
        node_grid,center_grid=grids
        x_,y_=center_grid
        if map_kind=='vy':
            data=data[3]
        elif map_kind=='vz':
            data=data[4]            
        else:
            data=data[2]
        ROI_data_list=[]
        square_ROI=False
        for ROI in ROI_list:
            x,y,dat,square_ROI=get_subblock_data(x_,y_,data,ROI)
            ROI_data_list.append([x,y,dat])

        if square_ROI:
            re_select=raw_input("WARNING: there is a square ROI. Do you want to select again? [y]/n ")
            if re_select!='n':
                not_found=True
                ROI_list=None #reset ROI to call get_ROI again
            else:
                not_found=False
        else:
            not_found=False

    return [ROI_data_list,ROI_list]

def avg_ROI_major_axis(ROI_data):
    """average data along the major axis of the ROI"""

    #get principal axis
    r,c=ROI_data[0].shape
    if r==c:
        print 'ERROR: square ROI'
        return
    elif r>c:
        major_ax,minor_ax=0,1
    elif r<c:
        major_ax,minor_ax=1,0

    #average
    avg_data=[]
    for d in ROI_data:
        avg_data.append(np.nanmean(d,axis=minor_ax))

    return {'data':avg_data,'major_ax':major_ax}

def compute_XY_flow(df,data_dir,line,orientation,frame,groups,window_size=None,timescale=1.,lengthscale=1.,z_depth=None,reg_data_bin=1.,plot_steps=False):
    """Compute the flow along the surface defined by a XY line. The first end of the line is x=0 for the plot. The cells crossing the line along the orientation (from first point to second) are counted
    as positive cells, the cells in the other are counted as negative. The count is integrated along a moving window along the line"""

    print '\rcomputing XY flow '+str(frame),
    sys.stdout.flush()

    group=groups.get_group(frame).reset_index(drop=True)

    #find intersection (beware vx is in um and coordinates in px) using formula from https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    x1=line[0][0];y1=line[0][1];x2=line[1][0];y2=line[1][1]
    Ox1=orientation[0][0];Oy1=orientation[0][1];Ox2=orientation[1][0];Oy2=orientation[1][1]

    for dim in ['x','y']:
        group['displ_'+dim]=group['v'+dim]*timescale*lengthscale 
        group[dim+'_prev']=group[dim]-group['displ_'+dim] #get previous timestep position
    group = group[(np.isfinite(group['x_prev'])|np.isfinite(group['y_prev']))] #remove nan

    #calculate coordinates of point (I) of intersection between line and displacement vector
    A=x1*y2-y1*x2; B=x1-x2; C=y1-y2
    group['intersec_denom']=B*group['displ_y'] - C*group['displ_x']
    group['intersec_x']=(A*group['displ_x']-B*(group['x']*group['y_prev']-group['y']*group['x_prev']))/group['intersec_denom']#intesection x_coord
    group['intersec_y']=(A*group['displ_y']-C*(group['x']*group['y_prev']-group['y']*group['x_prev']))/group['intersec_denom']#intesection y_coord
    if plot_steps:
        print "all intercepts"
        fig,ax,xmin,ymin,xmax,ymax,no_bkg=get_background(df,data_dir,frame)
        group.plot.scatter(x='intersec_x',y='intersec_y',ax=ax)
        ax.plot([x1,x2],[y1,y2])
        show(fig)


    # #check if I on line
    ind=((group['intersec_x']>=min(x1,x2)) & (group['intersec_x']<=max(x1,x2)) & (group['intersec_y']>=min(y1,y2))&(group['intersec_y']<=max(y1,y2)))
    group=group[ind]
    if plot_steps:
        print "on line intercepts"
        fig,ax,xmin,ymin,xmax,ymax,no_bkg=get_background(df,data_dir,frame)
        group.plot.scatter(x='intersec_x',y='intersec_y',ax=ax)
        ax.plot([x1,x2],[y1,y2])
        show(fig)

    #check if crossing line 
    group['intersec_vecx']=group['intersec_x']-group['x_prev']#vector I-x between intersection and previous point
    group['intersec_vecy']=group['intersec_y']-group['y_prev']
    group['converging']=group['intersec_vecx']*group['displ_x']+group['intersec_vecy']*group['displ_y'] #scalar product between (I-x) and displacement vectors.
    if plot_steps:
        print "all I-x"
        fig,ax,xmin,ymin,xmax,ymax,no_bkg=get_background(df,data_dir,frame)
        ax.quiver(group['x_prev'].values,group['y_prev'].values,group['intersec_vecx'].values,group['intersec_vecy'].values)
        ax.plot([x1,x2],[y1,y2])
        show(fig)

    group=group[group['converging']>0] #if >0 converging towards line. If not discard because crossing is impossible
    if plot_steps:
        print "converging I-x"
        fig,ax,xmin,ymin,xmax,ymax,no_bkg=get_background(df,data_dir,frame)
        ax.quiver(group['x_prev'].values,group['y_prev'].values,group['intersec_vecx'].values,group['intersec_vecy'].values)
        ax.plot([x1,x2],[y1,y2])
        show(fig)

    group['crossing']=(group['displ_x']**2+group['displ_y']**2)/(group['intersec_vecx']**2+group['intersec_vecy']**2) #|displ|^2/|I-x|^2
    group=group[group['crossing']>=1] #if displacement > distance to surface

    #compute orientation
    group['orientation']=group['displ_x']*(Ox2-Ox1)+group['displ_y']*(Oy2-Oy1) #scalar product between O and displ
    group['orientation']=group['orientation'].apply(lambda x:1 if x>0 else -1)

    #compute distance to line end
    group['line_abscissa']=np.sqrt((group['intersec_x']-x1)**2+(group['intersec_y']-y1)**2)

    #return data
    data=group[['line_abscissa','orientation']].values
    data[:,0]/=lengthscale #rescale data in um
    abs_length = np.sqrt((line[1][0]-line[0][0])**2+(line[1][1]-line[0][1])**2)/lengthscale
    axis_data=arange(0,abs_length,reg_data_bin)
    data=regularized_rolling_mean(data,axis_data,window_size,reg_data_bin)
    if window_size is not None:
        window_area=window_size*z_depth if z_depth is not None else window_size
        data[:,1]/=(timescale*window_area)
    else:
        data[:,1]/=timescale
    return data

def regularized_rolling_mean(data,axis_data,window_size=None,reg_data_bin=1.):
    """Compute the rolling sum of a discrete set of data along a regularized axis with a step of reg_data_bin"""
    if window_size is None:
        return data

    #regularize data (fill with zeros missing data)
    reg_data=array([axis_data,np.zeros(axis_data.shape[0])]).T
    for i in range(data.shape[0]):
        ind=((reg_data[:,0]>=data[i,0]) & (reg_data[:,0]<data[i,0]+reg_data_bin)) #find index in new
        reg_data[ind,1]+=data[i,1]

    #rolling
    reg_data=pd.DataFrame(reg_data,columns=list('xy'))
    reg_data['y']=reg_data['y'].rolling(window_size,min_periods=1).mean()

    return reg_data.values

def select_frame_list(df,frame_subset=None):
    """Make a list of frame list with interactive input if needed. frame_subset can be a number a list [first,last_included] or None for interactive"""
    if frame_subset is None:
        typing=True
        while typing:
            try:
                frame_subset=input("Give the frame subset you want to analyze as [first,last] or unique_frame, if you want them all just press ENTER: ")
                if type(frame_subset) is list:
                    if frame_subset[0] in df['frame'].unique() and frame_subset[1] in df['frame'].unique():
                        typing=False
                    else:
                        print "WARNING: the subset is invalid, please try again"
                elif type(frame_subset) is int:
                    if frame_subset in df['frame'].unique():
                        typing=False
                    else:
                        print "WARNING: the subset is invalid, please try again"
                else:
                    print "WARNING: the subset is invalid, please try again"
            except:
                typing=False
                frame_subset=None
 
    elif type(frame_subset) is list:
        if not frame_subset[0] in df['frame'].unique() or not frame_subset[1] in df['frame'].unique():
            return "WARNING: the subset is invalid, please try again"
    elif type(frame_subset) is int:
        if not frame_subset in df['frame'].unique():
            return "WARNING: the subset is invalid, please try again"
    else:
         return "WARNING: the subset is invalid, please try again"

    # select frame_list
    if frame_subset is None: #if no subset
        frame_list=df['frame'].unique()
    elif type(frame_subset) is list:
        frame_list = range(frame_subset[0],frame_subset[1]+1)
    elif type(frame_subset) is int:
        frame_list = [frame_subset]
    return frame_list

def compute_msd(trajectory, coords=['x', 'y']):
    '''Compute the MSD of a trajectory that potentially misses some steps. The trajectory steps MUST be constant'''
    dt_array=trajectory['t'][1:].values - trajectory['t'][:-1].values
    if dt_array[dt_array==0].size!=0: #if there is an overlap
        print "WARNING: overlap in the traj"
    dt=dt_array[dt_array!=0].min()
    max_shift=np.floor(trajectory['t'].values.max()/dt).astype(np.int)
    shifts=range(0,max_shift+1)
    tau=np.array(shifts)*dt
    msds = np.zeros(tau.size)
    msds_std = np.zeros(tau.size)
    numpoints = np.zeros(tau.size)
    #initialize dictionary of square displacements (displ(t)-displ(t-delta_t))
    sqdists={}
    for delta in tau:
        delta='%.2f'%delta
        sqdists[delta]=np.array([])
    
    for shift in shifts:
        #get the displacement if the shift is equal to the right delta
        shifted=trajectory.shift(-shift)-trajectory
        delta_set = shifted['t'].unique()
        for delta in delta_set[~np.isnan(delta_set)]:
            indices = shifted['t'] == delta
            delta='%.2f'%delta
            sqdists[delta] = np.append(sqdists[delta],np.square(shifted.loc[indices,coords]).sum(axis=1))

    for i,delta in enumerate(tau):
        delta='%.2f'%delta
        sqdist=sqdists[delta][~np.isnan(sqdists[delta])]
        msds[i] = sqdist.mean()
        msds_std[i] = sqdist.std()
        numpoints[i] = sqdist.size
    msds = pd.DataFrame({'msds': msds, 'tau': tau, 'msds_std': msds_std, 'numpoints':numpoints})
    return msds

def fit_msd(msd,trajectory,save_plot=False,model_bounds={'P':[0,300],'D':[0,1e8],'v':[0,1e8]},model='PRW',data_dir=None,traj=None):
    '''Fit MSD with a persistent random walk model and extract the persistence length '''

    if data_dir is not None:
        outdir=osp.join(data_dir,'MSD')
        if osp.isdir(outdir) is False:
            os.mkdir(outdir)
    else:
        outdir=None

    # n=msd['numpoints'][msd['numpoints']>msd['numpoints'].max()*0.7].size #fit MSD only on part with enough statistics
    #lmfit model
    mean_vel=trajectory['v'][1:].mean()
    success=False;best=None;logx=False;logy=False
    if model is not None:
        if model=='PRW':
            logx=True;logy=True
            func = lambda t,P:2*(mean_vel**2)*P*(t-P*(1-np.exp(-t/P)))
            func_model=Model(func)
            p=func_model.make_params(P=10)
            p['P'].set(min=model_bounds['P'][0],max=model_bounds['P'][1])
        elif model=='biased_diff':
            func = lambda t,D,v:4*D*t+v**2*t**2
            func_model=Model(func)
            p=func_model.make_params(D=1,v=1)
            p['D'].set(min=model_bounds['D'][0])
            p['v'].set(min=model_bounds['v'][0])
        elif model=="pure_diff":
            func = lambda t,D:4*D*t
            func_model=Model(func)
            p=func_model.make_params(D=1)
            p['D'].set(min=model_bounds['D'][0])
        try:
            msd['weights']=1./(msd['msds_std']+1) #to ensure no div by 0
            # best=func_model.fit(msd['msds'][0:n],t=msd['tau'][0:n],params=p)
            msd.dropna(inplace=True)
            best=func_model.fit(msd['msds'].values,t=msd['tau'].values,params=p,weights=msd['numpoints'])
            
            if best.success==False:
                print "WARNING: fit_msd failed"
            success=best.success
        except:
            best=Dummy_class()
            success=False

    if save_plot:
        ax = msd.plot.scatter(x="tau", y="msds", logx=logx, logy=logy)
        if success:
            fitted=func(msd['tau'],*best.best_values.values())
            fitted_df=pd.DataFrame({'fitted':fitted,'tau':msd['tau']})
            fitted_df.plot(x="tau", y="fitted", logx=logx, logy=logy, ax=ax)
            if model=='biased_diff':
                title_ = 'D=%0.2f'%best.best_values['D']+r' $\mu m^2/min$, '+'v=%0.2f'%best.best_values['v']+r' $\mu m/min$'
            elif model=='PRW':
                title_ = 'P=%0.2f min'%best.best_values['P']
            elif model=='pure_diff':
                title_ = 'D=%0.2f'%best.best_values['D']+r' $\mu m^2/min$'
            ax.set_title(title_)
        ax.set_xlabel('lag time (min)')
        ax.set_ylabel(r'MSD ($\mu m^2$)')
        if outdir is None:
            print "WARNING: can't save MSD plot, data_dit not provided"
        else:
            savefig(osp.join(outdir,'%d.svg'%traj), dpi=300, bbox_inches='tight')
            close()
    return best,mean_vel,success

def get_obj_persistence_length(track_groups,track,traj=None,save_plot=False,dim=3):
    '''This function fits an object MSD with a PRW model to extract its persistence length'''
    if traj is None:
        traj=get_obj_traj(track_groups,track,dim=dim)
    msd=compute_msd(traj)
    best,speed,success=fit_msd(msd,traj,save_plot=save_plot)
    if success:
        pers_time=best.best_values['P']
        pers_length=pers_time*speed
        return pers_length
    else:
        return np.nan

class Dummy_class:
    """Class used to create an object containing a single attribute success. Useful for fit_SLS for returning an object instead of the data container object"""
    def __init__ (self):
        self.success = False


#################################################################
###########   PLOT METHODS   ####################################
#################################################################

def plot_cmap(plot_dir,label,cmap,vmin,vmax):
    close('all')
    fig = figure(figsize=(8,3))
    ax = fig.add_axes([0.05,0.80,0.9,0.15])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,norm=norm,orientation='horizontal')
    ax.tick_params(labelsize=16)
    cb.set_label(label=label,size=24)
    filename=osp.join(plot_dir,'colormap.png')
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    close('all')

def plot_cells(df_list,groups_list,frame,data_dir,plot_traj=False,z_lim=[],hide_labels=False,no_bkg=False,lengthscale=1.,length_ref=0.75,display=False,plot3D=False,elevation=None,angle=None,dim=3,shift=None):
    """ Plot all cells of a given frame.
        Different groups of cells can be plotted with different colors (data contained in df_list)
        The trajectory of each cell can be plotted with plot_traj
        By default, the cells are plotted on the microscopy image that should be stored in the directory raw.
        It can be plotted in 3D with plot3D, elevation and angle set the 3D view
    """
    print '\rplotting cells '+str(frame),
    sys.stdout.flush()

    plot_dir=osp.join(data_dir,'traj')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    multiple_groups=False
    if len(df_list)>1:
        multiple_groups=True

    z_labeling=False
    if len(z_lim)>0 and multiple_groups is False: #z_labeling impossible if multiple groups
        z_labeling=True

    #import image
    if plot3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig,ax,xmin,ymin,xmax,ymax,no_bkg=get_background(df_list[0],data_dir,frame,no_bkg=no_bkg)
    for k,df in enumerate(df_list):
        groups=groups_list[k]
        group=groups.get_group(frame).reset_index(drop=True)
        r,c=group.shape

        if plot_traj:
            track_groups=df.groupby(['traj'])

        for i in range(0,r):
            #write label
            x=group.loc[i,'x']
            y=group.loc[i,'y']
            track=int(group.loc[i,'traj'])
            s='%d'%(track)
            if hide_labels is False:
                color_='k' if no_bkg else 'w'
                ax.text(x,y,s,fontsize=5,color=color_)
            if plot_traj:
                #traj size
                lw_ref=rcParams['lines.linewidth']
                ms_ref=rcParams['lines.markersize']
                size_factor=lengthscale*length_ref
                lw=lw_ref*size_factor; ms=ms_ref*size_factor
                #plot trajectory
                traj=get_obj_traj(track_groups,track,max_frame=frame,dim=dim,shift=shift)
                traj_length,c=traj.shape
                if traj_length>1:
                    if not plot3D:
                        if z_labeling:
                            X=traj['x'].values;Y=traj['y'].values;Z=traj['z_rel'].values; #convert to numpy to optimize speed
                            for j in range(1,traj_length):
                                ax.plot([X[j-1],X[j]],[Y[j-1],Y[j]],lw=lw,ls='-',color=get_cmap_color(Z[j],cm.plasma,vmin=z_lim[0],vmax=z_lim[1]))
                            ax.plot(X[traj_length-1],Y[traj_length-1],ms=ms,marker='.',color=get_cmap_color(Z[j],cm.plasma,vmin=z_lim[0],vmax=z_lim[1]))
                        elif multiple_groups:
                            ax.plot(traj['x'],traj['y'],lw=lw,ls='-',color=color_list[k%7])
                            ax.plot(traj['x'].values[-1],traj['y'].values[-1],ms=ms,marker='.',color=color_list[k%7])
                        else:
                            ax.plot(traj['x'],traj['y'],lw=lw,ls='-',color=color_list[track%7])
                            ax.plot(traj['x'].values[-1],traj['y'].values[-1],ms=ms,marker='.',color=color_list[track%7])                       
                        ax.axis([xmin,xmax,ymin,ymax])
                    else:
                        if z_labeling:
                            X=traj['x'].values;Y=traj['y'].values;Z=traj['z_rel'].values; #convert to numpy to optimize speed
                            for j in range(1,traj_length):
                                ax.plot([X[j-1],X[j]],[Y[j-1],Y[j]],[Z[j-1],Z[j]],lw=lw,ls='-',color=get_cmap_color(Z[j],cm.plasma,vmin=z_lim[0],vmax=z_lim[1]))
                            ax.scatter(X[traj_length-1],Y[traj_length-1],Z[traj_length-1],s=10,color=get_cmap_color(Z[j],cm.plasma,vmin=z_lim[0],vmax=z_lim[1]))
                        elif multiple_groups:
                            ax.plot(traj['x'],traj['y'],lw=lw,ls='-',color=color_list[k%7])
                            ax.plot(traj['x'].values[-1],traj['y'].values[-1],ms=ms,marker='.',color=color_list[k%7])
                        else:
                            ax.plot(traj['x'],traj['y'],lw=lw,ls='-',color=color_list[track%7])
                            ax.plot(traj['x'].values[-1],traj['y'].values[-1],ms=ms,marker='.',color=color_list[track%7])                       
                            ax.axis([xmin,xmax,ymin,ymax])
                        ax.view_init(elev = elevation, azim=angle)
                        plt.axis('off')
    if display:
        plt.show()
        return                

    filename=osp.join(plot_dir,'%04d.png'%int(frame))
    fig.savefig(filename, dpi=300)
    close('all')

def plot_vfield(df,frame,data_dir,no_bkg=False,vlim=None,axis_on=False,plot_on_mean=False,black_arrows=False,vlim_mean=None):
    """ Plot velocity field and compute avg vfield on a grid"""
    close('all')
    print '\rplotting velocity field '+str(frame),
    sys.stdout.flush()
    plot_dir=osp.join(data_dir,'vfield')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    #import image
    if plot_on_mean:
        plot_fig,data_mean=plot_mean_vel(df,frame,data_dir,no_bkg=no_bkg,vlim=vlim_mean,axis_on=axis_on,save_plot=False)
        fig,ax,xmin,ymin,xmax,ymax=plot_fig
    else:
        fig,ax,xmin,ymin,xmax,ymax,no_bkg=get_background(df,data_dir,frame,no_bkg=no_bkg,axis_on=axis_on)
    data=get_map_data(plot_dir,frame)
    norm=plt.Normalize(vlim[0],vlim[1]) if vlim is not None else None
    if black_arrows:
        data=data[:4] #remove the vz data, so no color on arrows
    Q=ax.quiver(*data,units='x',cmap='plasma',norm=norm)

    if axis_on:
        ax.grid(False)
        ax.patch.set_visible(False)
        fig.set_tight_layout(True)

    filename=osp.join(plot_dir,'%04d.png'%int(frame))
    fig.savefig(filename,dpi=300)
    close()

def plot_div(df,frame,data_dir,no_bkg=False,vlim=None,axis_on=False):
    """ Plot 2D divergence"""
    close('all')
    print '\rplotting divergence field '+str(frame),
    sys.stdout.flush()
    plot_dir=osp.join(data_dir,'div')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    #import image
    fig,ax,xmin,ymin,xmax,ymax,no_bkg=get_background(df,data_dir,frame,no_bkg=no_bkg,axis_on=axis_on)
    X,Y,div=get_map_data(plot_dir,frame)
    div_masked = np.ma.array(div, mask=np.isnan(div))
    [vmin,vmax]= [div_masked.min(),div_masked.max()] if vlim is None else vlim
    cmap=cm.plasma; cmap.set_bad('w',alpha=0) #set NAN transparent
    C=ax.pcolormesh(X[1:-1,1:-1],Y[1:-1,1:-1],div_masked[1:-1,1:-1],cmap=cmap,alpha=0.5,vmin=vmin,vmax=vmax)
    ax.axis([xmin,xmax,ymin,ymax])

    if axis_on:
        ax.grid(False)
        ax.patch.set_visible(False)
        fig.set_tight_layout(True)
    filename=osp.join(plot_dir,'%04d.png'%int(frame))
    fig.savefig(filename, dpi=300)
    close()

def plot_mean_vel(df,frame,data_dir,no_bkg=False,vlim=None,axis_on=False,save_plot=True):
    close('all')
    print '\rplotting mean velocity '+str(frame),
    sys.stdout.flush()
    plot_dir=osp.join(data_dir,'mean_vel')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    #import image
    fig,ax,xmin,ymin,xmax,ymax,no_bkg=get_background(df,data_dir,frame,no_bkg=no_bkg,axis_on=axis_on)
    X,Y,mean_vel=get_map_data(plot_dir,frame)
    mean_vel_masked = np.ma.array(mean_vel, mask=np.isnan(mean_vel))
    [vmin,vmax]= [mean_vel_masked.min(),mean_vel_masked.max()] if vlim is None else vlim
    cmap=cm.plasma; cmap.set_bad('w',alpha=0) #set NAN transparent
    C=ax.pcolormesh(X,Y,mean_vel_masked,cmap=cmap,alpha=0.5,vmin=vmin,vmax=vmax)
    ax.axis([xmin,xmax,ymin,ymax])

    if axis_on:
        ax.grid(False)
        ax.patch.set_visible(False)
        fig.set_tight_layout(True)
    if save_plot:
        filename=osp.join(plot_dir,'%04d.png'%int(frame))
        fig.savefig(filename, dpi=300)

    plot_fig=[fig,ax,xmin,ymin,xmax,ymax]
    data=[X,Y,mean_vel,mean_vel_masked]
    return [plot_fig,data]

def plot_v_coord(df,frame,data_dir,no_bkg=False,vlim=None,axis_on=False,save_plot=True,coord='vx',node_grid=None):
    close('all')
    print '\rplotting '+coord+' '+str(frame),
    sys.stdout.flush()
    plot_dir=osp.join(data_dir,coord)
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)
    coord_={'vx':2,'vy':3,'vz':4}

    #import image
    fig,ax,xmin,ymin,xmax,ymax,no_bkg=get_background(df,data_dir,frame,no_bkg=no_bkg,axis_on=axis_on)
    data=get_map_data(osp.join(data_dir,'vfield'),frame)
    data=data[coord_[coord]]
    data_masked = np.ma.array(data, mask=np.isnan(data))
    [vmin,vmax]= [data_masked.min(),data_masked.max()] if vlim is None else vlim
    cmap=cm.plasma; cmap.set_bad('w',alpha=0) #set NAN transparent
    X,Y=node_grid
    C=ax.pcolormesh(X,Y,data_masked,cmap=cmap,alpha=0.5,vmin=vmin,vmax=vmax)
    ax.axis([xmin,xmax,ymin,ymax])

    if axis_on:
        ax.grid(False)
        ax.patch.set_visible(False)
        fig.set_tight_layout(True)
    if save_plot:
        filename=osp.join(plot_dir,'%04d.png'%int(frame))
        fig.savefig(filename, dpi=300)

    plot_fig=[fig,ax,xmin,ymin,xmax,ymax]
    data=[X,Y,data,data_masked]
    return [plot_fig,data]

def plot_z_flow(df,frame,data_dir,no_bkg=False,vlim=None,axis_on=False):
    """Plot the flow (defined as the net number of cells going through a surface element in the increasing z direction) through the plane of z=z0"""
    
    #Make sure these are 3D data
    if 'z' not in df.columns:
        print "Not a 3D set of data"
        return

    close('all')
    print '\rplotting z flow '+str(frame),
    sys.stdout.flush()
    plot_dir=osp.join(data_dir,'z_flow')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)
    
    fig,ax,xmin,ymin,xmax,ymax,no_bkg=get_background(df,data_dir,frame,no_bkg=no_bkg,axis_on=axis_on)
    X,Y,flow=get_map_data(plot_dir,frame)
    [vmin,vmax]= [flow.min(),flow.max()] if vlim is None else vlim
    cmap=cm.plasma
    C=ax.pcolormesh(X,Y,flow,cmap=cmap,alpha=0.5,vmin=vmin,vmax=vmax)
    ax.axis([xmin,xmax,ymin,ymax])

    if axis_on:
        ax.grid(False)
        ax.patch.set_visible(False)
        fig.set_tight_layout(True)
    filename=osp.join(plot_dir,'%04d.png'%int(frame))
    fig.savefig(filename, dpi=300)
    close()
    
    return flow

def plot_ROI_avg(df,data_dir,map_kind,frame,ROI_data_list,plot_on_map=False,plot_section=True):
    """ Plot average data along the major axis of a map at a given frame"""

    close('all')
    print '\rplotting ROI average '+str(frame),
    sys.stdout.flush()

    map_kind_ = 'vfield' if map_kind in ['vx','vy','vz'] else map_kind
    plot_dir=osp.join(data_dir,map_kind_,'ROI_avg')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    if plot_on_map:
        fig,ax,xmin,ymin,xmax,ymax,no_bkg=get_background(df,data_dir,frame)

    plot_data={}
    for i,ROI_data in enumerate(ROI_data_list):
        avg=avg_ROI_major_axis(ROI_data)
        if plot_on_map:
            ax.plot(avg['data'][0],avg['data'][1],color=color_list[i])
        fig2, ax2 = plt.subplots(1, 1)
        if avg['major_ax']==0:
            x_data=avg['data'][1]
            title='x = %d px'%avg['data'][0][0]
            xlab='y (px)'
        elif avg['major_ax']==1:
            x_data=avg['data'][0]
            title='y = %d px'%avg['data'][1][0]
            xlab='x (px)'
        
        filename_prefix=osp.join(plot_dir,'ROI_%d-%d-%d-%d'%(avg['data'][0].min(),avg['data'][0].max(),avg['data'][1].min(),avg['data'][1].max()))
        plot_data[str(i)]={'x_data':x_data,'y_data':avg['data'][2],'title':title,'xlab':xlab,'filename_prefix':filename_prefix}
        
        if plot_section:
            ax2.plot(x_data,avg['data'][2])
            ax2.set_title(title)
            ax2.set_xlabel(xlab)
            ax2.set_ylabel(map_dic[map_kind]['cmap_label'])
            filename=filename_prefix+'_frame_%04d.png'%frame
            fig2.savefig(filename,dpi=300,bbox_inches='tight')

    if plot_on_map:
        filename=osp.join(plot_dir,'sections_'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'.png') #filename with the time to get a specific name not to overwrite the different sections files
        fig.savefig(filename,dpi=300,bbox_inches='tight')
        close()

    return plot_data

def plot_XY_flow(df,data_dir,line,orientation,frame,groups,window_size=None,timescale=1.,lengthscale=1.,z_depth=None,plot_on_map=False):
    """Plot the flow along the surface defined by a XY line. The first end of the line is x=0 for the plot. The cells crossing the line along the orientation (from first point to second) are counted
    as positive cells, the cells in the other are counted as negative. The count is integrated along a moving window along the line (line in um)"""
    close('all')

    plot_dir=osp.join(data_dir,'XY_flow')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    image_dir=osp.join(data_dir,'raw')
    if osp.isdir(image_dir)==False:
        image_dir=osp.join(data_dir,'traj')

    data=compute_XY_flow(df,data_dir,line,orientation,frame,groups,window_size=window_size,timescale=timescale,lengthscale=lengthscale,z_depth=z_depth)

    title='line_%d-%d_%d-%d'%(line[0][0],line[0][1],line[1][0],line[1][1])
    xlab=r'x ($\mu m$)'
    if window_size is not None:
        ylab=r'flow ($cells.\mu m^{-2}.min^{-1}$)' if z_depth is not None else r'flow ($cells.\mu m^{-1}.min^{-1}$)'
    else:
        ylab=r'flow ($cells.min^{-1}$)'
    filename_prefix=osp.join(plot_dir,'line_%d-%d_%d-%d'%(line[0][0],line[0][1],line[1][0],line[1][1]))
    plot_data={'x_data':data[:,0],'y_data':data[:,1],'title':title,'xlab':xlab,'ylab':ylab,'filename_prefix':filename_prefix}
    
    #abscissa length 
    abs_length = np.sqrt((line[1][0]-line[0][0])**2+(line[1][1]-line[0][1])**2)/lengthscale

    fig, ax = plt.subplots(1, 1)
    if window_size is None:
        ax.scatter(data[:,0],data[:,1])
    else:
        ax.plot(data[:,0],data[:,1])
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(0,abs_length)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,4))
    filename=filename_prefix+'_frame_%04d.png'%frame
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    close()

    if plot_on_map:
        fig,ax,xmin,ymin,xmax,ymax,no_bkg=get_background(df,data_dir,frame)
        ax.plot(line[:,0],line[:,1])
        ax.arrow(orientation[0,0],orientation[0,1],orientation[1,0]-orientation[0,0],orientation[1,1]-orientation[0,1],shape='full',length_includes_head=True,width=10,color='k')
        filename=osp.join(plot_dir,filename_prefix+'section.png')
        fig.savefig(filename,dpi=300,bbox_inches='tight')
        close()

    return plot_data

def plot_hist_persistence_length(data_dir,track_groups,tracks,minimal_traj_length=40,normalize=True,dim=3):
    close('all')
    pers_length_dict={}
    for track in tracks:
        traj=get_obj_traj(track_groups,track,dim=dim)
        traj_length,c=traj.shape
        if traj_length>minimal_traj_length:
            pers_length_dict[track]=get_obj_persistence_length(track_groups,track,traj,dim=dim)

    pers_lengths=pd.Series(pers_length_dict)
    if normalize:
        pers_lengths.plot.hist(weights=np.ones_like(pers_lengths*100)/len(pers_lengths))
        ylabel('trajectories proportion ')
    else:
        pers_lengths.plot.hist()
        ylabel('trajectories count')
    xlabel('persistence length ($\mu m$) ')
    filename = osp.join(data_dir,'persistence_lenght.svg')
    savefig(filename, dpi=300, bbox_inches='tight')
    close()


#### PLOT_ALL methods

def plot_all_cells(df_list,data_dir,plot_traj=False,z_lim=[],hide_labels=False,no_bkg=False,parallelize=False,lengthscale=1.,length_ref=0.75,plot3D=False,elevation=None,angle=None,dim=3,shift=None):
    plot_dir=osp.join(data_dir,'traj')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    groups_list=[df.groupby('frame') for df in df_list]

    if len(z_lim)==2:
        plot_cmap(plot_dir,'$z\ (\mu m)$',cm.plasma,z_lim[0],z_lim[1])

    if parallelize:
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(plot_cells)(df_list,groups_list,frame,data_dir,plot_traj,z_lim,hide_labels,no_bkg,lengthscale) for frame in df['frame'].unique())
    else:
        for frame in df['frame'].unique():
            plot_cells(df_list,groups_list,frame,data_dir,plot_traj=plot_traj,z_lim=z_lim,hide_labels=hide_labels,no_bkg=no_bkg,lengthscale=lengthscale,length_ref=length_ref,plot3D=plot3D,elevation=elevation,angle=angle,dim=dim,shift=shift)

def plot_all_vfield(df,data_dir,grids=None,no_bkg=False,parallelize=False,dim=3,refresh=False,axis_on=False,plot_on_mean=False,black_arrows=False,manual_vlim=False,force_vlim=None):
    # Maps of all frames are computed through the compute_vlim function
    groups=df.groupby('frame')
    plot_dir=osp.join(data_dir,'vfield')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    if osp.isdir(osp.join(plot_dir,'data')) is False:
        refresh=True
    if refresh:
        # compute data
        # if parallelize:
        #     num_cores = multiprocessing.cpu_count()
        #     Parallel(n_jobs=num_cores)(delayed(compute_vfield)(df,groups,frame,data_dir,grids) for frame in df['frame'].unique())
        # else:
        #     vlim=compute_vlim(df,compute_vfield,groups,data_dir,grids)
        #     pickle.dump(vlim,open(osp.join(plot_dir,'data','vlim.p'),"wb"))

        #compute data
        vlim=compute_vlim(df,compute_vfield,groups,data_dir,grids,-1,dim=dim)
        pickle.dump(vlim,open(osp.join(plot_dir,'data','vlim.p'),"wb"))

    vlim=pickle.load( open(osp.join(plot_dir,'data','vlim.p'), "rb" ))
    if force_vlim is not None:
        if force_vlim['vfield'] is not None:
            vlim=force_vlim['vfield']
    #plot colorbar
    if dim==3 and not black_arrows:
        plot_cmap(plot_dir,r'$v_z\ (\mu m.min^{-1})$',cm.plasma,vlim[0],vlim[1])

    if plot_on_mean:
        mean_dir=osp.join(data_dir,'mean_vel')
        if osp.isdir(mean_dir)==False:
            os.mkdir(mean_dir)

        if osp.isdir(osp.join(mean_dir,'data')) is False:
            refresh=True
        if refresh:
            #compute data
            if parallelize:
                num_cores = multiprocessing.cpu_count()
                Parallel(n_jobs=num_cores)(delayed(map_dic['mean_vel']['compute_func'])(df,groups,frame,data_dir,grids,lengthscale) for frame in df['frame'].unique())
            else:
                vlim_mean=compute_vlim(df,map_dic['mean_vel']['compute_func'],groups,data_dir,grids,-1,show_hist=manual_vlim,dim=dim)
                pickle.dump(vlim_mean,open(osp.join(mean_dir,'data','vlim.p'),"wb"))

        vlim_mean=pickle.load( open(osp.join(mean_dir,'data','vlim.p'), "rb" ))
        if force_vlim is not None:
            if force_vlim['mean_vel'] is not None:
                vlim_mean=force_vlim['mean_vel']
        #plot colorbar
        plot_cmap(plot_dir,map_dic['mean_vel']['cmap_label'],cm.plasma,vlim_mean[0],vlim_mean[1])
    else:
        vlim_mean=None

    #plot maps
    if parallelize:
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(plot_vfield)(df,frame,data_dir,no_bkg,vlim,axis_on,plot_on_mean,black_arrows) for frame in df['frame'].unique())
    else:
        for i,frame in enumerate(df['frame'].unique()):
            plot_vfield(df,frame,data_dir,no_bkg=no_bkg,vlim=vlim,axis_on=axis_on,plot_on_mean=plot_on_mean,black_arrows=black_arrows,vlim_mean=vlim_mean)

def plot_all_maps(df,data_dir,grids,map_kind,refresh=False,no_bkg=False,parallelize=False,dim=3,manual_vlim=False,axis_on=False,force_vlim=None,**kwargs):

    groups=df.groupby('frame')

    plot_dir=osp.join(data_dir,map_kind)
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    #in the case of velocity coordinate don't recompute the data but get them from the vfield folder
    if map_kind in ['vx','vy','vz'] and osp.isdir(osp.join(data_dir,'vfield','data')):
        get_former_data=True
        plot_dir_=osp.join(data_dir,'vfield')
    else:
        get_former_data=False
        plot_dir_=None

    #force refresh if the data does not exist
    if osp.isdir(osp.join(plot_dir,'data')) is False:
        os.mkdir(osp.join(plot_dir,'data'))
        refresh=True
    if refresh:
        #compute data
        if parallelize:
            num_cores = multiprocessing.cpu_count()
            Parallel(n_jobs=num_cores)(delayed(map_dic[map_kind]['compute_func'])(df,groups,frame,data_dir,grids,lengthscale) for frame in df['frame'].unique())
        else:
            if map_kind=='vx':
                data_coord=2
            elif map_kind=='vy':
                data_coord=3
            else:
                data_coord=-1
            vlim=compute_vlim(df,map_dic[map_kind]['compute_func'],groups,data_dir,grids,data_coord,dim,show_hist=manual_vlim,get_former_data=get_former_data,plot_dir=plot_dir_,**kwargs)
            pickle.dump(vlim,open(osp.join(plot_dir,'data','vlim.p'),"wb"))

    pickle.dump(grids,open(osp.join(plot_dir,'data','grids.p'),"wb"))
    vlim=pickle.load( open(osp.join(plot_dir,'data','vlim.p'), "rb" ))
    if force_vlim is not None:
        if force_vlim[map_kind] is not None:
            vlim=force_vlim[map_kind]
    #plot colorbar
    plot_cmap(plot_dir,map_dic[map_kind]['cmap_label'],cm.plasma,vlim[0],vlim[1])

    #plot maps
    if parallelize:
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(plot_z_flow)(df,frame,data_dir,no_bkg,vlim) for frame in df['frame'].unique())
    else:
        for i,frame in enumerate(df['frame'].unique()):
            if map_kind in ['vx','vy','vz']:
                map_dic[map_kind]['plot_func'](df,frame,data_dir,no_bkg,vlim,axis_on,coord=map_kind,node_grid=grids[0])
            else:
                map_dic[map_kind]['plot_func'](df,frame,data_dir,no_bkg,vlim,axis_on)

def plot_all_avg_ROI(df,data_dir,map_kind,frame_subset=None,selection_frame=None,ROI_list=None,plot_on_map=False,plot_section=True,cumulative_plot=True,avg_plot=True,timescale=1.):

    close('all')

    map_kind_ = 'vfield' if map_kind in ['vx','vy','vz'] else map_kind 

    plot_dir=osp.join(data_dir,map_kind_,'ROI_avg')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    if selection_frame is None:
        selection_frame=input("Give the frame number on which you want to draw your ROIs: ")
    
    [ROI_data_list,ROI_list]=select_map_ROI(data_dir,map_kind,selection_frame,ROI_list)

    frame_list=select_frame_list(df,frame_subset)

    plot_data_list=[]
    for i,frame in enumerate(frame_list):
        [ROI_data_list,ROI_list]=select_map_ROI(data_dir,map_kind,frame,ROI_list)
        if i>0:
            plot_on_map=False #plot it only once
        plot_data_list.append(plot_ROI_avg(df,data_dir,map_kind,frame,ROI_data_list,plot_on_map=plot_on_map,plot_section=plot_section))

    if cumulative_plot:
        time_min=min(frame_list)*timescale; time_max=max(frame_list)*timescale
        #colorbar
        Z = [[0,0],[0,0]]
        levels=array(frame_list)*timescale
        CS3 = plt.contourf(Z, levels, cmap=cm.plasma)
        plt.clf()
        for i,ROI_data in enumerate(ROI_data_list):
            x_data_l=[p[str(i)]['x_data'] for p in plot_data_list]
            y_data_l=[p[str(i)]['y_data'] for p in plot_data_list]
            title=plot_data_list[0][str(i)]['title'];xlab=plot_data_list[0][str(i)]['xlab']
            fig, ax = plt.subplots(1, 1)
            for j in range(len(x_data_l)):
                time=frame_list[j]*timescale
                ax.plot(x_data_l[j],y_data_l[j],color=get_cmap_color(time, cm.plasma, vmin=time_min, vmax=time_max))
            ax.set_title(title)
            ax.set_xlabel(xlab)
            ax.set_ylabel(map_dic[map_kind]['cmap_label'])
            cb=fig.colorbar(CS3)
            cb.set_label(label='time (min)')
            filename=plot_data_list[0][str(i)]['filename_prefix']+'_cumulative.png'
            fig.savefig(filename,dpi=300,bbox_inches='tight')
            close()

    if avg_plot:
        for i,ROI_data in enumerate(ROI_data_list):
            x_data=plot_data_list[0][str(i)]['x_data']
            y_data_l=[p[str(i)]['y_data'] for p in plot_data_list]
            title=plot_data_list[0][str(i)]['title'];xlab=plot_data_list[0][str(i)]['xlab']
            fig, ax = plt.subplots(1, 1)
            sns.tsplot(y_data_l,time=x_data,ax=ax,estimator=np.nanmean,err_style="unit_traces")
            ax.set_title(title)
            ax.set_xlabel(xlab)
            ax.set_ylabel(map_dic[map_kind]['cmap_label'])
            filename=plot_data_list[0][str(i)]['filename_prefix']+'_average.png'
            fig.savefig(filename,dpi=300,bbox_inches='tight')
            close()

def plot_all_XY_flow(df,data_dir,line=None,orientation=None,frame_subset=None,window_size=None,selection_frame=None,timescale=1.,lengthscale=1.,z_depth=None):
    """Plot the flow through a vertical surface define by a XY line (the first point of line is the start of the abscissa axis). 
    The orientation of the flow is given by the orientation vector (pointing towards the 2nd point). If line and orientations are not given they are manually drawn with get_ROI"""

    close('all')

    #get parameters
    plot_dir=osp.join(data_dir,'XY_flow')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    image_dir=osp.join(data_dir,'raw')
    if osp.isdir(image_dir)==False:
        image_dir=osp.join(data_dir,'traj')


    if line is None and orientation is None:
        if selection_frame is None:
            selection_frame=input("Give the frame number on which you want to draw your ROIs: ")
        print "Draw two lines (and press ENTER to validate each one of them). \n The first defines your vertical surface (the first point will be the origin of the plot). The second needs to be approximatively perpendicular to the first one. It gives the orientation to the flow (going from the first point to the 2nd)"
        line,orientation=get_ROI(image_dir,selection_frame,tool=LineTool) #lines coordinates are given 2nd point first and 1st point second
        line=line[::-1,:]; orientation=orientation[::-1,:] #flip order of points coordinates 

    #plot data
    frame_list=select_frame_list(df,frame_subset)
    groups=df.groupby('frame')

    plot_data_list=[]
    for i,frame in enumerate(frame_list):
        plot_on_map=True if i==0 else False #plot on map only once
        plot_data_list.append(plot_XY_flow(df,data_dir,line,orientation,frame,groups,window_size=window_size,timescale=timescale,lengthscale=lengthscale,z_depth=z_depth,plot_on_map=plot_on_map))

    #cumulative plot
    abs_length = np.sqrt((line[1][0]-line[0][0])**2+(line[1][1]-line[0][1])**2)/lengthscale
    time_min=min(frame_list)*timescale; time_max=max(frame_list)*timescale
    #colorbar
    Z = [[0,0],[0,0]]
    levels=array(frame_list)*timescale
    CS3 = plt.contourf(Z, levels, cmap=cm.plasma)
    plt.clf()
    x_data_l=[p['x_data'] for p in plot_data_list]
    y_data_l=[p['y_data'] for p in plot_data_list]
    fig, ax = plt.subplots(1, 1)
    for j in range(len(x_data_l)):
        time=frame_list[j]*timescale
        if window_size is None:
            ax.scatter(x_data_l[j],y_data_l[j],color=get_cmap_color(time, cm.plasma, vmin=time_min, vmax=time_max))
        else:
            ax.plot(x_data_l[j],y_data_l[j],color=get_cmap_color(time, cm.plasma, vmin=time_min, vmax=time_max))
    ax.set_title(plot_data_list[0]['title'])
    ax.set_xlabel(plot_data_list[0]['xlab'])
    ax.set_ylabel(plot_data_list[0]['ylab'])
    ax.set_xlim(0,abs_length)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,4))
    cb=fig.colorbar(CS3)
    cb.set_label(label='time (min)')
    filename=plot_data_list[0]['filename_prefix']+'_cumulative.png'
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    close()

    #average plot
    if window_size is not None:
        x_data=plot_data_list[0]['x_data']
        fig, ax = plt.subplots(1, 1)
        sns.tsplot(y_data_l,time=x_data,ax=ax,estimator=np.nanmean,err_style="unit_traces")
        ax.set_title(plot_data_list[0]['title'])
        ax.set_xlabel(plot_data_list[0]['xlab'])
        ax.set_ylabel(plot_data_list[0]['ylab'])
        ax.set_xlim(0,abs_length)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,4))
        filename=plot_data_list[0]['filename_prefix']+'_average.png'
        fig.savefig(filename,dpi=300,bbox_inches='tight')
        close()

    return plot_data_list

def plot_all_MSD(df_list,data_dir,fit_model=None,dim=3,save_plot=False,to_csv=True,plot_along_Y=True,origins=None,lengthscale=1.,shift=None,avg_wind=None,):
    if fit_model=="biased_diff":
        cols=['traj','xmean','ymean','D','v','D_err','v_err']
        param_list=['D','v']
    elif fit_model=="PRW":
        cols=['traj','xmean','ymean','P','P_err']
        param_list=['P']
    elif fit_model=="pure_diff":
        cols=['traj','xmean','ymean','D','D_err']
        param_list=['D']
    df_out=pd.DataFrame(columns=cols)
    i=0
    for df in df_list:
        track_groups=df.groupby(['traj'])
        track_list=df['traj'].unique()
        for track in track_list:
            traj=get_obj_traj(track_groups,track,dim=dim,shift=shift)
            msd=compute_msd(traj)
            best,speed,success=fit_msd(msd,traj,save_plot=save_plot,model=fit_model,data_dir=data_dir,traj=track)
            if success:
                param_val=[best.best_values[param] for param in param_list]
                errors=[np.nan]*len(param_list)
                if best.covar is not None:
                    errors=list(sqrt(best.covar).diagonal())
                df_out.loc[i,cols]=[track,traj['x'].mean(),traj['y'].mean()]+param_val+errors
                i+=1

    df_out=df_out.apply(pd.to_numeric,args=('ignore',None))

    #moving average along Y axis with regular spacing dY defined as 1/5 of avg_window
    if avg_wind is not None:
        dY = avg_wind/5.
        reg_Y = arange(0,df_out['ymean'].max(),dY)
        cols_=['y']+[p+'_mean' for p in param_list]+[p+'_std' for p in param_list]
        init_array=np.empty((reg_Y.shape[0],len(cols_)-1)) * np.nan
        init_array=np.concatenate((reg_Y[:,None],init_array),axis=1)
        df_reg=pd.DataFrame(init_array,columns=cols_)
        for i,y in enumerate(reg_Y):
            ind = ((df_out['ymean']>=y) & ((df_out['ymean']<y+avg_wind)))
            mean_ = df_out[ind][param_list].mean()
            std_ = df_out[ind][param_list].std()
            for param in param_list:
                df_reg.loc[i,param+'_mean']=mean_[param]
                df_reg.loc[i,param+'_std']=std_[param]
        df_reg['y']=df_reg['y']+avg_wind/2. #shift y in the middle of the window
        
    if origins is not None:
        df_out['ymean']=abs(df_out['ymean']-origins)/lengthscale

    if to_csv:
        outdir=osp.join(data_dir,'MSD')
        if osp.isdir(outdir) is False:
            os.mkdir(outdir)
        df_out.to_csv(osp.join(outdir,'all_MSD_fit.csv'))
        df_reg.to_csv(osp.join(outdir,'mean_along_Y.csv'))

    if plot_along_Y is not None:
        lab_dict={'v':r'$\langle v \rangle \ \mu m/min$','D':r'$D \ \mu m^2/min$'}
        for param in param_list:
            fig,ax=plt.subplots(1,1)
            df_out.plot.scatter(x='ymean',y=param,ax=ax)
            ax.set_ylabel(lab_dict[param])
            xlab='position along Y axis (px)' if origins is None else r'distance to anterior ($\mu m$)'
            ax.set_xlabel(xlab)
            fig.savefig(osp.join(outdir,param+'_along_Y.svg'))

            if avg_wind is not None:
                fig,ax=plt.subplots(1,1)
                df_reg.plot.scatter(x='y',y=param+'_mean',yerr=param+'_std',ax=ax)
                ax.set_ylabel(lab_dict[param])
                xlab='position along Y axis (px)' if origins is None else r'distance to anterior ($\mu m$)'
                ax.set_xlabel(xlab)
                fig.savefig(osp.join(outdir,param+'_mean_along_Y.svg'))

    return df_out


#################################################################
###########   CONTAINER METHODS   ###############################
#################################################################

def cell_analysis(data_dir,refresh=False,parallelize=False,plot_traj=True,hide_labels=True,no_bkg=False,linewidth=1.,plot3D=False,MSD_fit=None,z_lim=None,shift_traj=False,save_MSD_plot=False,min_traj_len=None,avg_wind_along_Y=None,frame_subset=None):
    df,lengthscale,timescale,columns,dim=get_data(data_dir,refresh=refresh)
    if z_lim is None:
        z_lim=[df['z_rel'].min(),df['z_rel'].max()] if dim==3 else []
    df_list=[]

    if min_traj_len is None:
        subset=raw_input('By what do you want to filter? Type: none, ROI, track_length, frame_subset. \t ')
        if subset=='none':
            df_list=[df]
        elif subset=='ROI':
            df_list=filter_by_ROI(df,data_dir)
        elif subset=='track_length':
            min_traj_len=input('Give the minimum track length? (number of frames): ')
            df_list=[filter_by_traj_len(df,min_traj_len=min_traj_len)]
        elif subset=='frame_subset':
            frame_subset=input('Give the frame subset? (first,last): ')
            df=df[((df['frame']>=frame_subset[0]) & (df['frame']<=frame_subset[1]))]
            df_list=[df]
        else:
            print 'ERROR: not a valid answer'
            return
    else:
        if frame_subset is not None:
            df=df[((df['frame']>=frame_subset[0]) & (df['frame']<=frame_subset[1]))]
        df_list=[filter_by_traj_len(df,min_traj_len=min_traj_len)]
    
    if frame_subset is not None:
        df_list_=[]
        for df in df_list:
            df=df[((df['frame']>=frame_subset[0]) & (df['frame']<=frame_subset[1]))]
            df_list_.append(df)
        df_list=df_list_

    shift=get_shift(data_dir,timescale,lengthscale) if shift_traj else None

    plot_all_cells(df_list,data_dir,plot_traj=plot_traj,z_lim=z_lim,hide_labels=hide_labels,no_bkg=no_bkg,parallelize=parallelize,lengthscale=lengthscale,length_ref=0.75/linewidth,plot3D=plot3D,dim=dim,shift=None)
    
    if MSD_fit is not None:
        plot_all_MSD(df_list,data_dir,fit_model=MSD_fit,dim=dim,lengthscale=lengthscale,save_plot=save_MSD_plot,shift=shift,origins=shift.loc[0,'y0'],avg_wind=avg_wind_along_Y)

def map_analysis(data_dir,refresh=False,parallelize=False,x_grid_size=10,no_bkg=False,z0=None,dimensions=None,axis_on=False,plot_on_mean=True,black_arrows=True,manual_vlim=False):
    df,lengthscale,timescale,columns,dim=get_data(data_dir,refresh=refresh)
    if osp.isdir(osp.join(data_dir,'raw'))==False:
        info=get_info(data_dir)
        dimensions=info['dimensions']
    else:
        dimensions=None
    force_vlim=get_vlim(data_dir)
    grids=make_grid(x_grid_size,data_dir,dimensions=dimensions)
    plot_all_vfield(df,data_dir,grids=grids,no_bkg=no_bkg,parallelize=parallelize,dim=dim,refresh=refresh,axis_on=axis_on,plot_on_mean=plot_on_mean,black_arrows=black_arrows,manual_vlim=manual_vlim,force_vlim=force_vlim)
    if grids is not None:
        plot_all_maps(df,data_dir,grids,'div',refresh=refresh,no_bkg=no_bkg,parallelize=parallelize,dim=dim,axis_on=axis_on,lengthscale=lengthscale,force_vlim=force_vlim)
        plot_all_maps(df,data_dir,grids,'mean_vel',refresh=refresh,no_bkg=no_bkg,parallelize=parallelize,dim=dim,manual_vlim=manual_vlim,axis_on=axis_on,force_vlim=force_vlim)
        plot_all_maps(df,data_dir,grids,'vx',refresh=refresh,no_bkg=no_bkg,parallelize=parallelize,dim=dim,manual_vlim=manual_vlim,axis_on=axis_on,force_vlim=force_vlim)
        plot_all_maps(df,data_dir,grids,'vy',refresh=refresh,no_bkg=no_bkg,parallelize=parallelize,dim=dim,manual_vlim=manual_vlim,axis_on=axis_on,force_vlim=force_vlim)
        if dim==3:
            plot_all_maps(df,data_dir,grids,'vz',refresh=refresh,no_bkg=no_bkg,parallelize=parallelize,dim=dim,manual_vlim=manual_vlim,axis_on=axis_on,force_vlim=force_vlim)
        # if z0 is None:
        #     z0= df['z_rel'].min() + (df['z_rel'].max()-df['z_rel'].min())/2.
        #     print 'z0=%f'%z0
        # plot_all_maps(df,data_dir,grids,'z_flow',refresh=refresh,no_bkg=no_bkg,parallelize=parallelize,z0=z0,timescale=timescale)

def avg_ROIs(data_dir,frame_subset=None,selection_frame=None,ROI_list=None,plot_on_map=True,plot_section=True,cumulative_plot=True,avg_plot=True,refresh=False,map_kind=None):
    df,lengthscale,timescale,columns,dim=get_data(data_dir,refresh=refresh)
    if map_kind is None:
        map_kind=raw_input("Give the map wou want to plot your ROIs on (div,mean_vel,z_flow,vx,vy,vz): ")
    plot_all_avg_ROI(df,data_dir,map_kind,frame_subset=frame_subset,selection_frame=selection_frame,ROI_list=ROI_list,plot_on_map=plot_on_map,plot_section=plot_section,cumulative_plot=cumulative_plot,avg_plot=avg_plot,timescale=timescale)

def XY_flow(data_dir,window_size=None,refresh=False,line=None,orientation=None,frame_subset=None,selection_frame=None,z_depth=None):
    df,lengthscale,timescale,columns,dim=get_data(data_dir,refresh=refresh)
    
    if z_depth is None:
        z_lim=[df['z_rel'].min(),df['z_rel'].max()] if dim==3 else []
        z_depth=None if len(z_lim)==0 else z_lim[1]-z_lim[0]

    if window_size is None:
        window_size=input("Give the window size you want to calculate the flow on (in um and must be an integer >= 1): ")
    
    plot_all_XY_flow(df,data_dir,line=line,orientation=orientation,frame_subset=frame_subset,window_size=window_size,selection_frame=selection_frame,timescale=timescale,lengthscale=lengthscale,z_depth=z_depth)
    

###############################################

map_dic={'div':{'compute_func':compute_div,'plot_func':plot_div,'cmap_label':r'$div(\overrightarrow{v})\ (min^{-1})$'},
     'mean_vel':{'compute_func':compute_mean_vel,'plot_func':plot_mean_vel,'cmap_label':r'$v\ (\mu m.min^{-1})$'},
     'z_flow':{'compute_func':compute_z_flow,'plot_func':plot_z_flow,'cmap_label':'cell flow $(min^{-1})$'},
     'vx':{'compute_func':compute_vfield,'plot_func':plot_v_coord,'cmap_label':r'$v_x\ (\mu m.min^{-1})$'},
     'vy':{'compute_func':compute_vfield,'plot_func':plot_v_coord,'cmap_label':r'$v_y\ (\mu m.min^{-1})$'},
     'vz':{'compute_func':compute_vfield,'plot_func':plot_v_coord,'cmap_label':r'$v_z\ (\mu m.min^{-1})$'}}


###############################################
########### IDEAS FOR THE FUTURE ##############
###############################################

def plot_superimposed_traj(df,data_dir,traj_list,center_origin=True,fn_end=''):
    """ Plot a set of trajectories without any background. If center_origin all first points are located in the center"""
    close('all')
    
    track_groups=df.groupby(['traj'])

    if center_origin:
        for i,traj_id in enumerate(traj_list):
            traj=get_obj_traj(track_groups,traj_id)
            if i==0:
                minx=traj['x'].min(); maxx=traj['x'].max(); miny=traj['y'].min(); maxy=traj['y'].max()
            else:
                minx=traj['x'].min() if traj['x'].min()<minx else minx
                maxx=traj['x'].max() if traj['x'].max()>maxx else maxx
                miny=traj['y'].min() if traj['y'].min()<miny else miny
                maxy=traj['y'].max() if traj['y'].max()>maxy else maxy
                midx=minx+(maxx-minx)/2.
                midy=miny+(maxy-miny)/2.

    fig,ax=plt.subplots(1,1)
    ax.axis('off')
    for i,traj_id in enumerate(traj_list):
        traj=get_obj_traj(track_groups,traj_id)
        traj_length = traj.shape[0]
        if center_origin:
            traj['x']=traj['x']-midx; traj['y']=traj['y']-midy
        ax.plot(traj['x'],traj['y'],ls='-',color=color_list[i%7])
        ax.plot(traj.loc[traj_length,'x'],traj.loc[traj_length,'y'],ls='none',marker='.',c='k')
    centered='centered' if center_origin else ''
    filename=osp.join(data_dir,'superimposed'+fn_end+centered+'.svg')
    fig.savefig(filename, dpi=300)
    close(fig)

def traj_plot(data_dir):
    df,lengthscale,timescale,columns,dim=get_data(data_dir,refresh=False)
    traj_list=raw_input('give the list of traj you want to plot (sep: comas) if none given they will all be plotted: ')
    traj_list=traj_list.split(',')
    if len(traj_list)==0:
        traj_list=df['traj'].values
    else:
        traj_list=[int(t) for t in traj_list]
    plot_superimposed_traj(df,data_dir,traj_list)

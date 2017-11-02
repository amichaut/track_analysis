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

plt.style.use('ggplot') # Make the graphs a bit prettier

color_list=[c['color'] for c in list(plt.rcParams['axes.prop_cycle'])]

welcome_message="""\n\n WELCOME TO TRACK_ANALYSIS \n Developped and maintained by Arthur Michaut: arthur.michaut@gmail.com \n Released on 10-14-2017\n\n\n     _''_\n    / o  \\\n  <       |\n    \\    /__\n    /       \\-----\n    /    \\    \\   \\__\n    |     \\_____\\  __>\n     \\--       ___/  \n        \\     /\n         || ||\n         /\\ /\\\n\n"""
usage_message="""Usage: \n- plot cells analysis using cell_analysis(data_dir,refresh,parallelize,plot_traj,hide_labels,no_bkg,linewidth) \t data_dir: data directory, refresh (default False) to refresh the table values, parallelize (default False) to run analyses in parallel, 
plot_traj (default true) to print the cell trajectories, hide_labels (default True) to hide the cell label, no_bkg (default False) to remove the image background, linewidth being the trajectories width (default=1.0) \n
- plot maps using map_analysis(data_dir,refresh,parallelize,x_grid_size,no_bkg,z0,dimensions,axis_on) \t data_dir: data directory, refresh (default False) to refresh the table values, parallelize (default False) to run analyses in parallel, 
x_grid_size: number of columns in the grid (default 10), no_bkg (default False) to remove the image background, z0: altitude of the z_flow surface (default None => center of z axis), dimensions ([row,column] default None) to give the image dimension in case of no_bkg, axis_on: display axes along maps (default False) 
- plot average ROIs using avg_ROIs(data_dir,frame_subset=None,selection_frame=None,ROI_list=None,plot_on_map=True,plot_section=True,cumulative_plot=True,avg_plot=True) \t data_dir: data directory, frame_subset is a list [first,last], default None: open interactive choice"""


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

    #relative z: centered around mean
    df['z_rel']=df['z_scaled']-df['z_scaled'].mean()
    
def get_info(data_dir):
    """info.txt gives the lengthscale in um/px, the frame intervalle delta_t in min and the column names of the table"""
    filename=osp.join(data_dir,"info.txt")
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

def get_data(data_dir,refresh=False):
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
        #scale data
        dimensions=['x','y','z'] if 'z' in columns else ['x','y']
        dim=len(dimensions)
        scale_dim(df,dimensions,timescale,lengthscale)
        compute_parameters(df)
        #update pickle
        pickle.dump([df,lengthscale,timescale,columns,dim], open( osp.join(data_dir,"data_base.p"), "wb" ) )
    else:
        [df,lengthscale,timescale,columns,dim]=pickle.load( open( pickle_fn, "rb" ))
    
    return df,lengthscale,timescale,columns,dim

def get_obj_traj(track_groups,track,max_frame=None):
    '''gets the trajectory of an object. track_groups is the output of a groupby(['relabel'])'''
    group=track_groups.get_group(track)
    trajectory=group[['frame','t','x','y','z','z_scaled','z_rel','v']].copy()
    if max_frame is not None:
        trajectory=trajectory[trajectory['frame']<=max_frame]
    return trajectory.reset_index(drop=True)

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
    return fig,ax,xmin,ymin,xmax,ymax

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
    node_grid=meshgrid(arange(xmin,xmax+step,step),arange(ymin,ymax+step,step))
    center_grid=meshgrid(arange(xmin+step/2.,xmax,step),arange(ymin+step/2.,ymax,step))
    return node_grid,center_grid

def compute_vfield(df,frame,groups,data_dir,grids=None):
    print '\rcomputing velocity field '+str(frame),
    sys.stdout.flush()

    plot_dir=osp.join(data_dir,'vfield')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    dim=2 if 'z' not in df.columns else 3 #2d or 3D
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

def compute_div(df,frame,groups,data_dir,grids,lengthscale):
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

def compute_mean_vel(df,frame,groups,data_dir,grids):
    print '\rcomputing mean velocity field '+str(frame),
    sys.stdout.flush()

    plot_dir=osp.join(data_dir,'mean_vel')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    #get avg_vfield
    data=get_map_data(osp.join(data_dir,'vfield'),frame)
    avg_vfield=data[2:]

    dim=2 if 'z' not in df.columns else 3 #2d or 3D

    #compute avg
    V=0
    for k in range(dim):
        V+=avg_vfield[k]**2
    mean_vel=sqrt(V)

    #save data in pickle
    X,Y=grids[1]
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

def get_vlim(df,compute_func,groups,data_dir,grids,show_hist=False,**kwargs):
    vmin=np.nan;vmax=np.nan #boudaries of colorbar
    for i,frame in enumerate(df['frame'].unique()):
        data=compute_func(df,frame,groups,data_dir,grids,**kwargs)
        data=data[-1]
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
    if osp.isdir(image_dir1)==False:
        image_dir1=osp.join(data_dir,map_kind)

    image_dir=osp.join(data_dir,map_kind)
    not_found=True
    while not_found:
        if ROI_list is None:
            ROI_list=get_ROI(image_dir1,frame,tool=RectangleTool)
        data=get_map_data(image_dir,frame)
        X=data[0];Y=data[1];data=data[-1]
        ROI_data_list=[]
        square_ROI=False
        for ROI in ROI_list:
            x,y,dat,square_ROI=get_subblock_data(X,Y,data,ROI)
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

def compute_XY_flow(df,data_dir,line,orientation,frame,groups,window_size=None,timescale=1.,lengthscale=1.,z_depth=None,reg_data_bin=1.):
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

    # #check if I on line
    ind=((group['intersec_x']>=min(x1,x2)) & (group['intersec_x']<=max(x1,x2)) & (group['intersec_y']>=min(y1,y2))&(group['intersec_y']<=max(y1,y2)))
    group=group[ind]

    #check if crossing line 
    group['intersec_vecx']=group['intersec_x']-group['x_prev']#vector I-x between intersection and previous point
    group['intersec_vecy']=group['intersec_y']-group['y_prev']
    group['converging']=group['intersec_vecx']*group['displ_x']+group['intersec_vecy']*group['displ_y'] #scalar product between (I-x) and displacement vectors. 
    group=group[group['converging']>0] #if >0 converging towards line. If not discard because crossing is impossible
    group['crossing']=group['displ_x']**2+group['displ_y']**2/(group['intersec_vecx']**2+group['intersec_vecy']**2) #|displ|^2/|I-x|^2
    group=group[group['crossing']>=1] #if displacement > distance to surface

    #compute orientation
    group['orientation']=group['displ_x']*(Ox2-Ox1)+group['displ_y']*(Oy2-Oy1) #scalar product between O and displ
    group['orientation']=group['orientation'].apply(lambda x:1 if x>0 else -1)

    #compute distance to line end
    group['line_abscissa']=np.sqrt((group['intersec_x']-x1)**2+(group['intersec_y']-y1)**2)

    #return data
    data=group[['line_abscissa','orientation']].values
    if data.shape[0]==0:
        return data
    data[:,0]/=lengthscale #rescale data in um
    abs_length = np.sqrt((line[1][0]-line[0][0])**2+(line[1][1]-line[0][1])**2)/lengthscale
    axis_data=arange(0,abs_length)
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
        reg_data[ind,1]=data[i,1]

    #rolling
    reg_data=pd.DataFrame(reg_data,columns=list('xy'))
    reg_data['y'].rolling(window_size).mean()
    reg_data=reg_data[reg_data['y']!=0]

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

def plot_cells(df_list,groups_list,frame,data_dir,plot_traj=False,z_lim=[],hide_labels=False,no_bkg=False,lengthscale=1.,length_ref=0.75,display=False):
    """ Print the tracked pictures with updated (=relinked) tracks"""
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
    fig,ax,xmin,ymin,xmax,ymax=get_background(df_list[0],data_dir,frame,no_bkg=no_bkg)
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
                ax.text(x,y,s,fontsize=5,color='w')
            if plot_traj:
                #traj size
                lw_ref=rcParams['lines.linewidth']
                ms_ref=rcParams['lines.markersize']
                size_factor=lengthscale*length_ref
                lw=lw_ref*size_factor; ms=ms_ref*size_factor
                #plot trajectory
                traj=get_obj_traj(track_groups,track,max_frame=frame)
                traj_length,c=traj.shape
                if traj_length>1:
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
    if display:
        plt.show()
        return                

    filename=osp.join(plot_dir,'%04d.png'%int(frame))
    fig.savefig(filename, dpi=300)
    close('all')

def plot_vfield(df,frame,data_dir,no_bkg=False,vlim=None,axis_on=False):
    """ Plot velocity field and compute avg vfield on a grid"""
    close('all')
    print '\rplotting velocity field '+str(frame),
    sys.stdout.flush()
    plot_dir=osp.join(data_dir,'vfield')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    #import image
    fig,ax,xmin,ymin,xmax,ymax=get_background(df,data_dir,frame,no_bkg=no_bkg,axis_on=axis_on)
    data=get_map_data(plot_dir,frame)
    norm=plt.Normalize(vlim[0],vlim[1]) if vlim is not None else None
    Q=quiver(*data,units='x',cmap='plasma',norm=norm)

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
    fig,ax,xmin,ymin,xmax,ymax=get_background(df,data_dir,frame,no_bkg=no_bkg,axis_on=axis_on)
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

def plot_mean_vel(df,frame,data_dir,no_bkg=False,vlim=None,axis_on=False):
    close('all')
    print '\rplotting mean velocity '+str(frame),
    sys.stdout.flush()
    plot_dir=osp.join(data_dir,'mean_vel')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    #import image
    fig,ax,xmin,ymin,xmax,ymax=get_background(df,data_dir,frame,no_bkg=no_bkg,axis_on=axis_on)
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
    filename=osp.join(plot_dir,'%04d.png'%int(frame))
    fig.savefig(filename, dpi=300)
    close()

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
    
    fig,ax,xmin,ymin,xmax,ymax=get_background(df,data_dir,frame,no_bkg=no_bkg,axis_on=axis_on)
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
    plot_dir=osp.join(data_dir,map_kind,'ROI_avg')
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    if plot_on_map:
        fig,ax,xmin,ymin,xmax,ymax=get_background(df,data_dir,frame)

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
    ax.scatter(data[:,0],data[:,1])
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(0,abs_length)
    filename=filename_prefix+'_frame_%04d.png'%frame
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    close()

    if plot_on_map:
        fig,ax,xmin,ymin,xmax,ymax=get_background(df,data_dir,frame)
        ax.plot(line[:,0],line[:,1])
        ax.arrow(orientation[0,0],orientation[0,1],orientation[1,0]-orientation[0,0],orientation[1,1]-orientation[0,1],shape='full',length_includes_head=True,width=10,color='k')
        filename=osp.join(plot_dir,filename_prefix+'section.png')
        fig.savefig(filename,dpi=300,bbox_inches='tight')
        close()

    return plot_data


#### PLOT_ALL methods


def plot_all_cells(df_list,data_dir,plot_traj=False,z_lim=[],hide_labels=False,no_bkg=False,parallelize=False,lengthscale=1.,length_ref=0.75):
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
            plot_cells(df_list,groups_list,frame,data_dir,plot_traj,z_lim,hide_labels,no_bkg,lengthscale,length_ref)

def plot_all_vfield(df,data_dir,grids=None,no_bkg=False,parallelize=False,refresh=False,axis_on=False):
    groups=df.groupby('frame')
    dim=2 if 'z' not in df.columns else 3 #2d or 3D
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
        #     vlim=get_vlim(df,compute_vfield,groups,data_dir,grids)
        #     pickle.dump(vlim,open(osp.join(plot_dir,'data','vlim.p'),"wb"))

        vlim=get_vlim(df,compute_vfield,groups,data_dir,grids)
        pickle.dump(vlim,open(osp.join(plot_dir,'data','vlim.p'),"wb"))

    vlim=pickle.load( open(osp.join(plot_dir,'data','vlim.p'), "rb" ))
    #plot colorbar
    if dim==3:
        plot_cmap(plot_dir,'$v_z\ (\mu m.min^{-1})$',cm.plasma,vlim[0],vlim[1])

    #plot maps
    if parallelize:
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(plot_vfield)(df,frame,data_dir,no_bkg,vlim) for frame in df['frame'].unique())
    else:
        for i,frame in enumerate(df['frame'].unique()):
            plot_vfield(df,frame,data_dir,no_bkg=no_bkg,vlim=vlim,axis_on=axis_on)

def plot_all_maps(df,data_dir,grids,map_kind,refresh=False,no_bkg=False,parallelize=False,manual_vlim=False,axis_on=False,**kwargs):

    groups=df.groupby('frame')

    plot_dir=osp.join(data_dir,map_kind)
    if osp.isdir(plot_dir)==False:
        os.mkdir(plot_dir)

    if osp.isdir(osp.join(plot_dir,'data')) is False:
        refresh=True
    if refresh:
        #compute data
        if parallelize:
            num_cores = multiprocessing.cpu_count()
            Parallel(n_jobs=num_cores)(delayed(map_dic[map_kind]['compute_func'])(df,groups,frame,data_dir,grids,lengthscale) for frame in df['frame'].unique())
        else:
            vlim=get_vlim(df,map_dic[map_kind]['compute_func'],groups,data_dir,grids,show_hist=manual_vlim,**kwargs)
            pickle.dump(vlim,open(osp.join(plot_dir,'data','vlim.p'),"wb"))

    vlim=pickle.load( open(osp.join(plot_dir,'data','vlim.p'), "rb" ))
    #plot colorbar
    plot_cmap(plot_dir,map_dic[map_kind]['cmap_label'],cm.plasma,vlim[0],vlim[1])

    #plot maps
    if parallelize:
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(plot_z_flow)(df,frame,data_dir,no_bkg,vlim) for frame in df['frame'].unique())
    else:
        for i,frame in enumerate(df['frame'].unique()):
            map_dic[map_kind]['plot_func'](df,frame,data_dir,no_bkg,vlim,axis_on)

def plot_all_avg_ROI(df,data_dir,map_kind,frame_subset=None,selection_frame=None,ROI_list=None,plot_on_map=False,plot_section=True,cumulative_plot=True,avg_plot=True,timescale=1.):

    close('all')

    plot_dir=osp.join(data_dir,map_kind,'ROI_avg')
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
        ax.scatter(x_data_l[j],y_data_l[j],color=get_cmap_color(time, cm.plasma, vmin=time_min, vmax=time_max))
    ax.set_title(plot_data_list[0]['title'])
    ax.set_xlabel(plot_data_list[0]['xlab'])
    ax.set_ylabel(plot_data_list[0]['ylab'])
    cb=fig.colorbar(CS3)
    cb.set_label(label='time (min)')
    filename=plot_data_list[0]['filename_prefix']+'_cumulative.png'
    fig.savefig(filename,dpi=300,bbox_inches='tight')
    close()

    # #average plot ==> need to interpolate to average
    # x_data=plot_data_list[0]['x_data']
    # y_data_l=[p['y_data'] for p in plot_data_list]
    # fig, ax = plt.subplots(1, 1)
    # sns.tsplot(y_data_l,time=x_data,ax=ax,estimator=np.nanmean,err_style="unit_traces")
    # ax.set_title(plot_data_list[0]['title'])
    # ax.set_xlabel(plot_data_list[0]['xlab'])
    # ax.set_ylabel(plot_data_list[0]['ylab'])
    # filename=plot_data_list[0]['filename_prefix']+'_average.png'
    # fig.savefig(filename,dpi=300,bbox_inches='tight')
    # close()



#################################################################
###########   CONTAINER METHODS   ###############################
#################################################################

def cell_analysis(data_dir,refresh=False,parallelize=False,plot_traj=True,hide_labels=True,no_bkg=False,linewidth=1.):
    df,lengthscale,timescale,columns,dim=get_data(data_dir,refresh=refresh)
    z_lim=[df['z_rel'].min(),df['z_rel'].max()] if dim==3 else []
    df_list=[]

    subset=raw_input('By what do you want to filter? Type: none, ROI, track_length. \t ')
    if subset=='none':
        df_list=[df]
    elif subset=='ROI':
        df_list=filter_by_ROI(df,data_dir)
    elif subset=='track_length':
        min_traj_len=input('Give the minimum track length? (number of frames): ')
        df_list=[filter_by_traj_len(df,min_traj_len=min_traj_len)]
    else:
        print 'ERROR: not a valid answer'
        return
    
    plot_all_cells(df_list,data_dir,plot_traj=plot_traj,z_lim=z_lim,hide_labels=hide_labels,no_bkg=no_bkg,parallelize=parallelize,lengthscale=lengthscale,length_ref=0.75/linewidth)

def map_analysis(data_dir,refresh=False,parallelize=False,x_grid_size=10,no_bkg=False,z0=None,dimensions=None,axis_on=False):
    df,lengthscale,timescale,columns,dim=get_data(data_dir,refresh=refresh)
    grids=make_grid(x_grid_size,data_dir,dimensions=dimensions)
    plot_all_vfield(df,data_dir,grids=grids,no_bkg=no_bkg,parallelize=parallelize,refresh=refresh,axis_on=axis_on)
    if grids is not None:
        plot_all_maps(df,data_dir,grids,'div',refresh=refresh,no_bkg=no_bkg,parallelize=parallelize,axis_on=axis_on,lengthscale=lengthscale)
        plot_all_maps(df,data_dir,grids,'mean_vel',refresh=refresh,no_bkg=no_bkg,parallelize=parallelize,manual_vlim=True,axis_on=axis_on)
        if z0 is None:
            z0= df['z_rel'].min() + (df['z_rel'].max()-df['z_rel'].min())/2.
            print 'z0=%f'%z0
        plot_all_maps(df,data_dir,grids,'z_flow',refresh=refresh,no_bkg=no_bkg,parallelize=parallelize,z0=z0,timescale=timescale)

def avg_ROIs(data_dir,frame_subset=None,selection_frame=None,ROI_list=None,plot_on_map=True,plot_section=True,cumulative_plot=True,avg_plot=True,refresh=False):
    df,lengthscale,timescale,columns,dim=get_data(data_dir,refresh=refresh)
    map_kind=raw_input("Give the map wou want to plot your ROIs on (div,mean_vel,z_flow,vfield): ")
    plot_all_avg_ROI(df,data_dir,map_kind,frame_subset=frame_subset,selection_frame=selection_frame,ROI_list=ROI_list,plot_on_map=plot_on_map,plot_section=plot_section,cumulative_plot=cumulative_plot,avg_plot=avg_plot,timescale=timescale)

def XY_flow(data_dir,window_size=None,refresh=False,line=None,orientation=None,frame_subset=None,selection_frame=None,z_depth=None):
    df,lengthscale,timescale,columns,dim=get_data(data_dir,refresh=refresh)
    
    if z_depth is None:
        z_lim=[df['z_rel'].min(),df['z_rel'].max()] if dim==3 else []
        z_depth=None if len(z_lim)==0 else z_lim[1]-z_lim[0]

    if window_size is None:
        window_size=input("Give the window size you want to calculate the flow on (in um): ")
    
    plot_all_XY_flow(df,data_dir,line=line,orientation=orientation,frame_subset=frame_subset,window_size=window_size,selection_frame=selection_frame,timescale=timescale,lengthscale=lengthscale,z_depth=z_depth)



###############################################

map_dic={'div':{'compute_func':compute_div,'plot_func':plot_div,'cmap_label':'$div(\overrightarrow{v})\ (min^{-1})$'},
     'mean_vel':{'compute_func':compute_mean_vel,'plot_func':plot_mean_vel,'cmap_label':'$v\ (\mu m.min^{-1})$'},
     'z_flow':{'compute_func':compute_z_flow,'plot_func':plot_z_flow,'cmap_label':'cell flow $(min^{-1})$'},
     'vfield':{'compute_func':compute_vfield,'plot_func':plot_vfield,'cmap_label':'$v_z\ (\mu m.min^{-1})$'}}


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
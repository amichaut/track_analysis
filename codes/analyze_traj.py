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
import multiprocessing
from joblib import Parallel, delayed

plt.style.use('ggplot') # Make the graphs a bit prettier

color_list=[c['color'] for c in list(plt.rcParams['axes.prop_cycle'])]


#################################################################
###########   PREPARE METHODS   #################################
#################################################################

def get_cmap_color(value, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(value))

def scale_dim(df,dimensions=['x','y','z'],timescale=1.,lengthscale=1.):
    #time
    df['t']=df['frame']*timescale
    #length 
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

def get_avg_vfields(df,data_dir,grids=None,refresh=False):
    refresh=False
    pickle_fn=osp.join(data_dir,'avg_fields.p')
    #check pickle exists
    if osp.exists(pickle_fn)==False:
        refresh=True
        
    if refresh:
        if grids is None:
            print "ERROR: provide a grid to calculate field"
            return
        plot_all_frame(plot_vfield,df,data_dir,parallelize=True,grids=grids,plot_field=False)
    else:
        avg_vfields=pickle.load(open(pickle_fn,"rb"))
    
    return avg_vfields

def get_data(data_dir,refresh=False,plot_frame=True,plot_data=True,plot_modified_tracks=False,plot_all_traj=False):
    #import
    pickle_fn=osp.join(data_dir,"data_base.p")
    if osp.exists(pickle_fn)==False or refresh:
        #data=loadtxt(osp.join(data_dir,'test_data.txt'))
        data=loadtxt(osp.join(data_dir,'table.txt'))
        info=get_info(data_dir)
        for inf in ['lengthscale','delta_t','columns']:
            if info[inf]==-1:
                print "WARNING: "+inf+" not provided in info.txt"
        lengthscale=info["lengthscale"];timescale=info["delta_t"];columns=info["columns"]
        df=pd.DataFrame(data[:,1:],columns=columns) 
        #scale data
        dimensions=['x','y','z'] if 'z' in columns else ['x','y']
        scale_dim(df,dimensions,timescale,lengthscale)
        compute_parameters(df)
        #update pickle
        pickle.dump([df,lengthscale,timescale,columns], open( osp.join(data_dir,"data_base.p"), "wb" ) )
    else:
        [df,lengthscale,timescale,columns]=pickle.load( open( pickle_fn, "rb" ))
    
    return df,lengthscale,timescale,columns

def get_vlim(plot_func,df,data_dir,avg_vfields,**kwargs):
    groups=df.groupby('frame')
    vmin=1000;vmax=-1000
    for i,frame in enumerate(avg_vfields['frame_list']):
        data=plot_func(df,groups,frame,data_dir,avg_vfields['vfield_list'][i],plot_field=False,**kwargs)
        data = np.ma.array (data, mask=np.isnan(data))
        vmin=min(vmin,data.min());vmax=max(vmax,data.max())
    return [vmin,vmax]

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

def get_background(df,data_dir,frame,no_bkg=False):
    """Get image background or create white backgound if no_bkg"""
    if not osp.exists(osp.join(data_dir,'raw')): #check if images exist 
       no_bkg=True
    if no_bkg:
        #get approximative image size
        m=int(df['x'].max());n=int(df['y'].max())
        im = ones((n,m,3)) #create white background ==> not ideal, it would be better not to use imshow and to modify axes rendering
    else:
        filename=osp.join(data_dir,'raw/%04d.png'%int(frame))
        im = io.imread(filename)
        n=im.shape[0]; m=im.shape[1]
    fig=figure(frameon=False)
    fig.set_size_inches(m/300,n/300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im,aspect='auto',origin='lower')
    xmin, ymin, xmax, ymax=ax.axis('off')
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
    center_grid=meshgrid(arange(xmin+step/2,xmax,step),arange(ymin+step/2,ymax,step))
    return node_grid,center_grid

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
    fig.savefig(filename, dpi=300)
    close('all')

def plot_cells(df,groups,frame,data_dir,plot_traj=False,z_lim=[],hide_labels=False,no_bkg=False):
    """ Print the tracked pictures with updated (=relinked) tracks"""
    print '\rplotting frame '+str(frame),
    sys.stdout.flush()

    track_dir=osp.join(data_dir,'traj')
    if osp.isdir(track_dir)==False:
        os.mkdir(track_dir)

    group=groups.get_group(frame).reset_index(drop=True)
    r,c=group.shape

    if plot_traj:
        track_groups=df.groupby(['traj'])
    z_labeling=False
    if len(z_lim)>0:
        z_labeling=True
        z_map=linspace(z_lim[0],z_lim[1],1000)
        #create z color map
        fig2=figure()
        Z = [[0,0],[0,0]]
        cont = plt.contourf(Z, z_map, cmap=cm.plasma)
        plt.close(fig2)

    #import image
    fig,ax,xmin,ymin,xmax,ymax=get_background(df,data_dir,frame,no_bkg=no_bkg)
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
                    X=traj['x'].values;Y=traj['y'].values;Z=traj['z_rel'].values; #convert to numpy to optimize speed
                    for j in range(1,traj_length):
                        ax.plot([X[j-1],X[j]],[Y[j-1],Y[j]],ls='-',color=get_cmap_color(Z[j],cm.plasma,vmin=z_lim[0],vmax=z_lim[1]))
                    ax.plot(X[traj_length-1],Y[traj_length-1],marker='.',color=get_cmap_color(Z[j],cm.plasma,vmin=z_lim[0],vmax=z_lim[1]))
                else:
                    ax.plot(traj['x'],traj['y'],ls='-',color=color_list[track%7])
                    ax.plot(traj['x'].values[-1],traj['y'].values[-1],marker='.',color=color_list[track%7])
                ax.axis([xmin, ymin, xmax, ymax])

    if z_labeling:
        cbaxes = fig.add_axes([0.4, 0.935, 0.025, 0.05])
        cbar = fig.colorbar(cont,cax = cbaxes,label='$z\ (\mu m)$')
        cbaxes.tick_params(labelsize=5,color='w')
        cbaxes.yaxis.label.set_color('white')
    filename=osp.join(track_dir,'traj_%04d.png'%int(frame))
    fig.savefig(filename, dpi=300)
    close('all')

def plot_vfield(df,groups,frame,data_dir,grids=None,plot_field=True,no_bkg=False):
    """ Plot velocity field and compute avg vfield on a grid"""
    close('all')
    print '\rplotting frame '+str(frame),
    sys.stdout.flush()
    track_dir=osp.join(data_dir,'vfield')
    if osp.isdir(track_dir)==False:
        os.mkdir(track_dir)

    dim=2 if 'z' not in df.columns else 3 #2d or 3D
    coord_list=['vx','vy','vz']
        
    group=groups.get_group(frame).reset_index(drop=True)

    #import image
    fig,ax,xmin,ymin,xmax,ymax=get_background(df,data_dir,frame,no_bkg=no_bkg)
    #plot quiver
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
        Q=quiver(x,y,*v_field,units='x',cmap='plasma')
    else:
        v_field=[group[coord_list[k]] for k in range(dim)]
        Q=quiver(group['x'],group['y'],*v_field,units='x',cmap='plasma')

    if plot_field:
        if dim==3:
            cbaxes = fig.add_axes([0.4, 0.935, 0.025, 0.05])
            cbar = fig.colorbar(Q,cax = cbaxes,label='$v_z\ (\mu m.min^{-1})$')
            cbaxes.tick_params(labelsize=5,color='w')
            cbaxes.yaxis.label.set_color('white')
        ax.axis([xmin, ymin, xmax, ymax])
        filename=osp.join(track_dir,'vfield_%04d.png'%int(frame)) if grids is None else osp.join(track_dir,'avg_vfield_%04d.png'%int(frame))
        fig.savefig(filename, dpi=600)
    close()
    
    #save data in pickle
    pickle_fn=osp.join(data_dir,'avg_fields.p')
    if osp.exists(pickle_fn)==False:
        data={str(frame):v_field}
        pickle.dump(data,open(pickle_fn,"wb"))
    else:
        data=pickle.load(open(pickle_fn,"rb"))
        data[str(frame)]=v_field
        pickle.dump(data,open(pickle_fn,"wb"))

def plot_div(df,groups,frame,data_dir,grids,lengthscale,fixed_vlim=None,plot_field=True,no_bkg=False):
    """ Plot 2D divergence"""
    close('all')
    print '\rplotting frame '+str(frame),
    sys.stdout.flush()
    track_dir=osp.join(data_dir,'divergence')
    if osp.isdir(track_dir)==False:
        os.mkdir(track_dir)

    #get avg_vfield
    avg_vfields=get_avg_vfields(df,data_dir)
    avg_vfield=avg_vfields[str(frame)]

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

    if plot_field:
        fig,ax,xmin,ymin,xmax,ymax=get_background(df,data_dir,frame,no_bkg=no_bkg)
        div_masked = np.ma.array (div, mask=np.isnan(div))
        if fixed_vlim is not None:
            vmin=fixed_vlim[0];vmax=fixed_vlim[1]
        else:
            vmin=div_masked.min();vmax=div_masked.max()
        cmap=cm.plasma; cmap.set_bad('w',alpha=0) #set NAN transparent
        C=ax.pcolormesh(x[1:-1,1:-1],y[1:-1,1:-1],div_masked[1:-1,1:-1],cmap=cmap,alpha=0.5,vmin=vmin,vmax=vmax)
        cbaxes = fig.add_axes([0.4, 0.935, 0.025, 0.05])
        cbar = fig.colorbar(C,cax = cbaxes,label='$div(\overrightarrow{v})\ (min^{-1})$')
        cbaxes.tick_params(labelsize=5,color='w')
        cbaxes.yaxis.label.set_color('white')
        ax.axis([xmin, ymin, xmax, ymax])
        filename=osp.join(track_dir,'div_%04d.png'%int(frame))
        fig.savefig(filename, dpi=300)
    close()
    return div

def plot_mean_vel(df,groups,frame,data_dir,grids,fixed_vlim=None,plot_field=True,no_bkg=False):
    close('all')
    print '\rplotting frame '+str(frame),
    sys.stdout.flush()
    track_dir=osp.join(data_dir,'mean_vel')
    if osp.isdir(track_dir)==False:
        os.mkdir(track_dir)

    #get avg_vfield
    avg_vfields=get_avg_vfields(df,data_dir)
    avg_vfield=avg_vfields[str(frame)]

    dim=2 if 'z' not in df.columns else 3 #2d or 3D

    #compute avg
    V=0
    for k in range(dim):
        V+=avg_vfield[k]**2
    mean_vel=sqrt(V)

    X,Y=grids[1]

    if plot_field:
    	fig,ax,xmin,ymin,xmax,ymax=get_background(df,data_dir,frame,no_bkg=no_bkg)
        mean_vel_masked = np.ma.array (mean_vel, mask=np.isnan(mean_vel))
        if fixed_vlim is not None:
            vmin=fixed_vlim[0];vmax=fixed_vlim[1]
        else:
            vmin=mean_vel_masked.min();vmax=mean_vel_masked.max()
        cmap=cm.plasma; cmap.set_bad('w',alpha=0) #set NAN transparent
        C=ax.pcolormesh(X,Y,mean_vel_masked,cmap=cmap,alpha=0.5,vmin=vmin,vmax=vmax)
        cbaxes = fig.add_axes([0.4, 0.935, 0.025, 0.05])
        cbar = fig.colorbar(C,cax = cbaxes,label='$v\ (\mu m.min^{-1})$')
        cbaxes.tick_params(labelsize=5,color='w')
        cbaxes.yaxis.label.set_color('white')
        ax.axis([xmin, ymin, xmax, ymax])
        filename=osp.join(track_dir,'mean_vel_%04d.png'%int(frame))
        fig.savefig(filename, dpi=300)
    close()
    return mean_vel

def plot_z_flow(df,groups,frame,data_dir,grids,z0,plot_field=True,no_bkg=False):
    """Plot the flow (defined as the net number of cells going through a surface element in the increasing z direction) through the plane of z=z0"""
    
    #Make sure these are 3D data
    if 'z' not in df.columns:
        print "Not a 3D set of data"
        return

    close('all')
    print '\rplotting frame '+str(frame),
    sys.stdout.flush()
    track_dir=osp.join(data_dir,'z_flow')
    if osp.isdir(track_dir)==False:
        os.mkdir(track_dir)

    group=groups.get_group(frame).reset_index(drop=True)

    df_layer=group[abs(group['vz'])>=abs(z0-group['z_rel'])] #layer of cells crossing the surface
    df_ascending=df_layer[((df_layer['vz']>=0) & (df_layer['z_rel']<=z0))] #ascending cells below the surface
    df_descending=df_layer[((df_layer['vz']<=0) & (df_layer['z_rel']>=z0))] #descending cells above the surface
    
    #calculate the intersection coordinates (x0,y0) of the vector and the surface (calculate homothety coefficient alpha)
    for df_ in [df_ascending,df_descending]:
        df_['alpha']=(z0-df_['z_rel'])/df_['vz']
        df_['x0']=df_['x']+df_['alpha']*df_['vx']
        df_['y0']=df_['y']+df_['alpha']*df_['vy']
    
    node_grid,center_grid=grids   
    X,Y=node_grid
    x,y=center_grid
    flow = zeros((x.shape[0],x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ind_asc=((df_ascending['x0']>=X[i,j]) & (df_ascending['x0']<X[i,j+1]) & (df_ascending['y0']>=Y[i,j]) & (df_ascending['y0']<Y[i+1,j]))
            ind_des=((df_descending['x0']>=X[i,j]) & (df_descending['x0']<X[i,j+1]) & (df_descending['y0']>=Y[i,j]) & (df_descending['y0']<Y[i+1,j]))
            flow[i,j]=df_ascending[ind_asc].shape[0]-df_descending[ind_des].shape[0]
    
    if plot_field:
        fig,ax,xmin,ymin,xmax,ymax=get_background(df,data_dir,frame,no_bkg=no_bkg)
        vmin=flow.min();vmax=flow.max()
        cmap=cm.plasma
        C=ax.pcolormesh(x,y,flow,cmap=cmap,alpha=0.5,vmin=vmin,vmax=vmax)
        cbaxes = fig.add_axes([0.4, 0.935, 0.025, 0.05])
        cbar = fig.colorbar(C,cax = cbaxes,label='cell flow $(min^{-1})$')
        cbaxes.tick_params(labelsize=5,color='w')
        cbaxes.yaxis.label.set_color('white')
        ax.axis([xmin, ymin, xmax, ymax])
        filename=osp.join(track_dir,'flow_%04d.png'%int(frame))
        fig.savefig(filename, dpi=300)
    close()
    
    return flow

def plot_all_frame(plot_func,df,data_dir,parallelize=True,**kwargs):
    groups=df.groupby('frame')
    if parallelize:
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(plot_func)(df,groups,frame,data_dir,**kwargs) for frame in df['frame'].unique())
    else:
        for frame in df['frame'].unique():
            plot_func(df,groups,frame,data_dir,**kwargs)

#################################################################
###########   CONTAINER METHODS   ###############################
#################################################################

def cell_analysis(data_dir,refresh=False,min_traj_len=24,parallelize=False,x_grid_size=10,plot_traj=True,hide_labels=True,no_bkg=False,z0=None,dimensions=None):
    df,lengthscale,timescale,columns=get_data(data_dir,refresh=refresh)
    df2=filter_by_traj_len(df,min_traj_len=min_traj_len)
    print "plotting cells trajectories"
    z_lim=[df['z_rel'].min(),df['z_rel'].max()]
    plot_all_frame(plot_cells,df2,data_dir,parallelize=parallelize,plot_traj=plot_traj,z_lim=z_lim,hide_labels=hide_labels,no_bkg=no_bkg)

def map_analysis(data_dir,refresh=False,min_traj_len=24,parallelize=False,x_grid_size=10,plot_traj=True,hide_labels=True,no_bkg=False,z0=None,dimensions=None):
    df,lengthscale,timescale,columns=get_data(data_dir,refresh=refresh)
    print "plotting velocity fields"
    grids=make_grid(x_grid_size,data_dir,dimensions=dimensions)
    plot_all_frame(plot_vfield,df,data_dir,parallelize=parallelize,grids=grids,no_bkg=no_bkg)
    print "plotting divergence"
    plot_all_frame(plot_div,df,data_dir,parallelize=parallelize,grids=grids,lengthscale=lengthscale,no_bkg=no_bkg)
    print "plotting mean velocity map"
    plot_all_frame(plot_mean_vel,df,data_dir,parallelize=parallelize,grids=grids,no_bkg=no_bkg)
    print "plotting z flow map"
    if z0 is None:
        z0= z_lim[0] + (z_lim[1]-z_lim[0])/2.
    plot_all_frame(plot_z_flow,df,data_dir,parallelize=parallelize,grids=grids,z0=z0,no_bkg=no_bkg)


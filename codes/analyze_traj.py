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

    #relative z: centered around mean
    df['z_rel']=df['z_scaled']-df['z_scaled'].mean()
    
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

def get_avg_vfields(data_dir,df=None,avg_grid=10):
    refresh=False
    pickle_fn=osp.join(data_dir,'avg_fields.p')
    #check pickle exists
    if osp.exists(pickle_fn)==False:
        refresh=True
    if df is not None:
        refresh=True
        
    if refresh:
        avg_vfields={'avg_grid':avg_grid,'frame_list':[],'vfield_list':[]}
        groups=df.groupby('frame')
        for frame in df['frame'].unique():
            avg_vfields['frame_list'].append(frame)
            avg_vfields['vfield_list'].append(plot_vfield(df,groups,frame,data_dir,avg_grid=avg_grid,plot_field=False))
        #update pickle
        pickle.dump(avg_vfields,open(pickle_fn,"wb"))
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
        scale_dim(df,timescale,lengthscale)
        compute_parameters(df)
        #update pickle
        pickle.dump(df, open( osp.join(data_dir,"data_base.p"), "wb" ) )
    else:
        df=pickle.load( open( pickle_fn, "rb" ))
    
    return df

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

def get_background(df,dirdata,frame,no_bkg=False):
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
        n,m,d = im.shape
    fig=figure(frameon=False)
    fig.set_size_inches(m/300,n/300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im,aspect='auto',origin='lower')
    xmin, ymin, xmax, ymax=ax.axis('off')
    return fig,ax,xmin,ymin,xmax,ymax

def make_grid(xres,dimensions=None,lengthscale=None):
    """make a meshgrid. The boundaries can be passed by dimensions as [xmin,xmax,ymin,ymax] or using the raw image dimensions. xres is the number of cells in the grid along the x axis.
	It returns two grids: the node_grid with the positions of the nodes of each cells, and the center_grid with the position of the center of each cell"""
    if dimensions is None:
        if not osp.exists(osp.join(data_dir,'raw')):
            print """ERROR: the grid can't be created, no dimensions are available"""
            return
        else:
            filename=osp.join(data_dir,'raw/%04d.png'%int(frame))
            im = io.imread(filename)
            ymax,xmax,d = im.shape
            xmin=0;ymin=0
            ymax/=lengthscale; xmax/=lengthscale
    else:
        [xmin,xmax,ymin,ymax] = dimensions

    step=float(xmax-xmin)/xres
    node_grid=meshgrid(arange(xmin,xmax+step,step),arange(ymin,ymax+step,step))
    center_grid=meshgrid(arange(xmin+step/2,xmax,step),arange(ymin+step/2,ymax,step))
    return node_grid,center_grid

#################################################################
###########   PLOT METHODS   ####################################
#################################################################

def plot_cells(df,groups,frame,data_dir,plot_traj=False,z_lim=[],filtered_tracks=None,hide_labels=False,no_bkg=False):
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
    fig,ax,xmin,ymin,xmax,ymax=get_background(df,dirdata,frame,no_bkg=no_bkg)
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
                        X=traj['x'].values;Y=traj['y'].values;Z=traj['z_rel'].values; #convert to numpy to optimize speed
                        for j in range(1,traj_length):
                            ax.plot([X[j-1],X[j]],[Y[j-1],Y[j]],color=get_cmap_color(Z[j],cm.plasma, vmin=z_lim[0], vmax=z_lim[1]))
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
                        if z_labeling:
                            X=traj['x'].values;Y=traj['y'].values;Z=traj['z_rel'].values; #convert to numpy to optimize speed
                            for j in range(1,traj_length):
                                ax.plot([X[j-1],X[j]],[Y[j-1],Y[j]],color=get_cmap_color(Z[j],cm.plasma, vmin=z_lim[0], vmax=z_lim[1]))
                        else:
                            ax.plot(traj['x'],traj['y'],ls='-',color=color_list[track%7])
                    ax.axis([xmin, ymin, xmax, ymax])
    if z_labeling:
        cbaxes = fig.add_axes([0.4, 0.935, 0.025, 0.05])
        cbar = fig.colorbar(cont,cax = cbaxes,label='$z\ (\mu m)$')
        cbaxes.tick_params(labelsize=5,color='w')
        cbaxes.yaxis.label.set_color('white')
    filename=osp.join(track_dir,'traj_%04d.png'%int(frame))
    fig.savefig(filename, dpi=300)
    close('all')

def plot_vfield(df,groups,frame,data_dir,avg_grid=None,filtered_tracks=False,plot_field=True,no_bkg=False):
    """ Plot velocity field and compute avg vfield on a grid if avg_grid is a int giving the number of square in the X direction"""
    close('all')
    print '\rplotting frame '+str(frame),
    sys.stdout.flush()
    if filtered_tracks:
        track_dir=osp.join(data_dir,'vfield_subset')
    else:
        track_dir=osp.join(data_dir,'vfield')
        
    if osp.isdir(track_dir)==False:
        os.mkdir(track_dir)
    group=groups.get_group(frame).reset_index(drop=True)
    #import image
    fig,ax,xmin,ymin,xmax,ymax=get_background(df,dirdata,frame,no_bkg=no_bkg)
    #plot quiver
    if avg_grid is not None:
        if avg_grid<1:
            print "ERROR: avg_grid <1"
            return
        res = avg_grid+1 #number of bounds = nmuber of cells + 1
        xsubgrid=linspace(xmin,ymin,res)
        cell_size = xsubgrid[1]-xsubgrid[0]
        ysubgrid=arange(xmax,ymax,cell_size)
        X=[];Y=[];VX=[];VY=[];VZ=[] #new data
        for i,xg in enumerate(xsubgrid[:-1]):
            for j,yg in enumerate(ysubgrid[:-1]):
                xg1=xsubgrid[i+1];yg1=ysubgrid[j+1]
                ind=((group['x']>=xg) & (group['x']<xg1) & (group['y']>=yg) & (group['y']<yg1))
                VX.append(group[ind]['vx'].mean());VY.append(group[ind]['vy'].mean());VZ.append(group[ind]['vz'].mean())
                X.append(xg+(xg1-xg)*0.5);Y.append(yg+(yg1-yg)*0.5) #center of the cell
        
        avg_vfield = pd.DataFrame({'x':X,'y':Y,'vx':VX,'vy':VY,'vz':VZ})
        Q=quiver(avg_vfield['x'],avg_vfield['y'],avg_vfield['vx'],avg_vfield['vy'],avg_vfield['vz'],units='x',cmap='plasma')
    else:
        Q=quiver(group['x'],group['y'],group['vx'],group['vy'],group['vz'],units='x',cmap='plasma')

    if plot_field:
        cbaxes = fig.add_axes([0.4, 0.935, 0.025, 0.05])
        cbar = fig.colorbar(Q,cax = cbaxes,label='$v_z\ (\mu m.min^{-1})$')
        cbaxes.tick_params(labelsize=5,color='w')
        cbaxes.yaxis.label.set_color('white')
        ax.axis([xmin, ymin, xmax, ymax])
        filename=osp.join(track_dir,'vfield_%04d.png'%int(frame)) if avg_grid is None else osp.join(track_dir,'avg_vfield_%04d.png'%int(frame))
        fig.savefig(filename, dpi=600)
    close()
    if avg_grid is not None:
        return avg_vfield

def plot_div(df,groups,frame,data_dir,avg_vfield,fixed_vlim=None,plot_field=True,no_bkg=False):
    """ Plot 2D divergence"""
    close('all')
    print '\rplotting frame '+str(frame),
    sys.stdout.flush()
    track_dir=osp.join(data_dir,'divergence')
    if osp.isdir(track_dir)==False:
        os.mkdir(track_dir)

    #compute div
    x_array=avg_vfield['x'].unique(); y_array=avg_vfield['y'].unique()
    X, Y = np.meshgrid(x_array,y_array)
    div = zeros((X.shape[0],X.shape[1]))
    for i in range(1,X.shape[0]-1):
        for j in range(1,X.shape[1]-1):
            dy=Y[i,j]-Y[i-1,j]; dx=X[i,j]-X[i,j-1]
            vx1=avg_vfield[((avg_vfield['x']==X[i,j+1]) & (avg_vfield['y']==Y[i,j+1]))]['vx'].values[0]
            vx_1=avg_vfield[((avg_vfield['x']==X[i,j-1]) & (avg_vfield['y']==Y[i,j-1]))]['vx'].values[0]
            vy1=avg_vfield[((avg_vfield['x']==X[i+1,j]) & (avg_vfield['y']==Y[i+1,j]))]['vy'].values[0]
            vy_1=avg_vfield[((avg_vfield['x']==X[i-1,j]) & (avg_vfield['y']==Y[i-1,j]))]['vy'].values[0]
            Dvx=(vx1-vx_1)/(2*dx);Dvy=(vy1-vy_1)/(2*dy)
            div[i-1,j-1]=Dvx+Dvy

    if plot_field:
        fig,ax,xmin,ymin,xmax,ymax=get_background(df,dirdata,frame,no_bkg=no_bkg)
        div_masked = np.ma.array (div, mask=np.isnan(div))
        if fixed_vlim is not None:
            vmin=fixed_vlim[0];vmax=fixed_vlim[1]
        else:
            vmin=div_masked.min();vmax=div_masked.max()
        cmap=cm.plasma; cmap.set_bad('w',alpha=0) #set NAN transparent
        C=ax.pcolormesh(X[1:-1,1:-1],Y[1:-1,1:-1],div_masked[1:-1,1:-1],cmap=cmap,alpha=0.5,vmin=vmin,vmax=vmax)
        cbaxes = fig.add_axes([0.4, 0.935, 0.025, 0.05])
        cbar = fig.colorbar(C,cax = cbaxes,label='$div(\overrightarrow{v})\ (min^{-1})$')
        cbaxes.tick_params(labelsize=5,color='w')
        cbaxes.yaxis.label.set_color('white')
        ax.axis([xmin, ymin, xmax, ymax])
        filename=osp.join(track_dir,'div_%04d.png'%int(frame))
        fig.savefig(filename, dpi=300)
    close()
    return div

def plot_mean_vel(df,groups,frame,data_dir,avg_vfield,fixed_vlim=None,plot_field=True,no_bkg=False):
    close('all')
    print '\rplotting frame '+str(frame),
    sys.stdout.flush()
    track_dir=osp.join(data_dir,'mean_vel')
    if osp.isdir(track_dir)==False:
        os.mkdir(track_dir)

    #compute avg
    x_array=avg_vfield['x'].unique(); y_array=avg_vfield['y'].unique()
    X, Y = np.meshgrid(x_array,y_array)
    mean_vel = zeros((X.shape[0],X.shape[1]))
    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[1]):
            vx=avg_vfield[((avg_vfield['x']==X[i,j]) & (avg_vfield['y']==Y[i,j]))]['vx'].values[0]
            vy=avg_vfield[((avg_vfield['x']==X[i,j]) & (avg_vfield['y']==Y[i,j]))]['vy'].values[0]
            vz=avg_vfield[((avg_vfield['x']==X[i,j]) & (avg_vfield['y']==Y[i,j]))]['vz'].values[0]
            mean_vel[i,j]=sqrt(vx**2+vy**2+vz**2)

    if plot_field:
    	fig,ax,xmin,ymin,xmax,ymax=get_background(df,dirdata,frame,no_bkg=no_bkg)
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

def plot_all_frame(plot_func,df,data_dir,parallelize=True,avg_vfields=None,**kwargs):
    groups=df.groupby('frame')
    if parallelize:
        num_cores = multiprocessing.cpu_count()
        if avg_vfields is None:
            Parallel(n_jobs=num_cores)(delayed(plot_func)(df,groups,frame,data_dir,**kwargs) for frame in df['frame'].unique())
        else:
            Parallel(n_jobs=num_cores)(delayed(plot_func)(df,groups,frame,data_dir,avg_vfields['vfield_list'][i],**kwargs) for i,frame in enumerate(avg_vfields['frame_list']))	
    else:
        for frame in df['frame'].unique():
            plot_func(df,groups,frame,data_dir,**kwargs)

def plot_z_flow(df,frame,z0,grid,plot_field,no_bkg=False):
    """Plot the flow (defined as the net number of cells going through a surface element in the increasing z direction) through the plane of z=z0"""
    
    close('all')
    print '\rplotting frame '+str(frame),
    sys.stdout.flush()
    track_dir=osp.join(data_dir,'z_flow')
    if osp.isdir(track_dir)==False:
        os.mkdir(track_dir)

    df_layer=df[abs(df['vz'])>=abs(z0-df['z'])] #layer of cells crossing the surface
    df_ascending=df_layer[((df_layer['vz']>=0) & (df_layer['z']<=z0))] #ascending cells below the surface
    df_descending=df_layer[((df_layer['vz']<=0) & (df_layer['z']>=z0))] #descending cells above the surface
    
    #calculate the intersection coordinates (x0,y0) of the vector and the surface (calculate homothety coefficient alpha)
    for df_ in [df_ascending,df_descending]:
        df_['alpha']=(z0-df_['z'])/df_['vz']
        df_['x0']=df_['x']+df_['alpha']*df_['vx']
        df_['y0']=df_['y']+df_['alpha']*df_['vy']
        
    X,Y=grid
    x = zeros((X.shape[0]-1,X.shape[1]-1)); y = zeros((X.shape[0]-1,X.shape[1]-1)); flow = zeros((X.shape[0]-1,X.shape[1]-1))
    for i in range(X.shape[0]-1):
        for j in range(X.shape[1]-1):
            ind_asc=((df_ascending['x0']>=X[i,j]) & (df_ascending['x0']<X[i,j+1]) & (df_ascending['y0']>=Y[i,j]) & (df_ascending['y0']<Y[i+1,j]))
            ind_des=((df_descending['x0']>=X[i,j]) & (df_descending['x0']<X[i,j+1]) & (df_descending['y0']>=Y[i,j]) & (df_descending['y0']<Y[i+1,j]))
            x[i,j]=X[i,j]+(X[i,j+1]-X[i,j])*0.5; y[i,j]=Y[i,j]+(Y[i+1,j]-Y[i,j])*0.5
            flow[i,j]=df_ascending[ind_asc].shape[0]-df_descending[ind_des].shape[0]
    
    if plot_field:
        fig,ax,xmin,ymin,xmax,ymax=get_background(df,dirdata,frame,no_bkg=no_bkg)
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
    
    return (x,y,flow)
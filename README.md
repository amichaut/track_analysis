# README #

This code provides different functions to analyze tracking data of cells over time and to plot this analysis as an overlay on the raw movie. Please not that 3D rendering and parallelized analysis are currently not available.

IMPORTANT: this project was released for work presented in this Ph.D. thesis: https://publication-theses.unistra.fr/public/theses_doctorat/2018/MICHAUT_Arthur_2018_ED414.pdf. 
It led to an active project in development available here: https://gitlab.pasteur.fr/track-analyzer/track-analyzer. 

### What should be in the data directory ###

* a 'table.txt' file containing the data of the tracking (without the columns name)
* a 'info.txt' file containing the timestep length in min, the lengthscale in um/px and the name of the columns. Stick with the format presented in example/
* (optional) a 'raw' directory containing the pictures of the movie. Only 'png' format and labeled with a four-digit number starting at 0 (ie. as 0000.png for the first one)
* (optional) a 'vlim.txt' containing the color limits for the color map. One map per line with the following format: <map_name> : lower_lim,upper_lim

### How do I run the code ###
* Open 'analyze_traj.py' in a IPython terminal: open terminal and run the command: %run <path_to_analyze_traj.py> (you can just drag and drop the file on the terminal after %run )
* run the commands: 
- cell_analysis(data_dir,refresh,parallelize,plot_traj,hide_labels,no_bkg,linewidth,plot3D,frame_subset): cell trajectories
- map_analysis(data_dir,refresh,parallelize,x_grid_size,no_bkg,z0,dimensions,axis_on,plot_on_mean,black_arrows): maps analyzing the velocity field
- avg_ROIs(data_dir,frame_subset=None,selection_frame=None,ROI_list=None,plot_on_map=True,plot_section=True,cumulative_plot=True,avg_plot=True): plotting sections of map making an average along the major axis of a ROI
- XY_flow(data_dir,window_size=None,refresh=False,line=None,orientation=None,frame_subset=None,selection_frame=None,z_depth=None): plot XY flow through a vertical surface defined by a XY line using


with the mandatory argument: data_dir: data directory and the optional arguments: refresh (default False) to refresh the table values, parallelize (default False) to run analyses in parallel, plot_traj (default true) to print the cell trajectories, hide_labels (default True) to hide the cell label, no_bkg (default False) to remove the image background, linewidth (default 1.0) width of the trajectories, dimensions ([row,column] default None) to give the image dimension in case of no_bkg, x_grid_size: number of columns in the grid (default 10), z0: altitude of the z_flow surface (default None => center of z axis), axis_on: display axes along maps (default False).
frame_subset: subset of frames to be plotted [first,last] (default None: open interactive choice), selection_frame: ROI selection frame (if None: chosen interactively), ROI_list: list of ROIs, a ROI=[xmin,xmax,ymin,ymax] in pixel (if None: chosen interactively), plot_on_map: plot section line on a map, plot_section: plot ROI average along major axis, cumulative_plot: all plots on same plot time color coded, avg_plot= average of all plots, window_size = rolling average window in um, default None => interactive choice,plot_on_mean: plot vfield on mean_vel map (default=True),black_arrows: don't use vz to color code vfield arrows (default=True)

### Dependencies ###

analyze_traj depends only on Python packages. Please install: pandas, skimage, numpy, matplotlib, scipy, multiprocessing, joblib, seaborn and lmfit.

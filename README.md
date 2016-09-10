# README #

This codes provides different functions to analyze tracking data of cells over time and to plot this analysis as an overlay on the raw movie. Plots of trajectories, maps of the velocities and divergence are currently available.

### What should be in the data directory ###

* a 'table.txt' file containing the data of the tracking (without the columns name)
* a 'info.txt' file containing the timestep length in min, the lengthscale in um and the name of the columns. Stick with the format presented in example/
* (optional) a 'raw' directory containing the pictures of the movie. Only 'png' format and labeled with a four-digit number starting at 0 (ie. as 0000.png for the first one)

### How do I run the code ###
* Open 'analyze_traj.py' in a IPython terminal: open terminal and run the command: %run <path_to_analyze_traj.py> (you can just drag and drop the file on the terminal after %run )
* run the commands: 
- cell_analysis(data_dir,refresh,parallelize,plot_traj,hide_labels,no_bkg,dimensions)
- map_analysis(data_dir,refresh,parallelize,x_grid_size,no_bkg,z0,dimensions)

With the mandatory argument: data_dir: data directory
And the optional arguments: refresh (default False) to refresh the table values, parallelize (default False) to run analyses in parallel, plot_traj (default true) to print the cell trajectories, hide_labels (default True) to hide the cell label, no_bkg (default False) to remove the image background, dimensions ([row,column] default None) to give the image dimension in case of no_bkg, x_grid_size: number of columns in the grid (default 10), z0: altitude of the z_flow surface (default None => center of z axis)
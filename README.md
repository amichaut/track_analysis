# README #

This codes provides different functions to analyze tracking data of cells over time and to plot this analysis as an overlay on the raw movie. Plots of trajectories, maps of the velocities and divergence are currently available.

### What should be in the data directory ###

* a 'table.txt' file containing the data of the tracking (without the columns name)
* a 'info.txt' file containing the timestep length in min, the lengthscale in um and the name of the columns. Stick with the format presented in example/
* (optional) a 'raw' directory containing the pictures of the movie. Only 'png' format and labeled with a four-digit number starting at 0 (ie. as 0000.png for the first one)

### How do I run the code ###
* Open 'analyze_traj.py' in a IPython terminal
* run the command: run_analysis(data_dir) with data_dir being the path of the data directory.
The main argument to pass are: min_traj_len (the minimum length you want for plotting the trajectories), x_grid_size (the number of column in the grid), z0 (the plan defining the z flow)
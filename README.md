# CISC813-Project-USV-Nav
Project for CISC813 at Queen's University, "Dynamic Obstacle Avoidance by a USV with Moving Obstacles". This project is in the RDDL domain.

The goal of this project is to implement a path-planning problem involving safe movement of a Unmanned Surface Vehicle (USV) around moving obstacles, using Collision Regulations (COLREGS) to determine the correct movements for safe avoidance. This consists of a 2-dimensional cartesian plane, in which obstacles are given (limited) headings which change probabilistically. If the USV gets too close to an object, it flags a damaged boolean failing the condition. Ideally the reward function will make it such that the USV reaches the goal as quickly as possible. 

Currently there are 3 versions:
  Version 1 is a depreceated grid and integer based model, do not use this.
  Version 2 is the general dynamic obstacle avoidance model, without COLREGS. This generally works well, but can experience issues with a lot of obstacles or too much randomness.
  Version 3 is the full model with COLREGS. Again, this is best suited to a limited number of obstacles, without too high a probability distribution.

An associated domain, problem, and config file are all provided. RDDL can be run in Google Colab (example [here](https://colab.research.google.com/drive/1v_YVCxloAhojR2pNzxH4MIWItuhU-pg1)). Please see the attached sample python RDDL file for reference. This has the required visualization tool and imports for this model to work appropriately. Alternatively, this python file can be run in any other IDE.

This [link](https://colab.research.google.com/drive/1D-PHlP4tly_Z2GzgdnREkZA9nVC59EFB?usp=sharing) should also work for an already working implementation in Colab.

To switch between random heading and unchanging heading for the obstacles, remove the Normal(0,0.001) with a 0. 

Visualizations can be found in the CISC813 gifs folder, outlining certain situations. The problem files should be configured for a working version of the model already. Performance is often dependent on the configuration of the obstacle vessels, sometimes tuning of hyperparameters (small variation in batch size and epochs) is required but this should not be necessary.

To use the online planner, change the offline to online in the planner section, and add a rollout_horizon to the optimizer config. I have not yet got this to solve the problem successfully.

This is my first foray into RDDL, so much of this is not perfectly optimized as I learn more about the language and the tools (some of which are new to the domain) which are available. 




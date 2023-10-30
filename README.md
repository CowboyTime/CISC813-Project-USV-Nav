# CISC813-Project-USV-Nav
Project for CISC813 at Queen's University, "Dynamic Obstacle Avoidance by a USV with Moving Obstacles". This project is in the RDDL domain.

The goal of this project is to implement a path-planning problem involving safe movement of a Unmanned Surface Vehicle (USV) around moving obstacles, using Collision Regulations (COLREGS) to determine the correct movements for safe avoidance. This consists of a 2-dimensional cartesian plane, in which obstacles are given (limited) headings and speeds which will change probabilistically. If the USV gets too close to an object, it flags a damaged boolean failing the condition. Ideally the reward function will make it such that the USV reaches the goal as quickly as possible and stays there (unless an obstacle moves to the goal location).

This is my first foray into RDDL, so much of this is not optimized as I learn more about the language and the tools (some of which are new to the domain) which are available. Currently, the status of this implementation is 1 ASV and 2 moving obstacles, though performance is not ideal. The ASV will seek the goal location, and avoid obstacles, but shoot right through the goal. 

Currently on Version 1, though I plan on doing a complete overhaul of this code now that I have learned a lot from my first attempt of this problem in RDDL.

This project should be complete by the end of Nov. 2023.


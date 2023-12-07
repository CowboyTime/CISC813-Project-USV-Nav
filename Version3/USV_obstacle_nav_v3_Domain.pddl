domain USV_obstacle_nav_v3 {

    requirements = {
        concurrent,
        reward-deterministic,
        cpf-deterministic,
				constrained-state,
        multivalued,
        continuous
    };

    types {
      obs : object;
      USV : object;
    };

    pvariables {
        //Goal coordinates
        XG : {non-fluent, real, default = 100.0};
        YG : {non-fluent, real, default = 100.0};
        //Damage Radius
        dmgRad : {non-fluent, real, default = 5.0};
        //Colreg radius
        crgRad : {non-fluent, real, default = 10.0};

        //Boolean for collision
        damaged(USV) : {state-fluent, bool, default = false};

        //Distance to goal
        distance(USV) : {state-fluent, real, default = 0.0};

        //Time ticker (unused currently)
        time : {state-fluent, real, default = 0.0};

        //ASV Position and Velocity
        xPos(USV) : {state-fluent, real, default = 0.0};
        yPos(USV) : {state-fluent, real, default = 0.0};
        xVel(USV) : {state-fluent, real, default = 0.0};
        yVel(USV) : {state-fluent, real, default = 0.0};

        //ASV Inputs
        heading(USV) : {action-fluent, real, default = 0.0};
        tVel(USV) : {action-fluent, real, default = 0.0};

        headingLast(USV) : {state-fluent, real, default = 0.0};

        //COLREG Penalty fluent (used in reward)
        crgPen(USV) : {state-fluent, real, default = 0.0};
        //relative angle used for colreg determiniation
        relAngle(USV,obs) : {state-fluent, real, default = 0.0};

        //Obstacle states
        obsXp(obs) : {state-fluent, real, default = 0.0};
        obsYp(obs) : {state-fluent, real, default = 0.0};
	      obsXv(obs) : {interm-fluent, real};
	      obsYv(obs) : {interm-fluent, real};
        obsV(obs) : {state-fluent, real, default = 0.0};
        obsHead(obs) : {state-fluent, real, default = 0.0};


        anglecheck(USV,obs) : {interm-fluent, real};

    };

    cpfs {

        //update position counters
        xPos'(?b) = xPos(?b) + cos[heading(?b)]*tVel(?b)*0.5;
        yPos'(?b) = yPos(?b) + sin[heading(?b)]*tVel(?b)*0.5;

        //update velocities
        xVel'(?b) = cos[heading(?b)]*tVel(?b);
        yVel'(?b) = sin[heading(?b)]*tVel(?b);

        headingLast'(?b) = heading(?b);

        //Calculate distance to goal
        distance'(?b) = hypot[(XG - xPos(?b)) , (YG - yPos(?b))];

        //Apply COLREGS if within 15m
        //Not sure if i wrapped all my angles here, or if it matters...

        crgPen'(?b) = if (exists_{?o : obs}(hypot[xPos(?b) - obsXp(?o), yPos(?b) - obsYp(?o)] <= crgRad ^ relAngle(?b, ?o) < 0.2617994 ^ relAngle(?b, ?o) > -0.2617994))
                      //Head On
                      then -headingLast(?b) + heading(?b)
                    else if (exists_{?o : obs}(hypot[xPos(?b) - obsXp(?o), yPos(?b) - obsYp(?o)] <= crgRad ^ relAngle(?b, ?o) < 1.745329 ^ relAngle(?b, ?o) > 0.2617994))
                      //Crossing Left
                      then headingLast(?b) - heading(?b)
                    else if (exists_{?o : obs}(hypot[xPos(?b) - obsXp(?o), yPos(?b) - obsYp(?o)] <= crgRad ^ relAngle(?b, ?o) > -1.745329 ^ relAngle(?b, ?o) < -0.2617994))
                      //Crossing Right
                      then -headingLast(?b) + heading(?b)
                    else if (exists_{?o : obs}(hypot[xPos(?b) - obsXp(?o), yPos(?b) - obsYp(?o)] <= crgRad))
                      //Passing/Overtaking
                      then 0
                    else
                      //No Colreg
                      0;


        //relative angle between position of vessels adjusted for obstacle vessel heading
        relAngle'(?b,?o) = if (obsXp(?o) - xPos(?b) > 0 ^ obsYp(?o) - yPos(?b) > 0)
				                    then atan[(obsYp(?o) - yPos(?b))/(obsXp(?o) - xPos(?b))] - obsHead(?o)
			                    else if (obsXp(?o) - xPos(?b) < 0 ^ obsYp(?o) - yPos(?b) > 0)
				                    then atan[(obsYp(?o) - yPos(?b))/(obsXp(?o) - xPos(?b))] - obsHead(?o) + 3.1415
                          else if (obsXp(?o) - xPos(?b) < 0 ^ obsYp(?o) - yPos(?b) < 0)
				                    then atan[(obsYp(?o) - yPos(?b))/(obsXp(?o) - xPos(?b))] - obsHead(?o) - 3.1415
                          else if (obsXp(?o) - xPos(?b) == 0 ^ obsYp(?o) - yPos(?b) > 0)
				                    then  1.57079632679 - obsHead(?o)
                          else if (obsXp(?o) - xPos(?b) == 0 ^ obsYp(?o) - yPos(?b) < 0)
				                    then  -1.57079632679 - obsHead(?o)
			                    else
				                    0;
        
        anglecheck(?b,?o) = if (obsXp(?o) - xPos(?b) > 0 ^ obsYp(?o) - yPos(?b) > 0)
				                    then atan[(obsYp(?o) - yPos(?b))/(obsXp(?o) - xPos(?b))] - obsHead(?o)
			                    else if (obsXp(?o) - xPos(?b) < 0 ^ obsYp(?o) - yPos(?b) > 0)
				                    then atan[(obsYp(?o) - yPos(?b))/(obsXp(?o) - xPos(?b))] - obsHead(?o) + 3.1415
			                    else
				                    atan[(obsYp(?o) - yPos(?b))/(obsXp(?o) - xPos(?b))] - obsHead(?o) - 3.1415;

        relAngle'(?b,?o) = if (anglecheck(?b,?o) < -3.14159265)
                              then anglecheck(?b,?o) + 6.28318531
                           else if (anglecheck(?b,?o) > 3.14159265)
                              then anglecheck(?b,?o) - 6.28318531
                           else anglecheck(?b,?o);
                          

        //Apply damage if collided
        //In this case radius is 5m
        damaged'(?b) = if (exists_{?o : obs}(hypot[xPos(?b) - obsXp(?o), yPos(?b) - obsYp(?o)] <= dmgRad))
                    then true
                 else
                    if (damaged(?b))
                      then true
                    else
                      false;

        //Required to keep constant
        obsHead'(?o) = obsHead(?o) + 0;
        obsV'(?o) = obsV(?o) + 0;

        //Update obstacle positions
	      obsXv(?o) = cos[obsHead(?o)]*obsV(?o);
	      obsYv(?o) = sin[obsHead(?o)]*obsV(?o);
        obsXp'(?o) = obsXp(?o) + obsXv(?o)*0.5;
        obsYp'(?o) = obsYp(?o) + obsYv(?o)*0.5;

        //tick time
        time' = time + 1;
    };

    action-preconditions {
		//Speed and heading change cap!
    //Used only for random actor
        forall_{?b : USV}[tVel(?b) <= 2];
        forall_{?b : USV}[tVel(?b) >= -0.5];
		};

    //reward function

    reward = -sum_{?b:USV}(distance(?b) + damaged(?b)*10000 + crgPen(?b)*10000);
    //reward = -sum_{?b:USV}(distance(?b));  + (heading(?b) - headingLast(?b))*1000
}

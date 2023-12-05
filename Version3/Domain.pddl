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

        //COLREG Penalty fluent (used in reward)
        crgPen(USV) : {state-fluent, real, default = 0.0};
        //relative angle used for colreg determiniation
        relAngle(USV, obs) : {state-fluent, real, default = 0.0};

        //Obstacle states
        obsXp(obs) : {state-fluent, real, default = 0.0};
        obsYp(obs) : {state-fluent, real, default = 0.0};
        obsV(obs) : {state-fluent, real, default = 0.0};
        obsHead(obs) : {state-fluent, real, default = 0.0};

    };

    cpfs {

        //update position counters
        xPos'(?b) = xPos(?b) + xVel(?b)*0.5;
        yPos'(?b) = yPos(?b) + yVel(?b)*0.5;

        //update velocities
        xVel'(?b) = cos[heading(?b)]*tVel(?b);
        yVel'(?b) = sin[heading(?b)]*tVel(?b);

        //Calculate distance to goal
        distance'(?b) = hypot[(XG - xPos(?b)) , (YG - yPos(?b))];

        //Apply COLREGS if within 15m
        //Not sure if i wrapped all my angles here, or if it matters...
        crgPen'(?b) = if (exists_{?o : obs}(hypot[xPos(?b) - obsXp(?o), yPos(?b) - obsYp(?o)] <= 15))
                        then if (relAngle(?b, ?o) < 0.2617994 ^ relAngle(?b, ?o) > -0.2617994)
                            //Head on
                            //Difference between angles and 30 degrees to right
                            then if (obsHead(?o) < 0)
                              then abs(0.5235988 - (obsHead(?o) + 3.1415 - heading(?b))
                            else
                              abs(-0.5235988 - (obsHead(?o) - 3.1415 - heading(?b))
                          else if (relAngle(?b, ?o) < 1.963495 ^ relAngle(?b, ?o) > 0.2617994)
                            //Cross Right (30 degrees past perpendicular, to left)
                            then if (obsHead(?o) < 0)
                              then abs(0.5235988 - (obsHead(?o) - 1.570796 - heading(?b))
                            else
                              abs(-0.5235988 - (obsHead(?o) + 1.570796 - heading(?b))
                          else if (relAngle(?b, ?o) > -1.963495 ^ relAngle(?b, ?o) < -0.2617994)
                            //Cross left (30 degrees past perpendicular, to right)
                            then if (obsHead(?o) < 0)
                              then abs(0.5235988 - (obsHead(?o) + 1.570796 - heading(?b))
                            else
                              abs(-0.5235988 - (obsHead(?o) - 1.570796 - heading(?b))
                          else
                            //Overtake (30 degrees to right)
                            if (obsHead(?o) < 0)
                              then abs(0.5235988 - (obsHead(?o) - heading(?b))
                            else
                              abs(-0.5235988 - (obsHead(?o) - heading(?b))
                      else
                        0;

        //relative angle between position of vessels adjusted for obstacle vessel heading
        relAngle'(?b,?o) = atan(obsYp(?o) - yPos(?b), obsXp(?o) - xPos(?b)) - obsHead(?o);

        //Apply damage if collided
        //In this case radius is 5m
        damaged'(?b) = if (exists_{?o : obs}((hypot[xPos(?b) - obsXp(?o), yPos(?b) - obsYp(?o)] <= 5) | damaged(?b))
                    then true
                 else
                    false;

        //Required to keep constant
        obsHead'(?o) = obsHead(?o) + 0;
        obsV'(?o) = obsV(?o) + 0;

        //Update obstacle positions
        obsXp'(?o) = obsXp(?o) + cos[obsHead(?o)]*obsV(?o)*0.5;
        obsYp'(?o) = obsYp(?o) + sin[obsHead(?o)]*obsV(?o)*0.5;
        //obsXp'(?o) = obsXp(?o) + 0;
        //obsYp'(?o) = obsYp(?o) + 0;

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

    reward = -sum_{?b:USV}(distance(?b) + damaged(?b)*100000 + crgPen(?b)*100);
    //reward = -sum_{?b:USV}(distance(?b));
}

!pip install --upgrade ipykernel
!pip install pyRDDLGym

!pip install jax>=0.3.25
!pip install optax>=0.1.4
!pip install dm-haiku>=0.0.9
!pip install tensorflow>=2.11.0
!pip install tensorflow-probability>=0.19.0


from IPython.display import Image as oldIM # for displaying gifs in colab
import time
import jax
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange
from pprint import pprint

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Visualizer.MovieGenerator import MovieGenerator
#from pyRDDLGym import ExampleManager

from pyRDDLGym.Core.Policies.Agents import RandomAgent

from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import load_config
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOfflineController
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOnlineController


from matplotlib.patches import Arrow
from matplotlib import collections  as mc
from PIL import Image
from pyRDDLGym.Core.Compiler.RDDLModel import PlanningModel
from pyRDDLGym.Visualizer.StateViz import StateViz

DOMAIN = r"""
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

        //difference between vectors
        anglecheck(USV,obs) : {interm-fluent, real};

    };

    cpfs {

        //update position counters
        xPos'(?b) = xPos(?b) + cos[heading(?b)]*tVel(?b)*0.5;
        yPos'(?b) = yPos(?b) + sin[heading(?b)]*tVel(?b)*0.5;

        //update velocities
        xVel'(?b) = cos[heading(?b)]*tVel(?b);
        yVel'(?b) = sin[heading(?b)]*tVel(?b);

        //previous heading value (used for derivative)
        headingLast'(?b) = heading(?b);

        //Calculate distance to goal
        distance'(?b) = hypot[(XG - xPos(?b)) , (YG - yPos(?b))];

        //Apply COLREGS if within given radius
        crgPen'(?b) = if (exists_{?o : obs}(hypot[xPos(?b) - obsXp(?o), yPos(?b) - obsYp(?o)] <= crgRad ^ relAngle(?b, ?o) < 0.2617994 ^ relAngle(?b, ?o) > -0.2617994))
                      //Head On (turn right)
                      then headingLast(?b) - heading(?b)
                    else if (exists_{?o : obs}(hypot[xPos(?b) - obsXp(?o), yPos(?b) - obsYp(?o)] <= crgRad ^ relAngle(?b, ?o) < 2.617994 ^ relAngle(?b, ?o) > 0.2617994))
                      //Crossing Left (turn left)
                      then -headingLast(?b) + heading(?b) 
                    else if (exists_{?o : obs}(hypot[xPos(?b) - obsXp(?o), yPos(?b) - obsYp(?o)] <= crgRad ^ relAngle(?b, ?o) > -2.617994 ^ relAngle(?b, ?o) < -0.2617994))
                      //Crossing Right (turn right)
                      then headingLast(?b) - heading(?b)
                    else if (exists_{?o : obs}(hypot[xPos(?b) - obsXp(?o), yPos(?b) - obsYp(?o)] <= crgRad))
                      //Passing/Overtaking (do nothing)
                      then 0.0
                    else
                      //No Colreg
                      0.0;


        //relative angle between position of vessels adjusted for obstacle vessel heading
        //this performs an Atan2
        anglecheck(?b,?o) = if (xPos(?b) - obsXp(?o) > 0 ^ yPos(?b) - obsYp(?o) > 0)
				                    then atan[(yPos(?b) - obsYp(?o))/(xPos(?b) - obsXp(?o))] - obsHead(?o)
			                    else if (xPos(?b) - obsXp(?o) < 0 ^ yPos(?b) - obsYp(?o) > 0)
				                    then atan[(yPos(?b) - obsYp(?o))/(xPos(?b) - obsXp(?o))] - obsHead(?o) + 3.1415
			                    else
				                    atan[(yPos(?b) - obsYp(?o))/(xPos(?b) - obsXp(?o))] - obsHead(?o) - 3.1415;

        //round to +-pi
        relAngle'(?b,?o) = if (anglecheck(?b,?o) < -3.14159265)
                              then anglecheck(?b,?o) + 6.28318531
                           else if (anglecheck(?b,?o) > 3.14159265)
                              then anglecheck(?b,?o) - 6.28318531
                           else anglecheck(?b,?o);


        //Apply damage if collided
        damaged'(?b) = if (exists_{?o : obs}(hypot[xPos(?b) - obsXp(?o), yPos(?b) - obsYp(?o)] <= dmgRad))
                    then true
                 else
                    if (damaged(?b))
                      then true
                    else
                      false;

        //Required to keep constant
        //obsHead'(?o) = obsHead(?o) + 0;
        obsHead'(?o) = obsHead(?o) + Normal(0,0.001);
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
		//Speed cap!
    //Used only for random actor
        forall_{?b : USV}[tVel(?b) <= 2];
        forall_{?b : USV}[tVel(?b) >= -0.5];
		};

    //reward function

    reward = -sum_{?b:USV}(distance(?b) + damaged(?b)*1000 + crgPen(?b)*10000);

    //Unused
    //reward = -sum_{?b:USV}(distance(?b));  + (heading(?b) - headingLast(?b))*1000
}
"""

with open('domain.rddl', 'w') as f:
    f.write(DOMAIN)
  
PROBLEM = r"""
non-fluents nf_USV_obstacle_nav_v3 {
    domain = USV_obstacle_nav_v3;

    objects{
      obs : {o1, o2, o3};
      USV : {b1};
    };
    non-fluents {
      //define goal coordinates
      XG = 100;
      YG = 100;
      dmgRad = 7;
      crgRad = 15;
    };

}

instance USV_obstacle_nav_inst_v3 {
    domain = USV_obstacle_nav_v3;
    non-fluents = nf_USV_obstacle_nav_v3;
    init-state{

      //inital positioning of ASV
      xPos(b1) = 0.0;
      yPos(b1) = 0.0;

      //obstacle states
      obsXp(o1) = 40.0;
      obsYp(o1) = 40.0;
      obsV(o1) = 0.75;
      obsHead(o1) = -2.356194;

      obsXp(o2) = 60.0;
      obsYp(o2) = 30.0;
      obsV(o2) = 0.75;
      obsHead(o2) = 2.356194;

      obsXp(o3) = 70.0;
      obsYp(o3) = 50.0;
      obsV(o3) = 0.5;
      obsHead(o3) = 0.7853982;



    };
    //horizon = timesteps
    max-nondef-actions = 2;
    horizon = 120;
    discount = 1.0;
}
"""

with open('problem.rddl', 'w') as f:
    f.write(PROBLEM)
JAXCONFIG = r"""
[Model]
logic='FuzzyLogic'
logic_kwargs={'weight': 850}
tnorm='ProductTNorm'
tnorm_kwargs={}

[Optimizer]
method='JaxDeepReactivePolicy'
method_kwargs={'topology': [128, 64]}
optimizer='rmsprop'
optimizer_kwargs={'learning_rate': 0.001}
batch_size_train=10
batch_size_test=10
action_bounds={'tVel': (0, 3), 'heading' : (-3.14, 3.14)}
rollout_horizon = 15

[Training]
key=42
epochs=300
train_seconds=300
"""

with open('jax.cfg', 'w') as f:
    f.write(JAXCONFIG)

class USVVisualizer(StateViz):
    #Credit to https://github.com/ataitler/pyRDDLGym for the vector based car visualization, modified for this USV simulation
    def __init__(self, model: PlanningModel,
                 figure_size=(4, 4),
                 car_radius=2,
                 vector_len=5,
                 wait_time=100) -> None:
        self._model = model
        self._objects = model.objects
        self._figure_size = figure_size
        self._car_radius = car_radius
        self._vector_len = vector_len
        self._wait_time = wait_time

        self._nonfluents = model.groundnonfluents()

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = plt.gca()

        # draw goal
        goal = plt.Circle((self._nonfluents['XG'], self._nonfluents['YG']),
                          3,
                          color='g')
        self.ax.add_patch(goal)

        # velocity vector
        self.arrow = Arrow(0, 0, 0, 0,
                           width=0.2 * self._vector_len,
                           color='black')
        self.move = self.ax.add_patch(self.arrow)

        self.ax.set_xlim([-10,110])
        self.ax.set_ylim([-10,110])

    def convert2img(self, canvas):
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(canvas.get_width_height()[::-1] + (3,))
        return Image.fromarray(data)

    def render(self, state):
        for dk in self._objects['USV']:
          vel = np.array([state[f'xVel___{dk}'], state[f'yVel___{dk}']])
          if np.max(np.abs(vel)) > 0:
              vel /= np.linalg.norm(vel)
          vel *= self._vector_len
          dx = vel[0]
          dy = vel[1]
          #self.move.remove()
          self.arrow = Arrow(state[f'xPos___{dk}'], state[f'yPos___{dk}'], dx, dy,
                             width=0.2 * self._vector_len,
                             color='black')
          self.move = self.ax.add_patch(self.arrow)

          car = plt.Circle((state[f'xPos___{dk}'], state[f'yPos___{dk}']), self._car_radius)
          self.ax.add_patch(car)
          self.fig.canvas.draw()

        self.move1 = [None] * len(self._objects['obs']);
        obs = [None] * len(self._objects['obs']);
        obs1 = [None] * len(self._objects['obs']);
        obs2 = [None] * len(self._objects['obs']);
        obs3 = [None] * len(self._objects['obs']);
        for dk in self._objects['obs']:
          k = self._objects['obs'].index(dk)
          vel = np.array([np.cos(state[f'obsHead___{dk}'])*state[f'obsV___{dk}'], np.sin(state[f'obsHead___{dk}'])*state[f'obsV___{dk}']])
          if np.max(np.abs(vel)) > 0:
              vel /= np.linalg.norm(vel)
          vel *= self._vector_len
          dx = vel[0]
          dy = vel[1]
          self.arrow1 = Arrow(state[f'obsXp___{dk}'], state[f'obsYp___{dk}'], dx, dy,
                             width=0.2 * self._vector_len,
                              color='red')
          self.move1[k] = self.ax.add_patch(self.arrow1)
          obs[k] = plt.Circle((state[f'obsXp___{dk}'], state[f'obsYp___{dk}']), 2, color = 'r')
          self.ax.add_patch(obs[k])
          obs1[k] = plt.Circle((state[f'obsXp___{dk}'], state[f'obsYp___{dk}']), self._nonfluents['dmgRad'] - 1, color = 'r', alpha = 0.3)
          self.ax.add_patch(obs1[k])
          obs2[k] = plt.Circle((state[f'obsXp___{dk}'], state[f'obsYp___{dk}']), self._nonfluents['crgRad'], color = 'r', alpha = 0.1)
          self.ax.add_patch(obs2[k])
          obs3[k] = self.ax.text(0, 100-k*8, "%0.2f" % state[f'relAngle___b1__{dk}'] )
          self.fig.canvas.draw()
          img = self.convert2img(self.fig.canvas)

        for dk in self._objects['obs']:
          #self.move1[k].remove()
          k = self._objects['obs'].index(dk)
          obs[k].remove()
          obs1[k].remove()
          obs2[k].remove()
          obs3[k].remove()
        car.remove()
        del car
        del obs


        return img

base_path = 'rddl'
ENV = 'test'

myEnv = RDDLEnv.RDDLEnv(domain='domain.rddl', instance='problem.rddl')
MovieGen = MovieGenerator('', ENV, myEnv.horizon)
myEnv.set_visualizer(USVVisualizer, movie_gen=MovieGen, movie_per_episode=False)
gif_name = f'{ENV}_chart'

agent = None

def use_random():
    global agent
    agent = RandomAgent(action_space=myEnv.action_space,
                        num_actions=myEnv.numConcurrentActions,
                        seed=42)

def use_jax():

    global agent

    planner_args, _, train_args = load_config('jax.cfg')

    planner = JaxRDDLBackpropPlanner(rddl=myEnv.model, **planner_args)
    agent = JaxOnlineController(planner, **train_args)

use_jax()
#use_random()


stats = agent.evaluate(myEnv, ground_state=False, episodes=1, verbose=True, render=True)
from pprint import pprint
pprint(stats)

# Plot the behaviour
MovieGen.save_animation(gif_name)
oldIM(open(gif_name+'.gif','rb').read())

gif_name

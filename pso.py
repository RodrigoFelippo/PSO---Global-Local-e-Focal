# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:55:45 2019

@author: suporte
"""

#--- IMPORT DEPENDENCIES ----+
from __future__ import division
import random
import math
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
 
#--- COST FUNCTION ----+
def sphere(x):
	total = 0
	for	i in range(len(x)): 
		total += x[i]**2
	return total
	
def rastrigin(x):
	return 10*len(x) + sum([(xi**2 - 10 * np.cos(2 * math.pi * xi)) for xi in x])
	
def rosenbrock(x):
	total = 0 
	for i in range(len(x)-1):
		total += 100*((x[i]**2 - x[i+1])**2) + (1-x[i])**2 
	return total

class Topology(Enum):
	GLOBAL = 0
	LOCAL = 1
	FOCAL = 2
    
class Velocity(Enum):
	INERTIA = 0
	DECAY_INERTIA = 1
	CLERC = 2    

class Fitness(Enum):
	SPHERE = 0
	RASTRIGIN = 1
	ROSENBROCK = 2    

class Particle:
    def __init__(self, x0, v0):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity 
        self.pos_best_i = []  # best position individual 
        self.err_best_i = -1  # best error individual 
        self.err_i = -1 		 # error individual
        
        for i in range(0,num_dimensions):
            #self.velocity_i.append(random.uniform(-1,1))
            #self.position_i.append(random.uniform(1,5))
            self.velocity_i.append(v0[i])
            self.position_i.append(x0[i])
		
    # evaluate current fitness
    def evaluate(self,costFunc):
    		self.err_i = costFunc(self.position_i)
    		# check to see if the current position is an individual best
    		if self.err_i < self.err_best_i or self.err_best_i == -1:
    			self.pos_best_i = self.position_i
    			self.err_best_i = self.err_i
   
                
    #update new particles velocity 
    def update_velocity(self,pos_best_g, velocity, decay_inertia):
        
       if(velocity == Velocity.INERTIA):
           #w = 0.8  # constant inertia weight
           w = 1
       elif(velocity == Velocity.DECAY_INERTIA):
           w = decay_inertia
           
       
       c1 = 2.05   # cognative constant 
       c2 = 2.05   # social constant
       
       y = c1+c2
       
       x = 2/ abs((2 - y - math.sqrt(y**2 - (4*y))) )
                    
       for i in range(0,num_dimensions): 
           rl = random.random()
           r2 = random.random()
           vel_cognitive = c1*rl*(self.pos_best_i[i] - self.position_i[i])
           vel_social = c2*r2*(pos_best_g[i] - self.position_i[i])
           
           if(velocity == Velocity.CLERC):
               self.velocity_i[i] = x*(self.velocity_i[i] + vel_cognitive + vel_social)
           else:    
               self.velocity_i[i] = w*self.velocity_i[i] + vel_cognitive + vel_social
         
    
    # update the particle position based off new velocity updates 
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]
            
            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]
                
            # adjust minimum position if necessary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0] 	

class PSO():
    def __init__(self, costName, costFunc, x0, v0, bounds, num_particles, maxiter, topology, velocity):
        global num_dimensions
        
        self.costName = costName
        self.array_err = []
        self.err_best_g = -1         # best error for group 
        self.pos_best_g = []         # best position for group
                        
        num_dimensions = len(x0[0])
        swarm = []              # establish the swarm
        
        max_decay_inertia = 0.9
        min_decay_inertia = 0.4
        
        decay_inertia = (max_decay_inertia - min_decay_inertia)/maxiter
        
        
        for i in range(0, num_particles):
            swarm.append(Particle(x0[i],v0[i]))
        i = 0  #begin optimization loop
        
        while i < maxiter:
           
          if(topology == Topology.GLOBAL):
                                    # cicle through particles in swarm and evaluate fitness
            for j in range(0, num_particles):
                
                swarm[j].evaluate(costFunc)
       
                #determine if current particle is the best (globably)
                if swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                   self.pos_best_g = list(swarm[j].position_i) 
                   self.err_best_g = float(swarm[j].err_i)
               
             #cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                  swarm[j].update_velocity(self.pos_best_g, velocity, max_decay_inertia)
                  swarm[j].update_position(bounds)
            
          elif(topology == Topology.LOCAL):
                    
            for j in range(0, num_particles):
                
                 swarm[j].evaluate(costFunc)
                 
                 if(j == 0):
                     previousParticle = swarm[num_dimensions - 1]
                     afterParticle = swarm[j+1]
                        
                 elif(j == (num_dimensions - 1)):
                     afterParticle = swarm[0]
                     previousParticle = swarm[j-1]
                        
                 else:
                     previousParticle = swarm[j-1]
                     afterParticle = swarm[j+1]
                 
                 afterParticle.evaluate(costFunc) 
                 previousParticle.evaluate(costFunc) 
                 
                 if swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g = list(swarm[j].position_i) 
                    self.err_best_g = float(swarm[j].err_i)                           
                 
                 if previousParticle.err_i < afterParticle.err_i:
                     if previousParticle.err_i < swarm[j].err_i:
                          swarm[j].update_velocity(previousParticle.position_i, velocity, max_decay_inertia)
                          swarm[j].update_position(bounds)
                          
                 else:                     
                     if afterParticle.err_i < swarm[j].err_i:
                          swarm[j].update_velocity(afterParticle.position_i, velocity, max_decay_inertia)
                          swarm[j].update_position(bounds)
                 
                 
                    
          elif(topology == Topology.FOCAL):
              
              for j in range(0, num_particles):
                
                 swarm[j].evaluate(costFunc)
                 
                 if(self.err_best_g == -1):
                     focalParticle = swarm[0]
                 
                 if swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g = list(swarm[j].position_i) 
                    self.err_best_g = float(swarm[j].err_i)
               
                 
                 if(focalParticle.err_i < swarm[j].err_i):
                     swarm[j].update_velocity(focalParticle.position_i, velocity, max_decay_inertia)
                     swarm[j].update_position(bounds)
                     
                 else:
                     focalParticle = swarm[j]
                
                
          self.array_err.append(self.err_best_g)
          max_decay_inertia -= decay_inertia
          i += 1    
            
                        
        
if __name__ == "__PSO__":
    main() 

# --- RUN ---------+	
    


topology_global = Topology.GLOBAL   # topology type global
topology_local = Topology.LOCAL   # topology type local
topology_focal = Topology.FOCAL   # topology type focal

initial = [] # initial starting location

init_vel = [] # initial starting location

bounds = [] # input bounds 

particles = 30
num_dimensions = 30

def pso_fitness_name(pso_fitness):
    if(pso_fitness == Fitness.SPHERE):
        return "SPHERE"
    elif(pso_fitness == Fitness.RASTRIGIN):
        return "RASTRIGIN"
    elif(pso_fitness == Fitness.ROSENBROCK):
        return "ROSENBROCK"
    
def pso_velocity_name(pso_velocity):
    if(pso_velocity == Velocity.INERTIA):
        return "INERTIA"
    elif(pso_velocity == Velocity.DECAY_INERTIA):
        return "DECAY_INERTIA"
    elif(pso_velocity == Velocity.CLERC):
        return "CLERC"


"""
for a in range(0, particles):
    aux_init_vel = []
    for b in range(0, num_dimensions):
        aux_init_vel.append(random.uniform(-1,1))
    init_vel.append(aux_init_vel)
    
data_vel = np.asarray(init_vel)
np.savetxt("pso_velocity.csv", data_vel, delimiter=",")

for i in range(0, particles):
    aux_initial = []
    for j in range(0, num_dimensions):
        aux_initial.append(random.uniform(1,5))
    initial.append(aux_initial)
    
data_pos = np.asarray(initial)
np.savetxt("pso_position.csv", data_pos, delimiter=",")  
"""  
    
for k in range(0,num_dimensions):    
    bounds.append((-5,5))


init_vel = genfromtxt('pso_velocity.csv', delimiter=',')

initial = genfromtxt('pso_position.csv', delimiter=',')

inter = 10000
pso_fitness = [Fitness.SPHERE, Fitness.RASTRIGIN, Fitness.ROSENBROCK]
pso_velocity = [Velocity.INERTIA]

test = 2 #0 - test topology
         #1 - test velocity
         #2 - test default

for i in range(0, len(pso_fitness)):
    for j in range(0, len(pso_velocity)):
        
        if(pso_fitness[i] == Fitness.SPHERE): #PSO SPHERE
        
            if(test == 0):
                pso_global = PSO("sphere",sphere,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=pso_velocity[j])
                pso_local = PSO("sphere",sphere,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_local, velocity=pso_velocity[j])
                pso_focal = PSO("sphere",sphere,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_focal, velocity=pso_velocity[j])
            elif(test == 1):
                pso_inertia = PSO("sphere",sphere,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=Velocity.INERTIA)
                pso_decay_inertia = PSO("sphere",sphere,init_vel,initial,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=Velocity.DECAY_INERTIA)
                pso_clerc = PSO("sphere",sphere,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=Velocity.CLERC)
            elif(test == 2):
                pso_global = PSO("sphere",sphere,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=pso_velocity[j])
        
        elif(pso_fitness[i] == Fitness.RASTRIGIN):  #PSO RASTRIGIN
        
            if(test == 0):
               pso_global = PSO("rastrigin",rastrigin,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=pso_velocity[j])
               pso_local = PSO("rastrigin",rastrigin,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_local, velocity=pso_velocity[j])
               pso_focal = PSO("rastrigin",rastrigin,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_focal, velocity=pso_velocity[j])
            elif(test == 1):
               pso_inertia = PSO("rastrigin",rastrigin,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=Velocity.INERTIA)
               pso_decay_inertia = PSO("rastrigin",rastrigin,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=Velocity.DECAY_INERTIA)
               pso_clerc = PSO("rastrigin",rastrigin,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=Velocity.CLERC)
            elif(test == 2):
               pso_global = PSO("rastrigin",rastrigin,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=pso_velocity[j])
               
            
        elif(pso_fitness[i] == Fitness.ROSENBROCK):  #PSO ROSENBROCK 
            
            if(test == 0):
                pso_global = PSO("rosenbrock",rosenbrock,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=pso_velocity[j])
                pso_local = PSO("rosenbrock",rosenbrock,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_local, velocity=pso_velocity[j])
                pso_focal = PSO("rosenbrock",rosenbrock,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_focal, velocity=pso_velocity[j])
            elif(test == 1):
                pso_inertia = PSO("rosenbrock",rosenbrock,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=Velocity.INERTIA)
                pso_decay_inertia = PSO("rosenbrock",rosenbrock,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=Velocity.DECAY_INERTIA)
                pso_clerc = PSO("rosenbrock",rosenbrock,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=Velocity.CLERC)
            elif(test == 2):
                pso_global = PSO("rosenbrock",rosenbrock,initial,init_vel,bounds,num_particles=particles,maxiter=inter,topology=topology_global, velocity=pso_velocity[j])
               
    
        if(test == 0):
            plt.plot(np.arange(0, inter, 1), pso_global.array_err)
            plt.plot(np.arange(0, inter, 1), pso_local.array_err)
            plt.plot(np.arange(0, inter, 1), pso_focal.array_err)
            
            plt.title(pso_fitness_name(pso_fitness[i]))
            plt.ylabel("Fitness")
            plt.xlabel("Interaction")
            
            
            fig, ax = plt.subplots()
            ax.set_title(pso_fitness_name(pso_fitness[i]))
            
            ax.set_xlabel("Global                           Local                            Focal")
            ax.set_ylabel("Fitness")
            
            ax.boxplot([pso_global.array_err
                        , pso_local.array_err
                        , pso_focal.array_err], showfliers=False)
    
            plt.show() 
                     
            
        elif(test == 1):            
            plt.plot(np.arange(0, inter, 1), pso_inertia.array_err)
            plt.plot(np.arange(0, inter, 1), pso_decay_inertia.array_err)
            plt.plot(np.arange(0, inter, 1), pso_clerc.array_err)
            
            plt.title(pso_fitness_name(pso_fitness[i]))
            plt.ylabel("Fitness")
            plt.xlabel("Interaction")
            
            fig, ax = plt.subplots()
            ax.set_title(pso_fitness_name(pso_fitness[i]))
            
            ax.set_xlabel("w = 0.8                   w = 0.9 - 0.4                     Clerc")
            ax.set_ylabel("Fitness")
            
            ax.boxplot([pso_inertia.array_err
                        , pso_decay_inertia.array_err
                        , pso_clerc.array_err], showfliers=False)
    
            plt.show() 
            
        elif(test == 2):  
            
            plt.plot(np.arange(0, inter, 1), pso_global.array_err)
            plt.title(pso_fitness_name(pso_fitness[i]))
            plt.ylabel("Fitness")
            plt.xlabel("Interaction")
            
            
            fig, ax = plt.subplots()
            ax.set_title(pso_fitness_name(pso_fitness[i]))
            
            ax.set_xlabel("Global")
            ax.set_ylabel("Fitness")
            
            ax.boxplot([pso_global.array_err], showfliers=False)
    
            plt.show() 
        
        
                
            
        
        
        

"""
fig, ax = plt.subplots()
ax.set_title('boxplot')
ax.boxplot([pso_sphere_global.array_err, pso_sphere_local.array_err], showfliers=False)

"""
"""
#print final results
print ("*******\n")
print ("Funtcion: " + str(pso_rastrigin_global.costName))
print ("Best position: " + str(pso_rastrigin_global.pos_best_g) )
print ("Function solution: " + str(pso_rastrigin_global.err_best_g) )
print ("") 
"""

# --- END ----------+
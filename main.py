import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#defining the PSO algorithm

class PSO:
    def __init__(self, noP = 30, maxiter = 500, wmax = 0.9, wmin = 0.2, c1 = 2, c2 = 2):

        self.noP = noP                # number of particles
        self.maxiter = maxiter        # maximum no. of iterations
        self.wmax = wmax              # maximum value of inertia
        self.wmin = wmin              # minimum value of inertia
        self.c1 = c1                  # cognitive component
        self.c2 = c2                  # social component

        #defining the details of objective functions
        self.n_variables = 10                                                      #total no. of variables 
        self.upper_bound = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])                #upper bound of each variable
        self.lower_bound = np.array([-10, -10, -10, -10, -10, -10, -10, -10, -10, -10])      #lower bound of each variable
        
        #initializing the swarm particles
        self.swarm = self.initialize_particles()
        return
    
    
    def objective_func(self, x):
        ob = np.sum(x**2)
        return ob
    

    def get_random_position(self):
        li = []
        for i in range(self.n_variables):
            li.append(random.randint(self.lower_bound[i], self.upper_bound[i]))
        return np.array(li)


    def get_particles(self):
        particles = []
        for _ in range(self.noP):

            particle = {}
            particle['X'] = self.get_random_position()                                   #position
            particle['V'] = np.zeros(self.n_variables)                                   #velocity
            particle['PBEST'] = self.get_best()                                          #personal best
            
            particles.append(particle)
        
        return particles


    def get_best(self):

        d = {}
        d['X'] = np.zeros(self.n_variables)                     #position
        d['O'] = float('inf')                                   #objective value

        return d
        

    def initialize_particles(self):

        swarm = {}
        swarm['Particles'] = self.get_particles()                   # list of all particles
        swarm['GBEST'] = self.get_best()                            # global best
        
        return swarm
    

    def run(self):
        for t in range(self.maxiter):
            
            #calculate the objective values
            for j in range(self.noP):
                # print(' current particle ', j)
                current_position = self.swarm['Particles'][j]['X']
                # print('current position', current_position)
                self.swarm['Particles'][j]['O'] = self.objective_func(current_position)
                # print('obj', self.swarm['Particles'][j]['O'])

                #update the personal best
                if self.swarm['Particles'][j]['O'] < self.swarm['Particles'][j]['PBEST']['O']:
                    # print('PBEST')
                    self.swarm['Particles'][j]['PBEST']['X'] = current_position
                    self.swarm['Particles'][j]['PBEST']['O'] = self.swarm['Particles'][j]['O'] 


                #update global best
                if self.swarm['Particles'][j]['O'] < self.swarm['GBEST']['O']:
                    # print('GBEST')
                    self.swarm['GBEST']['X'] = current_position
                    self.swarm['GBEST']['O'] = self.swarm['Particles'][j]['O']
                
                # print(self.swarm)
            

            #update the X and V vectors
            w = self.wmax - t*((self.wmax - self.wmin)/self.maxiter)
            # print(w)


            for k in range(self.noP):
                # print('Current particle ', k)
                self.swarm['Particles'][k]['V'] = w*(self.swarm['Particles'][k]['V']) + self.c1*np.random.rand(1,self.n_variables)*(self.swarm['Particles'][k]['PBEST']['X'] - self.swarm['Particles'][k]['X']) + self.c2*np.random.rand(1,self.n_variables)*(self.swarm['GBEST']['X'] - self.swarm['Particles'][k]['X'])
                self.swarm['Particles'][k]['X'] = self.swarm['Particles'][k]['X'] + self.swarm['Particles'][k]['V']
            

            #print the results
            print('iteration = ' + str(t) + ' ,GBEST objective value = ' + str(self.swarm['GBEST']['O']))

        return

ps = PSO()
ps.run()
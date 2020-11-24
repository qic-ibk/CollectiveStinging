# -*- coding: utf-8 -*-
"""
Copyright 2020 Andrea López Incera.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Andrea López Incera, used and analysed in, 

'Honeybee communication during collective defence is shaped by predation.'
Andrea López-Incera, Morgane Nouvian, Katja Ried, Thomas Müller and Hans J. Briegel.

"""

import numpy as np


import beesting_predator_pheromone
import ps_agent_bee


#parameters we set for environment:
group_size=100
num_bins=10 
min_units_perbin=3
full_resolution=False
range_predators=[16,40]
perc_false_alarms=0
rew_scaling='linear'

#predator's parameters:
kill_rate=1 
time_attack=0 
visual_time_delay=10

#parameters we set for the PS agent:
gamma_damping=0.003#float between 0 and 1. Forgetting.
eta_glow_damping=0#float between 0 and 1. Setting it to 1 effectively deactivates glow. 0 means that there is no damping (remembers everything with equal intensity).
policy_type='standard'#usual computation of prob distribution according to h matrix.
beta_softmax=1#irrelevant if policy_type is standard.
num_reflections=0 #effectively deactivates reflections.
init_mode='standard'#standard means that the initial probabilities are 0.5 for stinging and 0.5 for chilling.
init_psting=0.2

#simulation parameters:
num_trials=80000 #number of trials (defensive events).
num_pop=50 #number of populations (trained independently).

#initialization of the "environment".
env=beesting_predator_pheromone.Beesting_predator(group_size, num_bins, min_units_perbin, full_resolution,range_predators,perc_false_alarms,rew_scaling)

#record of performance for all the populations.
learning_curve_allpop=np.zeros([num_pop,num_trials])
prob_stinging_allpop=np.zeros([num_pop,env.num_percepts_list[0]])
number_stung_allpop=np.zeros([num_pop,num_trials])
which_sting=np.zeros([num_pop,group_size])
predator_sth_allpop=np.zeros([num_pop,num_trials])
predator_kills_allpop=np.zeros([num_pop,num_trials])

for pop in range(num_pop):
    
    #initialize ensemble of PS agents
    agent_list=[]
    for i in range(group_size):
        agent_list.append(ps_agent_bee.BasicPSAgent(env.num_actions,env.num_percepts_list,\
        gamma_damping, eta_glow_damping, policy_type, beta_softmax, num_reflections,init_mode,init_psting))
        
    #initialize a record of performance for this population.
    learning_curve=np.zeros(num_trials)
    number_stung=np.zeros(num_trials)
    which_stingevol=np.zeros([1000,group_size])
    sth=np.zeros(num_trials)
    kills=np.zeros(num_trials)
    ps_evolution=np.zeros([num_trials,env.num_percepts_list[0]])
    
    #interaction of this population
    
    for i_trial in range(num_trials):
        
        for i in range(group_size):#reset g matrix to not mix the actions of current trial with past trials.
            agent_list[i].g_matrix=np.zeros((agent_list[i].num_actions, agent_list[i].num_percepts), dtype=np.float64) 
        
        #define global g matrix, where the collective performance will be stored.
        global_g_matrix=np.zeros((agent_list[i].num_actions, agent_list[i].num_percepts), dtype=np.float64) 
        
        #initialize defensive event
        test_sting=np.zeros(group_size) #array with the boolean value of each agent's action (0 hasn't stung yet, 1 has stung).
        test_killed=0 #no kills yet.
        test_predator=1 #predator is there.
        predator_leaving=0 #predator is not leaving.
        counter_pher=0    #no pheromone in the air.
        counter_rounds_nopredator=0
        
        ps=np.zeros(env.num_percepts_list[0]) #initialize record of p_s for each percept at the current trial.
       
        predator_resistance=env.rchoose() #random choice of predator (s_th)

        #sequential decision process of the group.
        for i in range(group_size):
            
            action=agent_list[i].deliberate_and_learn(env.get_percept(counter_pher,predator_leaving),0)
            counter_pher+=np.copy(action)
            test_sting[i]=action
                
            test_predator=env.scare_predator(np.sum(test_sting),predator_resistance)  #check if predator is scared away.   
            if test_predator and i>=time_attack:#predator's attack, only if it is not already scared away.  
                test_killed+=kill_rate
            
            if (test_predator+1)%2: #if predator is scared away...
                counter_rounds_nopredator+=1
                if counter_rounds_nopredator>=visual_time_delay:
                    predator_leaving=1 #...bee perceives the visual stimulus of "predator is leaving" after some time delay.
                    
        reward=env.get_reward(group_size-min(group_size,np.sum(test_sting)+np.sum(test_killed))) 
        
        #save performance for this trial
        learning_curve[i_trial]=reward
        number_stung[i_trial]=np.sum(test_sting)
        which_stingevol[i_trial%1000]=test_sting
        sth[i_trial]=predator_resistance
        kills[i_trial]=np.sum(test_killed)
        
        for i in range(env.num_percepts_list[0]): #save current probability of stinging for each percept.
            ps[i]=agent_list[0].h_matrix[1,i]/np.sum(agent_list[0].h_matrix[:,i])
    
        ps_evolution[i_trial]=ps
        
        #update h matrix
        for i in range(group_size):#sum up all g matrices into one. 
            global_g_matrix += agent_list[i].g_matrix
            
        for i in range(group_size):#update the h matrix.
            agent_list[i].h_matrix =  agent_list[i].h_matrix - agent_list[i].gamma_damping * (agent_list[i].h_matrix - 1.) + global_g_matrix * reward
        
    #save data for this population
    learning_curve_allpop[pop]=learning_curve
    for i in range(env.num_percepts_list[0]):
        prob_stinging_allpop[pop,i]=agent_list[0].h_matrix[1,i]/np.sum(agent_list[0].h_matrix[:,i])#all agents have same h matrix.
     
    number_stung_allpop[pop]=number_stung
    which_sting[pop]=np.sum(which_stingevol,axis=0)/1000
    predator_sth_allpop[pop]=sth
    predator_kills_allpop[pop]=kills
        

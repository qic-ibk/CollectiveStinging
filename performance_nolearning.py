# -*- coding: utf-8 -*-
"""
Copyright 2020 Andrea López Incera.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Please acknowledge the authors when re-using this code and maintain this notice intact.
Code written by Andrea López Incera, used and analysed in, 

'Honeybee communication during collective defence is shaped by predation.'
Andrea López-Incera, Morgane Nouvian, Katja Ried, Thomas Müller and Hans J. Briegel.

"""
#%% performance prior to learning
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
init_psting=0.5

#simulation parameters:
num_trials=5000
num_pop=1

#initialization of the object "environment".
env=beesting_predator_pheromone.Beesting_predator(group_size, num_bins, min_units_perbin, full_resolution,range_predators,perc_false_alarms,rew_scaling)

#initializes record of performance.
learning_curve_allpop=np.zeros([num_pop,num_trials])
prob_stinging_allpop=np.zeros([num_pop,env.num_percepts_list[0]])
number_stung_allpop=np.zeros([num_pop,num_trials])
which_sting=np.zeros([num_pop,group_size])
predator_sizes_allpop=np.zeros([num_pop,num_trials])
predator_kills_allpop=np.zeros([num_pop,num_trials])

for pop in range(num_pop):
    
    #initialize agents.
    agent_list=[]
    for i in range(group_size):
        agent_list.append(ps_agent_bee.BasicPSAgent(env.num_actions,env.num_percepts_list,\
        gamma_damping, eta_glow_damping, policy_type, beta_softmax, num_reflections,init_mode,init_psting))
        
    #initialize a record of performance for this population.
    learning_curve=np.zeros(num_trials)
    number_stung=np.zeros(num_trials)
    which_stingevol=np.zeros([1000,group_size])
    sizes=np.zeros(num_trials)
    kills=np.zeros(num_trials)
    
    #interaction
    
    for i_trial in range(num_trials):
        
        for i in range(group_size):#reset g matrix to not mix the actions of current trial with past trials.
            agent_list[i].g_matrix=np.zeros((agent_list[i].num_actions, agent_list[i].num_percepts), dtype=np.float64) 
        
        #Initialize global g matrix.
        global_g_matrix=np.zeros((agent_list[i].num_actions, agent_list[i].num_percepts), dtype=np.float64) 
        
        test_sting=np.zeros(group_size)
        test_killed=0
        test_predator=1 #predator is there.
        predator_leaving=0 #predator is not leaving.
        counter_pher=0    
        counter_rounds_nopredator=0
               
        predator_resistance=env.rchoose() #random choice of predator.

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
                    predator_leaving=1
        
        #obtain the reward
        reward=env.get_reward(group_size-min(group_size,np.sum(test_sting)+np.sum(test_killed))) 
        
        #store data
        learning_curve[i_trial]=reward
        number_stung[i_trial]=np.sum(test_sting)
        which_stingevol[i_trial%1000]=test_sting
        sizes[i_trial]=predator_resistance
        kills[i_trial]=np.sum(test_killed)
        
        
       
    #save data for this population
    learning_curve_allpop[pop]=learning_curve
    for i in range(env.num_percepts_list[0]):
        prob_stinging_allpop[pop,i]=agent_list[0].h_matrix[1,i]/np.sum(agent_list[0].h_matrix[:,i])#all agents have same h matrix.
     
    number_stung_allpop[pop]=number_stung
    which_sting[pop]=np.sum(which_stingevol,axis=0)/1000
    predator_sizes_allpop[pop]=sizes
    predator_kills_allpop[pop]=kills
    
    # np.savetxt('live_nolearning_ps=0.5.txt',learning_curve_allpop)
#    np.savetxt('number_stung_nolearning.txt',number_stung_allpop)
    # np.savetxt('predator_sizes_nolearning_ps=0.5.txt',predator_sizes_allpop)
#    np.savetxt('predator_kills_nolearning.txt',predator_kills_allpop)

    
#%% study of performance after learning (used in case study: African vs European bees)
    
import numpy as np

import beesting_predator_pheromone


#parameters we set for environment:
group_size=200
num_bins=10 
min_units_perbin=3
full_resolution=False
range_predators=[55,55]
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
num_trials=100
num_pop=50

#initialization of the object "environment" and load learned probabilities.
env=beesting_predator_pheromone.Beesting_predator(group_size, num_bins, min_units_perbin, full_resolution,range_predators,perc_false_alarms,rew_scaling)
prob_stinging_allpop=np.loadtxt('prob_stinging_sth1525_g003_Dtv10_tatt0_G200.txt')
#load here learned probabilities of populations trained with a different range of predators than the ones that will attack on this simulation.

for pop in range(num_pop): #the p_s for the pheromone percepts that are never perceived during the learning process is changed from 0.5 to the value p_s for the "last" pheromone percept that was commonly perceived (last point of the decaying part of the curve).
    for i in range(1,len(prob_stinging_allpop[0])):
        if prob_stinging_allpop[pop][i] == 0.5:
           prob_stinging_allpop[pop][i] = prob_stinging_allpop[pop][i-1]

alive_bees_allpop=np.zeros([num_pop,num_trials])
stings_allpop=np.zeros([num_pop,num_trials])


for pop in range(num_pop):
    
    #initialize a record of performance for this population.
    alive_bees=np.zeros(num_trials)
    stings=np.zeros(num_trials)

    #interaction
   
    for i_trial in range(num_trials):
        
        test_sting=np.zeros(group_size)
        test_killed=0
        test_predator=1 #predator is there.
        predator_leaving=0 #predator is not leaving.
        counter_pher=0    
        counter_rounds_nopredator=0
        
        
        predator_resistance=env.rchoose()

        for i in range(group_size):
            
            action=np.random.choice([0,1],p=[1-prob_stinging_allpop[pop][env.get_percept(counter_pher,predator_leaving)[0]],prob_stinging_allpop[pop][env.get_percept(counter_pher,predator_leaving)[0]]])
            counter_pher+=np.copy(action)
            test_sting[i]=action
                
            test_predator=env.scare_predator(np.sum(test_sting),predator_resistance)  #check if predator is scared away.   

            if test_predator and i>=time_attack:#predator's attack, only if it is not already scared away.  
                test_killed+=1
            if (test_predator+1)%2:
                counter_rounds_nopredator+=1
                if counter_rounds_nopredator>=visual_time_delay:
                    predator_leaving=1
                
        reward=env.get_reward(group_size-min(group_size,np.sum(test_sting)+np.sum(test_killed))) 
        
        alive_bees[i_trial]=reward
        stings[i_trial]=np.sum(test_sting)
       
    #save data for this population
    alive_bees_allpop[pop]=alive_bees
    stings_allpop[pop]=stings

    
    

# np.savetxt('alive_bees_trainedEURO_tested55.txt',alive_bees_allpop)
# np.savetxt('stings_trainedEURO_tested55.txt',stings_allpop)
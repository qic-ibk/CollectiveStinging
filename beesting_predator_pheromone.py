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


class Beesting_predator(object):
    
    
    def __init__(self, group_size, num_bins, min_units_perbin,full_resolution,range_predators,perc_false_alarms,rew_scaling):
        """Initialises the environment. Arguments:
        Group size: size of the bee colony (number N of PS agents in the ensemble).
        num_bins: initial request for number of logarithmic bins in which the percept space is split. The minimum size of each bin is also stablished, so the number of bins may change after bins with less pheromone units are merged.
        min_units_perbin: minimum number of pheromone units per bin.
        full_resolution: True if bee can distinguish every released pheromone unit (not in ranges), False otherwise.
        range_predators: range of number of stings (s_th) needed to scare away predators. 
        perc_false_alarms: percentage of trials where there is a false alarm (r_f) and no predator (predator_size=0).
        rew_scaling: if the reward function is linear or quadratic."""
        
        self.group_size = group_size;
        self.num_bins = num_bins;
        self.min_units_perbin = min_units_perbin;
        self.full_resolution = full_resolution;
        self.range_predators = range_predators;
        self.perc_false_alarms = perc_false_alarms;
        self.rew_scaling = rew_scaling;


        self.num_actions=2 #chill (0), sting  (1) .
        
        self.pher_concentration=self.create_bins() #splitting of the N pheromone units into bins of logarithmic size.
        
        #total number of percepts
        if self.full_resolution:
            self.num_percepts_list = [self.group_size+1]
        else:
            self.num_percepts_list = [len(self.pher_concentration)+2] #the logarithmic bins, plus the percept corresponding to 0 pheromone units, plus the visual percept v_ESC.
        
    def create_bins(self):
        """It splits the range of pheromone units into logarithmic bins with at least min_units_perbin units.
        Output
        -------
        pher_concentration (np.array): an array with the lower limits of each bin."""
        
        units_perbin=np.zeros(self.num_bins-1)
        
        for i in range(self.num_bins-1):
            for unit in range(1,self.group_size):
                if unit>=np.logspace(0,np.log10(self.group_size), self.num_bins,endpoint=False)[i] and unit<np.logspace(0,np.log10(self.group_size), self.num_bins,endpoint=False)[i+1]:
                    units_perbin[i]+=1
        
        pher_concentration=[]
        a=units_perbin[0]
        for i in range(len(units_perbin)-1):            
            
            if a<self.min_units_perbin:
                a+=units_perbin[i+1]
            else:
                pher_concentration.append(a)
                a=units_perbin[i+1]
        
        pher_concentration.append(units_perbin[len(units_perbin)-1])
        pher_concentration=np.cumsum(np.array([0]+pher_concentration)) 
        
        return (pher_concentration)
        
    def get_percept(self,counter_pher,predator_leaving):
        """It computes the percept considering that counter_pher units of pheromone are already released. In addition, if predator is
        leaving, the corresponding visual percept is activated.
        
        Input
        -------
        counter_pher (int): released pheromone units
        predator_leaving (bool): True if predator is perceived leaving.
        
        Output
        -------
        List [percept] with the percept number."""
        
        if predator_leaving:
            return ([self.num_percepts_list[0]-1])
        
    
        elif self.full_resolution:
            return([counter_pher])
            
        else:
            if counter_pher==0:
                percept=0
            else:
                for i in range(len(self.pher_concentration)):
                    if counter_pher>self.pher_concentration[i]:
                        percept=(i+1)
              
            return([percept])
            
    def rchoose(self):
        """It outputs the predator resistance (int), i.e. the number of stings s_th needed for predator to stop its attack.
        Each s_th is chosen randomly from the range self.range_predators. Predator_resistance=0 (false alarm) with a probability specified in perc_false_alarms. """
            
        return np.random.choice([0,np.random.randint(self.range_predators[0],self.range_predators[1]+1)],p=[self.perc_false_alarms,1-self.perc_false_alarms])
    
    
    def scare_predator(self,counter_sting,predator_resistance):
        """Given the number of bees that already stung, it computes if the predator keeps attacking (1) or is scared away (0).
        Input
        -------
        counter_sting (int): number of bees that have already stung.
        predator_resistance (int): s_th of the predator in the current trial.
        
        Output
        -------
        1: predator keeps attacking.
        0: predator is scared away. """
        
        if counter_sting>=predator_resistance: 
            return 0
        else:
            return 1
        
    def get_reward(self,alive_bees):
        """Given how many bees are alive, it computes the reward.
        Input
        -------
        alive_bees (int)
        
        Output
        -------
        Reward (float), 0<=R<=1. """
        
        if self.rew_scaling=='linear':
            return(alive_bees/self.group_size)
            
        if self.rew_scaling=='quadratic':
            return((alive_bees/self.group_size)**2)
        
    
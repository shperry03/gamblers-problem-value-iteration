'''
Gamblers problem for sutton 4.9
Value iteration

'''
import numpy as np
import csv

class Gamble:
    
    def __init__(self,pHeads) -> None:
        self.pHeads = pHeads
        self.states = list(range(0,100))
        self.values = np.zeros(101)
        self.rewards = np.zeros(101)


    def calculate_action_val(self,discount,action,s):

        sum = (self.pHeads * (self.rewards[action+s] + discount * self.values[action+s])) + ((1-self.pHeads) * (self.rewards[s-action] + discount * self.values[s-action]))
        return sum


    def val_iter(self,theta):
        self.rewards[100] = 1
 
        while True:
            delta = 0
            for s in self.states:
                actions = range(0,min(s,100-s)+1)
                v = self.values[s]
                val_list = np.zeros(101)
                for a in actions:
                    val_list[a] = self.calculate_action_val(1,a,s)
                self.values[s] = np.max(val_list)
                delta = max(delta,abs(v - self.values[s]))


            if theta > delta:
                break

gam_25 = Gamble(0.25)
gam_25.val_iter(0.001)
gam_25_01 = Gamble(0.25)
gam_25_01.val_iter(0.1)
gam_55 = Gamble(0.55)
gam_55.val_iter(0.001)
gam_55_01 = Gamble(0.55)
gam_55_01.val_iter(0.1)

with open("gam_25.csv","w") as file:
    writer = csv.writer(file)
    writer.writerow(gam_25.values)
with open("gam_25_01.csv","w") as file:
    writer = csv.writer(file)
    writer.writerow(gam_25_01.values)
with open("gam_55.csv","w") as file:
    writer = csv.writer(file)
    writer.writerow(gam_55.values)
with open("gam_55_01.csv","w") as file:
    writer = csv.writer(file)
    writer.writerow(gam_55_01.values)


# run this script to analyze the exe time to obtain f(X) with reasonable RSE
import sys
sys.path.append("src")
import SupplyChainModelv2 as model
import numpy as np
import pandas as pd
import os
from csv import writer
#import multiprocessing as mp
#print("Number of processors: ", mp.cpu_count())
distributor1 = {'name':'D1',
                'S': 500, # inventory capacity
                's': 350, # inventory threshold
                'H': 1, # inventory holding cost
                'C': [500], # delivery cost from manufacturer
                'D': [7] # delivery time
               }

distributor2 = {'name':'D2',
                'S': 500,
                's': 350,
                'H': 1,
                'C': [500],
                'D': [8]}

retailer1 = {'name':"R1",
             'S': 500,
             's': 300,
             'H': 10,
             'C':[5000,6000], # delivery cost per distributor
             'D':[2,3] # delivery time per distributor
            } 
retailer2 = {'name':"R2",
             'S': 500,
             's': 300,
             'H': 10,
             'C':[7000,5500],
             'D':[3,2]}

R_list = [retailer1,retailer2]
D_list = [distributor1,distributor2]

# this fun sets the design parameters of the supply chain
def N_sim_runs(D_list,R_list,
               S_D1, s_D1, S_D2, s_D2, S_R1, s_R1, S_R2, s_R2,
               arr_rate,p,NUM_OF_DAYS,Profit,
               NUM_OF_SIMS):
    R_list[0]['S'] = S_R1
    R_list[0]['s'] = s_R1
    R_list[1]['S'] = S_R2
    R_list[1]['s'] = s_R2
    D_list[0]['S'] = S_D1
    D_list[0]['s'] = s_D1
    D_list[1]['S'] = S_D2
    D_list[1]['s'] = s_D2
    
    avg_stats = []
    avg_nstats = []
    for i in range(NUM_OF_SIMS):
        frac_cust_ret,avg_profit,avg_hold_c,avg_del_c,timed_avg_nitems,avg_net_profit,nwise_stats = model.single_sim_run(lam=arr_rate, D_list=D_list, R_list=R_list, p=p, NUM_OF_DAYS=NUM_OF_DAYS, P=Profit)
        avg_stats.append([frac_cust_ret,avg_profit,avg_hold_c,avg_del_c,timed_avg_nitems,avg_net_profit])
        for i in range(len(nwise_stats)):
            if(len(avg_nstats)<=i):
                nwise_stats[i].pop(0)
                avg_nstats.append([x/NUM_OF_SIMS for x in nwise_stats[i]])
            else:
                for j in range(1,len(nwise_stats[i])):
                    avg_nstats[i][j-1] = avg_nstats[i][j-1] + nwise_stats[i][j]/NUM_OF_SIMS
    avg_stats = np.array((avg_stats))
    avg_stats = np.mean(avg_stats,axis=0)
    temp = []
    for i in avg_nstats:
        for j in i:
            temp.append(j)
    return [S_R1,s_R1,S_R2,s_R2,S_D1,s_D1,S_D2,s_D2,*avg_stats,*temp]

# parameters
lambda_arr_rate = 20
p = [0.5,0.5]
Profit = 100
num_days = 90
num_sims = 60

import time
start_time = time.time()


R_list[0]['S'] = 300
R_list[0]['s'] = 200
R_list[1]['S'] = 350
R_list[1]['s'] = 200
D_list[0]['S'] = 700
D_list[0]['s'] = 350
D_list[1]['S'] = 650
D_list[1]['s'] = 400

filename = 'data/exe_time_rse_ana.csv'
print("days    sims    std    err    rse    mean    time")
#with open(filename, 'a', newline='') as f_object:
#    writer_object = writer(f_object)
#    writer_object.writerow(["days","sims","std","err","rse","mean","time"])
#    f_object.close()
for num_sims in [100, 200, 400]:
    for num_days in range(100,1100,100):
        avg_stats_arr = []
        for i in range(0,num_sims):
            frac_cust_ret, avg_profit, avg_hold_c, avg_del_c, timed_avg_nitems, avg_net_profit, nwise_stats = model.single_sim_run(lam=lambda_arr_rate, D_list=D_list, R_list=R_list, p=p, NUM_OF_DAYS=num_days, P=Profit)
            avg_stats_arr.append(avg_net_profit)
        std_pnet = np.std(avg_stats_arr)
        err_pnet = std_pnet/np.sqrt(num_sims)
        rse_pnet = err_pnet*100/np.mean(avg_stats_arr)
        exe_time = time.time()-start_time
        #print("days=",num_days," sims=",i," std=",std_pnet," err=",err_pnet," rse=",rse_pnet," mean=",np.mean(avg_stats_arr)," time=", exe_time)
        print(f"{num_days},   {num_sims},   {std_pnet:.2f},   {err_pnet:.2f},   {rse_pnet:.2f},   {np.mean(avg_stats_arr):.2f},   {exe_time:.2f}")
        with open(filename, 'a', newline='') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([num_days,num_sims,std_pnet,err_pnet,rse_pnet,np.mean(avg_stats_arr),exe_time])
            f_object.close()
#start_time = time.time()
#num_days = 90
#num_sims = 60
#N_sim_runs(D_list=D_list,R_list=R_list,S_D1=600, s_D1=350, S_D2=650, s_D2=400, S_R1=350, s_R1=300, S_R2=300, s_R2=100,arr_rate=lambda_arr_rate,p=p,NUM_OF_DAYS=num_days,Profit=Profit,NUM_OF_SIMS=num_sims)
#print("time using N_sim fun = ", time.time()-start_time)

# output_filename = 'exe_time_rse_ana.csv'
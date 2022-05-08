import os
from typing import *
import numpy as np
import random
import numpy.linalg as LA
import numba as nb
import pandas as pd
from numba import jit,float32,int32 
import types
import pickle
import numpy as np
from collections import defaultdict
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import sys
import sklearn
from sklearn.datasets import load_iris,load_wine,load_digits
import time
import argparse
import concurrent.futures 

data=None

# calulating the feedback of the datapoint ( given true label and pred label)
@jit(nopython=True)
def feedback(true_label:int,predicted_label:int)->int:
  fb=0
  if true_label==predicted_label:
    fb=1
  return fb

# returns the prob of selecting each class label 
@jit(nopython=True)
def cal_prob(cal_label:int,gamma:float,k:int)->np.ndarray:
  prob= np.ones(k)
  prob = prob*gamma/k
  prob[cal_label]+= 1 -gamma
  # print("Prob",prob)
  return prob

# random sample a class label according to the prob dist 
@jit(nopython=True)
def random_sample(prob:np.ndarray)->int:
  number = float32(random.random()) * np.sum(prob)
  for i in range(0,prob.shape[0]):
    if number < prob[i]:
      return i
    number -= prob[i]
  return prob.shape[0]-1

@jit(nopython=True)
def Run(
  k:int,
  d:int,
  size:int,
  gamma:float,
  delta:float,
  lamda:float,
  )->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,int]:

  weight_matrix=np.zeros((k,d))
  incorrect_classified=0
  error_rate=0
  correct_classified=0
  error_rate_list=np.zeros((size))
  query_cnt=0
  query_rate=np.zeros((size))
  query_error_rate=np.zeros((size))
  mt=1/np.sqrt(lamda)
  loss=float32(0)
  
  for i in range(0,size):
    num=random.randint(0,data.shape[0]-1)
    entry= data[num,:]
    feature_vector=np.reshape(entry[0:d],(d,1))
    true_label=int(entry[d])
    score=np.reshape((np.dot(weight_matrix,feature_vector)),-1)
    y_hat=np.argmax(score)   # calculated label
    prob=cal_prob(y_hat,gamma,k)    
    y_tilde= random_sample(prob) #predicted label 

    cp_score=score.copy()
    cp_score.sort()
    
    # deciding whether to query or not 
    pt=max(0,cp_score[-1]-cp_score[-2])   
    prob_Q= delta/(delta+pt)
    cutoff=random.random()
    Q=0
    if cutoff<prob_Q:
      Q=1


    if true_label==y_tilde:    
      correct_classified+=1
    else:
      incorrect_classified+=1

    error_rate=incorrect_classified/(i+1)
    error_rate_list[i] = error_rate
    
    if Q==1 or y_tilde!=y_hat:
      query_error_rate[query_cnt]=error_rate
      query_cnt+=1
      nt=1/(lamda*(query_cnt))
      fb=feedback(true_label,y_tilde)
      loss= 1+ (1- 2*fb)*score[y_tilde]
      if loss>0.0:
        basis_vec=np.zeros((k,1))
        if fb==1:
          basis_vec[y_tilde,0]+=nt
        else:
          basis_vec[y_tilde,0]-=nt
        weight_matrix= (1-nt*lamda)*weight_matrix + np.kron(basis_vec,feature_vector.T)
      else:
         weight_matrix= (1-nt*lamda)*weight_matrix
      mn=LA.norm(weight_matrix) 
      if mn !=0:
        weight_matrix*=min(1,mt/mn)
    query_rate[i] = query_cnt/(i+1)

  return query_rate,error_rate_list,query_error_rate,query_cnt



if __name__=="__main__":
  parser = argparse.ArgumentParser()

  # hyperparameters sent by the client are passed as command-line arguments to the script.
  parser.add_argument("--data", type=str,default='synsep')
  parser.add_argument("--dim", type=int,default=400)
  parser.add_argument("--num_class", type=int,default=10)
  parser.add_argument("--repition", type=int, default=20)
  parser.add_argument("--size", type=float,default=1e5)
  parser.add_argument("--ll_gamma", type=int,default=-8)
  parser.add_argument("--ul_gamma",type=int,default=0)
  parser.add_argument("--ll_delta", type=int, default=-10)
  parser.add_argument("--ul_delta", type=int, default=-1)
  parser.add_argument("--ll_lamda", type=int, default=-12)
  parser.add_argument("--ul_lamda", type=int, default=7)

  args, _ = parser.parse_known_args()


  prefix=f"/home/{os.getenv('USER')}/exp/dataset"
# Dataset 
  # SynSep Data
  if args.data=='synsep':
    data=np.load(f'{prefix}/syn_sep.npy','r')
  # SynNonSep Data
  elif args.data=='synnonsep':
    data=np.load(f'{prefix}/syn_nonsep.npy','r') 
  # Fashion MNIST
  elif args.data=='fashion':
    data=np.load(f'{prefix}/fashion.npy','r') 
  # USPS dataset 
  elif args.data=='usps':
    dataset=[]
    with open(f'{prefix}/upsp.csv','r') as f:
      lines=f.readlines()
      for i in lines:
        temp=i.strip().split(',')
        dataset.append([np.float64(i) for i in temp])
    data=np.array(dataset)
  # Ecoli dataset 
  elif args.data=='ecoli':
    data=np.load(f'{prefix}/ecoli.npy','r')
  # Abalone dataset 
  elif args.data=='abalone':
    data=np.load(f'{prefix}/abalone.npy','r')
 
  data=np.float64(data)



  # Hyper-parameters
  k=args.num_class
  d=args.dim
  repition=args.repition
  size=int(args.size)
  # gamma
  gamma_values=2.0**np.arange(args.ll_gamma,args.ul_gamma)
  # delta
  delta_values=[2**i for i in range(args.ll_delta,args.ul_delta)]
  # lamda 
  lamda_values=[2**i for i in range(args.ll_lamda,args.ul_lamda)]


  avg_time=0
  query_rate ,error_rate,query_error_rate=np.zeros([repition,size]),np.zeros([repition,size]),np.zeros([repition,size])
  avg_query_rate ,avg_error_rate,avg_query_error_rate=np.zeros([2,size]),np.zeros([2,size]),np.zeros([2,size])
  best_query_rate ,best_error_rate,best_query_error_rate=None,None,None

  min_query_cnt=size
  best_error_rate=1
  best_gamma=1
  best_lamda=1
  best_delta=1
  best_avg_time=0

  data_dict={
    'gamma':[],
    'delta':[],
    'lamda':[],
    'error_rate':[],
    'avg_time':[],
    'query_rate':[],
    'min_query_cnt':[],
  }
  for gamma in gamma_values:
    for delta in delta_values:
      for lamda in lamda_values:
        # Parallel
        t0= time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor() as executor:
          output = [executor.submit(Run,k,d,size,gamma,delta,lamda)for _ in range(0,repition)]
          results = [f.result() for f in concurrent.futures.as_completed(output)]
        t1 = round(time.perf_counter() - t0,2)
        avg_time=t1
        for ind,mat in enumerate(results):
          query_rate[ind]=mat[0]
          error_rate[ind]=mat[1]
          query_error_rate[ind]=mat[2]
          min_query_cnt=min(min_query_cnt,mat[3]) 
       
        avg_query_rate[0]=query_rate.mean(axis=0)
        avg_query_rate[1]=query_rate.std(axis=0)
        avg_error_rate[0]=error_rate.mean(axis=0)
        avg_error_rate[1]=error_rate.std(axis=0)
        avg_query_error_rate[0]=query_error_rate.mean(axis=0)
        avg_query_error_rate[1]=query_error_rate.std(axis=0)
        
        print(f"{len(results)} Runs Completed for gamma:{gamma}, delta:{delta} and lamda:{lamda} with error_rate {avg_error_rate[0,-1]} and querry rate {avg_error_rate[0,-1]} in {avg_time} seconds")
        
        data_dict['gamma'].append(gamma)
        data_dict['delta'].append(delta)
        data_dict['lamda'].append(lamda)
        data_dict['error_rate'].append(avg_error_rate[0,-1])
        data_dict['avg_time'].append(avg_time)
        data_dict['min_query_cnt'].append(min_query_cnt)
        data_dict['query_rate'].append(avg_query_rate[0,-1])
        
        if avg_error_rate[0,-1] < best_error_rate:
          best_gamma=gamma
          best_delta=delta
          best_lamda=lamda
          best_avg_time=avg_time
          best_error_rate=avg_error_rate.copy()
          best_query_rate=query_rate.copy()
          best_query_error_rate=avg_query_error_rate.copy()
          best_error_rate=avg_error_rate[0,-1]

  cwd = os.getcwd()
  cwd= cwd+f'/{args.data}'
  if not os.path.exists(cwd):
    print("making dir",cwd)
    os.mkdir(cwd)

  np.save(f'{cwd}/query_rate', best_query_rate)
  np.save(f'{cwd}/error_rate', best_error_rate)
  np.save(f'{cwd}/query_error_rate', best_query_error_rate)
  with open(f'{cwd}/param.txt','w') as f:
    f.write("AvgTime: "+str(best_avg_time)+"\n")
    f.write("ErrorRate: "+str(best_error_rate)+"\n")
    f.write("BestGamma: "+str(best_gamma)+"\n")
    f.write("BestDelta: "+str(best_delta)+"\n")
    f.write("BestLamda: "+str(best_lamda)+"\n")

# Saving data
  df=pd.DataFrame.from_dict(data_dict)
  df.to_parquet(cwd+'/data.parquet')

## Pegasos following 

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
import multiprocessing as mp
import   concurrent.futures 

data=None


@jit(nopython=True)
def feedback(true_label:int,predicted_label:int)->int:
  fb=0
  if true_label==predicted_label:
    fb=1
  return fb

@jit(nopython=True)
def cal_prob(cal_label:int,gamma:float,k:int)->np.ndarray:
  prob= np.ones(k)
  prob = prob*gamma/k
  prob[cal_label]+= 1 -gamma
  # print("Prob",prob)
  return prob

@jit(nopython=True)
def random_sample(prob:np.ndarray)->int:
  number = float32(random.random()) * np.sum(prob)
  # print("Sum prob",sum(prob), number)
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
  )->Tuple[np.ndarray,np.ndarray]:

  weight_matrix=np.zeros((k,d))
  incorrect_classified=0
  error_rate=0
  correct_classified=0
  error_rate_list=np.zeros((size))
  q_cnt=0
  frac_qcnt_list=np.zeros((size))
  qerr=np.zeros((size))
  mt=1/np.sqrt(lamda)
  max_norm=0
  L=float32(0)
  for i in range(0,size):
    num=random.randint(0,data.shape[0]-1)
    entry= data[num,:]
    feature_vector=np.reshape(entry[0:d],(d,1))
    true_label=int(entry[d])
    val_f=np.reshape((np.dot(weight_matrix,feature_vector)),-1)
    y_hat=np.argmax(val_f)   # calculated label
    prob=cal_prob(y_hat,gamma,k)    # hyperparameters sent by the client are passed as command-line arguments to the script.
    y_tilde= random_sample(prob) #predicted label 

    Q=0
    temp=np.reshape(np.dot(weight_matrix,feature_vector),-1)
    temp.sort()
    pt=max(0,temp[-1]-temp[-2])   # does not account for the case when y_tilde!=y_hat ( in that case pt would be 0, so Qt is 1)

    prZ= delta/(delta+pt)
    cutoff=random.random()
    
    if cutoff<prZ:
      Q=1

    if true_label==y_tilde:    
      correct_classified+=1
    else:
      incorrect_classified+=1

    error_rate=incorrect_classified/(i+1)
    error_rate_list[i] = error_rate
    
    if Q==1 or y_tilde!=y_hat:
      qerr[q_cnt]=error_rate
      q_cnt+=1
      nt=1/(lamda*(q_cnt))
      fb=feedback(true_label,y_tilde)
      L= 1+ (1- 2*fb)*val_f[y_tilde]
      if L>0.0:
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
      max_norm=max(max_norm,LA.norm(weight_matrix))
    frac_qcnt_list[i] = q_cnt/(i+1)
  return frac_qcnt_list,error_rate_list,qerr,q_cnt,max_norm



if __name__=="__main__":
  parser = argparse.ArgumentParser()

  # hyperparameters sent by the client are passed as command-line arguments to the script.
  parser.add_argument("--data", type=str,default='digits')
  parser.add_argument("--dim", type=int,default=64)
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

  print(args)

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
  elif args.data=='ecoli':
    data=np.load(f'{prefix}/ecoli.npy','r')
  elif args.data=='abalone':
    data=np.load(f'{prefix}/abalone.npy','r')
 
  # print(data.shape)  
  data=np.float64(data)



  # Hyper-parameters
  k=args.num_class
  d=args.dim
  repition=args.repition
  size=int(args.size)
  # gamma
  # g_val=2.0**np.arange(args.ll_gamma,args.ul_gamma)
  # # parameter delta
  # d_val=[2**i for i in range(args.ll_delta,args.ul_delta)]
  # # d_val=[0]
  # l_val=[2**i for i in range(args.ll_lamda,args.ul_lamda)]


  g_val=[args.gamma]
  # parameter delta
  d_val=[2**(-1*args.delta)]
  # d_val=[0]
  l_val=[args.lamda]
  print(g_val,d_val,l_val)
  avg_time=0
  frac_query_rate ,final_list,query_error_rate=np.zeros([repition,size]),np.zeros([repition,size]),np.zeros([repition,size])
  avg_frac_query_rate ,avg_final_list,avg_final_qer=np.zeros([2,size]),np.zeros([2,size]),np.zeros([2,size])
  best_avg_frac_query_rate ,best_avg_final_list,best_avg_final_qer=None,None,None

  best_qcnt=10
  avg_qcnt=size
  best_error_rate=10
  best_gamma=1
  best_lamda=1
  best_avg_time=0
  best_delta=0
  file_name=""
  data_file1=""
  data_file2=""
  max_norm=0
  avg_max_norm=0
  data_dict={
    'gamma':[],
    'delta':[],
    'lamda':[],
    'error_rate':[],
    'avg_time':[],
    'frac_query_rate':[],
    'avg_cnt':[],
  }
  for exp in g_val:
    for de in d_val:
      for lm in l_val:
        # Parallel
        t0= time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor() as executor:
          output = [executor.submit(Run,k, d,size,exp,de,lm)for _ in range(0,repition)]
          results = [f.result() for f in concurrent.futures.as_completed(output)]
        t1 = round(time.perf_counter() - t0,2)
        avg_time=t1
        max_norm=0
        for ind,mat in enumerate(results):
          frac_query_rate[ind]=mat[0]
          final_list[ind]=mat[1]
          query_error_rate[ind]=mat[2]
          avg_qcnt=min(avg_qcnt,mat[3]) 
          max_norm+=mat[4] 
       
        avg_max_norm=avg_max_norm/repition
        avg_frac_query_rate[0]=frac_query_rate.mean(axis=0)
        avg_frac_query_rate[1]=frac_query_rate.std(axis=0)
        avg_final_list[0]=final_list.mean(axis=0)
        avg_final_list[1]=final_list.std(axis=0)
        avg_final_qer[0]=query_error_rate.mean(axis=0)
        avg_final_qer[1]=query_error_rate.std(axis=0)
        print(f"{len(results)} Runs Completed for gamma:{exp}, delta:{de} and lamda:{lm} with error_rate {avg_final_list[0,-1]} and querry rate {avg_frac_query_rate[0,-1]} norm ({avg_max_norm}) in {avg_time} seconds")
        data_dict['gamma'].append(exp)
        data_dict['delta'].append(de)
        data_dict['lamda'].append(lm)
        data_dict['error_rate'].append(avg_final_list[0,-1])
        data_dict['avg_time'].append(avg_time)
        data_dict['avg_cnt'].append(avg_qcnt)
        data_dict['frac_query_rate'].append(avg_frac_query_rate[0,-1])
        
        if avg_final_list[0,-1] < best_error_rate:
          best_gamma=exp
          best_delta=de
          best_lamda=lm
          best_avg_time=avg_time
          best_avg_final_list=avg_final_list.copy()
          best_avg_frac_query_rate=frac_query_rate.copy()
          best_avg_final_qer=query_error_rate.copy()
          best_error_rate=avg_final_list[0,-1]

  cwd = os.getcwd()
  cwd= cwd+f'/{args.data}'
  if not os.path.exists(cwd):
    print("making dir",cwd)
    os.mkdir(cwd)

  file_name ="param.txt"
  data_file1="frac_query_rate"
  data_file2="error_rate"
  data_file3="query_error_rate"

  np.save(f'{cwd}/{data_file1}', best_avg_frac_query_rate)
  np.save(f'{cwd}/{data_file2}', best_avg_final_list)
  np.save(f'{cwd}/{data_file3}', best_avg_final_qer)
  with open(f'{cwd}/{file_name}','w') as f:
    f.write("AvgTime: "+str(best_avg_time)+"\n")
    f.write("ErrorRate: "+str(best_error_rate)+"\n")
    f.write("BestGamma: "+str(best_gamma)+"\n")
    f.write("BestDelta: "+str(best_delta)+"\n")
    f.write("BestLamda: "+str(best_lamda)+"\n")

  df=pd.DataFrame.from_dict(data_dict)
  df.to_parquet(cwd+'/data.parquet')

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 18:16:14 2023

@author: Zhou xi
"""
import numpy as np
import tensorflow as tf
import utils
from scipy.sparse import vstack
from evalModel import train_and_evaluate
import scipy.io
from scipy.sparse import lil_matrix
import warnings
warnings.filterwarnings("ignore")

tf.set_random_seed(0)
np.random.seed(0)


source= 'squirrel1' 
target = 'squirrel2'

emb_filename=str(source)+'_'+str(target)

###model config
config=dict()
config['clf_type'] = 'multi-class'
config['batch_size'] = 100
config['num_epoch'] = 50
config['n_hidden'] = [512,128]
config['n_emb'] = 128
config['emb_filename'] =emb_filename
config['l2_w'] = 0.0005
config['FP_w'] = 1e-4
config['lr_ini'] = 0.001
config['dropout'] = 0.1
config['kstep']=3

    
resultfilename='./DM_GNN_'+str(source)+'_'+str(target)+'_results.txt'
f=open(resultfilename,'a')
f.write('{}'.format(config))
f.write('\n')

Kstep=config['kstep']

####################
# Load source data
####################
A_s, X_s, Y_s= utils.load_network('./input/'+str(source)+'.mat') 
num_nodes_S=X_s.shape[0]

####################
# Load target data
####################
A_t, X_t, Y_t = utils.load_network('./input/'+str(target)+'.mat')
num_nodes_T=X_t.shape[0]

'''compute PPMI'''
A_k_s=utils.AggTranProbMat(A_s, Kstep) 
PPMI_s=utils.ComputePPMI(A_k_s) 
n_PPMI_s=utils.MyScaleSimMat(PPMI_s)   # row normalized PPMI         
X_n_s=np.matmul(n_PPMI_s,lil_matrix.toarray(X_s)) #neibors' attribute matrix   

'''compute PPMI'''
A_k_t=utils.AggTranProbMat(A_t, Kstep) 
PPMI_t=utils.ComputePPMI(A_k_t)
n_PPMI_t=utils.MyScaleSimMat(PPMI_t)   # row normalized PPMI
X_n_t=np.matmul(n_PPMI_t,lil_matrix.toarray(X_t)) #neibors' attribute matrix


##input data
input_data=dict()
input_data['PPMI_S']=PPMI_s
input_data['PPMI_T']=PPMI_t 
input_data['attrb_S']=X_s
input_data['attrb_T']=X_t
input_data['attrb_nei_S']=X_n_s
input_data['attrb_nei_T']=X_n_t
input_data['label_S']=Y_s
input_data['label_T']=Y_t
    


         
numRandom=5
microAllRandom=[]
macroAllRandom=[]
best_microAllRandom=[]
best_macroAllRandom=[]

for random_state in range(numRandom):

    print("%d-th random initialization for unsupervised domain adaptation" %(random_state+1))
    micro_t,macro_t, best_micro_t, best_macro_t=train_and_evaluate(input_data, config, random_state) 
    microAllRandom.append(micro_t)
    macroAllRandom.append(macro_t)
    best_microAllRandom.append(best_micro_t)
    best_macroAllRandom.append(best_macro_t)        

'''avg F1 scores over 5 random splits'''
micro=np.mean(microAllRandom)
macro=np.mean(macroAllRandom)
micro_sd=np.std(microAllRandom)
macro_sd=np.std(macroAllRandom)

best_micro=np.mean(best_microAllRandom)
best_macro=np.mean(best_macroAllRandom)
best_micro_sd=np.std(best_microAllRandom)
best_macro_sd=np.std(best_macroAllRandom)

print ("source: %s and target: %s "%(source,target))    
print("The avergae micro and macro F1 scores over %d random initializations for unsupervised domain adaptation are:  %f + %f and %f + %f: " %(numRandom, micro, micro_sd, macro, macro_sd))   
print("The BEST avergae micro and macro F1 scores over %d random initializations for unsupervised domain adaptation are:  %f + %f and %f + %f: " %(numRandom, best_micro, best_micro_sd, best_macro, best_macro_sd))   

f.write("source: %s and target: %s "%(source,target))
f.write('\n')
f.write("The avergae micro and macro F1 scores over %d random initializations for unsupervised domain adaptation are:  %f + %f and %f + %f: " %(numRandom, micro, micro_sd, macro, macro_sd))
f.write('\n')
f.write("The BEST avergae micro and macro F1 scores over %d random initializations for unsupervised domain adaptation are:  %f + %f and %f + %f: " %(numRandom, best_micro, best_micro_sd, best_macro, best_macro_sd))
f.write('\n')
f.close()

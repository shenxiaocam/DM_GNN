# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 18:16:14 2023

@author: Zhou xi
"""

import numpy as np
import tensorflow as tf
import utils
from scipy.sparse import vstack
from functools import partial
import scipy.io
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from DM_GNN_model import DM_GNN
import numpy as np






def train_and_evaluate(input_data, config, random_state=0):
    
    ###get input data
    PPMI_s=input_data['PPMI_S']
    PPMI_t=input_data['PPMI_T']
    X_s=input_data['attrb_S']
    X_t=input_data['attrb_T']
    X_n_s= input_data['attrb_nei_S']
    X_n_t=input_data['attrb_nei_T']
    Y_s=input_data['label_S']
    Y_t=input_data['label_T']


    ###model config
    clf_type = config['clf_type'] 
    dropout = config['dropout'] 
    num_epoch = config['num_epoch'] 
    batch_size = config['batch_size']
    n_hidden = config['n_hidden'] 
    n_emb = config['n_emb'] 
    l2_w = config['l2_w'] 
    FP_w = config['FP_w']
    emb_filename = config['emb_filename'] 
    lr_ini = config['lr_ini'] 
    
    n_input = X_s.shape[1]
    num_class = Y_s.shape[1] 
    num_nodes_S=X_s.shape[0]
    num_nodes_T=X_t.shape[0]

    ##observable label matrix of target network
    Y_t_o=np.zeros(np.shape(Y_t))


    
    ##whether two nodes have same labels or not
    O_s=np.matmul(Y_s,Y_s.transpose())
    np.fill_diagonal(O_s, 0)  
    O_s[O_s>0]=1

    

    n_PPMI_s=utils.MyScaleSimMat(np.multiply(PPMI_s,O_s))
    n_PPMI_t=utils.MyScaleSimMat(PPMI_t)

 
    X_s_new=lil_matrix(np.concatenate((lil_matrix.toarray(X_s), X_n_s),axis=1))  
    X_t_new=lil_matrix(np.concatenate((lil_matrix.toarray(X_t), X_n_t),axis=1))

    whole_xs_xt_stt_self = utils.csr_2_sparse_tensor_tuple(vstack([X_s, X_t]))     
    whole_xs_xt_stt_nei = utils.csr_2_sparse_tensor_tuple(vstack([X_n_s, X_n_t]))  
             
    with tf.Graph().as_default():
        # Set random seed
        tf.set_random_seed(random_state)
        np.random.seed(random_state)
        model = DM_GNN(n_input, n_hidden, n_emb, num_class, clf_type, l2_w, FP_w)       
                      

    
        with tf.Session() as sess:
            # Random initialize
            sess.run(tf.global_variables_initializer())
    
            best_epoch = 0
            best_micro_f1 = 0
            best_macro_f1 = 0



            for cEpoch in range(num_epoch):  
                S_batches = utils.batch_generator([X_s_new,Y_s], int(batch_size / 2), shuffle=True)
                T_batches = utils.batch_generator([X_t_new,Y_t_o], int(batch_size / 2), shuffle=True) 
                
                num_batch=round(max(num_nodes_S/(batch_size/2),num_nodes_T/(batch_size/2)))
                
                # Adaptation param and learning rate schedule as described in the DANN paper 
                p=float(cEpoch) / (num_epoch)
                lr=lr_ini / (1. + 10 * p)**0.75   
                grl_lambda =2. / (1. + np.exp(-10. * p)) - 1 #gradually change from 0 to 1


                
                ##in each epoch, train all the mini batches
                for cBatch in range(num_batch):
                    ### batch used to train domain classifier, half from source, half from target
                    xs_ys_batch, shuffle_index_s = next(S_batches)
                    xs_batch=xs_ys_batch[0]
                    ys_batch =xs_ys_batch[1]
                    
                    xt_yt_batch, shuffle_index_t = next(T_batches)
                    xt_batch=xt_yt_batch[0]


                    ##whether two nodes have same labels or not
                    os_batch=np.matmul(ys_batch,ys_batch.transpose())
                    np.fill_diagonal(os_batch, 0)
                    os_batch[os_batch>0]=1

                    


                    x_batch = vstack([xs_batch, xt_batch])
                    batch_csr=x_batch.tocsr()
                    xb_self=utils.csr_2_sparse_tensor_tuple(batch_csr[:,0:n_input])
                    xb_nei=utils.csr_2_sparse_tensor_tuple(batch_csr[:,-n_input:])    
                    

                    domain_label = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),np.tile([0., 1.], [batch_size // 2, 1])]) #[1,0] for source, [0,1] for target

                    ##topological proximity matrix between nodes in the mini-batch
                    a_s, a_t=utils.batchPPMI(batch_size,shuffle_index_s,shuffle_index_t,PPMI_s,PPMI_t)
                    a_s=utils.MyScaleSimMat(np.multiply(a_s,os_batch)) 
                    a_t=utils.MyScaleSimMat(a_t)
                    
                    
  
                    _ ,dloss,tloss= sess.run([model.train_op,model.domain_loss,model.total_loss], feed_dict={model.X_self: xb_self, model.X_nei:xb_nei, model.ys_true: ys_batch,  model.d_label: domain_label, model.A_s: a_s, model.A_t: a_t, model.learning_rate: lr, model.Ada_lambda:grl_lambda, model.dropout:dropout, model.n_s:batch_size//2, model.n_t:batch_size//2})

                    

                                                
                '''Compute evaluation on test data by the end of each epoch'''       

                pred_prob_xs_xt,entropy_domain, closs, lloss, FP_loss= sess.run([model.pred_prob,model.entropy_domain, model.clf_loss_s, model.l2_loss, model.FP_loss], feed_dict={model.X_self:whole_xs_xt_stt_self, model.X_nei:whole_xs_xt_stt_nei, model.Ada_lambda:1., model.dropout:0., model.n_s:num_nodes_S, model.n_t:num_nodes_T, model.ys_true:Y_s,  model.A_s:n_PPMI_s, model.A_t:n_PPMI_t})                                                  
                pred_prob_xs=pred_prob_xs_xt[0:num_nodes_S,:]
                pred_prob_xt=pred_prob_xs_xt[-num_nodes_T:,:]

                

                print ('epoch: ', cEpoch+1) 
                print('lr',lr)
                print('grl_lambda',grl_lambda)  
                print('entropy_domain',entropy_domain)  
                print('clf loss',closs)      
                print('l2 loss',lloss) 
                print('FP_loss',FP_loss) 

                F1_s=utils.f1_scores(pred_prob_xs,Y_s)
                print('Source micro-F1: %f, macro-F1: %f' %(F1_s[0],F1_s[1]))            
                F1_t=utils.f1_scores(pred_prob_xt,Y_t)
                print('Target testing micro-F1: %f, macro-F1: %f' %(F1_t[0],F1_t[1]))


                if F1_t[1]>best_macro_f1:                    
                    best_micro_f1 = F1_t[0]
                    best_macro_f1 = F1_t[1]
                    best_epoch = cEpoch+1
                
                
        
            ''' save final evaluation on test data by the end of all epoches'''
            micro=float(F1_t[0])
            macro=float(F1_t[1])
            print('Target best epoch %d, micro-F1: %f, macro-F1: %f' %(best_epoch,best_micro_f1,best_macro_f1))
            
        
            
            ##save embedding features 
#            emb=sess.run(model.emb, feed_dict={model.X_self:whole_xs_xt_stt_self, model.X_nei:whole_xs_xt_stt_nei, model.Ada_lambda:u_0, model.dropout:0.})
#            hs=emb[0:num_nodes_S,:]
#            ht=emb[-num_nodes_T:,:]
#            print(np.shape(hs))
#            print(np.shape(ht))    
#            scipy.io.savemat(emb_filename+'_emb.mat', {'rep_S':hs, 'rep_T':ht})

            
    
    return micro,macro,float(best_micro_f1),float(best_macro_f1)





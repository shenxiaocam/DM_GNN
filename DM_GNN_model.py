"""
Created on Mon May 27 19:18:27 2020

@author: Shen xiao
"""


import numpy as np
import tensorflow as tf
import utils
from flip_gradient import flip_gradient




class DM_GNN(object):
    def __init__(self, n_input, n_hidden, n_emb, num_class, clf_type, l2_w,  FP_w):
        

        self.X_self = tf.sparse_placeholder(dtype=tf.float32) #input attr of all nodes       
        self.X_nei = tf.sparse_placeholder(dtype=tf.float32) #input attr of all nodes' neighbors
        self.ys_true = tf.placeholder(dtype=tf.float32)
        self.d_label = tf.placeholder(dtype=tf.float32) #domain label, source [1 0] or target [0 1]
        self.Ada_lambda = tf.placeholder(dtype=tf.float32) #grl_lambda Gradient reversal scaler   
        self.dropout = tf.placeholder(tf.float32)
        self.A_s=tf.placeholder(dtype=tf.float32) #network proximity matrix
        self.A_t=tf.placeholder(dtype=tf.float32) #network proximity matrix        
        self.learning_rate = tf.placeholder(dtype=tf.float32) 
#        self.mask= tf.placeholder(dtype=tf.float32) #check a node is with observable label or not      
        self.n_s = tf.placeholder(dtype=tf.int32)
        self.n_t = tf.placeholder(dtype=tf.int32)
        


        




        with tf.name_scope('Network_Embedding'):     
            ##feature exactor 1
            h1_self = utils.fc_layer(self.X_self, n_input, n_hidden[0], layer_name='hidden1_self', input_type='sparse')
            self.h2_self = utils.fc_layer(h1_self, n_hidden[0], n_hidden[1], layer_name='hidden2_self',drop=self.dropout)
             
            ##feature exactor 2
            h1_nei = utils.fc_layer(self.X_nei, n_input, n_hidden[0], layer_name='hidden1_nei', input_type='sparse') 
            self.h2_nei = utils.fc_layer(h1_nei, n_hidden[0], n_hidden[1], layer_name='hidden2_nei',drop=self.dropout)

   
            ##concatenation layer, final embedding vector representation
            self.emb = utils.fc_layer(tf.concat([self.h2_self, self.h2_nei], 1), n_hidden[-1]*2, n_emb, layer_name='concat')                
            emb_s=tf.slice(self.emb, [0, 0], [self.n_s, -1])
            emb_t=tf.slice(self.emb, [self.n_s, 0], [self.n_t, -1])         

            
            
            ### mini-batch feature propagation loss, a node's emb should be similar to its neighbors' weighted emb 
            A_s_rowsum=tf.reduce_sum(self.A_s,axis=1)
            I_A_s_rowsum= tf.cast(tf.greater(A_s_rowsum, 0.), dtype= A_s_rowsum.dtype) ##Indicator vector, the ith row is 1, if v_i has at least one neigbor in this mini-batch; otherwise, the i-th row is 0               
            emb_nei_s=tf.matmul(self.A_s, emb_s) #compute nei_rep based on self_rep(emb) and topological proximities with neighbors(A_s)
            FP_loss_s=tf.nn.l2_loss((emb_s-emb_nei_s)*I_A_s_rowsum[:,None])/tf.cast(self.n_s, tf.float32)
            
            A_t_rowsum=tf.reduce_sum(self.A_t,axis=1)
            I_A_t_rowsum= tf.cast(tf.greater(A_t_rowsum, 0.), dtype= A_t_rowsum.dtype)                     
            emb_nei_t=tf.matmul(self.A_t, emb_t) 
            FP_loss_t=tf.nn.l2_loss((emb_t-emb_nei_t)*I_A_t_rowsum[:,None])/tf.cast(self.n_t, tf.float32)

            self.FP_loss=FP_w * (FP_loss_s+FP_loss_t)


 

     

            
        with tf.name_scope('Node_Classifier'):
            ##node classification
            W_clf = tf.Variable(tf.truncated_normal([n_emb, num_class], stddev=1. / tf.sqrt(n_emb/2.)), name='clf_weight')
            b_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='clf_bias')               
            
            label_logit_s = tf.matmul(emb_s, W_clf) + b_clf 
            label_logit_t = tf.matmul(emb_t, W_clf) + b_clf  
            label_logit = tf.concat([label_logit_s, label_logit_t], 0)
  
    
            ###label propagation: compute nei's label_logit based on self_label_logit and topological proximities with neighbors
            label_logit_nei_s=tf.matmul(self.A_s, label_logit_s)
            label_logit_nei_t=tf.matmul(self.A_t, label_logit_t)    
            label_logit_nei =tf.concat([label_logit_nei_s, label_logit_nei_t], 0)

            
            
            
            if clf_type == 'multi-class':
                ### multi-class, softmax output
                self.pred_prob =tf.nn.softmax(label_logit + label_logit_nei) 
                loss_s = tf.nn.softmax_cross_entropy_with_logits_v2(logits=label_logit_s+label_logit_nei_s, labels=tf.stop_gradient(self.ys_true))           


            elif clf_type == 'multi-label':
            ### multi-label, sigmod output
                self.pred_prob =tf.sigmoid(label_logit + label_logit_nei)
                loss_s = tf.nn.sigmoid_cross_entropy_with_logits(logits=label_logit_s+label_logit_nei_s, labels=tf.stop_gradient(self.ys_true))
                
                
            self.clf_loss_s=  tf.reduce_sum(loss_s)/tf.cast(self.n_s,tf.float32)
            
            




            
        with tf.name_scope('Domain_Discriminator'):
            emb_grl = flip_gradient(self.emb, self.Ada_lambda) 
            ### Follow CDAN paper: https://github.com/thuml/CDAN/issues/6#issuecomment-652908849
            ### do not use domain loss to update node classifier, but use domain loss to update network embedding
            label_logit1=tf.matmul(emb_grl, tf.stop_gradient(W_clf)) + tf.stop_gradient(b_clf)             
            label_logit1_s=label_logit1[0:self.n_s,:]
            label_logit1_t=label_logit1[-self.n_t:,:]
            
            label_logit_nei1_s=tf.matmul(self.A_s, label_logit1_s) 
            label_logit_nei1_t=tf.matmul(self.A_t, label_logit1_t)
            label_logit_nei1 =tf.concat([label_logit_nei1_s, label_logit_nei1_t], 0)
            
            if clf_type == 'multi-class':
                pred_prob1=tf.nn.softmax(label_logit1+label_logit_nei1)                
            elif clf_type == 'multi-label':
                pred_prob1=tf.sigmoid(label_logit1+label_logit_nei1)
            
            ###conditional domain adaptation
            ## tensor product between predict probs and embedding reps as the input of domain discriminator 
            h_c_all=tf.matmul(tf.expand_dims(pred_prob1, 2), tf.expand_dims(emb_grl, 1))
            h_c_all=tf.reshape(h_c_all, [self.n_s+self.n_t, n_emb*num_class])        
            ##MLP for domain classification
            h_dann_1 = utils.fc_layer(h_c_all, n_emb*num_class, 128, layer_name='dann_fc_1')    
            h_dann_2 = utils.fc_layer(h_dann_1, 128, 128, layer_name='dann_fc_2') 

            W_domain = tf.Variable(tf.truncated_normal([128, 2], stddev=1. / tf.sqrt(128 / 2.)), name='dann_weight')
            b_domain = tf.Variable(tf.constant(0.1, shape=[2]), name='dann_bias')
            d_logit = tf.matmul(h_dann_2, W_domain) + b_domain
            self.d_softmax = tf.nn.softmax(d_logit)
            self.domain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_logit, labels=tf.stop_gradient(self.d_label)))
            self.entropy_domain= tf.reduce_mean(utils.entropy_compute(self.d_softmax)) #high if node reps are domain-uncertain
            
            




                
        all_variables = tf.trainable_variables()
        self.l2_loss =   l2_w * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name]) 
        
        self.total_loss = self.FP_loss + self.clf_loss_s  + self.domain_loss + self.l2_loss
         
        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.total_loss)        


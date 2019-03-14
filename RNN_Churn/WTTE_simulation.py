# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Masking
from keras.optimizers import RMSprop
from keras import backend as k
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import matplotlib.pyplot as plt


import tensorflow as tf
import theano as T
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense

from keras.layers import LSTM,GRU
from keras.layers import Lambda
from keras.layers.wrappers import TimeDistributed

from keras.optimizers import RMSprop,adam
from keras.callbacks import History, TensorBoard


N_feature = 2
N_sample = 1000
n_timesteps = 100
coefA = [ 1., 3.]
coefB = [ 2., 1.]


# ##  Use pure Weibull data censored at C(ensoring point). 
# ## Should converge to the generating A(alpha) and B(eta) for each timestep

def generate_parameter(N_sample, N_feature, N_timesteps, coefA, coefB,  C, discrete_time):
    np.random.seed(1)
    shape =[N_sample,N_timesteps,N_feature]         
    A2 = np.random.uniform(0, 5, size = N_feature * N_sample* N_timesteps).reshape(N_sample, N_timesteps, N_feature)
    A3 = (A2*coefA).sum(axis = 2).reshape(N_sample, N_timesteps, 1)   
    B2 = np.random.uniform(2, 3, size = N_feature * N_sample* N_timesteps).reshape(N_sample, N_timesteps, N_feature)
    B3 = (B2 * coefB).sum(axis = 2).reshape(N_sample, N_timesteps, 1)
    X = np.concatenate((A2, B2), axis = 2)
    W = np.sort(A3*np.power(-np.log(np.random.uniform(0,1,[N_sample,N_timesteps,1] )),1/B3))
    if discrete_time:
        C = np.floor(C)
        W = np.floor(W)

    U = np.less_equal(W, C)*1
    Y = np.minimum(W,C)    
    return X, W,Y,U


def _keras_unstack_hack(ab):
    """Implements tf.unstack(y_true_keras, num=2, axis=-1).
       Keras-hack adopted to be compatible with theano backend.
    """
    ndim = len(K.int_shape(ab))
    if ndim == 0:
        print('can not unstack with ndim=0')
    else:
        a = ab[..., 0]
        b = ab[..., 1]
    return a, b

def weibull_loss_discrete(y_true, y_pred, name=None):
    """calculates a keras loss op designed for the sequential api.
    
        Discrete log-likelihood for Weibull hazard function on censored survival data.
        For math, see 
        https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)
        
        Args:
            y_true: tensor with last dimension having length 2
                with y_true[:,...,0] = time to event, 
                     y_true[:,...,1] = indicator of not censored
                
            y_pred: tensor with last dimension having length 2 
                with y_pred[:,...,0] = alpha, 
                     y_pred[:,...,1] = beta

        Returns:
            A positive `Tensor` of same shape as input
            
    """    
    y,u = _keras_unstack_hack(y_true)
    a,b = _keras_unstack_hack(y_pred)

    hazard0 = K.pow((y + 1e-35) / a, b)
    hazard1 = K.pow((y + 1.0) / a, b)
    
    loglikelihoods = u * K.log(K.exp(hazard1 - hazard0) - 1.0) - hazard1
    loss = -1 * K.mean(loglikelihoods)
    return loss


def output_lambda(x, init_alpha=1.0, max_beta_value=5.0, max_alpha_value=None):
    """Elementwise (Lambda) computation of alpha and regularized beta.

        Alpha: 
        (activation) 
        Exponential units seems to give faster training than 
        the original papers softplus units. Makes sense due to logarithmic
        effect of change in alpha. 
        (initialization) 
        To get faster training and fewer exploding gradients,
        initialize alpha to be around its scale when beta is around 1.0,
        approx the expected value/mean of training tte. 
        Because we're lazy we want the correct scale of output built
        into the model so initialize implicitly; 
        multiply assumed exp(0)=1 by scale factor `init_alpha`.

        Beta: 
        (activation) 
        We want slow changes when beta-> 0 so Softplus made sense in the original 
        paper but we get similar effect with sigmoid. It also has nice features.
        (regularization) Use max_beta_value to implicitly regularize the model
        (initialization) Fixed to begin moving slowly around 1.0

        Assumes tensorflow backend.

        Args:
            x: tensor with last dimension having length 2
                with x[...,0] = alpha, x[...,1] = beta

        Usage:
            model.add(Dense(2))
            model.add(Lambda(output_lambda, arguments={"init_alpha":100., "max_beta_value":2.0}))
        Returns:
            A positive `Tensor` of same shape as input
    """
    a, b = _keras_unstack_hack(x)

    # Implicitly initialize alpha:
    if max_alpha_value is None:
        a = init_alpha * K.exp(a)
    else:
        a = init_alpha * K.clip(x=a, min_value=K.epsilon(),
                                max_value=max_alpha_value)

    m = max_beta_value
    if m > 1.05:  # some value >>1.0
        # shift to start around 1.0
        # assuming input is around 0.0
        _shift = np.log(m - 1.0)

        b = K.sigmoid(b - _shift)
    else:
        b = K.sigmoid(b)

    # Clipped sigmoid : has zero gradient at 0,1
    # Reduces the small tendency of instability after long training
    # by zeroing gradient.
    b = m * K.clip(x=b, min_value=K.epsilon(), max_value=1. - K.epsilon())

    x = K.stack([a, b], axis=-1)

    return x


n_feature = 2
n_sample = 10000
n_timesteps = 100
 
coefA = [ 1., 3.]
coefB = [ 2., 1.]

X, W, Y, U  = generate_parameter(N_sample = n_sample, N_feature = n_feature,
                                             N_timesteps = n_timesteps, coefA = coefA, coefB= coefB, 
                                             C= 15, discrete_time=True)




train_simux, test_simux = train_test_split(X, test_size = 0.3, random_state=1)


"""
train_simux, test_simux = train_test_split(X, test_size = 0.3, random_state=1)
train_simuy, test_simuy = train_test_split(np.append(Y, np.ones_like(U),axis=-1),test_size = 0.3, random_state=1) 
"""

alpha = (X[:, :, 0:2]*coefA).sum(axis=2).reshape(n_sample, n_timesteps, 1)
beta =  (X[:, :, 2:]*coefB).sum(axis=2).reshape(n_sample, n_timesteps, 1)                    



train_simux = X
train_simuy = np.append(Y,U,axis=-1)
test_simux = X
test_simuy = np.append(W, np.ones_like(W),axis=-1)
alpha = (X[:, :, 0:2]*coefA).sum(axis=2).reshape(n_sample, n_timesteps, 1)
beta =  (X[:, :, 2:]*coefB).sum(axis=2).reshape(n_sample, n_timesteps, 1)                    



tte_mean_train = np.nanmean(train_simuy[:,:,0])
init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0) )
init_alpha = init_alpha/np.nanmean(train_simuy[:,:,1]) # use if lots of censoring
print ('init_alpha: ', init_alpha)


np.random.seed(1)
# Store some history
history = History()

# Start building the model
simu_model = Sequential()
#model.add(TimeDistributed(Dense(2), input_shape=(None, n_features)))
simu_model.add(GRU(1, input_shape=(n_timesteps, X.shape[2]),activation='tanh',return_sequences=True))

simu_model.add(Dense(2))
simu_model.add(Lambda(output_lambda, arguments={"init_alpha":init_alpha, 
                                               "max_beta_value":10.0}))

simu_model.compile(loss=weibull_loss_discrete, optimizer=adam(lr=.01))

simu_model.summary()

# Fit! (really don't need to train this long)
np.random.seed(1)
simu_model.fit(train_simux, train_simuy,
          epochs=60, 
          batch_size=train_simux.shape[0]//10, 
          verbose=2, 
          validation_data=(test_simux, test_simuy),
          callbacks=[history]
          )

#check the loss function of training data and validation data
plt.plot(history.history['loss'],    label='training')
plt.plot(history.history['val_loss'],label='validation')
plt.xlabel('traing iteration')
plt.ylabel('value of loss')
plt.legend()



"""
predictions
"""
print ('All test cases (no noise)')
print ('each horizontal line is a sequence')
predicted_simu = simu_model.predict(test_simux)
predicted_simu = simu_model.predict(train_simux)


def weibull_quantiles(a, b, p):
    return a*np.power(-np.log(1.0-p),1.0/b)

def weibull_mode(a, b):
    # Continuous mode. 
    # TODO (mathematically) prove how close it is to discretized mode
    mode = a*np.power((b-1.0)/b,1.0/b)
    mode[b<=1.0]=0.0
    return mode

def weibull_mean(a, b):
    # Continuous mean. Theoretically at most 1 step below discretized mean 
    # E[T ] <= E[Td] + 1 true for positive distributions. 
    from scipy.special import gamma
    return a*gamma(1.0+1.0/b)

# TTE, Event Indicator, Alpha, Beta
drawstyle = 'steps-post'

print ('one training case:')
batch_indx =10
a = predicted_simu[batch_indx,:,0]
b = predicted_simu[batch_indx,:,1]
this_alpha = alpha[batch_indx, :, :]
this_beta = beta[batch_indx, :, :]


this_x_train = train_simux[batch_indx,:,:].mean(axis=1)
this_x_test =  test_simux[batch_indx,:,:].mean(axis=1)

this_tte_train = train_simuy[batch_indx,:,0]
this_tte_test =  test_simuy[batch_indx,:,0]

plt.plot(a,drawstyle='steps-post')
plt.title('predicted alpha')
plt.show()
plt.plot(b,drawstyle='steps-post')
plt.title('predicted beta')
plt.show()



plt.plot(this_tte_train,label='censored time to event',color='red',linestyle='dashed',linewidth=2,drawstyle=drawstyle)

plt.plot(this_tte_test ,label='actual time to event',color='black',linewidth=2,drawstyle=drawstyle)
#plt.plot(weibull_quantiles(a,b,0.75),color='blue',label='pred <0.75',drawstyle=drawstyle)
plt.plot(weibull_mode(a, b), color='blue',linewidth=2,label='pred mode/peak prob',drawstyle=drawstyle)
#plt.plot(weibull_mean(a, b), color='green',linewidth=1,label='pred mean',drawstyle='steps-post')
#plt.plot(weibull_quantiles(a,b,0.25),color='green',label='pred <0.25',drawstyle=drawstyle)
plt.xlabel('time')
plt.ylabel('time to event')
plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
plt.show()


plt.plot(a, label = 'predicted alpha', drawstyle='steps-post')
plt.plot(this_alpha,  label = 'true alpha',  drawstyle='steps-post')
plt.xlabel('time')
plt.ylabel('alpha')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

plt.plot(b, label = 'predicted beta',  drawstyle='steps-post')
plt.plot(this_beta,  label = 'true beta', drawstyle='steps-post')
plt.xlabel('time')
plt.ylabel('beta')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


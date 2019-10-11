# create custom loss function for training

import keras.backend as K
def audio_discriminate_loss2(gamma=0.1,beta = 2*0.1,people_num=2):
    def loss_func(S_true,S_pred,gamma=gamma,beta=beta,people_num=people_num):
        sum_mtr = K.zeros_like(S_true[:,:,:,:,0])
        for i in range(people_num):
            sum_mtr += K.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,i])
            for j in range(people_num):
                if i != j:
                    sum_mtr -= gamma*(K.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,j]))

        for i in range(people_num):
            for j in range(i+1,people_num):
                #sum_mtr -= beta*K.square(S_pred[:,:,:,i]-S_pred[:,:,:,j])
                #sum_mtr += beta*K.square(S_true[:,:,:,:,i]-S_true[:,:,:,:,j])
                pass
        #sum = K.sum(K.maximum(K.flatten(sum_mtr),0))

        loss = K.mean(K.flatten(sum_mtr))

        return loss
    return loss_func





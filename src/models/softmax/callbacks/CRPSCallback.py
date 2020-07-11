from keras.callbacks import Callback
from src.models.softmax.abstract.AbstractMetrics import AbstractMetrics



class CRPSCallback(Callback,AbstractMetrics):
    
    # init object
    def __init__(self,validation, predict_batch_size=20, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch
        print('validation shape',len(self.validation))

    
    def on_train_begin(self, logs={}):
        if not ('CRPS_score_val' in self.params['metrics']):
            self.params['metrics'].append('CRPS_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['CRPS_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):


        logs['CRPS_score_val'] = float('-inf')
            
        if (self.validation):
            X_valid, y_valid = self.validation[0], self.validation[1]
            val_s =  self.get_model_score(self.model,X_valid,y_valid)
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s
       
          
    
    def on_batch_begin(self, batch, logs={}):

        pass


    
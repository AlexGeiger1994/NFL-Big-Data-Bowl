from src.models.softmax.callbacks.CRPSCallback import CRPSCallback


class KerasModel(object):
    

    def __init__(self,TR_DATA):

        # setup column space & input layers
        self.dim0 = TR_DATA[0].shape[1]
        self.dim1 = TR_DATA[1].shape[1]
        self.modelName = 'best_model.h5'
        self.monitor   = 'CRPS_score_val'
        self.batchSize = 1024


    def getModel(self):

        # setup column space & input layers
        inp0 = Input(shape = (self.dim0,))
        inp1 = Input(shape = (self.dim1,))
    
        # define hidden layer one
        x0 = Dense(512, input_dim=self.dim0, activation='relu')(inp0)
        x1 = Dense(512, input_dim=self.dim1, activation='selu')(inp1)
    
        # Augment the static hidden Layer 1
        x0 = GaussianDropout(0.375)(x0)
        x0 = BatchNormalization()(x0)
    
        # Augment the player intuition hidden layer 1
        x1 = GaussianNoise(0.7)(x1)
        x1 = BatchNormalization()(x1)
        x1 = GaussianDropout(0.25)(x1)
        x1 = BatchNormalization()(x1)
    
        # concatentate the static and intution hidden layers
        x = Concatenate(axis=1)([x0,x1])
    
        # pass the concatenated layers through hidden layer 2
        x = Dense(256, activation='relu')(x)
    
        # apply gaussian drop to hidden layer 2
        x = GaussianDropout(0.5)(x)
        x = BatchNormalization()(x)
    
        # pass data to hidden layer hidden layer 3
        x = Dense(256, activation='sigmoid')(x)
    
        # augment model results
        x = GaussianDropout(0.5)(x)
        x = BatchNormalization()(x)
    
        # pass results to the output layer
        out = Dense(199, activation='softmax')(x)
        model = Model([inp0,inp1],out)

        # setup model call backs
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
        es = keras_call.EarlyStopping(monitor='CRPS_score_val',  mode='min',  restore_best_weights=True, verbose=False,  patience=10)
        mc = ModelCheckpoint(self.modelName,monitor=self.monitor,mode='min', save_best_only=True, verbose=False, save_weights_only=True)

        # return model
        return model
    

    def predict(self,TR_DATA,TR_TARGET,VAL_DATA,VAL_TARGET):
        
        # build model
        model = self.buildModel()
    
        # train model
        model.fit(TR_DATA, TR_TARGET,callbacks=[CRPSCallback(validation = (VAL_DATA,VAL_TARGET)),es,mc], epochs=100, batch_size=self.batchSize,verbose=False)
        
        # load model
        model.load_weights(self.modelName)
        
        # calculate feature scores
        tr_s  = np.round(getScore(model,TR_DATA,TR_TARGET),6)
        val_s = np.round(getScore(model,VAL_DATA,VAL_TARGET),6)
    
        # return model score
        return model, val_s, tr_s


    def getScore(model,x,y):

        y_pred  = model.predict(x)
        y_true  = np.clip(np.cumsum(y, axis=1), 0, 1)
        y_pred  = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])



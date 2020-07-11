


class AbstractMetrics(object):
    
    @staticmethod
    def get_model_score(model,x,y):
        y_pred  = model.predict(x)
        y_true  = np.clip(np.cumsum(y, axis=1), 0, 1)
        y_pred  = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])












class AbstractMetrics(object):
    
    @staticmethod
    def model_eval(model, dataset, data_loader, batch_size):
        model.eval()
        preds = np.zeros((len(dataset), 199))
        with torch.no_grad():
            for i, eval_x_batch in enumerate(data_loader):
                    eval_values = eval_x_batch[0].float()
                    pred = model(eval_values)
                    preds[i * batch_size:(i + 1) * batch_size] = pred
        return preds






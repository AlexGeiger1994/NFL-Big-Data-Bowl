class PytorchModel(object):
    

    def train(self,train_dataset, valid_dataset, train_loader, valid_loader,torch_feat_len,epoch,batch_size,n_fold):
    
            # setup model
            early_stopping = EarlyStoppingIV(patience=15, verbose=False)
            model          = NFL_NN(torch_feat_len)
            criterion      = nn.MSELoss()
            optimizer      = torch.optim.Adam(model.parameters(), lr=0.003)
            out_features   = 199
    
            # loop through each epoch
            for idx in range(epoch):
                train_batch_loss_sum = 0
                for param in model.parameters():
                    param.requires_grad = True

                model.train()
                for x_batch, y_batch in train_loader:
                    y_pred = model(x_batch.float())
                    loss = torch.sum((y_pred.float()-y_batch.view((len(y_batch), out_features)).float()).pow(2))/(199*len(y_pred))
                    train_batch_loss_sum += loss.item()

                    del x_batch
                    del y_batch

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    torch.cuda.empty_cache()
                    gc.collect()

                train_epoch_loss = train_batch_loss_sum / len(train_loader)
                valid_y_pred = model_eval(model, valid_dataset, valid_loader, batch_size)
                valid_crps = np.sum(np.power(valid_y_pred - valid_dataset[:][1].data.cpu().numpy(), 2))/(199*len(valid_dataset))
                model_save_name = 'checkpoint_fold_{}.pt'.format(n_fold+1)
                early_stopping(valid_crps, model, model_save_name)
        
        
                nfl_pred   = pytorch_transform(model_eval(model, train_dataset, train_loader, batch_size))
                train_crps = np.sum(np.power(nfl_pred     - train_dataset[:][1].data.cpu().numpy(), 2))/(199*len(train_dataset))
        
                if early_stopping.early_stop:
                    break

            # create model & load model with weights
            model = NFL_NN(torch_feat_len)
            model.load_state_dict(torch.load(model_save_name))

            nfl_pred       = pytorch_transform(model_eval(model, train_dataset, train_loader, batch_size))
            valid_y_pred   = pytorch_transform(model_eval(model, valid_dataset, valid_loader, batch_size))
    
            valid_crps = np.sum(np.power(valid_y_pred - valid_dataset[:][1].data.cpu().numpy(), 2))/(199*len(valid_y_pred))
            train_crps = np.sum(np.power(nfl_pred     - train_dataset[:][1].data.cpu().numpy(), 2))/(199*len(train_dataset))
    
            print('Offical Epoch Loss: {:.5f}, Valid CRPS: {:.5f}, Train CRPS: {:.5f}'.format(train_epoch_loss, valid_crps,train_crps))
            del criterion, optimizer
            gc.collect()
    
            return model
    
    
    def getDataset(self,df, y_val, batch_size, tr_ix, val_ix):
    
        # create training data & datasets
        tr_x, tr_y = torch.from_numpy(df.iloc[tr_ix].values), torch.from_numpy(y_val[tr_ix])
        val_x, val_y = torch.from_numpy(df.iloc[val_ix].values), torch.from_numpy(y_val[val_ix] )
        tr_dataset, val_dataset = TensorDataset(tr_x, tr_y), TensorDataset(val_x, val_y)

        # create data loaders
        tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return tr_dataset, val_dataset, tr_loader, val_loader





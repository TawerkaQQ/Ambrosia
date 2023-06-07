import os
from tqdm import tqdm
import pickle
import argparse
import time
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch import nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


from utils import set_seed, load_model, save, get_model, update_optimizer, get_data
from epoch import train_epoch, val_epoch, test_epoch
from cli import add_all_parsers

arrval = []
lossval = []
arrtrain = []
losstrain = []
arrepoch = []

def train(args):
    set_seed(args, use_gpu=torch.cuda.is_available())
    train_loader, val_loader, test_loader, dataset_attributes = get_data(args.root, args.image_size, args.crop_size,
                                                                         args.batch_size, args.num_workers, args.pretrained)

    model = get_model(args, n_classes=dataset_attributes['n_classes'])
    criteria = CrossEntropyLoss()

    if args.use_gpu:
        print('USING GPU')
        torch.cuda.set_device(0)
        model.cuda()
        criteria.cuda()

    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004, foreach=None, differentiable=False)

    # Containers for storing metrics over epochs
    loss_train, acc_train, topk_acc_train = [], [], []
    loss_val, acc_val, topk_acc_val, avgk_acc_val, class_acc_val = [], [], [], [], []

    save_name = args.save_name_xp.strip()
    save_dir = os.path.join(os.getcwd(), 'results', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('args.k : ', args.k)

    lmbda_best_acc = None
    best_val_acc = float('-inf')
    

    for epoch in tqdm(range(args.n_epochs), desc='epoch', position=0):
        t = time.time()
        optimizer = update_optimizer(optimizer, lr_schedule=args.epoch_decay, epoch=epoch)

        loss_epoch_train, acc_epoch_train, topk_acc_epoch_train = train_epoch(model, optimizer, train_loader,
                                                                              criteria, loss_train, acc_train,
                                                                              topk_acc_train, args.k,
                                                                              dataset_attributes['n_train'],
                                                                              args.use_gpu)

        loss_epoch_val, acc_epoch_val, topk_acc_epoch_val, \
        avgk_acc_epoch_val, lmbda_val = val_epoch(model, val_loader, criteria,
                                                  loss_val, acc_val, topk_acc_val, avgk_acc_val,
                                                  class_acc_val, args.k, dataset_attributes, args.use_gpu)

        # save model at every epoch
        save(model, optimizer, epoch, os.path.join(save_dir, save_name + '_weights.tar'))

        # save model with best val accuracy
        if acc_epoch_val > best_val_acc:
            best_val_acc = acc_epoch_val
            lmbda_best_acc = lmbda_val
            save(model, optimizer, epoch, os.path.join(save_dir, save_name + '_weights_best_acc.tar'))

        print()
        print(f'epoch {epoch} took {time.time()-t:.2f}')
        print(f'loss_train : {loss_epoch_train}')
        print(f'loss_val : {loss_epoch_val}')
        print(f'acc_train : {acc_epoch_train} / topk_acc_train : {topk_acc_epoch_train}')
        print(f'acc_val : {acc_epoch_val} / topk_acc_val : {topk_acc_epoch_val} / '
              f'avgk_acc_val : {avgk_acc_epoch_val}')
        
        # Save results for create a graphics 
        
        arrval.append(acc_epoch_val)
        arrtrain.append(acc_epoch_train)
        lossval.append(loss_epoch_val)
        losstrain.append(loss_epoch_train)
        arrepoch.append(epoch)
        
        # Create a graphics

        # Graph 1: Loss Val/Loss Train

        fig, loss = plt.subplots()
        loss.plot(arrepoch, losstrain, label='LossTrain')
        loss.plot(arrepoch, lossval, label='LossVal')  
        plt.scatter(arrepoch, losstrain, label='LossTrain') 
        plt.scatter(arrepoch, lossval, label='LossVal')  
        loss.set_xlabel('epoch')  
        loss.set_ylabel('loss val and train') 
        plt.title('Loss Val/Loss Train')
        plt.legend()
        plt.savefig('/home/cyber/Desktop/Ambrosia_NN/PlantNet-300K-main/graphics/Loss_Val_Train.png') 
        
        # Graph 2: acc Val/Loss Val

        fig, graph2 = plt.subplots() 
        graph2.plot(arrepoch, arrval, label='accVal')  
        graph2.plot(arrepoch, lossval, label='LossVal')  
        plt.scatter(arrepoch, arrval, label='accVal')
        plt.scatter(arrepoch, lossval, label='LossVal') 
        graph2.set_xlabel('epoch')
        graph2.set_ylabel('loss val and acc val')
        plt.title('acc Val/Loss Val')
        plt.legend()
        plt.savefig('/home/cyber/Desktop/Ambrosia_NN/PlantNet-300K-main/graphics/Acc_Loss_Val.png')

        # Graph 3: Acc Train/Loss Train

        fig, graph3 = plt.subplots() 
        graph3.plot(arrepoch, arrtrain, label='accTrain')
        graph3.plot(arrepoch, losstrain, label='LossTrain') 
        plt.scatter(arrepoch, arrtrain, label='accTrain') 
        plt.scatter(arrepoch, losstrain, label='LossTrain')
        graph3.set_xlabel('epoch')
        graph3.set_ylabel('loss train and acc train')
        plt.title('Acc Train/Loss Train')
        plt.legend()
        plt.savefig('/home/cyber/Desktop/Ambrosia_NN/PlantNet-300K-main/graphics/Acc_Loss_Train.png')


    load_model(model, os.path.join(save_dir, save_name + '_weights_best_acc.tar'), args.use_gpu)
    loss_test_ba, acc_test_ba, topk_acc_test_ba, \
    avgk_acc_test_ba, class_acc_test = test_epoch(model, test_loader, criteria, args.k,
                                                  lmbda_best_acc, args.use_gpu,
                                                  dataset_attributes)
    # Save the results as a dictionary and save it as a pickle file in desired location

    results = {'loss_train': loss_train, 'acc_train': acc_train, 'topk_acc_train': topk_acc_train,
               'loss_val': loss_val, 'acc_val': acc_val, 'topk_acc_val': topk_acc_val, 'class_acc_val': class_acc_val,
               'avgk_acc_val': avgk_acc_val,
               'test_results': {'loss': loss_test_ba,
                                'accuracy': acc_test_ba,
                                'topk_accuracy': topk_acc_test_ba,
                                'avgk_accuracy': avgk_acc_test_ba,
                                'class_acc_dict': class_acc_test},
               'params': args.__dict__}

    with open(os.path.join(save_dir, save_name + '.pkl'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_all_parsers(parser)
    args = parser.parse_args()
    train(args)
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from mobilenet import MobileNet
from utils import plot_loss_acc
from preprocessing import get_aug_train_val_loader, pre_process_mean_std, get_train_val_loader, get_test_loader
from training_model import train_model, test_model

#REMEMBER TO ENABLE CUDA BACK 

def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    args.dataset_dir = './cifar100data'
    args.batch_size  = 128
    args.mixup = False
    alpha = 0   
    
    
    '''
    #model params for section 3
    args.lr = [0.5, 0.05, 0.01]
    args.epochs = 15
    args.wd = 0
    args.lr_scheduler = False
    '''

    '''
    #model params for section 4
    args.lr = [0.05, 0.05]
    args.epochs = 300
    args.wd = 0    
    '''
    '''
    #model params for section 5
    args.lr = 0.05
    args.epochs = 300
    #args.wd = [5e-4, 1e-4]
    args.wd = [1e-4]
    args.lr_scheduler = True
    '''
    
    
    #model params for section 6
    args.lr = 0.05
    args.epochs = 300
    args.wd = [5e-4]
    args.lr_scheduler = True
    args.mixup = True
    alpha = 0.2
    args.test = True

    #Load train and val loader
    train_loader, _ = get_train_val_loader(args.dataset_dir, args.batch_size, True, args.seed, save_images=False) #un-augmented dataloader
    train_mean, train_std = pre_process_mean_std(train_loader)
    train_loader, val_loader = get_aug_train_val_loader(train_loader, args.batch_size, True, args.seed, save_images=args.save_images) #augmented dataloader
    
    if args.test:
        test_loader = get_test_loader(args.dataset_dir, args.batch_size, train_mean, train_std)


    for wd in args.wd:
      
      print ('args.mixup is: ', str(args.mixup))
      #model
      model = MobileNet(num_classes=100)
      #print(model)
      model.cuda()
    

      # criterion
      criterion = torch.nn.CrossEntropyLoss().cuda()
      #criterion = torch.nn.CrossEntropyLoss()
    
      #optimizer
      #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.wd) #For section 4
      optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=wd)

  

      if args.lr_scheduler:
          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
      else:
          scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epochs)

        #get losses and accuracies             
      stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc = train_model(
        model, train_loader, val_loader, optimizer, criterion, args.epochs, args.batch_size, scheduler, args.mixup, alpha
        )
      print(stat_training_loss,',', stat_val_loss, ',',  stat_training_acc, ',', stat_val_acc)

    
      # plot
      #args.fig_name = 'lr_ '+str(lr)+'.png' #For section 3
      args.fig_name = 'mixup_True.png' 
      plot_loss_acc(stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc, args.fig_name)
      print('saved the figure!')
        
    # test
    if args.test:
        test_loss = 0
        test_acc = 0
        test_samples = 0
        for test_imgs, test_labels in test_loader:
            batch_size = test_imgs.shape[0]
            test_logits = model.forward(test_imgs.cuda())
            #test_logits = model.forward(test_imgs)
            test_loss = criterion(test_logits, test_labels.cuda())
            _, top_class = test_logits.topk(1, dim=1)
            equals = top_class == test_labels.cuda().view(*top_class.shape)
            #equals = top_class == test_labels.view(*top_class.shape)
            test_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            test_loss += batch_size * test_loss.item()
            test_samples += batch_size
        assert test_samples == 10000
        print('Test loss: ', test_loss/test_samples)
        print('Test acc: ', test_acc/test_samples)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset_dir',type=str, help='')
    parser.add_argument('--batch_size',type=int, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--lr',type=float, help='')
    parser.add_argument('--wd',type=float, help='')
    parser.add_argument('--fig_name',type=str, help='')
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.set_defaults(lr_scheduler=False)
    parser.add_argument('--mixup', action='store_true')
    parser.set_defaults(mixup=False)
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--save_images', action='store_true')
    parser.set_defaults(save_images=False)
    parser.add_argument('--seed', type=int, default=0, help='')
    args = parser.parse_args()
    print(args)
    
    main(args)

    



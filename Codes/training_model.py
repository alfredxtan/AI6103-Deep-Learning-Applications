import torch
import numpy as np
import torch.nn as nn

def mixup_data(x, y, alpha = 0.2):
  if alpha > 0:
    lam = np.random.beta(alpha,alpha)
  else:
    lam = 1

  batch_size = x.size()[0]
  index = torch.randperm(batch_size)
  mixed_x = lam*x + (1-lam)*x[index, :]

  y_a, y_b = y, y[index]

  return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
  criterion = nn.CrossEntropyLoss()

  return lam*criterion(pred, y_a)+(1-lam)*criterion(pred, y_b)


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, batch_size, scheduler, mixup, alpha):
    stat_training_loss = []
    stat_val_loss = []
    stat_training_acc = []
    stat_val_acc = []
    for epoch in range(epochs):
        training_loss = 0
        training_acc = 0
        training_samples = 0
        val_loss = 0
        val_acc = 0
        val_samples = 0
        # training
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()

            #print("Shape of imgs:", imgs.shape)
            batch_size = imgs.shape[0]
            optimizer.zero_grad()
           
            if mixup:
              imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, alpha)
              logits = model.forward(imgs)
              loss = mixup_criterion(logits, labels_a, labels_b, lam)
              _, top_class = logits.topk(1, dim=1)
              equals = lam * (top_class == labels_a.view(*top_class.shape)) + (1 - lam) * (top_class == labels_b.view(*top_class.shape))
            else: 
              logits = model.forward(imgs)
              loss = criterion(logits, labels)
              _, top_class = logits.topk(1, dim=1)
              equals = top_class == labels.view(*top_class.shape)

            loss.backward()
            optimizer.step()

            training_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            training_loss += batch_size * loss.item()
            training_samples += batch_size
            
        # validation
        model.eval()
        for val_imgs, val_labels in val_loader:
            batch_size = val_imgs.shape[0]
            val_logits = model.forward(val_imgs.cuda())
           # val_logits = model.forward(val_imgs)
            loss = criterion(val_logits, val_labels.cuda())
            #loss = criterion(val_logits, val_labels)
            _, top_class = val_logits.topk(1, dim=1)
            equals = top_class == val_labels.cuda().view(*top_class.shape)
            #equals = top_class == val_labels.view(*top_class.shape)
            val_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            val_loss += batch_size * loss.item()
            val_samples += batch_size
        assert val_samples == 10000
        # update stats
        stat_training_loss.append(training_loss/training_samples)
        stat_val_loss.append(val_loss/val_samples)
        stat_training_acc.append(training_acc/training_samples)
        stat_val_acc.append(val_acc/val_samples)
        # print
        print(f"Epoch {(epoch+1):d}/{epochs:d}.. Learning rate: {scheduler.get_lr()[0]:.4f}.. Train loss: {(training_loss/training_samples):.4f}.. Train acc: {(training_acc/training_samples):.4f}.. Val loss: {(val_loss/val_samples):.4f}.. Val acc: {(val_acc/val_samples):.4f}")
        # lr scheduler
        scheduler.step()
        #print("Completed epoch ", epoch+1, "...")
        
    return stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc
        
        
def test_model(model, test_loader, criterion,):
    test_loss = 0
    test_acc = 0
    test_samples = 0
    for test_imgs, test_labels in test_loader:
        batch_size = test_imgs.shape[0]
        test_logits = model.forward(test_imgs.cuda())
        #test_logits = model.forward(test_imgs)
        test_loss = criterion(test_logits, test_labels.cuda())
        #test_loss = criterion(test_logits, test_labels)
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
    from mobilenet import MobileNet
    from preprocessing import get_train_val_loader
    import torch
    
    dataset_dir = './cifar100data'
    train_loader, val_loader=  get_train_val_loader(dataset_dir, 128, True, 0,)
    
    model = MobileNet(100)
    lr = 0.5
    wd = 0
    epochs = 1
    batch_size = 128
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = wd )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor = 1.0, total_iters = epochs)
    
    a,b,c,d = train_model(model, train_loader, val_loader, optimizer, criterion, epochs, batch_size, scheduler)
    
    print(a,b,c,d)
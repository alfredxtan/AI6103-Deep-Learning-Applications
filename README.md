This repository consists of codes and final report for the individual assignment for the course AI6103 Deep Learning Applications. 

In this assignment, we analyzed the effects of different hyperparameters in MobileNet, using CIFAR-100 as our dataset.
1. **Data preprocessing**: Perform image normalization, random horizontal flipping, and random cropping.
2. **Learning Rate**: For each learning rate in {0.5, 0.05, 0.01}, train for 15 epochs with batch size 128. Choose the best learning rate.
3. **Learning Rate Scheduler**: Train, for 300 epochs each at batch size of 128, using cosine annealing scheduler and without scheduler. Pick the best scheduler (or none).
4. **Weight Decay**: For each weight decay coefficient in {5e-4, 1e-4}, train for 300 epochs with batch size 128. Choose the best weight decay coefficient.
5. **Data Augmentation Using Mix Up**: Use mix-up augmentation and train the network for 300 epochs. Use alpha = 0.2 for the beta distribution.

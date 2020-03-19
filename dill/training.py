import torch
import torch.nn as nn
import time
import copy

class Trainer:
    def __init__(self, dataloaders, mlp_width=None, mode='oo', learning_rate=1e-3,
                 weight_decay=1e-2, pretrained_file=None,
                 optimizer_type=None, version=None, **kwargs):

        # Dataset
        self.dataloaders = dataloaders

        # Hyperparameters
        self.mlp_width = mlp_width
        self.mode = mode
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pretrained_file = pretrained_file
        self.optimizer_type = optimizer_type

        self.model, self.criterion, self.optimizer =  create_net(
            mlp_width=mlp_width,
            mode=mode, learning_rate=learning_rate,
            weight_decay=weight_decay, pretrained_file=pretrained_file,
            optimizer_type=optimizer_type, **kwargs)

        # Stored training metrics
        self.val_acc_history = []
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.num_epochs_trained = 0
        self.best_acc = 0.0
        self.grad_norms = []

        # Version
        self.version = version

    def train_model(self, num_epochs=25, eval_only=False):
        """
        Function taken from Pytorch documentation, has lots of nice functionality in it.
        But I didn't write this so it has a lot.
        """
        since = time.time()

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            self.num_epochs_trained = self.num_epochs_trained + 1

            # Each epoch has a training and validation phase
            phases = ['val'] if eval_only else ['train', 'val']
            for phase in phases:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                epoch_start_time = time.time()

                # Iterate over data.
                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if use_cuda and torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # loss.register_hook(lambda grad: print(grad))

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                            if (i+1) % 1000 == 0:
                                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data))


                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                epoch_time_elapsed = time.time() - epoch_start_time
                print('Epoch complete in {:.0f}m {:.0f}s'.format(epoch_time_elapsed // 60, epoch_time_elapsed % 60))
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                norms = [compute_gradient_norm(m) for m in self.model.classifier.children()]
                self.grad_norms.append(norms)
                print('Gradient norms of: classifier layers - {}, features - {}'.format(norms, compute_gradient_norm(self.model.features)))


                # deep copy the model
                if phase == 'val' and epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    self.val_acc_history.append(epoch_acc)


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.best_acc))

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)
        return self.model, self.val_acc_history

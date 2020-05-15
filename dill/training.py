import torch
import torch.nn as nn
import time
import copy
from .utils import compute_layer_norm
import numpy as np

class Trainer:
    def __init__(self, dataloaders, model, criterion, optimizer, scheduler=None, mlp_width=None, mode='oo', learning_rate=1e-3,
                 weight_decay=1e-2, pretrained_file=None,
                 optimizer_type=None, version=None, use_cuda=True,
                 record_rate=1000, print_rate=10000,
                 **kwargs):
        """
        :param record_rate: How often the trainer logs results (in epochs)
        """

        # Dataset
        self.dataloaders = dataloaders

        # Hyperparameters
        self.mlp_width = mlp_width
        self.mode = mode
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pretrained_file = pretrained_file
        self.optimizer_type = optimizer_type

        self.model, self.criterion, self.optimizer =  model, criterion, optimizer
        self.scheduler = scheduler
        # Stored training metrics
        self.val_loss = []
        self.train_loss = []
        self.weights_norms = []
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.num_epochs_trained = 0
        self.best_loss = 0.0
        self.grad_norms = []
        self.use_cuda = use_cuda
        self.record_rate = record_rate
        self.print_rate = print_rate

        # Version
        self.version = version

    def train(self, num_epochs=25, eval_only=False, train_only=False, verbose=True):
        """
        Function taken from Pytorch documentation, has lots of nice functionality in it.
        But I didn't write this so it has a lot.
        """
        since = time.time()

        for epoch in range(num_epochs):
            if verbose:
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
            self.num_epochs_trained = self.num_epochs_trained + 1

            # Each epoch has a training and validation phase
            if eval_only:
                phases = ['val']
            elif train_only:
                phases = ['train']
            else:
                phases = ['train', 'val']

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
                    if self.use_cuda and torch.cuda.is_available():
                        inputs = inputs.cuda().double()
                        labels = labels.cuda().double()
                    if len(inputs.shape) == 1:
                        inputs = inputs.reshape(inputs.shape[0],1)
                        labels = labels.reshape(labels.shape[0],1)

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

                        preds = outputs

                        # loss.register_hook(lambda grad: print(grad))

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                            if verbose:
                                if (i+1) % self.print_rate == 0:
                                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                                        %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data))


                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if self.scheduler is not None:
                    self.scheduler.step()

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                epoch_time_elapsed = time.time() - epoch_start_time

                if not verbose:
                    if (epoch+1) % self.print_rate == 0:
                        print('Epoch {} complete in {:.0f}m {:.0f}s'.format(epoch+1, epoch_time_elapsed // 60, epoch_time_elapsed % 60))
                        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                else:
                    print('Epoch complete in {:.0f}m {:.0f}s'.format(epoch_time_elapsed // 60, epoch_time_elapsed % 60))
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # norms = [compute_gradient_norm(m) for m in self.model.classifier.children()]
                # self.grad_norms.append(norms)
                # print('Gradient norms of: classifier layers - {}, features - {}'.format(norms, compute_gradient_norm(self.model.features)))


                # deep copy the model
                if phase == 'val' and epoch_loss > self.best_loss:
                    self.best_loss = epoch_loss
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

                if (epoch+1) % self.record_rate == 0:
                    # try: # why do I do these things to myself
                    base = compute_layer_norm(self.model, layer_name='linear0')
                    if len(base) > 0:
                        norms = base[0].detach().numpy()[np.newaxis][np.newaxis, :]
                        norms = np.concatenate((norms,
                        compute_layer_norm(self.model, layer_name='weights')[0].detach().numpy()[np.newaxis][np.newaxis, :]),
                        axis=0)
                    else:
                        norms = np.array(compute_layer_norm(self.model, layer_name='weights')[0].detach().numpy())

                    self.weights_norms.append(norms)
                    if phase == 'val':
                        self.val_loss.append((epoch+1, epoch_loss))
                    else:
                        self.train_loss.append((epoch+1, epoch_loss))


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # print('Best val Loss: {:4f}'.format(self.best_loss))

        # load best model weights
        # self.model.load_state_dict(self.best_model_wts)
        return self.model

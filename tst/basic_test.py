import dill.gdrive as gdrive
import pytest

# gdrive.authenticate()

# cifar_finetuned_parameters = 'alexnet_unmod_cifar10_lr1e12_mlpNone_epoch20_model_v14.pkl'
# partially_trained_mod_params = 'alexnet_modified_cifar10_mlp4_epoch10_model_v12.pkl'
# # trainer = Trainer(dataloaders, verbose=True, mode='oo',
# #                       pretrained_file=cifar_finetuned_parameters, optimizer_type='adam',
# #                       learning_rate=learning_rate, weight_decay=weight_decay)
# trainer = Trainer(dataloaders, mlp_width=mlp_width, verbose=True, mode='mo',
#                       pretrained_file=cifar_finetuned_parameters, optimizer_type='adam',
#                       learning_rate=learning_rate, weight_decay=weight_decay,
#                       version=version, cuda=True)
# # trainer = Trainer(dataloaders, mlp_width=mlp_width, verbose=True, mode='mm',
# #                       pretrained_file=partially_trained_mod_params, optimizer_type='adam',
# #                       learning_rate=learning_rate, weight_decay=weight_decay)


@pytest.mark.usefixtures("set_seed")
class TestFullModelCreationPath(object):

    @pytest.mark.parametrize("num_epochs, batch_size, learning_rate, weight_decay, mlp_width, model_type", [
        (10, 30, 1e-4, 1e-2, 4, 'oo'),
        ])
    def test_original_model_original_parameters(self, num_epochs, batch_size,
        learning_rate, weight_decay, mlp_width, model_type):
        dataloaders = get_fake_dataloaders(batch_size=batch_size)

        trainer = Trainer(dataloaders, verbose=True, mode=model_type,
                              learning_rate=learning_rate,
                               weight_decay=weight_decay)

        trainer.train_model(num_epochs=num_epochs)

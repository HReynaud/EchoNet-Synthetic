import argparse
import json
import os
import copy
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from echosyn.common import privacy_utils as Utils
from echosyn.privacy.shared import SiameseNetwork

class AgentSiameseNetwork:
    def __init__(self, config):
        self.config = config

        # set path used to save experiment-related files and results
        self.SAVINGS_PATH = './experiments/' + self.config['experiment_description'] + '/'
        self.IMAGE_PATH = self.config['image_path']

        # save configuration as config.json in the created folder
        with open(self.SAVINGS_PATH + 'config.json', 'w') as outfile:
            json.dump(self.config, outfile, indent='\t')
            outfile.close()

        # enable benchmark mode in cuDNN
        torch.backends.cudnn.benchmark = True

        # set all the important variables
        self.network = self.config['siamese_architecture']
        self.data_handling = self.config['data_handling']

        self.num_workers = self.config['num_workers']
        self.pin_memory = self.config['pin_memory']

        self.n_channels = self.config['n_channels']
        self.n_features = self.config['n_features']
        self.image_size = self.config['image_size']
        self.loss_method = self.config['loss']
        self.optimizer_method = self.config['optimizer']
        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        self.max_epochs = self.config['max_epochs']
        self.early_stopping = self.config['early_stopping']
        self.transform = self.config['transform']

        self.n_samples_train = self.config['n_samples_train']
        self.n_samples_val = self.config['n_samples_val']
        self.n_samples_test = self.config['n_samples_test']

        self.start_epoch = 0

        self.es = EarlyStopping(patience=self.early_stopping)
        self.best_loss = 100000
        self.loss_dict = {'training': [],
                          'validation': []}

        if self.data_handling == 'balanced':
            self.balanced = True
            self.randomized = False
        elif self.data_handling == 'randomized':
            self.balanced = True
            self.randomized = True

        # define the suffix needed for loading the checkpoint (in case you want to resume a previous experiment)
        if self.config['resumption'] is True:
            if self.config['resumption_count'] == 1:
                self.load_suffix = ''
            elif self.config['resumption_count'] == 2:
                self.load_suffix = '_resume'
            elif self.config['resumption_count'] > 2:
                self.load_suffix = '_resume' + str(self.config['resumption_count'] - 1)

        # define the suffix needed for saving the checkpoint (the checkpoint is saved at the end of each epoch)
        if self.config['resumption'] is False:
            self.save_suffix = ''
        elif self.config['resumption'] is True:
            if self.config['resumption_count'] == 1:
                self.save_suffix = '_resume'
            elif self.config['resumption_count'] > 1:
                self.save_suffix = '_resume' + str(self.config['resumption_count'])

        # Define the siamese neural network architecture
        self.net = SiameseNetwork(network=self.network, in_channels=self.n_channels, n_features=self.n_features).cuda()
        self.best_net = SiameseNetwork(network=self.network, in_channels=self.n_channels,
                                       n_features=self.n_features).cuda()

        # Choose loss function
        if self.loss_method == 'BCEWithLogitsLoss':
            self.loss = nn.BCEWithLogitsLoss().cuda()
        else:
            raise Exception('Invalid argument: ' + self.loss_method +
                            '\nChoose BCEWithLogitsLoss! Other loss functions are not yet implemented!')

        # Set the optimizer function
        if self.optimizer_method == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        else:
            raise Exception('Invalid argument: ' + self.optimizer_method +
                            '\nChoose Adam! Other optimizer functions are not yet implemented!')

        # load state dicts and other information in case a previous experiment will be continued
        if self.config['resumption'] is True:
            self.checkpoint = torch.load('./archive/' + self.config['previous_experiment'] + '/' + self.config[
                'previous_experiment'] + '_checkpoint' + self.load_suffix + '.pth')
            self.best_net.load_state_dict(torch.load(
                './archive/' + self.config['previous_experiment'] + '/' + self.config[
                    'previous_experiment'] + '_best_network' + self.load_suffix + '.pth'))
            self.net.load_state_dict(self.checkpoint['state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.best_loss = self.checkpoint['best_loss']
            self.loss_dict = self.checkpoint['loss_dict']
            self.es.best = self.checkpoint['best_loss']
            self.es.num_bad_epochs = self.checkpoint['num_bad_epochs']
            self.start_epoch = self.checkpoint['epoch']

        # Initialize transformations
        if self.transform == 'image_net':
            self.transform_train = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.transform_val_test = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.transform == "default":
            self.transform_train = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()
            ])
            self.transform_val_test = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()
            ])
        elif self.transform == "pmone": 
            self.transform_train = lambda x: (x  - x.min())/(x.max() - x.min()) * 2 - 1
            self.transform_val_test = lambda x: (x  - x.min())/(x.max() - x.min()) * 2 - 1

        elif self.transform == "none": 
            self.transform_train = lambda x: x
            self.transform_val_test = lambda x: x


        self.training_loader = Utils.get_data_loaders(phase='training', data_handling=self.data_handling,
                                                        n_channels=self.n_channels,
                                                        n_samples=self.n_samples_train,
                                                        transform=self.transform_train, image_path=self.IMAGE_PATH,
                                                        batch_size=self.batch_size, shuffle=True,
                                                        num_workers=self.num_workers, pin_memory=self.pin_memory,
                                                        save_path=None)
        self.validation_loader = Utils.get_data_loaders(phase='validation', data_handling=self.data_handling,
                                                        n_channels=self.n_channels,
                                                        n_samples=self.n_samples_val,
                                                        transform=self.transform_val_test,
                                                        image_path=self.IMAGE_PATH, batch_size=self.batch_size,
                                                        shuffle=False, num_workers=self.num_workers,
                                                        pin_memory=self.pin_memory, save_path=None, generator_seed=0)
        self.test_loader = Utils.get_data_loaders(phase='testing', data_handling='balanced',
                                                  n_channels=self.n_channels, n_samples=self.n_samples_test,
                                                  transform=self.transform_val_test, image_path=self.IMAGE_PATH,
                                                  batch_size=self.batch_size, shuffle=False,
                                                  num_workers=self.num_workers, pin_memory=self.pin_memory,
                                                  save_path=None, generator_seed=0)

    def training_validation(self):
        # Training and validation loop!
        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()

            training_loss = Utils.train(self.net, self.training_loader, self.n_samples_train, self.batch_size,
                                        self.loss, self.optimizer, epoch, self.max_epochs)

            self.validation_loader.dataset.reset_generator() # make sure this is deterministic for more accurate loss
            validation_loss = Utils.validate(self.net, self.validation_loader, self.n_samples_val, self.batch_size,
                                             self.loss, epoch, self.max_epochs)

            self.loss_dict['training'].append(training_loss)
            self.loss_dict['validation'].append(validation_loss)
            end_time = time.time()
            print('Time elapsed for epoch ' + str(epoch + 1) + ': ' + str(
                round((end_time - start_time) / 60, 2)) + ' minutes')

            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                self.best_net = copy.deepcopy(self.net)

            torch.save(self.best_net.state_dict(), self.SAVINGS_PATH + self.config[
                'experiment_description'] + '_best_network' + self.save_suffix + '.pth')

            Utils.save_loss_curves(self.loss_dict, self.SAVINGS_PATH, self.config['experiment_description'])
            Utils.plot_loss_curves(self.loss_dict, self.SAVINGS_PATH, self.config['experiment_description'])

            es_check = self.es.step(validation_loss)
            Utils.save_checkpoint(epoch, self.net, self.optimizer, self.loss_dict, self.best_loss,
                                  self.es.num_bad_epochs, self.SAVINGS_PATH + self.config[
                                      'experiment_description'] + '_checkpoint' + self.save_suffix + '.pth')

            if es_check:
                break

        print('Finished Training!')
    
    def testing_evaluation(self):
        # Testing phase!
        self.test_loader.dataset.reset_generator()
        y_true, y_pred = Utils.test(self.best_net, self.test_loader)
        y_true, y_pred = [y_true.numpy(), y_pred.numpy()]

        # Compute the evaluation metrics!
        fp_rates, tp_rates, thresholds = metrics.roc_curve(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_pred)
        y_pred_thresh = Utils.apply_threshold(y_pred, 0.5)
        accuracy, f1_score, precision, recall, report, confusion_matrix = Utils.get_evaluation_metrics(y_true,
                                                                                                       y_pred_thresh)
        auc_mean, confidence_lower, confidence_upper = Utils.bootstrap(10000,
                                                                       y_true,
                                                                       y_pred,
                                                                       self.SAVINGS_PATH,
                                                                       self.config['experiment_description'])

        # Plot ROC curve!
        Utils.plot_roc_curve(fp_rates, tp_rates, self.SAVINGS_PATH, self.config['experiment_description'])

        # Save all the results to files!
        Utils.save_labels_predictions(y_true, y_pred, y_pred_thresh, self.SAVINGS_PATH,
                                      self.config['experiment_description'])

        Utils.save_results_to_file(auc, accuracy, f1_score, precision, recall, report, confusion_matrix,
                                   self.SAVINGS_PATH, self.config['experiment_description'])

        Utils.save_roc_metrics_to_file(fp_rates, tp_rates, thresholds, self.SAVINGS_PATH,
                                       self.config['experiment_description'])

        # Print the evaluation metrics!
        print('EVALUATION METRICS:')
        print('AUC: ' + str(auc))
        print('Accuracy: ' + str(accuracy))
        print('F1-Score: ' + str(f1_score))
        print('Precision: ' + str(precision))
        print('Recall: ' + str(recall))
        print('Report: ' + str(report))
        print('Confusion matrix: ' + str(confusion_matrix))

        print('BOOTSTRAPPING: ')
        print('AUC Mean: ' + str(auc_mean))
        print('Confidence interval for the AUC score: ' + str(confidence_lower) + ' - ' + str(confidence_upper))
    
    def run(self):
        # Call training/validation and testing loop successively
        self.training_validation()
        self.testing_evaluation()

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


if __name__ == '__main__':
    # define an argument parser
    parser = argparse.ArgumentParser('Patient Verification')
    parser.add_argument('--config_path', default='./', help='the path where the config files are stored')
    parser.add_argument('--config', default='config.json', help='the hyper-parameter configuration and experiment settings')
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path + '\n--config: ' + args.config)

    # read config
    with open(args.config_path + args.config, 'r') as config:
        config = config.read()

    # parse config
    config = json.loads(config)

    # create folder to save experiment-related files
    os.makedirs(os.path.join('./experiments/' , config['experiment_description']), exist_ok=True)
    SAVINGS_PATH = './experiments/' + config['experiment_description'] + '/'

    # call agent and run experiment
    experiment = AgentSiameseNetwork(config)
    experiment.run()
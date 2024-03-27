import os
import torch
from torch.utils import data
from sklearn import metrics
import math
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from echosyn.common.datasets import SiameseUSDataset


"""
This file provides the most important functions that are used in our experiments. These functions are called in 
AgentSiameseNetwork.py which provides the actual training/validation loop and the code for evaluation.
"""


# Function to get the data loader.
def get_data_loaders(phase='training', data_handling='balanced', n_channels=3, n_samples=100000, transform=None,
                     image_path='./', batch_size=32, shuffle=True, num_workers=16, pin_memory=True, save_path=None, generator_seed=None):

    #dataset = SiameseDataset(phase=phase, data_handling=data_handling, n_channels=n_channels, n_samples=n_samples,
    #                         transform=transform, image_path=image_path, save_path=save_path)
    latents_csv = image_path
    training_latents_base_path = os.path.join(os.path.dirname(latents_csv), "Latents")
    dataset = SiameseUSDataset(phase=phase, transform=transform, latents_csv=latents_csv, training_latents_base_path=training_latents_base_path, in_memory=False, generator_seed=generator_seed)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1,
                                 pin_memory=pin_memory)

    return dataloader


# This function represents the training loop for the standard case where we have two input images and one output node.
def train(net, training_loader, n_samples, batch_size, criterion, optimizer, epoch, n_epochs):
    net.train()
    running_loss = 0.0

    print('Training----->')
    for i, batch in enumerate(training_loader):
        inputs1, inputs2, labels = batch
        inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs1, inputs2)
        outputs = outputs.squeeze()
        labels = labels.type_as(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f' % (epoch + 1, n_epochs, i + 1,
                                                                math.ceil(n_samples / batch_size),
                                                                loss.item()))

    # Compute the average loss per epoch
    training_loss = running_loss / math.ceil(n_samples / batch_size)
    return training_loss


# This function represents the validation loop for the standard case where we have two input images and one output node.
def validate(net, validation_loader, n_samples, batch_size, criterion, epoch, n_epochs):
    net.eval()
    running_loss = 0

    print('Validating----->')
    with torch.no_grad():
        for i, batch in enumerate(validation_loader):
            inputs1, inputs2, labels = batch
            inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()

            # forward
            outputs = net(inputs1, inputs2)
            outputs = outputs.squeeze()
            labels = labels.type_as(outputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f' % (epoch + 1, n_epochs, i + 1,
                                                                    math.ceil(n_samples / batch_size),
                                                                    loss.item()))

    # Compute the average loss per epoch
    validation_loss = running_loss / math.ceil(n_samples / batch_size)
    return validation_loss


# This function represents the test loop for the standard case where we have two input images and one output node.
# This function returns the true labels and the predicted values.
def test(net, test_loader):
    net.eval()
    y_true = None
    y_pred = None

    print('Testing----->')
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs1, inputs2, labels = batch

            if y_true is None:
                y_true = labels
            else:
                y_true = torch.cat((y_true, labels), 0)

            inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()
            outputs = net(inputs1, inputs2)
            outputs = torch.sigmoid(outputs)

            if y_pred is None:
                y_pred = outputs.cpu()
            else:
                y_pred = torch.cat((y_pred, outputs.cpu()), 0)

    y_pred = y_pred.squeeze()
    return y_true, y_pred


# This function computes some standard evaluation metrics given the true labels and the predicted values.
def get_evaluation_metrics(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    return accuracy, f1_score, precision, recall, report, confusion_matrix


# This function is used to apply a threshold to the predicted values before computing some evaluation metrics.
def apply_threshold(input_tensor, threshold):
    output = np.where(input_tensor > threshold, torch.ones(len(input_tensor)), torch.zeros(len(input_tensor)))
    return output


# This function implements bootstrapping, in order to get the mean AUC value and the 95% confidence interval.
def bootstrap(n_bootstraps, y_true, y_pred, path, experiment_description):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bootstrapped_scores = []

    f = open(path + experiment_description + '_AUC_bootstrapped.txt', "w+")
    f.write('AUC_bootstrapped\n')

    for i in range(n_bootstraps):
        indices = np.random.randint(0, len(y_pred) - 1, len(y_pred))
        auc = metrics.roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(auc)
        f.write(str(auc) + '\n')
    f.close()
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    auc_mean = np.mean(sorted_scores)
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    f = open(path + experiment_description + '_AUC_confidence.txt', "w+")
    f.write('AUC_mean: %s\n' % auc_mean)
    f.write('Confidence interval for the AUC score: ' + str(confidence_lower) + ' - ' + str(confidence_upper))
    f.close()
    return auc_mean, confidence_lower, confidence_upper


# This is a function that plots the loss curves.
def plot_loss_curves(loss_dict, path, experiment_description):
    plt.figure()
    plt.plot(range(1, len(loss_dict['training']) + 1), loss_dict['training'], label='Training Loss')
    plt.plot(range(1, len(loss_dict['validation']) + 1), loss_dict['validation'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss curves')
    plt.legend()
    plt.savefig(path + experiment_description + '_loss_curves.png')


# This is a function that plots the ROC curve.
def plot_roc_curve(fp_rates, tp_rates, path, experiment_description):
    plt.figure()
    plt.plot(fp_rates, tp_rates, label='ROC Curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(path + experiment_description + '_ROC_curve.png')


# This is a function that saves the evaluation metrics to a file.
def save_results_to_file(auc, accuracy, f1_score, precision, recall, report, confusion_matrix, path,
                         experiment_description):
    f = open(path + experiment_description + '_results.txt', "w+")
    f.write('AUC: %s\n' % auc)
    f.write('Accuracy: %s\n' % accuracy)
    f.write('F1-Score: %s\n' % f1_score)
    f.write('Precision: %s\n' % precision)
    f.write('Recall: %s\n' % recall)
    f.write('Classification report: %s\n' % report)
    f.write('Confusion matrix: %s\n' % confusion_matrix)
    f.close()


# This function saves the ROC metrics to a file.
def save_roc_metrics_to_file(fp_rates, tp_rates, thresholds, path, experiment_description):
    f = open(path + experiment_description + '_ROC_metrics.txt', "w+")
    f.write('FP_rate\tTP_rate\tThreshold\n')
    for i in range(len(fp_rates)):
        f.write(str(fp_rates[i]) + '\t' + str(tp_rates[i]) + '\t' + str(thresholds[i]) + '\n')
    f.close()


# This function saves the training and validation loss values to a file.
def save_loss_curves(loss_dict, path, experiment_description):
    f = open(path + experiment_description + '_loss_values.txt', "w+")
    f.write('TrainingLoss\tValidationLoss\n')
    for i in range(len(loss_dict['training'])):
        f.write(str(loss_dict['training'][i]) + '\t' + str(loss_dict['validation'][i]) + '\n')
    f.close()


# This function is used to save a checkpoint. This enables us to resume an experiment at a later point if wanted.
def save_checkpoint(epoch, model, optimizer, loss_dict, best_loss, num_bad_epochs, filename='checkpoint.pth'):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_dict': loss_dict,
        'best_loss': best_loss,
        'num_bad_epochs': num_bad_epochs
    }
    torch.save(state, filename)


# This function saves the true labels, the predicted vales, and the thresholded values to a file.
def save_labels_predictions(y_true, y_pred, y_pred_thresh, path, experiment_description):
    f = open(path + experiment_description + '_labels_predictions.txt', "w+")
    f.write('Label\tPrediction\tPredictionThresholded\n')
    for i in range(len(y_true)):
        f.write(str(y_true[i]) + '\t' + str(y_pred[i]) + '\t' + str(y_pred_thresh[i]) + '\n')
    f.close()


# This function was utilized to construct the positive and negative pairs needed for training and testing our
# verification network. This function takes filenames as an input argument and returns both the list of tuples and the
# list of corresponding labels. The constructed pairs were later saved to a .txt file. Now available in the folder
# './image_pairs/' (pairs_training.txt, pairs_validation.txt, pairs_testing.txt).
# Example:
#
# train_val_filenames = np.loadtxt('train_val_list.txt', dtype=str)
# test_filenames = np.loadtxt('test_list.txt', dtype=str)
# tuples_train, labels_train = get_tuples_labels(train_val_filenames[:75708])
# tuples_val, labels_val = get_tuples_labels(train_val_filenames[75708:])
# tuples_test, labels_test = get_tuples_labels(test_filenames)
#
def get_tuples_labels(filenames):
    tuples_list = []
    labels_list = []
    patients = []

    for i, element in enumerate(filenames):
        element = element[:-8]
        patients.append(element)

    unique_patients, counts_patients = np.unique(patients, return_counts=True)
    patients_dict = dict(zip(unique_patients, counts_patients))

    start_idx = 0
    patients_files_dict = {}

    for key in patients_dict:
        patients_files_dict[key] = filenames[start_idx:start_idx + patients_dict[key]]
        start_idx += patients_dict[key]

        if len(patients_files_dict[key]) > 1:
            # samples = list(itertools.product(patients_files_dict[key], patients_files_dict[key]))
            samples = list(itertools.combinations(patients_files_dict[key], 2))
            tuples_list.append(samples)
            labels_list.append(np.ones(len(samples)).tolist())

    tuples_list = list(itertools.chain.from_iterable(tuples_list))
    labels_list = list(itertools.chain.from_iterable(labels_list))

    N = len(tuples_list)
    i = 0

    while i < N:
        file1 = random.choice(filenames)
        file2 = random.choice(filenames)

        if file1[:-8] != file2[:-8]:
            sample = (file1, file2)
            tuples_list.append(sample)
            labels_list.append(0.0)
            i += 1

    return tuples_list, labels_list

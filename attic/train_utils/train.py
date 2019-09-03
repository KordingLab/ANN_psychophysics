
import torch
from torch.autograd import Variable

def train(model, epoch, features_loader, orientation_loader, optimizer, criterion,
        valid_features_loader, valid_orientation_loader, log_interval, batch_size):
    test_accuracy = []
    train_accuracy = []
    model.train()
    train_loss = 0
    i = 0
    for batch_idx, ((feats, _), orients) in enumerate(zip(features_loader, orientation_loader)):

        data, target = feats.cuda(), orients.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()

        train_loss += loss.data
        i += data.size()[0]

        optimizer.step()

        if (batch_idx > log_interval) and (batch_idx % log_interval == 0):
            train_loss /= i / batch_size
            i = 0

            train_accuracy.append(train_loss)
            test_loss = test(model,valid_features_loader, valid_orientation_loader, criterion)
            test_accuracy.append(test_loss)

            print('Epoch: {} Train Loss: {:.6f} || Test Loss: {:.6f}  '.format(
                epoch + 1,
                train_loss, test_loss))
            train_loss = 0

    return train_accuracy, test_accuracy


def test(model, valid_features_loader, valid_orientation_loader, criterion,):
    model.eval()
    test_loss = 0
    i = 0
    n_val = 2

    for batch_idx, ((feats, _), orients) in enumerate(zip(valid_features_loader, valid_orientation_loader)):
        data, target = feats.cuda(), orients.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.data
        i += 1
        if i > n_val:
            break

    test_loss /= i

    model.train()
    return test_loss
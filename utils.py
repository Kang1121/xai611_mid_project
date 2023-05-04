from braindecode.models import EEGNetv4
import torch.nn.functional as F
import torch
from torch.nn import init
import h5py
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.base import TransformerMixin
import logging
from sklearn.metrics import f1_score


def model_zoo(name, dataset):
    """Return a list of model names in the model zoo."""

    if dataset == 'bcic_iv_2a':
        channels = 22
        timepoints = 750
        num_classes = 4
    else:
        raise ValueError('Invalid dataset name: {}'.format(dataset))

    if name == 'EEGNetv4':
        model = EEGNetv4(in_chans=channels, n_classes=num_classes, input_window_samples=timepoints, final_conv_length='auto')
    else:
        raise ValueError('Invalid model name: {}'.format(name))

    logging.info(model)

    return init_weights(model, init_type='kaiming')


def load_data(dataset):
    """Load dataset."""

    if dataset == 'bcic_iv_2a':
        path = './data/BCICIV_2a_data_all.h5'
        data = h5py.File(path, 'r')
    else:
        raise ValueError('Invalid dataset name: {}'.format(dataset))

    return data


def data_split(train_index, valid_index, test_subj, data, cv_set):
    """Split the dataset into training, validation and test sets."""

    train_subjs = cv_set[train_index]
    valid_subjs = cv_set[valid_index]
    X_train, Y_train = get_multi_data(data, train_subjs)
    X_val, Y_val = get_multi_data(data, valid_subjs)
    X_test, Y_test = get_data(data, test_subj)
    X_test, Y_test = X_test[:], Y_test[:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def get_data(dfile, subj):

    dpath = '/s' + str(subj)
    x = dfile['{}/{}'.format(dpath, 'X')]
    y = dfile['{}/{}'.format(dpath, 'Y')]

    return x, y


def get_multi_data(dfile, subjs):

    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(dfile, s)
        Xs.append(x[:])
        Ys.append(y[:])

    return Xs, Ys


def clustering(data, label, prop):

    list_x, list_y = [], []
    for idx, (x, y) in enumerate(zip(data, label)):

        sorted_order = np.argsort(y)
        try:
            x, y = x[:][sorted_order], y[:][sorted_order]
        except IndexError:
            pass

        num = len(x) // (max(y) + 1)

        # random pick
        assert prop in [0.3, 0.6, 0.8]
        num_pick = int(num * prop)
        for i in range(max(y) + 1):
            data = x[num * i: num * (i + 1)]
            data = data[np.random.choice(int(data.shape[0]), num_pick)]
            list_x.append(data), list_y.append(np.ones((len(data)), dtype=np.int64) * i)

    return list_x, list_y


def accuracy(pred, label):

    count = 0
    for i in range(label.shape[0]):
        try:
            if np.argmax(pred[i]) == label[i]:
                count += 1
        except RuntimeError:
            if np.argmax(pred[i]) == np.argmax(label[i]):
                count += 1

    return count / label.shape[0]


def subs_preorder(dataset):

    if dataset in ['bcic_iv_2a']:
        return [i for i in range(1, 10)]
    else:
        raise ValueError('Invalid dataset name: {}'.format(dataset))


def training_module(
        loader,
        model,
        optimizer,
        train_mode,
        ):

    model.train() if train_mode else model.eval()

    loss_all, acc_all, f1score_all = 0, 0, 0
    for batch_idx, (data, label) in enumerate(loader):

        data, label = data.cuda(), label.cuda()
        data = data.permute(0, 3, 2, 1)

        output = squeeze_final_output(model(data))
        loss = F.nll_loss(output, label)

        pred = output.argmax(dim=1)

        loss_all += loss.item()
        acc_all += accuracy(output.detach().cpu().numpy(), label)
        f1score_all += f1_score(label.cpu().numpy(), pred.detach().cpu().numpy(), average='macro')

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_all / len(loader), acc_all / len(loader), f1score_all / len(loader)


def squeeze_final_output(x):
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """
    try:
        assert x.size()[3] == 1
        x = x[:, :, :, 0]
        if x.size()[2] == 1:
            x = x[:, :, 0]
    except IndexError:
        pass

    return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    logging.info('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

    return net


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # model.save_networks(self.name)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


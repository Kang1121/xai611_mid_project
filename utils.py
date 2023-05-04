from braindecode.models import Deep4Net, EEGNetv4, EEGInception, EEGResNet, EEGITNet, TIDNet, ShallowFBCSPNet, HybridNet
import torch.nn.functional as F
import torch
from torch.nn import init
import h5py
import os
from os.path import join as pjoin
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from captum.attr import LRP, LayerLRP
import mne
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin



CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9',
            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3',
            'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7',
            'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4']

CHANNELS_56 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9',
            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3',
            'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'TP7',
             'FT10', 'TP8', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4']

RESERVED = ['Fz', 'FC3', 'FC1', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz',
                'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P8', 'PO3', 'POz', 'PO4']


CHANNELS_39 = ['F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9',
            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'FC3',
            'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'TP7', 'TP8', 'PO3', 'PO4']

# find the position of the reserved channels in the original channel list
RESERVED_POS = [CHANNELS.index(ch) for ch in CHANNELS_39]
RESERVED_POS.sort()

def model_zoo(name, dataset):
    """Return a list of model names in the model zoo."""

    if dataset == 'sub54':
        channels = 39
        timepoints = 1000
        num_classes = 2
    elif dataset == 'iv_2a_2classes':
        channels = 22
        timepoints = 750
        num_classes = 2
    elif dataset == 'iv_2a_4classes':
        channels = 22
        timepoints = 750
        num_classes = 4
    else:
        raise ValueError('Invalid dataset name: {}'.format(dataset))

    if name == 'Deep4Net':
        model = Deep4Net(in_chans=channels, n_classes=num_classes, input_window_samples=timepoints, final_conv_length='auto')
    elif name == 'EEGNetv4':
        model = EEGNetv4(in_chans=channels, n_classes=num_classes, input_window_samples=timepoints, final_conv_length='auto')
    elif name == 'EEGInception':
        model = EEGInception(in_channels=channels, n_classes=num_classes, input_window_samples=timepoints)
    elif name == 'EEGResNet':
        model = EEGResNet(in_chans=channels, n_classes=num_classes, input_window_samples=timepoints, final_pool_length='auto', n_first_filters=30)
    elif name == 'EEGITNet':
        model = EEGITNet(in_channels=channels, n_classes=num_classes, input_window_samples=timepoints)
    elif name == 'TIDNet':
        model = TIDNet(in_chans=channels, n_classes=num_classes, input_window_samples=timepoints)
    elif name == 'ShallowFBCSPNet':
        model = ShallowFBCSPNet(in_chans=channels, n_classes=num_classes, input_window_samples=timepoints, final_conv_length='auto')
    elif name == 'HybridNet':
        model = HybridNet(in_chans=channels, n_classes=num_classes, input_window_samples=timepoints)
    else:
        raise ValueError('Invalid model name: {}'.format(name))

    print(model)

    return init_weights(model, init_type='kaiming')


def load_data(dataset):
    """Load dataset."""

    if dataset == 'sub54':
        path = './data/KU_mi_smt.h5'
        data = h5py.File(path, 'r')
    elif dataset == 'iv_2a_2classes':
        path = './data/BCICIV_2a_data_LR.h5'
        data = h5py.File(path, 'r')
    elif dataset == 'iv_2a_4classes':
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
    X = dfile[pjoin(dpath, 'X')]
    Y = dfile[pjoin(dpath, 'Y')]

    # use the reserved channels
    # X = X[:, RESERVED_POS, :]

    return X, Y


def get_multi_data(dfile, subjs):

    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(dfile, s)
        Xs.append(x[:])
        Ys.append(y[:])

    return Xs, Ys


def dbscan(data, eps, min_samples):
    """DBSCAN clustering with PCA"""

    data = StandardScaler().fit_transform(np.mean(data, axis=1))
    pca = PCA(n_components=10)
    data = pca.fit_transform(data)
    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    # print how many clusters are there in total and how many are noise points
    labels = db.labels_

    return db, labels


def clustering(X, Y, eps, min_samples, random_pick=False):

    list_x, list_y = [], []

    for _, (x, y) in enumerate(zip(X, Y)):

        idx = np.argsort(y)
        try:
            x, y = x[:][idx], y[:][idx]
        except IndexError:
            pass
        if not random_pick:
            # clustering in each class
            if max(y) == 1:
                # 2-class
                # class1, class2 = x[:200], x[200:]
                class1, class2 = x[:144], x[144:]

                dbs1, label1 = dbscan(class1, eps=eps, min_samples=min_samples)
                dbs2, label2 = dbscan(class2, eps=eps, min_samples=min_samples)

                class1, class2 = class1[label1 != -1], class2[label2 != -1]
                label1, label2 = np.zeros((len(label1[label1 != -1])), dtype=np.int64), \
                                 np.ones((len(label2[label2 != -1])), dtype=np.int64)
            elif max(y) == 3:
                # 4-class
                class1, class2, class3, class4 = x[:144], x[144:288], x[288:432], x[432:]

                dbs1, label1 = dbscan(class1, eps=eps, min_samples=min_samples)
                dbs2, label2 = dbscan(class2, eps=eps, min_samples=min_samples)
                dbs3, label3 = dbscan(class3, eps=eps, min_samples=min_samples)
                dbs4, label4 = dbscan(class4, eps=eps, min_samples=min_samples)

                class1, class2, class3, class4 = class1[label1 != -1], class2[label2 != -1], class3[label3 != -1], class4[label4 != -1]
                label1, label2, label3, label4 = np.zeros((len(label1[label1 != -1])), dtype=np.int64), \
                                                 np.ones((len(label2[label2 != -1])), dtype=np.int64), \
                                                 np.ones((len(label3[label3 != -1])), dtype=np.int64) * 2, \
                                                 np.ones((len(label4[label4 != -1])), dtype=np.int64) * 3
        else:
            # random pick
            if eps == 10:
                num = int(0.3 * 144)
            elif eps == 12:
                num = int(0.6 * 144)
            elif eps == 13.5:
                num = int(0.8 * 144)
            else:
                raise ValueError('Invalid eps: {}'.format(eps))

            class1, class2 = x[:144], x[144:]
            class1, class2 = class1[np.random.choice(class1.shape[0], num, replace=False)], \
                             class2[np.random.choice(class2.shape[0], num, replace=False)]
            label1, label2 = np.zeros((num), dtype=np.int64), np.ones((num), dtype=np.int64)
        if max(y) == 1:
            # concatenate
            list_x.append(class1), list_x.append(class2), list_y.append(label1), list_y.append(label2)
        elif max(y) == 3:
            list_x.append(class1), list_x.append(class2), list_x.append(class3), list_x.append(class4)
            list_y.append(label1), list_y.append(label2), list_y.append(label3), list_y.append(label4)

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


def subs_preorder():
    # return [35, 47, 46, 37, 13, 27, 12, 32, 53, 54, 4, 40, 19, 41, 18, 42, 34, 7, 49, 9, 5, 48, 29, 15, 21, 17, 31, 45,
    #         1, 38, 51, 8, 11, 16, 28, 44, 24, 52, 3, 26, 39, 50, 6, 23, 2, 14, 25, 20, 10, 33, 22, 43, 36, 30]
    return [1, 2, 3, 4, 5, 6, 7, 8, 9]

def training_module(
        loader,
        model,
        optimizer,
        device,
        train_mode,
        mixup=False,
        lrp=False,
        sub=None,
        path=None,
        ):

    if train_mode:
        model.train()
        # for param in model.parameters():
        #     param.requires_grad = True
    else:
        model.eval()
        # for param in model.parameters():
        #     param.requires_grad = False

    n_labels = 2
    attr_dict = {i: []  for i in range(n_labels)}
    loss_all, acc_all = 0, 0
    for batch_idx, (data, label) in enumerate(loader):

        data, label = data.cuda(), label.cuda()

        if model.__class__.__name__ in ['EEGInception', 'EEGITNet']:
            data = data.permute(0, 1, 3, 2)

        if mixup and train_mode:
            data, label_a, label_b, lam = mixup_data(data, label, 1.0, device=device)
            output = squeeze_final_output(model(data))
            loss = mixup_criterion(F.nll_loss, output, label_a, label_b, lam)
        else:
            output = squeeze_final_output(model(data))
            loss = F.nll_loss(output, label)

        pred = output.argmax(dim=1)
        if lrp:
            attr_dict = get_chan_attr(model, model.conv_time, data, pred, attr_dict)

        loss_all += loss.item()
        acc_all += accuracy(output.detach().cpu().numpy(), label)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if lrp:
        for i in range(n_labels):
            temp = sum(attr_dict[i]) / len(attr_dict[i])
            temp = temp.mean(1).squeeze().transpose(1, 0)
            get_topomap(temp, sub=sub, task=task_type(i), path=path)

    # print('LOADER:', len(loader), loader)
    return loss_all / len(loader), acc_all / len(loader)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0, device=None):
    """Returns mixed inputs, pairs of targets, and lambda"""

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def get_chan_attr(model, layer_name, data, label, attributions):

    layer_lrp = LayerLRP(model, layer_name)

    data_sorted = []
    for i in range(len(attributions.keys())):
        data_sorted.append(data[label == i])
        for j in range(len(data_sorted[i])):
            attr = layer_lrp.attribute(data_sorted[i][j].unsqueeze(0), i)
            attr = attr.detach().cpu().numpy()
            attributions[i].append(attr)

    return attributions


def get_topomap(data, sub=None, task=None, path=None):

    fig = plt.figure(figsize=(4, 3), dpi=500)
    ax1, ax2 = fig.add_axes([0.05, 0.1, 0.6, 0.8]), fig.add_axes([0.7, 0.1, 0.2, 0.8])
    # Convert to mne object
    # info = mne.create_info(ch_names=RESERVED, sfreq=250, ch_types='eeg')
    info = mne.create_info(ch_names=CHANNELS_39, sfreq=250, ch_types='eeg')
    montage = mne.channels.make_standard_montage(kind='standard_1020')
    evoked = mne.EvokedArray(data, info=info, tmin=0)
    times = np.arange(0.05, 3.95, 3.95)
    evoked = evoked.set_montage(montage)
    img = evoked.plot_topomap(times, ch_type='eeg', cmap='Spectral_r', res=16, image_interp='cubic', average=8, axes=[ax1, ax2], contours=4)

    # if dir is not exist, create it
    save_path = path + '/lrp'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img.savefig('{}/sub{}_{}.png'.format(save_path, sub, task), dpi=500)


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


def task_type(task_label):

    if task_label == 1:
        return 'left'
    elif task_label == 0:
        return 'right'


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

    print('initialize network with %s' % init_type)
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

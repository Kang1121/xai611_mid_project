import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='comparative runs')
# parser.add_argument('-dataset', type=str, default='sub54', help='dataset name')
parser.add_argument('-dataset', type=str, default='iv_2a_2classes', help='dataset name')
# parser.add_argument('-dataset', type=str, default='iv_2a_4classes', help='dataset name')
parser.add_argument('-model', type=str, default='EEGNetv4', help='model name')
parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-batch_size', type=int, default=16, help='batch size')

parser.add_argument('-fold', type=int, default=6, help='cross validation fold')
parser.add_argument('-outrm', action='store_true', help='outlier removal')
parser.add_argument('-eps', type=float, default=7, help='eps for DBSCAN')
parser.add_argument('-min_samples', type=int, default=5, help='min_samples for DBSCAN')
parser.add_argument('-n_epochs', type=int, default=3000, help='number of epochs')
parser.add_argument('-gpu', type=str, default='0', help='gpu device')
parser.add_argument('-mixup', action='store_true', help='mixup augmentation')
parser.add_argument('-test', action='store_true', help='mixup augmentation')
parser.add_argument('-lrp', action='store_true', help='Layer-wise Relevance Propagation')
parser.add_argument('-dependent', action='store_true', help='for LRP use')
parser.add_argument('-random_pick', action='store_true', help='for LRP use')

args, _ = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from utils import *
import numpy as np
from sklearn.model_selection import KFold
from braindecode.util import set_random_seeds
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler
import time


def main():

    set_random_seeds(seed=15485485, cuda=True)

    train(args)


def train(args):

    data = load_data(args.dataset)
    order = subs_preorder()
    kf = KFold(n_splits=args.fold)

    path1 = '/dependent/{}'.format(args.model) if args.dependent else '/{}'.format(args.model)
    path2 = '_or{}'.format(args.eps) if args.outrm else '_noor'
    path3 = '_mixup' if args.mixup else '_nomixup'

    out_path = './results{}{}{}'.format(path1, path2, path3)
    model_path = './checkpoints{}{}{}'.format(path1, path2, path3)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for idx, test_subj in enumerate(order):  # 54 subjects LOSO

        model = model_zoo(args.model, args.dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.cuda()

        cv_set = np.array(order[idx+1:] + order[:idx])
        for cv_index, (train_index, valid_index) in enumerate(kf.split(cv_set)):

            x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(train_index, valid_index, test_subj, data, cv_set)

            if args.outrm:
                x_train, y_train = clustering(x_train, y_train, args.eps, args.min_samples, random_pick=args.random_pick)
                # x_valid, y_valid = clustering(x_valid, y_valid, args.eps, args.min_samples)

            x_train, y_train = np.concatenate(x_train, axis=0), np.concatenate(y_train, axis=0)
            x_valid, y_valid = np.concatenate(x_valid, axis=0), np.concatenate(y_valid, axis=0)

            scaler = NDStandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_valid = scaler.fit_transform(x_valid)
            x_test = scaler.fit_transform(x_test)

            x_train, x_valid, x_test = x_train.transpose((0, 2, 1)), x_valid.transpose((0, 2, 1)), x_test.transpose((0, 2, 1))

            train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train).float().unsqueeze(1), torch.from_numpy(y_train).long()), batch_size=args.batch_size, shuffle=False, num_workers=0)
            valid_loader = DataLoader(TensorDataset(torch.from_numpy(x_valid).float().unsqueeze(1), torch.from_numpy(y_valid).long()), batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test).float().unsqueeze(1), torch.from_numpy(y_test).long()), batch_size=args.batch_size, shuffle=False)

            scheduler_steplr = StepLR(optimizer, step_size=100, gamma=0.1)
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20,
                                                      after_scheduler=scheduler_steplr)

            # this zero gradient update is needed to avoid a warning message, issue #8 in the git repo.
            optimizer.zero_grad()
            optimizer.step()

            if not args.test:
                early_stopping = EarlyStopping(patience=20, verbose=True, path='{}/checkpoint_sub{}.pth'.format(model_path, test_subj))

                for epoch in range(args.n_epochs):

                    scheduler_warmup.step(epoch)
                    start_time = time.time()
                    if args.dependent:
                        loss_train, acc_train = training_module(test_loader, model, optimizer, args.gpu, True, args.mixup)
                        loss_valid, acc_valid = training_module(test_loader, model, optimizer, args.gpu, True, args.mixup)
                    else:
                        loss_train, acc_train = training_module(loader=train_loader, model=model, optimizer=optimizer, device=args.gpu, train_mode=True, mixup=args.mixup)
                        loss_valid, acc_valid = training_module(loader=valid_loader, model=model, optimizer=optimizer, device=args.gpu, train_mode=False, mixup=args.mixup)

                    print('Epoch: {:03d}, LR: {:.5f}, Train Loss: {:.5f}, Train Acc: {:.5f}, Valid Loss: {:.5f}, Valid Acc: {:.5f}, Time Elapsed: {:.2f}'.format(epoch, optimizer.param_groups[0]['lr'], loss_train, acc_train, loss_valid, acc_valid, time.time() - start_time))

                    if epoch > 200:
                        early_stopping(loss_valid, model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
            else:
                model.load_state_dict(torch.load('{}/checkpoint_sub{}.pth'.format(model_path, test_subj)))

            _, acc_test = training_module(loader=test_loader, model=model, optimizer=optimizer, device=args.gpu, train_mode=False, mixup=args.mixup, lrp=args.lrp, sub=test_subj, path=out_path)
            # _, acc_test = training_module(loader=train_loader, model=model, optimizer=optimizer, device=args.gpu, train_mode=False, mixup=args.mixup, lrp=args.lrp, sub=test_subj, path=out_path)

            print('Test Acc: {:.5f}'.format(acc_test))

            with open(os.path.join(out_path, 'test_acc.txt'), 'a') as f:
                f.write('sub{}_fold{}: {}\n'.format(test_subj, cv_index, acc_test))
            f.close()

            # break


if __name__ == "__main__":

    # create folders
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./data'):
        os.makedirs('./data')

    main()

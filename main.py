import argparse
import os
from utils import *
import numpy as np
from sklearn.model_selection import KFold
from braindecode.util import set_random_seeds
from warmup_scheduler import GradualWarmupScheduler
import time
import logging

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='xai611_mid_project')
parser.add_argument('-dataset', type=str, default='bcic_iv_2a', help='dataset name')
parser.add_argument('-model', type=str, default='EEGNetv4', help='model name')
parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-batch_size', type=int, default=16, help='batch size')
parser.add_argument('-fold', type=int, default=8, help='cross validation fold')
parser.add_argument('-proportion', type=float, default=1, help='eps for DBSCAN')
parser.add_argument('-n_epochs', type=int, default=300, help='number of epochs')
parser.add_argument('-gpu', type=str, default='0', help='gpu device')
parser.add_argument('-test', action='store_true', help='mixup augmentation')
parser.add_argument('-random_pick', action='store_true', help='for LRP use')


def main(args):

    set_random_seeds(seed=15485485, cuda=True)
    data = load_data(args.dataset)
    order = subs_preorder(args.dataset)
    kf = KFold(n_splits=args.fold)

    out_path = './results/{}/{}_prop{}'.format(args.dataset, args.model, args.proportion)
    model_path = './checkpoints/{}/{}_prop{}'.format(args.dataset, args.model, args.proportion)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if os.path.exists(os.path.join(out_path, 'result_subs.csv')):
        os.remove(os.path.join(out_path, 'result_subs.csv'))
    if os.path.exists(os.path.join(out_path, 'result_all.csv')):
        os.remove(os.path.join(out_path, 'result_all.csv'))

    logging.basicConfig(handlers=[logging.StreamHandler(), logging.FileHandler('{}/log.txt'.format(out_path))],
                        level=logging.INFO)

    for idx, test_subj in enumerate(order):

        model = model_zoo(args.model, args.dataset)
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        cv_set = np.array(order[idx+1:] + order[:idx])
        for cv_index, (train_index, valid_index) in enumerate(kf.split(cv_set)):

            x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(train_index, valid_index, test_subj, data, cv_set)

            if args.random_pick:
                assert args.proportion < 1
                x_train, y_train = clustering(x_train, y_train, args.proportion)
                logging.info('Running with random pick, proportion: {}'.format(args.proportion))
            elif args.proportion < 1:
                raise ValueError('Random pick setting is not enabled, please specify -random_pick option')
            else:
                assert args.proportion == 1
                logging.warning('Running with all data, no random pick')

            x_train, y_train = np.concatenate(x_train, axis=0), np.concatenate(y_train, axis=0)
            x_valid, y_valid = np.concatenate(x_valid, axis=0), np.concatenate(y_valid, axis=0)

            scaler = NDStandardScaler()
            x_train, x_valid, x_test = scaler.fit_transform(x_train), \
                                       scaler.fit_transform(x_valid), \
                                       scaler.fit_transform(x_test)

            x_train, x_valid, x_test = x_train.transpose((0, 2, 1)), x_valid.transpose((0, 2, 1)), x_test.transpose((0, 2, 1))
            logging.info('Number of train samples: {}'.format(x_train.shape[0]))
            logging.info('Estimated number of train samples: {}'.format(int(len(train_index) * args.proportion * x_test.shape[0])))

            train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train).float().unsqueeze(1), torch.from_numpy(y_train).long()), batch_size=args.batch_size, shuffle=False, num_workers=0)
            valid_loader = DataLoader(TensorDataset(torch.from_numpy(x_valid).float().unsqueeze(1), torch.from_numpy(y_valid).long()), batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test).float().unsqueeze(1), torch.from_numpy(y_test).long()), batch_size=args.batch_size, shuffle=False)

            scheduler_steplr = StepLR(optimizer, step_size=60, gamma=0.1)
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20,
                                                      after_scheduler=scheduler_steplr)

            # this zero gradient update is needed to avoid a warning message, issue #8 in the git repo.
            optimizer.zero_grad()
            optimizer.step()

            if not args.test:
                early_stopping = EarlyStopping(patience=20, verbose=True, path='{}/checkpoint_sub{}.pth'.format(model_path, test_subj), trace_func=logging.info)

                for epoch in range(args.n_epochs):

                    scheduler_warmup.step()
                    start_time = time.time()

                    loss_train, acc_train, f1_train = training_module(loader=train_loader, model=model, optimizer=optimizer, train_mode=True)
                    loss_valid, acc_valid, f1_valid = training_module(loader=valid_loader, model=model, optimizer=optimizer, train_mode=False)

                    logging.info('Epoch: {:03d}, LR: {:.5f}, Train Loss: {:.5f}, Train Acc: {:.5f}, Valid Loss: {:.5f}, Valid Acc: {:.5f}, Time Elapsed: {:.2f}'.format(epoch, optimizer.param_groups[0]['lr'], loss_train, acc_train, loss_valid, acc_valid, time.time() - start_time))

                    if epoch > 120:
                        early_stopping(loss_valid, model)
                    if early_stopping.early_stop:
                        logging.info("Early stopping")
                        break
            else:
                model.load_state_dict(torch.load('{}/checkpoint_sub{}.pth'.format(model_path, test_subj)))

            _, acc_test, f1_test = training_module(loader=test_loader, model=model, optimizer=optimizer, train_mode=False)

            logging.info('Test Acc: {:.5f}'.format(acc_test))

            with open(os.path.join(out_path, 'result_subs.csv'), 'a') as f:
                if idx == 0 and cv_index == 0:
                    f.write('subject, fold, accuracy, f1-score\n')
                f.write('{}, {}, {}, {}\n'.format(test_subj, cv_index, acc_test, f1_test))
            f.close()

    with open(os.path.join(out_path, 'result_subs.csv'), 'r') as f:
        lines = f.readlines()
        acc = []
        f1 = []
        for line in lines:
            if line.split(',')[0] == 'subject':
                continue
            acc.append(float(line.split(',')[2]))
            f1.append(float(line.split(',')[3]))
    f.close()

    acc, f1 = np.array(acc), np.array(f1)
    logging.info('Average Acc: {:.5f}, Std: {:.5f}'.format(np.mean(acc), np.std(acc)))
    logging.info('Average F1: {:.5f}, Std: {:.5f}'.format(np.mean(f1), np.std(f1)))

    with open(os.path.join(out_path, 'result_all.csv'), 'a+') as f:
        f.write('subject, acc_mean, acc_std, f1_mean, f1_std\n')
        for idx in range(args.fold):
            f.write('{}, {}, {}, {}, {}\n'.format(
                idx+1,
                np.mean(acc[idx*args.fold:(idx+1)*args.fold]),
                np.std(acc[idx*args.fold:(idx+1)*args.fold]),
                np.mean(f1[idx*args.fold:(idx+1)*args.fold]),
                np.std(f1[idx*args.fold:(idx+1)*args.fold]))
            )
        f.seek(0)
        acc, f1 = [], []
        for line in f.readlines()[1:]:
            acc.append(float(line.split(',')[1]))
            f1.append(float(line.split(',')[3]))
        f.write('average, {}, {}, {}, {}\n'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1)))
    f.close()


if __name__ == "__main__":

    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim.lr_scheduler import StepLR

    main(args)


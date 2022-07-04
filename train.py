import numpy as np
import torch
from torch_geometric.data import DataLoader
import argparse
from models.GAT import GATModel
from utils.data_processing import load_data
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import time
import datetime
import warnings
warnings.filterwarnings("ignore")

def train(args):
    threshold = args.d

    # loading and spliting data

    fasta_path_train_positive = 'data/train_data/positive/XU_train_pos_1500.fasta'
    fasta_path_val_positive = 'data/train_data/positive/XU_val_pos_1500.fasta'
    npz_dir_positive = 'data/train_data/positive/npz/'
    data_train, _ = load_data(fasta_path_train_positive, npz_dir_positive, threshold, 1)
    data_val, _ = load_data(fasta_path_val_positive, npz_dir_positive, threshold, 1)

    fasta_path_train_negative = 'data/train_data/negative/XU_train_neg_1500.fasta'
    fasta_path_val_negative = 'data/train_data/negative/XU_val_neg_1500.fasta'
    npz_dir_negative = 'data/train_data/negative/npz/'
    neg_data_train, _ = load_data(fasta_path_train_negative, npz_dir_negative, threshold, 0)
    neg_data_val, _ = load_data(fasta_path_val_negative, npz_dir_negative, threshold, 0)

    data_train.extend(neg_data_train)
    data_val.extend(neg_data_val)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    node_feature_dim = data_train[0].x.shape[1]
    n_class = 2

    # tensorboard, record the change of auc, acc and loss
    writer = SummaryWriter()

    if args.pretrained_model is None:
        model = GATModel(node_feature_dim, args.hd, n_class, args.drop, args.heads).to(device)
    else:
        model = torch.load(args.pretrained_model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    criterion = torch.nn.CrossEntropyLoss()
    train_dataloader = DataLoader(data_train, batch_size=args.b)
    val_dataloader = DataLoader(data_val, batch_size=args.b)

    best_acc = 0
    best_auc = 0
    min_loss = 1000
    save_acc = '/'.join(args.save.split('/')[:-1]) + '/acc_' + args.save.split('/')[-1]
    save_auc = '/'.join(args.save.split('/')[:-1]) + '/auc_' + args.save.split('/')[-1]
    save_loss = '/'.join(args.save.split('/')[:-1]) + '/loss_' + args.save.split('/')[-1]

    for epoch in range(args.e):
        print('Epoch ', epoch)
        model.train()
        arr_loss = []
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            data = data.to(device)

            output = model(data.x, data.edge_index, data.batch)
            out = output[0]
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            arr_loss.append(loss.item())

        avgl = np.mean(arr_loss)
        print("Training Average loss :", avgl)

        model.eval()
        with torch.no_grad():
            total_num = 0
            total_correct = 0
            preds = []
            y_true = []
            arr_loss = []
            for data in val_dataloader:
                data = data.to(device)

                output = model(data.x, data.edge_index, data.batch)
                out = output[0]

                loss = criterion(out, data.y)
                arr_loss.append(loss.item())

                pred = out.argmax(dim=1)
                score = F.softmax(out, dim=1)[:, 1]
                correct = (pred == data.y).sum().float()
                total_correct += correct
                total_num += data.num_graphs
                preds.extend(score.cpu().detach().data.numpy())
                y_true.extend(data.y.cpu().detach().data.numpy())

            acc = (total_correct / total_num).cpu().detach().data.numpy()
            auc = roc_auc_score(y_true, preds)
            val_loss = np.mean(arr_loss)
            print("Validation accuracy: ", acc)
            print("Validation auc:", auc)
            print("Validation loss:", val_loss)

            writer.add_scalar('Loss', avgl, global_step=epoch)
            writer.add_scalar('acc', acc, global_step=epoch)
            writer.add_scalar('auc', auc, global_step=epoch)

            if acc > best_acc:
                best_acc = acc
                torch.save(model, save_acc)

            if auc > best_auc:
                best_auc = auc
                torch.save(model, save_auc)

            if np.mean(val_loss) < min_loss:
                min_loss = val_loss
                torch.save(model, save_loss)

            print('-' * 50)

        scheduler.step()

    print('best acc:', best_acc)
    print('best auc:', best_auc)
    if args.o is not None:
        with open(args.o, 'a') as f:
            localtime = time.asctime(time.localtime(time.time()))
            f.write(str(localtime) + '\n')
            f.write('args: ' + str(args) + '\n')
            f.write('auc result: ' + str(best_auc) + '\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-drop', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('-e', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('-b', type=int, default=512, help='Batch size')
    parser.add_argument('-hd', type=int, default=64, help='Hidden layer dim')

    parser.add_argument('-pretrained_model', type=str, default=None,
                        help='The path of pretraining model, if None, the model will be trained from scratch')
    parser.add_argument('-save', type=str, default='saved_models/samp.model',
                        help='The path saving the trained models')
    parser.add_argument('-heads', type=int, default=8, help='Hidden layer dim')

    parser.add_argument('-d', type=int, default=37, help='Distance threshold to construct a graph, 0-37')
    parser.add_argument('-o', type=str, default='results/log.txt', help='File saving prediction results')
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    train(args)

    end_time = datetime.datetime.now()
    print('End time(min):', (end_time - start_time).seconds / 60)
import numpy as np
import torch
from torch_geometric.data import DataLoader
import argparse
from models.GAT import GATModel
from utils.data_processing import load_data
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
import torch.nn.functional as F
import pandas as pd


def independent_test(args):

    threshold = args.d

    fasta_path_positive = args.pos_t
    npz_dir_positive = args.pos_npz

    data_list, labels = load_data(fasta_path_positive, npz_dir_positive, threshold, 1)

    fasta_path_negative = args.neg_t
    npz_dir_negative = args.neg_npz

    neg_data = load_data(fasta_path_negative, npz_dir_negative, threshold, 0)

    data_list.extend(neg_data[0])
    # labels = np.concatenate((labels, neg_data[1]), axis=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(args.save).to(device)

    test_dataloader = DataLoader(data_list, batch_size=args.b, shuffle=False)
    y_true = []
    y_pred = []
    prob = []

    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)

            output = model(data.x, data.edge_index, data.batch)
            out = output[0]

            pred = out.argmax(dim=1)
            score = F.softmax(out)[:, 1]

            prob.extend(score.cpu().detach().data.numpy().tolist())
            y_true.extend(data.y.cpu().detach().data.numpy().tolist())
            y_pred.extend(pred.cpu().detach().data.numpy().tolist())

        auc = roc_auc_score(y_true, prob)
        acc = accuracy_score(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)

        print("Test AUC: ", auc)
        print("ACC", acc)
        print("f1", f1)
        print("MCC", mcc)
        print("sn", sn)
        print("sp", sp)

        if args.o is not None:
            res_data = {'AMP_label': y_true, 'score': prob, 'pred': y_pred}
            df = pd.DataFrame(res_data)
            df.to_csv(args.o, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input file
    parser.add_argument('-pos_t', type=str, default='example/positive/example_pos.fasta',
                        help='Path of the positive test dataset')
    parser.add_argument('-pos_npz', type=str, default='example/positive/npz/',
                        help='Path of the positive npz folder, which saves the predicted structure')

    parser.add_argument('-neg_t', type=str, default='example/negative/example_neg.fasta',
                        help='Path of the negative test dataset')
    parser.add_argument('-neg_npz', type=str, default='example/negative/npz/', 
                        help='Path of the positive npz folder, which saves the predicted structure')

    parser.add_argument('-b', type=int, default=512, help='Batch size')
    parser.add_argument('-save', type=str, default='saved_models/samp.model',
                        help='The directory saving the trained models')
    parser.add_argument('-o', type=str, default='test_results.csv', help='Results file')
    parser.add_argument('-drop', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('-hd', type=int, default=64, help='Hidden layer dim')
    parser.add_argument('-heads', type=int, default=8, help='Number of heads')
    parser.add_argument('-d', type=int, default=37, help='Distance threshold')
    args = parser.parse_args()

    

    independent_test(args)

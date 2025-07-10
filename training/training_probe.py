import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import argparse
INPUT_DIM = 5120
HIDDEN_DIM = 256

class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class SemanticEntropyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))  # 输出概率
        return out.squeeze(-1)


def getting_training_examples(pkl_path,method):
    method = method
    x, y, z = [], [], []
    group_size = 21
    if os.path.getsize(pkl_path) > 0:
        with open(pkl_path, "rb") as f:
            generations = pickle.load(f)
    else:
        print('ggg')
    for i in range(0, len(generations), group_size):
        group = generations[i:i + group_size]
        for g in group[1:]:
            if g['probability_of_deberta'] is not None:
                if method == "last_hidden_state":
                    x.append(g['last_hidden_state'])
                elif method == "last_second_token":
                    x.append(g['sec_last_hidden_state'])
                elif method == "last_input_token":
                    x.append(g['last_input_token_state'])
                elif method == "output_last_hidden_list":
                    ## average pooling last hidden states of all tokens in the sequence, [len ,1 ,D] -> [1,D]
                    output_last_hidden_list = g['output_last_hidden_list'] # [len ,1 ,D]
                    output_last_hidden_list = output_last_hidden_list.squeeze(1) # [len ,D]
                    output_last_hidden_list = output_last_hidden_list.mean(dim=0, keepdim=True) #[1,D]
                    x.append(output_last_hidden_list)

                elif method == "":
                    #x.append(g[''])
                    pass
                elif method == "":
                    pass
                    #x.append(g[''])
                y.append(g['probability_of_deberta'])
                z.append(g['clustering-gpt-prompty_deberta'])

    assert len(x) == len(y)
    return x, y, z


def train_probe_regression(
    model, train_loader, val_loader, epochs=50, lr=1e-3,
    device='cuda', early_stop_rounds=5, save_pred_path=None,method="", dataset_name = "",model_name =""
):
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_counter = 0
    history = []
    best_preds = None

    for epoch in range(1, epochs+1):
        # 训练
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)
        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Valid Epoch {epoch}"):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                val_loss += loss.item() * len(x_batch)
                preds.append(pred.cpu().numpy())
                targets.append(y_batch.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        # 评估多指标
        val_mae = mean_absolute_error(targets, preds)
        val_r2 = r2_score(targets, preds)
        print(f"Epoch {epoch}: Train loss {train_loss:.4f}, Val MSE {val_loss:.4f}, Val MAE {val_mae:.4f}, Val R2 {val_r2:.4f}")

        # Early stopping机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            best_preds = preds.copy()
            torch.save(model.state_dict(), f'new_{model_name}_{dataset_name}_{method}_best_probe_mse.pt')
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_rounds:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_rounds} epochs)")
                break

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mse': val_loss,
            'val_mae': val_mae,
            'val_r2': val_r2
        })
    print(f"Best epoch: {best_epoch}, Best val MSE: {best_val_loss:.4f}")
    # 保存验证集预测
    if save_pred_path is not None:
        np.savez(save_pred_path, pred=best_preds, target=targets)
    return history

def create_Xs_and_ys(datasets, scores, val_test_splits=[0.2, 0.1], random_state=42):
    X = np.array([d[0] for d in datasets])
    y = np.array(scores)

    valid_size, test_size = val_test_splits


    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    val_adjusted = valid_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_adjusted, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test





def main(dataset,method,data_dir,model_name):

    start = 0
    end = 100
    X, Y = [], []
    skip_numbers = [1, 9, 10, 17, 18, 19, 21, 26, 30, 32, 36, 41, 43, 62, 64, 71, 80, 82, 88, 94, 96, 97]

    base_dir = data_dir
    for number in tqdm(range(start, end)):
        if number in skip_numbers:
            continue
        dirname = f'data-500-temp0_{number}'
        dir_path = os.path.join(base_dir, dirname)
        pkl_path = os.path.join(dir_path, f'new_generations_with_entropy_prob{number}.pkl')
        if not os.path.exists(pkl_path):
            print(f"File {pkl_path} does not exist!")
            continue


        x, y, z = getting_training_examples(pkl_path,method,)
        X.extend(x)
        Y.extend(y)
    print("X",len(X))
    X_train, X_val, X_test, y_train, y_val, y_test = create_Xs_and_ys(X, Y)
    train_set = ProbeDataset(X_train, y_train)
    val_set = ProbeDataset(X_val, y_val)
    test_set = ProbeDataset(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)

    # 模型与训练
    INPUT_DIM = X_train.shape[1]
    HIDDEN_DIM = 256
    model = SemanticEntropyModel(INPUT_DIM, HIDDEN_DIM)
    history = train_probe_regression(
        model, train_loader, val_loader, epochs=50, lr=1e-3,
        device='cuda', early_stop_rounds=7, save_pred_path=f'new_{model_name}_{dataset}_{method}_val_pred_results.npz',method=method,dataset_name = dataset,
        model_name = model_name
    )


    model.load_state_dict(torch.load(f'new_{model_name}_{dataset}_{method}_best_probe_mse.pt'))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Testing"):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            preds = model(x_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    test_mse = mean_squared_error(all_targets, all_preds)
    test_mae = mean_absolute_error(all_targets, all_preds)
    test_r2 = r2_score(all_targets, all_preds)
    print(f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")
    np.savez(f'new_{model_name}_{dataset}_{method}_test_pred_results.npz', pred=all_preds, target=all_targets)


    val_mse = [h['val_mse'] for h in history]
    val_mae = [h['val_mae'] for h in history]
    plt.figure()
    plt.plot(val_mse, label='Val MSE')
    plt.plot(val_mae, label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'new_{model_name}_{dataset}_{method}_Validation Loss Curve')
    plt.savefig(f'new_{model_name}_{dataset}_{method}_validation_loss_curve.png',dpi=200,bbox_inches='tight')
    plt.show()
#/data/ximing/aime
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # "last_hidden_state", "last_second_token", "last_input_token", "output_last_hidden_list"
    parser.add_argument("--dataset", type=str, required=True, help="dataset")
    parser.add_argument("--model", type=str, required=True, help="model")
    parser.add_argument("--method", type=str, required=True, help="method for X")
    parser.add_argument("--data_dir", type=str, required=True, help="method for X")
    args = parser.parse_args()
    main(args.dataset,args.method,args.data_dir,args.model)



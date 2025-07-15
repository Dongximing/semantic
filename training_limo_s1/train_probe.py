import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import argparse

class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class MultiLayerSemanticEntropyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        # 中间层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        # 输出层
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = torch.sigmoid(self.net(x))
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
                    output_last_hidden_list = g['output_last_hidden_list']
                    output_last_hidden_list = output_last_hidden_list.squeeze(1)
                    output_last_hidden_list = output_last_hidden_list.mean(dim=0, keepdim=True)
                    x.append(output_last_hidden_list)
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
        val_mae = mean_absolute_error(targets, preds)
        val_r2 = r2_score(targets, preds)
        print(f"Epoch {epoch}: Train loss {train_loss:.4f}, Val MSE {val_loss:.4f}, Val MAE {val_mae:.4f}, Val R2 {val_r2:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            best_preds = preds.copy()
            torch.save(model.state_dict(), f's1_valid_new_{model_name}_{dataset_name}_{method}_best_probe_mse.pt')
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
    if save_pred_path is not None:
        np.savez(save_pred_path, pred=best_preds, target=targets)
    return history

def create_Xs_and_ys(datasets, scores, val_test_splits=[0.2, 0.1], random_state=42):
    X = np.array([d[0] if isinstance(d, (list, tuple, np.ndarray)) and np.array(d).ndim==2 else d for d in datasets])
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
    end = 877
    X, Y = [], []

    base_dir = data_dir
    for number in tqdm(range(start, end)):
        dirname = f'data-877_{number}'
        dir_path = os.path.join(base_dir, dirname)
        pkl_path = os.path.join(dir_path, f'new_generations_with_entropy_prob{number}.pkl')
        if not os.path.exists(pkl_path):
            print(f"File {pkl_path} does not exist!")
            continue

        x, y, z = getting_training_examples(pkl_path, method)
        X.extend(x)
        Y.extend(y)
    print("X", len(X))

    X_train, X_val, X_test, y_train, y_val, y_test = create_Xs_and_ys(X, Y)
    train_set = ProbeDataset(X_train, y_train)
    val_set = ProbeDataset(X_val, y_val)
    test_set = ProbeDataset(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)

    # 结构配置
    configs = [
        {"hidden_dim": 256, "num_layers": 1},
        {"hidden_dim": 256, "num_layers": 2},
        {"hidden_dim": 512, "num_layers": 2},
        {"hidden_dim": 256, "num_layers": 3},
        {"hidden_dim": 512, "num_layers": 3},
    ]

    for cfg in configs:
        print(f"\n==== Try config: hidden_dim={cfg['hidden_dim']}, num_layers={cfg['num_layers']} ====")
        model = MultiLayerSemanticEntropyModel(
            input_dim=X_train.shape[1],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=0.3,
        )

        # 文件名带结构信息
        model_tag = f"{model_name}_hd{cfg['hidden_dim']}_nl{cfg['num_layers']}"
        history = train_probe_regression(
            model, train_loader, val_loader, epochs=50, lr=1e-3,
            device='cuda', early_stop_rounds=7,
            save_pred_path=f's1_valid_new_{model_tag}_{dataset}_{method}_val_pred_results.npz',
            method=method, dataset_name=dataset, model_name=model_tag
        )

        # 测试集评估
        model.load_state_dict(torch.load(f's1_valid_new_{model_tag}_{dataset}_{method}_best_probe_mse.pt'))
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
        print(f"[{model_tag}] Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")
        np.savez(f's1_valid_new_{model_tag}_{dataset}_{method}_test_pred_results.npz', pred=all_preds, target=all_targets)

        # 曲线保存
        val_mse = [h['val_mse'] for h in history]
        val_mae = [h['val_mae'] for h in history]
        plt.figure()
        plt.plot(val_mse, label='Val MSE')
        plt.plot(val_mae, label='Val MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{model_tag}_{dataset}_{method}_Validation Loss Curve')
        plt.savefig(f's1_valid_new_{model_tag}_{dataset}_{method}_validation_loss_curve.png', dpi=200, bbox_inches='tight')
        plt.close()  # 不用show，方便批量跑

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="dataset")
    parser.add_argument("--model", type=str, required=True, help="model")
    parser.add_argument("--method", type=str, required=True, help="method for X")
    parser.add_argument("--data_dir", type=str, required=True, help="data directory")
    args = parser.parse_args()
    main(args.dataset, args.method, args.data_dir, args.model)

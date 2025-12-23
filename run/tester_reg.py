import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from prettytable import PrettyTable
from tqdm import tqdm


class Tester_Reg(object):
    def __init__(self, model, device, test_dataloader, weight_path, alpha=1, **config):
        self.weight_path = weight_path
        self.model = model
        self.device = device
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.step = 0

        self.test_metrics = {}
        self.config = config
        self.loss_func = nn.MSELoss()

    def test(self, dataloader="test"):
        float2str = lambda x: '%0.4f' % x
        test_loss = 0
        feat_list = []
        y_label, y_pred = [], []
        data_loader = self.test_dataloader
        num_batches = len(data_loader)

        self.model.load_state_dict(torch.load(self.weight_path))
        self.model.to(self.device)

        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, v_d_aug, v_p_aug, labels, v_d_mask, v_p_mask) in enumerate(tqdm(data_loader)):
                v_d, v_p, v_d_aug, v_p_aug, labels, v_d_mask, v_p_mask = v_d.to(self.device), v_p.to(self.device), v_d_aug.to(self.device), v_p_aug.to(self.device), labels.float().to(self.device), v_d_mask.to(self.device), v_p_mask.to(self.device)

                v_d, v_p, f, score, z, z_aug = self.model(v_d, v_p, v_d_aug, v_p_aug, v_d_mask, v_p_mask)
                loss_con_aug = self.loss_func(z_aug, z)
                loss = self.loss_func(score, labels.unsqueeze(-1)) + 0.5* loss_con_aug
                test_loss += loss.item()

                feat_list += f.cpu().tolist()
                y_label += labels.cpu().tolist()
                y_pred += score.cpu().tolist()
        
        # ====== 计算整体指标 ======
        test_loss = test_loss / num_batches
        y_true_arr = np.array(y_label).flatten()
        y_pred_arr = np.array(y_pred).flatten()

        pcc, _ = pearsonr(y_true_arr, y_pred_arr)
        rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
        mae = mean_absolute_error(y_true_arr, y_pred_arr)
        r2 = r2_score(y_true_arr, y_pred_arr)


        # ====== 打印指标 ======
        print("MSE Loss", test_loss)
        print("PCC:", pcc)
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2:", r2)

        
        return test_loss, y_pred, y_label

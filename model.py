import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from layers import *
from utils import *
import pickle


class AIST(nn.Module):
    def __init__(self, ncfeat, nxfeat, gout, gatt, rhid, ratt, rlayer, rclass, bs, rts, city, tr, tc):
        super(AIST, self).__init__()
        self.ncfeat = ncfeat
        self.nxfeat = nxfeat
        self.gout = gout
        self.gatt = gatt
        self.rhid = rhid
        self.ratt = ratt
        self.rlayer = rlayer
        self.rclass = rclass
        self.bs = bs
        self.rts = rts
        self.tr = tr
        self.city = city
        self.tc = tc

        self.smod = Spatial_Module(self.ncfeat, self.nxfeat, self.gout, self.gatt, 0.5, 0.6, self.rts, self.bs,
                        self.tr, self.tc, self.city)
        self.sab1 = self_LSTM_sparse_attn_predict(2 * self.gout, self.rhid, self.rlayer, self.rclass,
                    truncate_length=5, top_k=4, attn_every_k=5, predict_m=10)
        self.sab2 = self_LSTM_sparse_attn_predict(1, self.rhid, self.rlayer, self.rclass,
                    truncate_length=5, top_k=4, attn_every_k=5, predict_m=10)
        self.sab3 = self_LSTM_sparse_attn_predict(1, self.rhid, self.rlayer, self.rclass,
                    truncate_length=1, top_k=3, attn_every_k=1, predict_m=10)

        self.fc1 = nn.Linear(self.rhid, 1)
        self.fc2 = nn.Linear(2 * self.rhid, self.rclass)
        self.fc3 = nn.Linear(3 * self.rhid, self.rclass)

        self.wv = nn.Linear(self.rhid, self.ratt)  # (S, E) x (E, 1) = (S, 1)
        self.wu = nn.Parameter(torch.zeros(size=(self.bs, self.ratt)))  # attention of the trends
        nn.init.xavier_uniform_(self.wu.data, gain=1.414)
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x_crime, x_crime_daily, x_crime_weekly, x_regions, x_sp_crime, x_ext, s_crime):
        x_crime = self.smod(x_sp_crime, x_regions, x_ext, s_crime)

        x_con, x_con_attn = self.sab1(x_crime)  # (bs, rts)
        x_con = self.dropout_layer(x_con)
        x_con = x_con.unsqueeze(1)

        x_daily, x_daily_attn = self.sab2(x_crime_daily)  # x_daily = (bs, dts=20)
        x_daily = self.dropout_layer(x_daily)
        x_daily = x_daily.unsqueeze(1)

        x_weekly, x_weekly_attn = self.sab3(x_crime_weekly)  # x_weekly = (bs, wts=3):
        x_weekly = self.dropout_layer(x_weekly)
        x_weekly = x_weekly.unsqueeze(1)

        x = torch.cat((x_con, x_daily, x_weekly), 1)

        um = torch.tanh(self.wv(x))  # (bs, 3, ratt)
        um = um.transpose(2, 1)  # (bs, ratt, 3)
        wu = self.wu.unsqueeze(1)
        alpha_m = torch.bmm(wu, um)  # (bs, 1, 3)
        alpha_m = alpha_m.squeeze(1)  # (bs, 3)
        alpha_m = torch.softmax(alpha_m, dim=1)
        attn_trend = alpha_m.detach()
        alpha_m = alpha_m.unsqueeze(1)

        x = torch.bmm(alpha_m, x)
        x = x.squeeze(1)
        x = torch.tanh(self.fc1(x))

        return x, attn_trend


class Spatial_Module(nn.Module):
    def __init__(self, ncfeat, nxfeat, nofeat, gatt, dropout, alpha, ts, bs, tr, tc, city):

        super(Spatial_Module, self).__init__()
        self.ncfeat = ncfeat
        self.nxfeat = nxfeat
        self.nofeat = nofeat
        self.att = gatt
        self.bs = bs
        self.ts = ts
        self.tr = tr
        self.tc = tc
        self.city = city
        self.gat = [GraphAttentionLayer(self.ncfeat, self.nxfeat, self.nofeat, self.att, self.bs, dropout=dropout,
                    alpha=alpha) for _ in range(self.ts)]
        for i, g in enumerate(self.gat):
            self.add_module('gat{}'.format(i), g)

    def forward(self, x_crime, x_regions, x_ext, s_crime):
        T = x_crime.shape[1]
        tem_x_regions = x_regions.copy()
        reg = gen_neighbor_index_zero_with_target(self.tr, self.city)
        label = torch.tensor(reg)
        label = label.repeat(T * self.bs, 1)  # (T*bs, N)
        label = label.view(label.shape[0] * label.shape[1], 1).long()  # (T * bs * N, 1)
        x_crime = x_crime.transpose(1, 0)  # (T, bs)
        tem_x_regions.append(x_crime)

        N = len(tem_x_regions)  # Num of actual nodes
        feat = torch.stack(tem_x_regions, 2)  # (T, bs, N)
        feat = feat.view(feat.shape[0] * feat.shape[1] * feat.shape[2], 1).long()  # (T*bs*N, 1)
        feat = torch.cat([label, feat], dim=1)  # (T*bs*N, 2) --> (Node Label, features)
        feat = feat.view(T, self.bs * N, 2)

        feat_ext = torch.stack(x_ext, 2)
        feat_ext = feat_ext.view(feat_ext.shape[0] * feat_ext.shape[1] * feat_ext.shape[2], -1).long()  # (T*bs*N, nxfeat)
        feat_ext = torch.cat([label, feat_ext], dim=1)  # (T*bs*N, 2)
        feat_ext = feat_ext.view(T, self.bs * N, self.nxfeat + 1)

        crime_side = torch.stack(s_crime, 2)
        crime_side = crime_side.view(crime_side.shape[0] * crime_side.shape[1] * crime_side.shape[2], -1).long()  # (T*bs*N, 1)
        crime_side = torch.cat([label, crime_side], dim=1)  # (T*bs*N, 2)
        crime_side = crime_side.view(T, self.bs * N, 2)  # (T, bs*N, 2)

        spatial_output = []
        j = 0
        for i in range(T-self.ts, T):
            np.savetxt("gat_feat.txt", feat[i], fmt='%d')
            np.savetxt("gat_feat_ext.txt", feat_ext[i], fmt='%d')
            np.savetxt("gat_crime_side.txt", crime_side[i], fmt='%d')
            adj, features, features_ext, crime_side_features = load_data_GAT()

            out, ext = self.gat[j](features, adj, features_ext, crime_side_features)  # (N, F')(N, N, dv)
            out = out[:, -1, :]
            ext = ext[:, -1, :]
            out = torch.stack((out, ext), dim=2)
            spatial_output.append(out)
            j = j + 1

        spatial_output = torch.stack(spatial_output, 1)
        return spatial_output

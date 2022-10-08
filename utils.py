import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(threshold=np.inf)


def create_inout_sequences(input_data, tw=120):
    forecast = 1  # Num of ts to forecast in the future

    # recent_temporal_data_generation
    in_seq1 = torch.from_numpy(np.ones((8000, tw), dtype=np.int))
    out_seq1 = torch.from_numpy(np.ones((8000, forecast), dtype=np.int))
    L = input_data.shape[0]
    for i in range(L - tw - forecast):
        train_seq = input_data[i:i + tw, :]
        in_seq1[i] = train_seq.view(train_seq.shape[0] * train_seq.shape[1])
        train_label = input_data[i + tw:i + tw + forecast, :]
        out_seq1[i] = train_label.view(train_label.shape[0] * train_label.shape[1])
    in_seq1 = in_seq1[:i + 1, :]
    out_seq1 = out_seq1[:i + 1, :]

    # daily_temporal_data_generation
    batch_size = in_seq1.shape[0]
    time_step_daily = int(tw / 6)
    in_seq2 = torch.from_numpy(np.ones((batch_size, time_step_daily), dtype=np.int))
    out_seq2 = out_seq1
    for i in range(batch_size):
        k = 0
        for j in range(tw):
            if j % 6 == 0:
                in_seq2[i][k] = in_seq1[i][j]
                k = k + 1

    # weekly_temporal_data_generation
    time_step_weekly = int(tw / (6 * 7)) + 1
    in_seq3 = torch.from_numpy(np.ones((batch_size, time_step_weekly), dtype=np.int))
    out_seq3 = out_seq1
    for i in range(batch_size):
        k = 0
        for j in range(tw):
            if j % (6 * 7) == 0:
                in_seq3[i][k] = in_seq1[i][j]
                k = k + 1
    return in_seq1, out_seq1, in_seq2, in_seq3


def load_data_GAT(bs):
    # build features
    idx_features_labels = np.genfromtxt("gat_feat.txt", dtype=np.dtype(str))  # (Nodes, NodeLabel+ features + label)
    features = sp.csr_matrix(idx_features_labels[:, 1:], dtype=np.float32)  # (Nodes, features)
    # build features_ext
    idx_features_labels_ext = np.genfromtxt("gat_feat_ext.txt",
                                            dtype=np.dtype(str))  # (Nodes, NodeLabel+ features + label)
    features_ext = sp.csr_matrix(idx_features_labels_ext[:, 1:], dtype=np.float32)  # (Nodes, features)
    # build features
    idx_crime_side_features_labels = np.genfromtxt("gat_crime_side.txt",
                                                   dtype=np.dtype(str))  # (Nodes, NodeLabel+ features + label)
    crime_side_features = sp.csr_matrix(idx_crime_side_features_labels[:, 1:], dtype=np.float32)  # (Nodes, features)

    # build graph
    num_reg = int(idx_features_labels.shape[0] / bs)
    idx = np.array(idx_features_labels[:num_reg, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("tem_gat_adj.txt", dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(num_reg, num_reg),
                        dtype=np.float32)  # replaced 5
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    features_ext = torch.FloatTensor(np.array(features_ext.todense()))
    crime_side_features = torch.FloatTensor(np.array(crime_side_features.todense()))

    return adj, features, features_ext, crime_side_features


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data_regions(bs, target_crime_cat, target_region, target_city, tw=120):
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    com = gen_neighbor_index_zero(target_region, target_city)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in com:
        loaded_data = torch.from_numpy(np.loadtxt("data/" + target_city + "/com_crime/r_" + str(i) + ".txt", dtype=np.float)).T
        loaded_data = loaded_data[:, target_crime_cat:target_crime_cat + 1]
        x, y, z, m = create_inout_sequences(loaded_data, tw)

        x = torch.from_numpy(scaler.fit_transform(x))
        z = torch.from_numpy(scaler.fit_transform(z))
        m = torch.from_numpy(scaler.fit_transform(m))
        y = torch.from_numpy(scaler.fit_transform(y))
        # Divide into train_test data
        train_x_size = int(x.shape[0] * .67)
        train_x = x[: train_x_size, :]  # (bs, tw) = (1386, 120)
        train_y = y[: train_x_size, :]  # (bs, 1) = (1386, 1)
        test_x = x[train_x_size:, :]  # (bs, tw) = (683, 120)
        test_x = test_x[:test_x.shape[0] - 11, :]  # (bs, tw): sub 11 to make it consistent with bs
        test_y = y[train_x_size:, :]  # (bs, 1) = (683, 1)
        test_y = test_y[:test_y.shape[0] - 11, :]

        train_x = train_x.view(int(train_x.shape[0] / bs), bs, tw)
        test_x = test_x.view(int(test_x.shape[0] / bs), bs, tw)

        train_x = train_x.transpose(2, 1)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)

    return batch_add_train, batch_add_test


def load_data_regions_external(bs, nxfeatures, target_region, target_city, tw=120):
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    com = gen_neighbor_index_one_with_target(target_region, target_city)
    poi_data = torch.from_numpy(np.loadtxt("data/" + target_city + "/poi.txt", dtype=np.int))

    for i in com:
        loaded_data = torch.from_numpy(np.loadtxt("data/" + target_city + "/act_ext/taxi" + str(i) + ".txt", dtype=np.int)).T
        loaded_data1 = loaded_data[:, 0:1]
        loaded_data2 = loaded_data[:, 1:2]
        x_in, y_in, z_in, m_in = create_inout_sequences(loaded_data1)
        x_out, y_out, z_out, m_out = create_inout_sequences(loaded_data2)

        x_in = x_in.unsqueeze(2).double()
        x_out = x_out.unsqueeze(2).double()
        poi = poi_data[i - 1].double()
        poi = poi.repeat(x_in.shape[0], tw, 1)

        x = torch.cat([x_in, x_out, poi], dim=2)

        # Divide into train_test data
        train_x_size = int(x.shape[0] * .67)
        train_x = x[: train_x_size, :, :]  # (bs, tw) = (1386, 120)
        test_x = x[train_x_size:, :, :]  # (bs, tw) = (683, 120)
        test_x = test_x[:test_x.shape[0] - 11, :, :]

        train_x = train_x.view(int(train_x.shape[0] / bs), bs, tw, nxfeatures)
        test_x = test_x.view(int(test_x.shape[0] / bs), bs, tw, nxfeatures)

        train_x = train_x.transpose(2, 1)  # (num_regions, tw, bs, nxfeatures)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)

    return batch_add_train, batch_add_test


def load_data_sides_crime(bs, target_crime_cat, target_region, target_city, tw=120):
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions

    com = gen_neighbor_index_zero_with_target(target_region, target_city)
    side = gen_com_side_adj_matrix(com, target_city)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    for i in range(len(com)):
        loaded_data = torch.from_numpy(np.loadtxt("data/" + target_city + "/side_crime/s_" + str(side[i]) + ".txt", dtype=np.int)).T
        loaded_data = loaded_data[:, target_crime_cat:target_crime_cat + 1]
        tensor_ones = torch.from_numpy(np.ones((loaded_data.size(0), loaded_data.size(1)), dtype=np.int))
        loaded_data = torch.where(loaded_data > 1, tensor_ones, loaded_data)
        x, y, z, m = create_inout_sequences(loaded_data)

        x = torch.from_numpy(scaler.fit_transform(x))
        z = torch.from_numpy(scaler.fit_transform(z))
        m = torch.from_numpy(scaler.fit_transform(m))
        y = torch.from_numpy(scaler.fit_transform(y))

        # Divide into train_test data
        train_x_size = int(x.shape[0] * .67)
        train_x = x[: train_x_size, :]
        train_y = y[: train_x_size, :]
        test_x = x[train_x_size:, :]
        test_x = test_x[:test_x.shape[0] - 11, :]
        test_y = y[train_x_size:, :]
        test_y = test_y[:test_y.shape[0] - 11, :]

        train_x = train_x.view(int(train_x.shape[0] / bs), bs, tw)
        test_x = test_x.view(int(test_x.shape[0] / bs), bs, tw)

        train_x = train_x.transpose(2, 1)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)
    return batch_add_train, batch_add_test


def gen_com_adj_matrix(target_region):
    adj_matrix = np.zeros((77, 77), dtype=np.int)
    edges_unordered = np.genfromtxt("data/com_adjacency.txt", dtype=np.int32)
    for i in range(edges_unordered.shape[0]):
        src = edges_unordered[i][0] - 1
        dst = edges_unordered[i][1] - 1
        adj_matrix[src][dst] = 1
        adj_matrix[src][dst] = 1
    np.savetxt("data/com_adj_matrix.txt", adj_matrix, fmt="%d")
    return


def gen_com_side_adj_matrix(regions, target_city):
    idx = np.loadtxt("data/" + target_city + "/side_com_adj.txt", dtype=np.int)
    idx_map = {j: i for i, j in iter(idx)}
    side = [idx_map.get(x + 1) % 101 for x in regions]  # As it starts with 0
    return side


def gen_neighbor_index_zero(target_region, target_city):
    adj_matrix = np.loadtxt("data/" + target_city + "/com_adj_matrix.txt")
    adj_matrix = adj_matrix[target_region]
    neighbors = []
    for i in range(adj_matrix.shape[0]):
        if adj_matrix[i] == 1:
            neighbors.append(i)
    return neighbors


def gen_neighbor_index_zero_with_target(target_region, target_city):
    neighbors = gen_neighbor_index_zero(target_region, target_city)
    neighbors.append(target_region)
    return neighbors


def gen_neighbor_index_one_with_target(target_region, target_city):
    neighbors = gen_neighbor_index_zero(target_region, target_city)
    neighbors.append(target_region)
    neighbors = [x + 1 for x in neighbors]
    return neighbors


def gen_gat_adj_file(target_city, target_region):
    neighbors = gen_neighbor_index_zero(target_region, target_city)
    adj_target = torch.zeros(len(neighbors), 2)
    for i in range(len(neighbors)):
        adj_target[i][0] = target_region
        adj_target[i][1] = neighbors[i]
    np.savetxt("tem_gat_adj.txt", adj_target, fmt="%d")
    return


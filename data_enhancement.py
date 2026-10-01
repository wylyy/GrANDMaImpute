
import numpy as np
import torch
from torch_geometric.data import Data



from scipy.sparse import coo_matrix
import dgl


dgl.random.seed(42)
def preprocess(graph):
    feat = graph.ndata["feat"]
    edge_index=graph.edge


    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat
    graph.edge= edge_index.to(torch.int64)

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()

    return graph
def knn_graph(data_z):

    data = data_z.copy(deep=True)

    pos = np.array(data)
    data1 = Data(pos=torch.tensor(pos, dtype=torch.float),
                 x = torch.tensor(pos, dtype=torch.float)

                 )

    graph_transform = GT.KNNGraph(k=3)
    gt_graph = graph_transform(data1)
    num_features = len(data.columns.tolist())

    node_num, _ = gt_graph.x.size()
    _, edge_num = gt_graph.edge_index.size()

    edge_index = gt_graph.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1


    qq = coo_matrix(adj)
    g = dgl.from_scipy(qq)

    feat = torch.from_numpy(np.array(data))

    g.ndata["feat"] = feat

    g.edge=torch.from_numpy(edge_index)

    g1=preprocess(g)

    return g1,num_features

def drop_nodes(data, aug_ratio):

    data2 = data.copy(deep=True)
    pos = np.array(data2)
    data1 = Data(pos=torch.tensor(pos, dtype=torch.float),
                 x=torch.tensor(pos, dtype=torch.float)
                 )

    graph_transform = GT.KNNGraph(k=3)
    gt_graph = graph_transform(data1)

    num_features = len(data.columns.tolist())



    node_num, _ = gt_graph.x.size()
    _, edge_num = gt_graph.edge_index.size()

    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()



    edge_index = gt_graph.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index_z = adj.nonzero().t()

    qq = coo_matrix(adj)
    g = dgl.from_scipy(qq)

    feat = torch.from_numpy(np.array(data))


    g.ndata["feat"] = feat[idx_nondrop]

    g.edge = edge_index_z

    g1 = preprocess(g)
    return g1, num_features

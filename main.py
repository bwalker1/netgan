from netgan import netgan
from netgan import utils
import networkx as nx
import scipy.sparse as sp
import numpy as np




# try out a simple netgan
if __name__=="__main__":
    # some parameters
    N = 100
    rw_len = 50
    
    # create an adjacency matrix
    G = nx.relaxed_caveman_graph(10,10,0.1)
    G.remove_edges_from(G.selfloop_edges())
    adj = nx.adjacency_matrix(G)    
    
    # split up edges apparently (would rather just split up graphs)
    val_share = 0.1
    test_share = 0.05
    train_ones, val_ones, val_zeros, test_ones, test_zeros = utils.train_val_test_split_adjacency(adj, val_share, test_share, undirected=True, connected=True, asserts=True)
    train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1]))).tocsr()

    
    # create the random walk generator
    walker = utils.RandomWalker(adj,rw_len)
    walk_generator = walker.walk
    
    ng = netgan.NetGAN(N,rw_len,walk_generator, gpu_id=None)
    
    stopping_criterion = "val"

    assert stopping_criterion in ["val", "eo"], "Please set the desired stopping criterion."

    if stopping_criterion == "val": # use val criterion for early stopping
        stopping = None
    elif stopping_criterion == "eo":  #use eo criterion for early stopping
        stopping = 0.5 # set the target edge overlap here
        
    log_dict = ng.train(A_orig=adj, val_ones=val_ones, val_zeros=val_zeros, stopping=stopping,max_iters=5)
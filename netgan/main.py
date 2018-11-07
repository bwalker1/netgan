from netgan import netgan
from utils import utils
import networkx as nx






# try out a simple netgan
if __name__=="__main__":
    # some parameters
    N = 100
    rw_len = 50
    
    # create an adjacency matrix
    G = nx.relaxed_caveman_graph(10,10,0.1)
    adj = nx.adjacency_matrix(G)    
    
    # create the random walk generator
    walker = RandomWalker(adj,rw_len)
    walk_generator = walker.walk
    
    ng = NetGAN(N,rw_len,walk_generator)
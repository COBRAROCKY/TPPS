# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
from average import average_weights
from MAB import MAB
from TPPS import TPPS

from dp.DP import DP
import random

class Edge():

    def __init__(self, id, cids, shared_layers, mab, args, learning_rate, edge_clients):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :param mab: MAB Object (the MAB Algorithm)
        :return:
        """
        self.id = id
        self.cids = cids
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.shared_state_dict = shared_layers.state_dict()
        self.clock = []
        self.edge_clients = edge_clients

        # COCS policy 对应算法 CC-MAB
        self.CUCB = object
        self.MAB = mab
        if args.using_dp and (args.alg == 'TPPS' or args.alg == 'TPPS2'):
            self.TPPS = TPPS(rate=args.TPPS_rate, total_rounds=args.num_edge_aggregation*args.num_communication, total_clients=self.edge_clients)
        
        self.using_dp = args.using_dp
        self.learning_rate = learning_rate
        if self.using_dp:
            self.DP = DP(   dp_mechanism=args.dp_mechanism,
                            times=args.num_local_update * args.num_edge_aggregation,
                            dp_epsilon=random.randint(2,10),
                            dp_delta=args.dp_delta,
                            min_idx_sample=args.batch_size,
                            learning_rate=self.learning_rate,
                            dp_clip=args.dp_clip,
                            using_dp=args.using_dp
                         )

    def refresh_edgeserver(self):
        # self.`receiver_buffer`.clear()
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        return None

    def receive_from_client(self, client_id, cshared_state_dict):
        self.receiver_buffer[client_id] = cshared_state_dict
        return None

    def aggregate(self, args):
        """ 
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w = received_dict,
                                                 s_num= sample_num)

    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_edge(edge_id=self.id,
                                eshared_state_dict= copy.deepcopy(
                                    self.shared_state_dict))
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        return None
    


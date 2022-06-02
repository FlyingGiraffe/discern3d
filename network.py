import xmlrpc
import random
import threading
import time
random.seed(1231920321)

class Router(object):
    def __init__(self, agent_ips, intragroup_link_failure_prob=0.0,
                 max_partitions=1, refresh_hz=1):
        self.agent_ips = agent_ips
        self.agentip2idx = {el: i for i, el in enumerate(agent_ips)}
        
        self.intragroup_link_failure_prob = intragroup_link_failure_prob
        self.max_partitions = max_partitions

        # preliminary partition
        self.group_number = [random.randint(0, self.max_partitions-1) for _ in range(len(self.agent_ips))]

        randomize_partition_kwargs = {'refresh_hz': refresh_hz}
        thread = threading.Thread(target=self.randomize_partitions, kwargs=randomize_partition_kwargs)
        thread.start()

    def randomize_partitions(self, refresh_hz):
        while True:
            print('repartitioning the network...')
            self.group_number = [random.randint(0, self.max_partitions-1) for _ in range(len(self.agent_ips))]
            time.sleep(1/refresh_hz)

    def is_connected(self, from_ip, to_ip):
        """Returns true if from_ip and to_ip can communicate with eachother"""
        from_group = self.group_number[self.agentip2idx[from_ip]]
        to_group = self.group_number[self.agentip2idx[to_ip]]
        if from_group != to_group:
            return False # replace with more sophisticated partitioning
        return random.uniform(0,1) > self.intragroup_link_failure_prob
        

    def attempt_ServerProxy(self, from_ip, to_ip):
        # if from_ip CAN talk to to_ip right now, then 
        # allow them to message pass.
        
        if self.is_connected(from_ip, to_ip):
            print('connection permitted between {} and {}'.format(from_ip, to_ip))
            s = xmlrpc.client.ServerProxy('http://{0}:{1}'.format(to_ip[0], to_ip[1]))
            return s

        print('connection broken between {} and {}'.format(from_ip, to_ip))
        raise ConnectionRefusedError('No connection permitted by Router.')

    

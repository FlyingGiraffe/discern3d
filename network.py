import xmlrpc

class Router(object):
    def __init__(self,):
        pass

    def is_connected(self, from_ip, to_ip):
        return True # replace with more sophisticated partitioning

    def attempt_ServerProxy(self, from_ip, to_ip):
        # if from_ip CAN talk to to_ip right now, then 
        # allow them to message pass.
        
        if self.is_connected(from_ip, to_ip):
            # print('connection permitted between {} and {}'.format(from_ip, to_ip))
            s = xmlrpc.client.ServerProxy('http://{0}:{1}'.format(to_ip[0], to_ip[1]))
            return s
        
        raise ConnectionRefusedError('No connection permitted by Router.')

    

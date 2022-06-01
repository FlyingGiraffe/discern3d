import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from datetime import datetime
import time
from vis import vis_voxel_grid
import queue
import threading

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import xmlrpc


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)



class Agent(object):
    def __init__(self, agent_idx, kwargs):
        self.grid_coarse = kwargs['grid_coarse']
        self.grid_fine = kwargs['grid_fine']

        self.num_points_view = kwargs['num_points_view']
        self.num_points_scan = kwargs['num_points_scan']

        # agent args
        self.T = kwargs['T']
        self.K = kwargs['K']
        self.agent_ips = kwargs['agent_ips']
        self.my_idx = agent_idx
        self.my_ip = self.agent_ips[agent_idx]

        # final course and fine representations
        self.repn_coarse = np.zeros((kwargs['grid_coarse'], kwargs['grid_coarse'], kwargs['grid_coarse']), dtype=bool)
        self.repn_fine_idx = -np.ones((kwargs['grid_coarse'], kwargs['grid_coarse'], kwargs['grid_coarse']), dtype=int)
        self.repn_fine = []

        # rep for building a dynamic priority list
        self.gossip_queue = queue.Queue()
        self.priority_list_timesteps = [[[[float('inf') for a in range(len(self.agent_ips))] for i in range(self.grid_coarse)] for j in range(self.grid_coarse)] for k in range(self.grid_coarse)]
        self.priority_list_locks = [[[threading.Condition() for i in range(self.grid_coarse)] for j in range(self.grid_coarse)] for k in range(self.grid_coarse)]

        # Queue and temporary state for handling fine representation
        self.received_packets = queue.Queue()
        self.temp_repn_fine_idx = -np.ones((kwargs['grid_coarse'], kwargs['grid_coarse'], kwargs['grid_coarse']), dtype=int)
        self.temp_repn_fine = []
        # TODO add list of locks per vox OR switch temp_repn_fine_idx to list implementation.

        # Thread management
        self.threads_lock = threading.RLock()
        self.threads = []


    def run(self, shape_data, view, view_transition, 
            scan_hz=1.0, gossip_hz=1.0, clean_hz=1.0, lowres_hz=1.0, 
            update_retry_hz=1.0):
        # step 1: init listening server loop
        thread = threading.Thread(target=self.init_server, args=())
        thread.start()
        self.threads.append(thread)

        # step 2: init gossiping loop
        gossip_kwargs = {'finished_callback' : (lambda :  False),
                         'freq': gossip_hz}
        thread = threading.Thread(target=self.gossip_loop, kwargs=gossip_kwargs) 
        thread.start()
        self.threads.append(thread)

        # step 3: init high-res update loop
        highres_update_kwargs = {'finished_callback': (lambda: False),
                                'update_retry_hz': update_retry_hz}        
        thread = threading.Thread(target=self.update_loop, kwargs=highres_update_kwargs) 
        thread.start()
        self.threads.append(thread)

        # step 4: initialize low-res get loop
        lowres_update_kwargs = {'finished_callback': (lambda: False),
                                'freq': lowres_hz}        
        thread = threading.Thread(target=self.get_loop, kwargs=lowres_update_kwargs) 
        thread.start()
        self.threads.append(thread)

        # step 5: init scan loop

        scan_kwargs={'shape_data': shape_data,
            'view': view,
            'view_transition': view_transition, 
            'finished_callback': (lambda : False), 
            'scan_hz': 20, 
            'vis': False}
        thread = threading.Thread(target=self.scan_loop, kwargs=scan_kwargs) 
        thread.start()
        self.threads.append(thread)


        # CLEANUP 
        thread = threading.Thread(target=self.cleanup, kwargs={'clean_hz': clean_hz}) 
        thread.start()
        self.threads.append(thread)

        
    def cleanup (self, clean_hz = 1.0):
        while True: # cleanup
            for thread in self.threads:
                if not thread.isAlive():
                    thread.join()
            time.sleep(1/clean_hz) 

    # =================================================================
    # ====================== BUILD RPC PROTOCOL ======================
    # =================================================================
    def init_server(self):
        with SimpleXMLRPCServer((self.my_ip[0], self.my_ip[1]), requestHandler=RequestHandler) as server:
            server.register_introspection_functions()
            server.register_function(self.get_rpc, 'get')
            print('Registered GET RPC')
            server.register_function(self.get_all_rpc, 'get_all')
            print('Registered GET_ALL RPC')
            server.register_function(self.get_priority_list_rpc, 'get_priority_list')
            print('Registered GET_PRIORITY_LIST RPC')
            server.register_function(self.update_priority_list_rpc, 'update_priority_list')
            print('Registered UPDATE_PRIORITY_LIST RPC')
            server.register_function(self.update_rpc, 'update')
            print('Registered UPDATE RPC')
            server.serve_forever()

    def get_rpc(self, course_idx):
        return self.repn_coarse[course_idx[0], course_idx[1], course_idx[2]]

    def get_all_rpc(self):
        return self.repn_coarse.tolist()

    def update_rpc(self, course_idx, packet_id, fine_scan):
        timestep = time.time()
        self.received_packets.put((course_idx, packet_id, np.array(fine_scan), timestep))
        return "success"

    def get_priority_list_rpc(self, coarse_idx):
        return self.priority_list_timesteps[coarse_idx[0]][coarse_idx[1]][coarse_idx[2]]

    def update_priority_list_rpc(self, coarse_idx, update):
        self.priority_list_timesteps[coarse_idx[0]][coarse_idx[1]][coarse_idx[2]] = update
        return "success"

    # =================================================================
    # ================== COURSE REPRESENTATION GET ====================
    # =================================================================
    def get_loop(self, finished_callback, freq=1.0):
        while not finished_callback():
            for i, agent_ip in enumerate(self.agent_ips):
                if i == self.my_idx:
                    continue
                try:
                    s = xmlrpc.client.ServerProxy('http://{0}:{1}'.format(agent_ip[0], agent_ip[1]))
                    coarse_grid = s.get_all()
                except ConnectionRefusedError as e:
                    continue
                
                self.repn_coarse |= np.array(coarse_grid)


            print('Visualization')
            vis_voxel_grid(self.repn_coarse) 

            time.sleep(1/freq)

    # =================================================================
    # ================== DYNAMIC PRIORITY LIST GOSSIP =================
    # =================================================================
    def gossip_loop(self, finished_callback, freq):
        """
        Iteratively communicates with all other agents. Adds any timestamp-observation updates to our priority queue.
        :param finished_callback:
        :param freq:
        :return:
        """

        while not finished_callback():
            observed_voxel_ids = grid2idx(self.repn_coarse)
            for voxel_id in observed_voxel_ids:
                for ip_address in self.agent_ips:
                    
                    try:
                        s = xmlrpc.client.ServerProxy('http://{0}:{1}'.format(ip_address[0], ip_address[1]))
                        their_priority_list_tstamps = s.get_priority_list(voxel_id.tolist())
                    except ConnectionRefusedError as e:
                        continue
                    
                    our_priority_list_tstamps = self.priority_list_timesteps[voxel_id[0]][voxel_id[1]][voxel_id[2]]
                    for i, tstamp in enumerate(their_priority_list_tstamps):
                        if tstamp < our_priority_list_tstamps[i]:
                            self.gossip_queue.put((voxel_id, tstamp, self.agent_ips[i]))
            time.sleep(1/freq)

    def update_our_priority_list(self, finished_callback):
        # get priority list of this agent and us
        while not finished_callback():
            priority_queue_update = self.gossip_queue.get(block=True)
            voxel_id, new_tstep, agent_id = priority_queue_update
            agent_idx = agent_id  # todo: fix me!

            # get current and updated set of agent timesteps
            self.priority_list_timesteps[voxel_id[0]][voxel_id[1]][voxel_id[2]][agent_idx] = new_tstep

            # check if we "lost" / not in top-k for this voxel-priority list
            agent_timestamps = self.priority_list_timesteps[voxel_id[0]][voxel_id[1]][voxel_id[2]] 
            fightingchance  = sum([el < agent_timestamps[self.my_idx] for el in agent_timestamps]) < self.K
            if not fightingchance and np.isinf(agent_timestamps[self.my_idx]):
                self.gossip_queue.put((voxel_id, time.time(), self.agent_ips[self.my_idx]))

            # ------ check if this affects our top-k agents
            # check 1: we added a new agent to the initially lacking priority list
            voxel_lock = self.priority_list_locks[voxel_id[0], voxel_id[1], voxel_id[2]]
            voxel_lock.notifyAll()

            # active_agents_now = np.sum(np.isfinite(new_priority_timesteps))
            # active_agents_before = np.sum(np.isfinite(old_priority_timesteps))
            # added_agent = active_agents_now <= self.K and active_agents_now > active_agents_before
            # if added_agent:
            #     voxel_lock = self.priority_list_locks[voxel_id[0], voxel_id[1], voxel_id[2]]
            #     voxel_lock.notifyAll()
            #
            # # check 2: we changed the order of the top-K agents in the priority list
            # topk_before = np.argsort(old_priority_timesteps)[:self.K]
            # topk_now = np.argsort(new_priority_timesteps)[:self.K]
            # if not np.all(topk_now == topk_before):
            #     voxel_lock = self.priority_list_locks[voxel_id[0], voxel_id[1], voxel_id[2]]
            #     voxel_lock.notifyAll()


    # =================================================================
    # ================== FINE REPRESENTATION UPDATE ===================
    # =================================================================
    def update_loop(self, finished_callback, update_retry_hz):
        """
        Listens to shared queue and spawns appropriate `update` calls
        :return:
        """
        while not finished_callback():
            received_packet = self.received_packets.get(block=True)
            thread = threading.Thread(target=self.update_packet, args=(received_packet, update_retry_hz))
            thread.start()

            self.threads_lock.acquire()
            self.threads.append(thread)
            self.threads_lock.release()

    def update_packet(self, data, update_retry_hz):
        """
        Given a data packet, adds it into "temporary" memory and transmits it to the appropriate agents.
        :param data: Tuple containing course_idx, packet_id, and voxel_data
        :return:
        """
        coarse_idx, packet_id, voxel_data, timestamp = data

        # we just received a new voxel/timestep --> add it to our gossip-queue to notify of priority-list updates
        cur_timestep = self.priority_list_timesteps[coarse_idx[0]][coarse_idx[1]][coarse_idx[2]][self.my_idx] 
        if timestamp < cur_timestep:
            self.gossip_queue.put((coarse_idx, timestamp, self.my_ip))

        # update temporary global fine representation
        fine_idx = self.temp_repn_fine_idx[coarse_idx[0], coarse_idx[1], coarse_idx[2]]
        if fine_idx == -1:
            self.temp_repn_fine_idx[coarse_idx[0], coarse_idx[1], coarse_idx[2]] = len(self.temp_repn_fine)  # todo: make thread safe!!
            self.temp_repn_fine.append({packet_id: voxel_data})
        else:
            self.temp_repn_fine[fine_idx][packet_id] = voxel_data

        # enter a loop that sends the data to appropriate agents
        thread = threading.Thread(target=self.update_packet_replication, args=(coarse_idx, packet_id, update_retry_hz))
        thread.start()
        self.threads.append(thread)

    def transmit_data(self, agent_ip, course_idx, packet_id):    
        try:
            s = xmlrpc.client.ServerProxy('http://{0}:{1}'.format(agent_ip[0], agent_ip[1]))
            fine_scan_idx = self.temp_repn_fine_idx[course_idx[0]][course_idx[1]][course_idx[2]]
            fine_scan = self.temp_repn_fine[fine_scan_idx][packet_id]
            res = s.update(course_idx, packet_id, fine_scan.tolist())
        except ConnectionRefusedError as e:  # TODO add error types here
            return False

        return res == 'success'


    def update_packet_replication(self, course_idx, packet_id, update_retry_hz):
        """
        Given the course-grid index and packet-id, this method appropriately transmits the voxel data-packet stored
        in self.temp_repn_fine to the appropriate agents. It will loop until it is confirmed that the data is sent
        to the K higher-priority agents, or until it can *confirm that it is in the top K agents.

        :param course_idx: tuple of (i,j,k) in the course grid
        :param packet_id: unique int identifier of this data packet
        :return: None
        """
        # iterate until our priority list convergesself.grid_coarse + 
        transmitted_history = {self.my_idx}
        
        agent_timestamps = self.priority_list_timesteps[course_idx[0]][course_idx[1]][course_idx[2]] 
        fightingchance = sum([el < agent_timestamps[self.my_idx] for el in agent_timestamps]) < self.K
        num_unknown = np.sum(np.isinf(agent_timestamps))
        our_cur_priority = np.where(np.argsort(agent_timestamps) == self.my_idx)[0]
        we_win = our_cur_priority < (self.K - num_unknown)  # we are in the top K*, and we know all except (K*-1) timsteps

        while fightingchance and not we_win:
            # np.any(np.isinf(self.priority_list_timesteps[course_idx[0]][course_idx[1]][course_idx[2]])): # TODO change this to be top K not infinity for each voxel
            # acquire coarse-grid lock
            this_lock = self.priority_list_locks[course_idx[0]][course_idx[1]][course_idx[2]]
            this_lock.acquire()

            # transmit to all agents in current priority list
            cur_priority_tsteps = self.priority_list_timesteps[course_idx[0]][course_idx[1]][course_idx[2]]
            cur_topk_agents = np.argsort(cur_priority_tsteps)[:self.K]
            for i in cur_topk_agents:
                agent_ip = self.agent_ips[i]
                if i in transmitted_history or np.isinf(cur_priority_tsteps[i]):
                    continue
                success = self.transmit_data(agent_ip, course_idx, packet_id)
                if success:
                    transmitted_history.add(i)

            # sleep until priority list is updated
            this_lock.wait()
            this_lock.release()

            # update while loop flags
            agent_timestamps = self.priority_list_timesteps_timesteps[course_idx[0]][course_idx[1]][course_idx[2]] 
            fightingchance = sum([el < agent_timestamps[self.my_idx] for el in agent_timestamps]) < self.K
            num_unknown = np.sum(np.isinf(agent_timestamps))
            our_cur_priority = np.where(np.argsort(agent_timestamps) == self.my_idx)[0]
            we_win = our_cur_priority < (self.K - num_unknown)  # we are in the top K*, and we know all except (K*-1) timsteps
   

        # once we break out of the loop, we know we have reached consensus on priority list
        k_winners = np.argsort(self.priority_list_timesteps[course_idx[0]][course_idx[1]][course_idx[2]])[:self.K]
        fully_replicated = set(k_winners).issubset(transmitted_history)
        while not fully_replicated:
            for agent_idx in k_winners:
                if agent_idx in transmitted_history:
                    continue
                success = self.transmit_data(self.agent_ips[agent_idx], course_idx, packet_id)
                if success:
                    transmitted_history.add(agent_idx)
            time.sleep(1/update_retry_hz)
            
            k_winners = sorted(self.priority_list_timesteps[course_idx[0]][course_idx[1]][course_idx[2]])[:self.K]
            fully_replicated = set(k_winners).issubset(transmitted_history)

        # if we are in the priority list, add the voxel to storage
        temp_fine_idx = self.temp_repn_fine_idx[course_idx[0], course_idx[1], course_idx[2]]
        voxel_data = self.temp_repn_fine[temp_fine_idx][packet_id]

        if self.my_ip in k_winners:
            fine_idx = self.repn_fine_idx[course_idx[0], course_idx[1], course_idx[2]]
            if fine_idx == -1:
                self.repn_fine_idx[course_idx[0], course_idx[1], course_idx[2]] = len(self.temp_repn_fine)  # todo: make thread safe!!
                self.repn_fine.append(voxel_data)
            else:
                self.temp_repn_fine[fine_idx] |= voxel_data

        # remove temporary storage of high-res voxel data
        del self.temp_repn_fine[temp_fine_idx][packet_id]

       #  vox_id = data['vox_id']
       #  vox_hash = datetime.now()  # idk, something that can be sorted by time.
       #
       #  if len(self.vox2agent[vox_id]) < self.K:
       #      # here, we have a decision to make. we can either pass this data to another agent,
       #      # or keep it for ourselves.
       #      a2v = self.get_agent2vox()
       #      cankeep, send_candidates = self.cankeep(self, vox_id, a2v)
       #
       #      # if we can get away with not sending it AND the data can fit in memory...
       #      if cankeep and len(self.data) < self.T:
       #          self.data[vox_id] = data['pointcloud']
       #          self.vox2agent[vox_id].append((self.id, vox_hash))
       #
       #          # now let's notify the other agents of this change...
       #          for agent_id in self.other_ids:
       #              other = self.id2agents[
       #                  agent_id]  # some function to return the actual agent object, or get access to some port...
       #              other.update("update_vox2agent", {'vox2agent': self.vox2agent})
       #
       #      # otherwise...
       #      else:
       #          # this means the memory of this agent is  full currently, so
       #          # let's pass this onto another agent. Let's pick a random agent
       #          # from send_candidates
       #          send_to = np.random.choice(send_candidates)
       #
       #          # then, do a quick sync with the other agent. Do they indeed have
       #          # enough space to store the data you're about to send them?
       #          # TODO. eugh.
       #
       #          # send them. TODO What should it do if it fails?
       #          self.id2agents[send_to].update(method, data)
       #
       #      for agent_id in self.other_ids:
       #          other = self.id2agents[
       #              agent_id]  # some function to return the actual agent object, or get access to some port...
       #          other.update("update_vox2agent", {'vox2agent': self.vox2agent})
       #
       # # reconcile the vox2agent from another agent to this...
       #  for key in self.vox2agent:
       #      other_vox2agent = data['vox2agent']
       #      mine = self.vox2agent[key]
       #      yours = other_vox2agent[key]
       #      union_agents = list(set(mine).union(set(yours)))
       #      # if the union <= length K, we don't have a problem... but...
       #      if len(union_agents) >= self.K:
       #          # we have to merge them, but in a way that will be consistent.
       #          merged = sorted(union_agents,
       #                          key=lambda x: x[1]) # we sort by the timestamp
       #          merged_topK = merged[:self.K]
       #          merged_remove = merged[self.K:]
       #
       #          # now of the ones that are going to be removed,
       #          # check if this agent belongs in there. Remove the
       #          # data if necessary.
       #          for el in merged_remove:
       #              if el[0] == self.id:
       #                  del self.data[key] # clear the data
       #
       #          self.vox2agent[key] = merged_topK
       #      else:  # we good. let' just merge it.
       #          self.vox2agent[key] = union_agents

    def get_agent2vox(self):
        agent2vox = {el: [] for el in self.id2agents}
        for vox_id, agent_list in self.vox2agent:
            for el in agent_list:
                agent_id = el[0]
                agent2vox[agent_id].append(vox_id)
        return agent2vox

    def cankeep(self, vox_id, a2v):
        """
        Args:
          vox_id: voxel id string.
          a2v: a mapping that maps each AID to a list of voxel id's.
        Returns:
          canKeep: True means that keeping the data onboard does not jeopardize
            crossing the assignment discrepancy threshold
          send_candidates: candidates that, if the agent chooses to send, are the most "beneficial" to send to
            ( the ones allocated to the least number of voxels. )
        """
        mustsend = False
        send_candidates = None

        candidate_agent_prior_commitment =  {}
        vox_current_holders = [el[0] for el in self.vox2agent[vox_id]]
        for el in [aid for aid in self.id2agents if aid not in vox_current_holders]:
            candidate_agent_prior_commitment[el] = len(a2v[vox_id])

        min_commitment = min(list(candidate_agent_prior_commitment.values()))
        send_candidates = [key for key, value in candidate_agent_prior_commitment.items()
                           if value == min_commitment and key != self.id]
        if self.id in candidate_agent_prior_commitment:
            current_agent_commitment = candidate_agent_prior_commitment[self.id]
            mustsend = (current_agent_commitment - min_commitment) >= self.allocation_discrepancy_threshold
        else:
            mustsend = True
        return (not mustsend), send_candidates

    # =================================================================
    # ==================== SCANNING FUNCTIONALITY =====================
    # =================================================================
    def scan_loop(self, shape_data, view, view_transition, finished_callback, scan_hz=20, vis=False):
        jj = 0
        while not finished_callback():
            # scan data
            course_scan, course_idxs, fine_scans = self.scan(shape_data, view)
            course_idxs = course_idxs.tolist()

            # update global coarse representation
            self.repn_coarse = self.repn_coarse | np.array(course_scan, dtype=np.bool)
            
            # add fine-scan packets to queue
            for i, course_idx in enumerate(course_idxs):
                packet_id = (time.time_ns() + hash(str(course_idx)))%(2**16)
                self.received_packets.put((course_idx, packet_id, fine_scans[i], time.time()))

            # update view
            view = view_transition(view)
            time.sleep(1/scan_hz)
            jj += 1
  
    def scan(self, shape_data, view):
        '''
        Args:
          shape_data: full object pointcloud, size (num_points, 3)
          view: a view point on the unit sphere, size (1, 3)
        Output:
          repn_coarse_new: new coarse voxel grids, size (grid_coarse, grid_coarse, grid_coarse)
          repn_fine_idx_tmp: temporary indices for new fine voxel grids, size (grid_coarse, grid_coarse, grid_coarse)
          repn_fine_new: new fine voxel grids, list [elements of size (grid_fine, grid_fine, grid_fine)]
        '''
        points_view = farthest_subsample_points(shape_data, view, self.num_points_view)
        scan_idx = np.random.choice(self.num_points_view, self.num_points_scan)
        points_scan = points_view[scan_idx]

        # create binary voxel grid at course resolution
        course_scan = np.zeros(self.repn_coarse.shape)
        perpoint_course_idxs = np.floor((points_scan + 1) * self.grid_coarse / 2).astype(int)
        course_idxs = np.unique(perpoint_course_idxs, axis=0)
        course_scan[course_idxs[:, 0], course_idxs[:, 1], course_idxs[:, 2]] = 1

        # loop through newly-populated "coarse" indices, and add to their fine representation
        res = self.grid_fine * self.grid_coarse
        perpoint_fine_idxs = np.floor((points_scan + 1) * res / 2).astype(int)
        fine_scan_subvoxels = []
        for i in range(course_idxs.shape[0]):
            fine2course = np.prod(perpoint_course_idxs == course_idxs[i:i+1, :], axis=1).astype(np.bool)  # (3,) array
            subvoxel_idxs = perpoint_fine_idxs[fine2course]
            subvoxel_idxs[:, 0] -= course_idxs[i, 0] * self.grid_fine
            subvoxel_idxs[:, 1] -= course_idxs[i, 1] * self.grid_fine
            subvoxel_idxs[:, 2] -= course_idxs[i, 2] * self.grid_fine

            # create fine-resolution voxel data chunk
            subvoxel = np.zeros((self.grid_fine, self.grid_fine, self.grid_fine), dtype=np.bool)
            subvoxel[subvoxel_idxs[:, 0], subvoxel_idxs[:, 1], subvoxel_idxs[:, 2]] = True
            fine_scan_subvoxels.append(subvoxel)
        
        # repn_coarse_new = np.zeros(self.repn_coarse.shape, dtype=bool)
        # coarse_new_idx = np.floor((points_scan + 1) * self.grid_coarse / 2).astype(int)
        # repn_coarse_new[coarse_new_idx[:, 0], coarse_new_idx[:, 1], coarse_new_idx[:, 2]] = True
        #
        # idx_occ = grid2idx(repn_coarse_new)
        #
        # repn_fine_new = []
        # for i in range(len(idx_occ)):
        #     repn_fine_box = np.zeros((self.grid_fine, self.grid_fine, self.grid_fine), dtype=bool)
        #     points_in_box_mask = np.linalg.norm(coarse_new_idx - idx_occ[i], axis=-1) == 0
        #     points_in_box = points_scan[points_in_box_mask]
        #     points_relative_coord = (points_in_box + 1) * self.grid_coarse / 2 - coarse_new_idx[points_in_box_mask]
        #     fine_box_idx = np.floor(points_relative_coord * self.grid_fine).astype(int)
        #     repn_fine_box[fine_box_idx[:, 0], fine_box_idx[:, 1], fine_box_idx[:, 2]] = True
        #     repn_fine_new.append(repn_fine_box)
        #
        # return repn_coarse_new, repn_fine_new, idx_occ

        return course_scan, course_idxs, fine_scan_subvoxels

    @property
    def fine_voxel_grid(self):
        res = self.grid_fine*self.grid_coarse
        fine_grid = np.zeros((res, res, res), dtype=bool)

        course_idxs = np.where(self.repn_fine_idx != -1)
        for i, j, k in zip(course_idxs[0], course_idxs[1], course_idxs[2]):
            fine_idx = self.repn_fine_idx[i, j, k]
            subvoxel = np.stack(self.repn_fine[fine_idx], axis=0)
            subvoxel = np.bitwise_or.reduce(subvoxel, axis=0)
            fine_grid[i*self.grid_fine: (i+1)*self.grid_fine,
                      j*self.grid_fine: (j+1)*self.grid_fine,
                      k*self.grid_fine: (k+1)*self.grid_fine] = subvoxel
        return fine_grid


def farthest_subsample_points(pointcloud, view, num_subsampled_points=768):
    num_points = pointcloud.shape[0]
    nbrs = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud)
    idx = nbrs.kneighbors(view, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud[idx, :]


def grid2idx(grid):
    '''
    Args:
      grid: dtype=bool, size (nb_grid, nb_grid, nb_grid)
    Output:
      idx_occ: indices for voxels of value 1, size (n_occ, 3)
    '''
    return np.stack(np.where(grid), axis=1)

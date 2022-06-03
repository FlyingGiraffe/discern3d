import os
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
import dill as pickle
from socket import error as SocketError

AGENT0_LOCK = threading.RLock()
AGENT0_FINISHED_THREADS = 0
AGENT_ALMOST_FINISHED_THREADS = 0
AGENT_FINISHED_PRIORITY_LISTS = 0
AGENT0_STARTED_THREADS = 0

from network import Router

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

class Agent(object):
    def __init__(self, router, agent_idx, kwargs):
        self.router = router
        self.output_dir = kwargs['output_dir']

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
        self.our_scanned_packets = set()

        # final course and fine representations
        self.repn_coarse = np.zeros((kwargs['grid_coarse'], kwargs['grid_coarse'], kwargs['grid_coarse']), dtype=bool)
        self.repn_fine_idx = -np.ones((kwargs['grid_coarse'], kwargs['grid_coarse'], kwargs['grid_coarse']), dtype=int)
        self.repn_fine_locks = [[[threading.RLock() for i in range(self.grid_coarse)] for j in range(self.grid_coarse)] for k in range(self.grid_coarse)]
        self.repn_fine = []

        # rep for building a dynamic priority list
        self.gossip_queue = queue.Queue()
        self.priority_list_timesteps = [[[[float('inf') for a in range(len(self.agent_ips))] for i in range(self.grid_coarse)] for j in range(self.grid_coarse)] for k in range(self.grid_coarse)]
        self.priority_list_locks = [[[threading.Condition() for i in range(self.grid_coarse)] for j in range(self.grid_coarse)] for k in range(self.grid_coarse)]

        # Queue and temporary state for handling fine representation
        self.received_packets = queue.Queue()
        self.temp_repn_fine = [[[dict() for i in range(self.grid_coarse)] for j in range(self.grid_coarse)] for k in range(self.grid_coarse)]
        # self.temp_repn_lock = threading.RLock()
        self.temp_repn_locks = [[[threading.RLock() for i in range(self.grid_coarse)] for j in range(self.grid_coarse)] for k in range(self.grid_coarse)]

        # Thread management
        self.threads_lock = threading.RLock()
        self.threads = []


    def run(self, shape_data, view, view_transition, 
            scan_hz=2.0, gossip_hz=1.0, clean_hz=1.0, lowres_hz=1.0,
            update_retry_hz=1.0):
        # step 1: init listening server loop
        thread = threading.Thread(target=self.init_server, args=())
        thread.start()
        self.threads.append(thread)

        # step 2-1: init gossiping loop
        gossip_kwargs = {'finished_callback' : (lambda :  False),
                         'freq': gossip_hz}
        thread = threading.Thread(target=self.gossip_loop, kwargs=gossip_kwargs) 
        thread.start()
        self.threads.append(thread)

        # step 2-2: init gossip listening loop
        gossip_handler_kwargs = {'finished_callback' : (lambda :  False)}
        thread = threading.Thread(target=self.gossip_handler, kwargs=gossip_handler_kwargs)
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
            'scan_hz': scan_hz,
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
        with SimpleXMLRPCServer((self.my_ip[0], self.my_ip[1]), requestHandler=RequestHandler, logRequests=False) as server:
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
            server.register_function(self.get_all_priority_lists_rpc, 'get_all_priority_lists')
            print('Registered UPDATE_ALL_PRIORITY_LISTS RPC')
            server.serve_forever()

    def get_rpc(self, course_idx):
        return self.repn_coarse[course_idx[0], course_idx[1], course_idx[2]]

    def get_all_rpc(self):
        return dense2sparse(self.repn_coarse).tolist()

    def update_rpc(self, course_idx, packet_id, fine_scan):
        timestep = time.time()
        fine_scan = sparse2dense(np.array(fine_scan), self.grid_fine)
        self.received_packets.put((course_idx, packet_id, fine_scan, timestep))
        return "success"

    def get_priority_list_rpc(self, coarse_idx):
        return self.priority_list_timesteps[coarse_idx[0]][coarse_idx[1]][coarse_idx[2]]

    def get_all_priority_lists_rpc(self):
        return self.priority_list_timesteps

    def update_priority_list_rpc(self, coarse_idx, update):
        self.priority_list_timesteps[coarse_idx[0]][coarse_idx[1]][coarse_idx[2]] = update
        return "success"

    # =================================================================
    # ================== COURSE REPRESENTATION GET ====================
    # =================================================================
    def get_loop(self, finished_callback, freq=1.0):
        global AGENT0_STARTED_THREADS, AGENT0_FINISHED_THREADS, AGENT_ALMOST_FINISHED_THREADS, AGENT_FINISHED_PRIORITY_LISTS
        while not finished_callback():
            for i, agent_ip in enumerate(self.agent_ips):
                if i == self.my_idx:
                    continue
                try:
                    # s = xmlrpc.client.ServerProxy('http://{0}:{1}'.format(agent_ip[0], agent_ip[1]))
                    s = self.router.attempt_ServerProxy(self.my_ip, agent_ip)
                    coarse_grid = s.get_all()
                    # print(coarse_grid)
                    coarse_grid = sparse2dense(np.array(coarse_grid), self.grid_coarse)
                except (ConnectionRefusedError, ConnectionResetError, TimeoutError) as e:
                    continue
                
                self.repn_coarse |= coarse_grid

            # print('Visualization')
            # print("Agent {0} Number of Fine Voxels {1}".format(self.my_idx, np.sum(self.fine_voxel_grid)))
            # vis_voxel_grid(self.fine_voxel_grid)

            voxel_data = {'coarse': self.repn_coarse, 'fine': self.fine_voxel_grid}
            self.save_voxel(voxel_data, self.my_idx, time.time())
            if self.my_idx == 1:
                print("Number of active threads: {0}, Number spawned {1}, Number ended {2}, Number almost ended {3}, Number priority-list ended {4}".format(threading.active_count(), len(self.threads), AGENT0_FINISHED_THREADS, AGENT_ALMOST_FINISHED_THREADS, AGENT_FINISHED_PRIORITY_LISTS))

            time.sleep(1/freq)

    def save_voxel(self, voxel_data, my_idx, time):
        save_dir = self.output_dir
        save_file = 'agent{}_{}.pkl'.format(my_idx, time)
        save_loc = os.path.join(save_dir, save_file)
        # print('Saving in {}'.format(save_loc))
        with open(save_loc, 'wb') as f:
            pickle.dump(voxel_data, f)

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
            all_voxel_ids = np.meshgrid(np.arange(self.grid_coarse), np.arange(self.grid_coarse), np.arange(self.grid_coarse))
            for ip_address in self.agent_ips:
                try:
                    # xmlrpc.client.ServerProxy('http://{0}:{1}'.format(ip_address[0], ip_address[1]))
                    s = self.router.attempt_ServerProxy(self.my_ip, ip_address)
                    all_their_priority_list_tstamps = s.get_all_priority_lists()
                except (ConnectionRefusedError, ConnectionResetError, TimeoutError) as e:
                    continue

                for i, j, k in zip(all_voxel_ids[0].flatten(), all_voxel_ids[1].flatten(), all_voxel_ids[2].flatten()):
                    our_priority_list_tstamps = self.priority_list_timesteps[i][j][k]
                    their_priority_list_tstamps = all_their_priority_list_tstamps[i][j][k]
                    for aidx, tstamp in enumerate(their_priority_list_tstamps):
                        if tstamp < our_priority_list_tstamps[aidx]:
                            # print("Updating priority list timesteps!")
                            self.gossip_queue.put(([i, j, k], tstamp, aidx))


            # for voxel_id in observed_voxel_ids:
            #     for ip_address in self.agent_ips:
            #         try:
            #             # xmlrpc.client.ServerProxy('http://{0}:{1}'.format(ip_address[0], ip_address[1]))
            #             s = self.router.attempt_ServerProxy(self.my_ip, ip_address)
            #             their_priority_list_tstamps = s.get_priority_list(voxel_id.tolist())
            #         except (ConnectionRefusedError, ConnectionResetError, TimeoutError) as e:
            #             continue
            #         # todo: this is massive!!!!

                    # our_priority_list_tstamps = self.priority_list_timesteps[voxel_id[0]][voxel_id[1]][voxel_id[2]]
                    # for i, tstamp in enumerate(their_priority_list_tstamps):
                    #     if tstamp < our_priority_list_tstamps[i]:
                    #         self.gossip_queue.put((voxel_id, tstamp, i))
            time.sleep(1/freq)

    def gossip_handler(self, finished_callback):
        # get priority list of this agent and us
        while not finished_callback():
            priority_queue_update = self.gossip_queue.get(block=True)
            voxel_id, new_tstep, agent_idx = priority_queue_update

            # get current and updated set of agent timesteps
            self.priority_list_timesteps[voxel_id[0]][voxel_id[1]][voxel_id[2]][agent_idx] = new_tstep

            # check if we "lost" / not in top-k for this voxel-priority list
            agent_timestamps = self.priority_list_timesteps[voxel_id[0]][voxel_id[1]][voxel_id[2]] 
            fightingchance = sum([el < agent_timestamps[self.my_idx] for el in agent_timestamps]) < self.K
            if not fightingchance and np.isinf(agent_timestamps[self.my_idx]):
                self.gossip_queue.put((voxel_id, time.time(), self.my_idx))

            # ------ check if this affects our top-k agents
            # check 1: we added a new agent to the initially lacking priority list
            # voxel_lock = self.priority_list_locks[voxel_id[0]][voxel_id[1]][voxel_id[2]]
            # voxel_lock.acquire()
            # voxel_lock.notifyAll()
            # voxel_lock.release()

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
            while threading.active_count() > 100:
                time.sleep(1)
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
            self.gossip_queue.put((coarse_idx, timestamp, self.my_idx))

        # update temporary global fine representation
        temp_repn_lock = self.temp_repn_locks[coarse_idx[0]][coarse_idx[1]][coarse_idx[2]]
        temp_repn_lock.acquire()
        self.temp_repn_fine[coarse_idx[0]][coarse_idx[1]][coarse_idx[2]][packet_id] = voxel_data
        temp_repn_lock.release()

        # enter a loop that sends the data to appropriate agents
        self.update_packet_replication(coarse_idx, packet_id, update_retry_hz)

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
        our_cur_priority = np.where(np.argsort(agent_timestamps) == self.my_idx)[0].item()
        we_win = our_cur_priority < (self.K - num_unknown)  # we are in the top K*, and we know all except (K*-1) timsteps

        init_time = time.time()
        MAX_WAIT_TIME = 5
        while fightingchance and not we_win:
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

            # HACKY: break out of infinite loops if we've waited too long --> could occur if we reach thread limit but haven't decided on new
            if time.time() - init_time > MAX_WAIT_TIME:
                cur_timestamps = self.priority_list_timesteps[course_idx[0]][course_idx[1]][course_idx[2]]
                unkown_agents = np.where(np.isinf(cur_timestamps))[0]
                if len(unkown_agents) > 0:
                    selected_agent_idx = np.random.choice(unkown_agents)
                    self.priority_list_timesteps[course_idx[0]][course_idx[1]][course_idx[2]][selected_agent_idx] = time.time()

            # sleep until priority list is updated
            time.sleep(1/update_retry_hz)
            this_lock.release()

            # update while loop flags
            agent_timestamps = self.priority_list_timesteps[course_idx[0]][course_idx[1]][course_idx[2]]
            fightingchance = sum([el < agent_timestamps[self.my_idx] for el in agent_timestamps]) < self.K
            num_unknown = np.sum(np.isinf(agent_timestamps))
            our_cur_priority = np.where(np.argsort(agent_timestamps) == self.my_idx)[0]
            we_win = our_cur_priority < (self.K - num_unknown)  # we are in the top K*, and we know all except (K*-1) timsteps

        if self.my_idx == 0:
            global AGENT0_LOCK, AGENT_FINISHED_PRIORITY_LISTS
            AGENT0_LOCK.acquire()
            AGENT_FINISHED_PRIORITY_LISTS += 1
            AGENT0_LOCK.release()

        # once we break out of the loop, we know we have reached consensus on priority list
        k_winners = np.argsort(self.priority_list_timesteps[course_idx[0]][course_idx[1]][course_idx[2]])[:self.K]
        fully_replicated = set(k_winners).issubset(transmitted_history)
        if packet_id in self.our_scanned_packets:
            attempts = 0
            while not fully_replicated:
                for agent_idx in k_winners:
                    if agent_idx in transmitted_history:
                        continue
                    success = self.transmit_data(self.agent_ips[agent_idx], course_idx, packet_id)
                    if success:
                        transmitted_history.add(agent_idx)
                time.sleep(1/update_retry_hz)
                fully_replicated = set(k_winners).issubset(transmitted_history)
                attempts += 1

        # if we are in the priority list, add the voxel to storage
        voxel_data = self.temp_repn_fine[course_idx[0]][course_idx[1]][course_idx[2]][packet_id]

        if self.my_idx == 0:
            global AGENT_ALMOST_FINISHED_THREADS
            AGENT0_LOCK.acquire()
            AGENT_ALMOST_FINISHED_THREADS += 1
            AGENT0_LOCK.release()

        # update fine-grained representation
        repn_lock = self.repn_fine_locks[course_idx[0]][course_idx[1]][course_idx[2]]
        repn_lock.acquire()
        if self.my_idx in k_winners:
            fine_idx = self.repn_fine_idx[course_idx[0], course_idx[1], course_idx[2]]
            if fine_idx == -1:
                self.repn_fine_idx[course_idx[0], course_idx[1], course_idx[2]] = len(self.repn_fine)
                self.repn_fine.append(voxel_data)
            else:
                self.repn_fine[fine_idx] |= voxel_data
        repn_lock.release()
        if self.my_idx == 0:
            global  AGENT0_FINISHED_THREADS
            AGENT0_LOCK.acquire()
            AGENT0_FINISHED_THREADS += 1
            AGENT0_LOCK.release()

        # remove temporary storage of high-res voxel data
        # TODO: deleting the data causes thread-safety problems...
        # temp_repn_lock = self.temp_repn_locks[course_idx[0]][course_idx[1]][course_idx[2]]
        # self.temp_repn_lock.acquire()
        # self.temp_repn_fine[course_idx[0]][course_idx[1]][course_idx[2]].pop(packet_id)
        # self.temp_repn_lock.release()

    def transmit_data(self, agent_ip, course_idx, packet_id):
        try:
            # s = xmlrpc.client.ServerProxy('http://{0}:{1}'.format(agent_ip[0], agent_ip[1]))
            s = self.router.attempt_ServerProxy(self.my_ip, agent_ip)
            fine_scan = self.temp_repn_fine[course_idx[0]][course_idx[1]][course_idx[2]][packet_id]
            fine_scan = dense2sparse(fine_scan)
            res = s.update(course_idx, packet_id, fine_scan.tolist())
        except (ConnectionRefusedError, ConnectionResetError, TimeoutError, SocketError, IOError) as e:
            # print("Error occured:")
            # print(e)
            return False

        return res == 'success'

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
                self.our_scanned_packets.add(packet_id)
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

        return course_scan, course_idxs, fine_scan_subvoxels

    @property
    def fine_voxel_grid(self):
        res = self.grid_fine*self.grid_coarse
        fine_grid = np.zeros((res, res, res), dtype=bool)

        course_idxs = np.where(self.repn_fine_idx != -1)
        for i, j, k in zip(course_idxs[0], course_idxs[1], course_idxs[2]):
            fine_idx = self.repn_fine_idx[i, j, k]
            fine_grid[i*self.grid_fine: (i+1)*self.grid_fine,
                      j*self.grid_fine: (j+1)*self.grid_fine,
                      k*self.grid_fine: (k+1)*self.grid_fine] = self.repn_fine[fine_idx]
        return fine_grid


def farthest_subsample_points(pointcloud, view, num_subsampled_points=768):
    # NOTE: following for sidestepping the bug from the uncommented code below
    points = np.random.randn(num_subsampled_points, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points
    
    # num_points = pointcloud.shape[0]
    # nbrs = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
    #                          metric=lambda x, y: minkowski(x, y)).fit(pointcloud)
    #
    # idx = nbrs.kneighbors(view, return_distance=False).reshape((num_subsampled_points,))
    # return pointcloud[idx, :]


def grid2idx(grid):
    '''
    Args:
      grid: dtype=bool, size (nb_grid, nb_grid, nb_grid)
    Output:
      idx_occ: indices for voxels of value 1, size (n_occ, 3)
    '''
    return np.stack(np.where(grid), axis=1)


def dense2sparse(grid):
    return np.stack(np.where(grid), axis=1)

def sparse2dense(idxs, res):
    fine_grid = np.zeros((res, res, res), dtype=bool)
    if len(idxs.shape) == 1:
        return fine_grid
    fine_grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = True
    return fine_grid

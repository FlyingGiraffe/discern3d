import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from datetime import datetime
import time
from vis import vis_voxel_grid

class Agent(object):
    def __init__(self, kwargs):
        self.grid_coarse = kwargs['grid_coarse']
        self.grid_fine = kwargs['grid_fine']

        self.num_points_view = kwargs['num_points_view']
        self.num_points_scan = kwargs['num_points_scan']
        
        # self.repn_coarse = np.zeros((kwargs['grid_coarse'], kwargs['grid_coarse'], kwargs['grid_coarse']), dtype=bool)
        # self.repn_fine = [[[None for i in range(kwargs['grid_coarse'])] \
        #                   for j in range(kwargs['grid_coarse'])] \
        #                   for k in range(kwargs['grid_coarse'])]

        self.repn_coarse = [np.zeros((kwargs['grid_coarse'], kwargs['grid_coarse'], kwargs['grid_coarse']), dtype=bool)]
        self.repn_fine_idx = -np.ones((kwargs['grid_coarse'], kwargs['grid_coarse'], kwargs['grid_coarse']), dtype=int)
        self.repn_fine = []

        # for the data...
        self.data = {}
        self.T = kwargs['T']
        self.K = kwargs['K']
        self.allocation_discrepancy_threshold = kwargs['allocation_discrepancy_threshold']


    def run(self, scan_hz=1.0):
        # step 1: init listening server loop

        # step 1.5: init high-res protocol loop

        # step 2: initialize course-rep get-scene loop

        # step 3: scan data
        pass


    def scan_loop(self, shape_data, view, view_transition, finished_callback, scan_hz=20, vis=False):
        timestep = 0
        while not finished_callback():
            # scan data
            course_scan, course_idxs, fine_scans = self.scan(shape_data, view)

            # update global coarse representation
            self.repn_coarse.append(course_scan)

            # update global fine representation
            for i, course_idx in enumerate(course_idxs):
                this_fine_idx = self.repn_fine_idx[course_idx[0], course_idx[1], course_idx[2]]
                if this_fine_idx == -1:
                    self.repn_fine_idx[course_idx[0], course_idx[1], course_idx[2]] = len(self.repn_fine)
                    self.repn_fine.append([fine_scans[i]])
                else:
                    self.repn_fine[this_fine_idx].append(fine_scans[i])

            # update view
            view = view_transition(view)
            time.sleep(1/scan_hz)
            print(timestep)

            if vis:
                vis_voxel_grid(self.fine_voxel_grid)
            timestep += 1

    def setup(self, kwargs):
        """ A function for setting up the ID's
        """
        self.id = kwargs['id']
        self.id2agents = kwargs['id2agents']
        self.other_ids = [el for el in self.id2agents if el!=self.id]

        self.vox_ids = kwargs['voxel_ids']
        self.vox2agent = {el: [] for el in self.vox_ids}

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

    def update(self, method, data):
        if method == 'scanned':
            # this means that new data is scanned, and the agent needs to decide
            # what to do with that scan.
            vox_id = data['vox_id']
            vox_hash = datetime.now() # idk, something that can be sorted by time.

            if len(self.vox2agent[vox_id]) < self.K:
                # here, we have a decision to make. we can either pass this data to another agent, 
                # or keep it for ourselves. 
                a2v = self.get_agent2vox()
                cankeep, send_candidates = self.cankeep(self, vox_id, a2v)
                
                # if we can get away with not sending it AND the data can fit in memory...
                if cankeep and len(self.data) < self.T:
                    self.data[vox_id] = data['pointcloud']
                    self.vox2agent[vox_id].append((self.id, vox_hash))

                    # now let's notify the other agents of this change...
                    for agent_id in self.other_ids:
                        other = self.id2agents[agent_id] # some function to return the actual agent object, or get access to some port...
                        other.update("update_vox2agent", {'vox2agent': self.vox2agent})

                # otherwise...
                else:
                    # this means the memory of this agent is  full currently, so
                    # let's pass this onto another agent. Let's pick a random agent 
                    # from send_candidates
                    send_to = np.random.choice(send_candidates)

                    # then, do a quick sync with the other agent. Do they indeed have
                    # enough space to store the data you're about to send them?
                    # TODO. eugh. 

                    # send them. TODO What should it do if it fails?
                    self.id2agents[send_to].update(method, data)
    
                for agent_id in self.other_ids:
                    other = self.id2agents[agent_id] # some function to return the actual agent object, or get access to some port...
                    other.update("update_vox2agent", {'vox2agent': self.vox2agent})

            else:
                print("voxel seems to be already covered by K agents -- tossing!")
            
           
        elif method == 'update_vox2agent':
            # reconcile the vox2agent from another agent to this...
            for key in self.vox2agent:
                other_vox2agent = data['vox2agent']
                mine = self.vox2agent[key]
                yours = other_vox2agent[key]
                union_agents = list(set(mine).union(set(yours)))
                # if the union <= length K, we don't have a problem... but...
                if len(union_agents) >= self.K:
                    # we have to merge them, but in a way that will be consistent.
                    merged = sorted(union_agents,
                                    key=lambda x: x[1]) # we sort by the timestamp
                    merged_topK = merged[:self.K]
                    merged_remove = merged[self.K:]

                    # now of the ones that are going to be removed,
                    # check if this agent belongs in there. Remove the
                    # data if necessary.
                    for el in merged_remove:
                        if el[0] == self.id:
                            del self.data[key] # clear the data
                    
                    self.vox2agent[key] = merged_topK
                else: # we good. let' just merge it.
                    self.vox2agent[key] = union_agents

  
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
        course_scan = np.zeros(self.repn_coarse[0].shape)
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

import math
import copy

def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step and curriculum[curriculum_step]['img_size'] > current_size:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step and curriculum[curriculum_step]['img_size'] == current_size:
            return curriculum_step
    return 0

def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step


def extract_metadata(curriculum, current_step):
    # step = get_current_step(curriculum, epoch)
    return_dict = {}
    # place standard
    for key, value in curriculum.items():
        if key != 'stages' and 'Blend' not in type(value).__name__:
            return_dict[key] = value
        elif 'Blend' in type(value).__name__:
            return_dict[key] = value.query(current_step)

    # place stages
    if 'stages' in curriculum:
        for curriculum_step in sorted(curriculum['stages'].keys(), reverse=True): # iterate over steps in stages dict
            if curriculum_step <= current_step: # find most recent stage
                for stage_key, stage_value in curriculum['stages'][curriculum_step].items(): # place into dict
                    if 'Blend' in type(stage_value).__name__:
                        return_dict[stage_key] = stage_value.query(current_step)
                    else:
                        return_dict[stage_key] = stage_value
                break
    return return_dict

# def extract_metadata(curriculum, current_step):
#     # step = get_current_step(curriculum, epoch)
#     return_dict = {}
#     for key in [k for k in curriculum.keys() if type(k) != int]:
#         #return_dict[key] = curriculum[key]
#         if 'Blend' in type(curriculum[key]).__name__:
#             return_dict[key] = curriculum[key].query(current_step)
#         else:
#             return_dict[key] = curriculum[key]
#     for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
#         if curriculum_step <= current_step:
#             for key, value in curriculum[curriculum_step].items():
#                 return_dict[key] = value
#             break
#     return return_dict

class LinearBlend:
    def __init__(self, initial, final, duration, start=0):
        self.initial = initial
        self.final = final
        self.duration = duration
        self.start = start
    def query(self, timestep):
        t = min((timestep - self.start)/self.duration, 1)
        return t * self.final + (1-t)*self.initial
    def __repr__(self):
        return f'LinearBlend(initial={self.initial}, final={self.final}, duration={self.duration}, start={self.start})'

class ExponentialBlend:
    def __init__(self, initial, final, period, start=0):
        self.initial = initial
        self.final = final
        self.period = period
        self.start = start
    def query(self, timestep):
        return self.final + (self.initial - self.final)*(0.5**((timestep-self.start)/self.period))

def override_dict(default, overriding):
    if overriding is None: return default
    new_dict = copy.deepcopy(default)
    for key, val in overriding.items():
        new_dict[key] = val
    return new_dict
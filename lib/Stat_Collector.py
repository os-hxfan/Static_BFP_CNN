import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

# Define Collector based on hook, which is used to record the intermediate result
class Stat_Collector:
    def __init__(self, m):
        self.handle = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, m, inp, outp):
        self.out_features = outp.clone()
        self.in_features = inp
        self.m = m
    def remove(self):
        self.handle.remove()

# Insert hook of every "target_module"
# Return the inserted model and intermediate result 
def insert_hook(model, target_module_list):
    intern_outputs = []
    for layer in model.modules():
        for target_module in target_module_list:
            if isinstance(layer, target_module):
                logging.debug("Collect: %s" % (layer))
                intern_outputs.append(Stat_Collector(layer))
    logging.info("Total %d hook inserted" % (len(intern_outputs)))
    return model, intern_outputs

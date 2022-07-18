import os
import json
from pathlib import Path

import numpy as np
import trimesh
import torch
class ClassificationMajorityVoting:
    def __init__(self, nclass):
        self.votes = {}
        self.nclass = nclass

    def vote(self, mesh_paths, preds, labels):
        
        if isinstance(preds, torch.Tensor):
            preds = preds.data
        if isinstance(labels, torch.Tensor):
            labels = labels.data

        for i in range(preds.shape[0]):
            name = (Path(mesh_paths[i]).stem).split('-')[0]
            if not name in self.votes:
                self.votes[name] = {
                    'polls': np.zeros(self.nclass, dtype=int),
                    'label': labels[i]
                }
            self.votes[name]['polls'][preds[i]] += 1

    def compute_accuracy(self):
        sum_acc = 0
        for name, vote in self.votes.items():
            pred = np.argmax(vote['polls'])
            sum_acc += pred == vote['label']
        return sum_acc / len(self.votes)

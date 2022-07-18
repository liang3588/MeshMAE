import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
import copy
import csv

if __name__ == '__main__':
    root = './ShapeNet'
    dataroot = Path(root)
    output_root_manifold = './ShapeNet_manifold'
    output_root_simplify = './ShapeNet_simplify'
    for obj_path in dataroot.iterdir():
        if obj_path.is_file() and str(obj_path)[-5:] == '-.obj':
            print('read', obj_path)
            mesh = trimesh.load_mesh(obj_path, process=False)
            oface_number = mesh.faces.shape[0]
            mface_number = oface_number * 1.2
            commandm = 'Manifold-master/build/manifold ' + str(
                obj_path) + ' ' + output_root_manifold + '/' + obj_path.name + ' ' + str(int(mface_number))
            print(commandm)
            try:
                status1 = os.system(commandm)
            except:
                if status1 != 0:
                    raise Exception('wrong, command=%s, status=%s' % (commandm, status1))

            commands = 'Manifold-master/build/simplify -i ' + output_root_manifold + '/' + obj_path.name + ' -o ' + output_root_simplify + '/' + obj_path.name + ' -m -f ' + str(
                500)
            print(commands)
            try:
                status = os.system(commands)
            except:
                if status != 0:
                    raise Exception('wrong, command=%s, status=%s' % (commands, status))


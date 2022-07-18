#generate the reconstruct mesh
import os
import json
from pathlib import Path
import numpy as np
import trimesh
import torch
import copy


def save_results(masked_indices, unmasked_indices, pred_vertices_coordinates, coordinates, paths):
    if not os.path.exists('result'):
        os.mkdir('result')
    name = 'ShapeNet'
    results_path = Path('result') / name

    print(masked_indices.shape, unmasked_indices.shape, pred_vertices_coordinates.shape, coordinates.shape)
    results_path.mkdir(parents=True, exist_ok=True)


    for i in range(masked_indices.shape[0]):
        unmask_vertice_coordinates = coordinates[i][unmasked_indices[i]].reshape(unmasked_indices[i].shape[0], -1, 3)

        unmask_vertice_coordinates = torch.unique(unmask_vertice_coordinates, dim=1)


        vertices = torch.cat([unmask_vertice_coordinates, pred_vertices_coordinates[i]], dim=0)
        vertices = vertices.reshape(-1, 3)
        unmask_vertice_coordinates = unmask_vertice_coordinates.reshape(-1, 3)
        vertices = vertices.cpu().detach().numpy()
        unmask_vertice_coordinates = unmask_vertice_coordinates.cpu().detach().numpy()
        mesh_path = Path(paths[i]).stem
        print(mesh_path)

        newmesh =  str(results_path / mesh_path) + '.txt'
        maskmesh = str(results_path / mesh_path) + 'mask.txt'
        print(newmesh)

        np.savetxt(newmesh, vertices, delimiter=' ')
        np.savetxt(maskmesh, unmask_vertice_coordinates, delimiter=' ')
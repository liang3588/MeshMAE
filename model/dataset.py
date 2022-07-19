# using the modelnet40 as the dataset, and using the processed feature matrixes
import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
import pygem
from pygem import FFD
import copy
import csv


def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.choice([0, 90, 180, 270]) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh1.vertices = rotation.apply(mesh1.vertices)
    return mesh1


def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh


def mesh_normalize(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    vertices = mesh1.vertices - mesh1.vertices.min(axis=0)
    vertices = vertices / vertices.max()
    mesh1.vertices = vertices
    return mesh1


def mesh_deformation(mesh: trimesh.Trimesh):
    ffd = FFD([2, 2, 2])
    random = np.random.rand(6) * 0.1
    ffd.array_mu_x[1, 1, 1] = random[0]
    ffd.array_mu_y[1, 1, 1] = random[1]
    ffd.array_mu_z[1, 1, 1] = random[2]
    ffd.array_mu_x[0, 0, 0] = random[3]
    ffd.array_mu_y[0, 0, 0] = random[4]
    ffd.array_mu_z[0, 0, 0] = random[5]
    vertices = mesh.vertices
    new_vertices = ffd(vertices)
    mesh.vertices = new_vertices
    return mesh


def load_mesh(path, augments=[], request=[], seed=None):
    label = 0
    if 'guitar' in str(path):
        label = 15
    elif 'door' in str(path):
        label = 30
    elif 'radio' in str(path):
        label = 38
    elif 'curtain' in str(path):
        label = 13
    elif 'dresser' in str(path):
        label = 7
    elif 'bookshelf' in str(path):
        label = 9
    elif 'tent' in str(path):
        label = 21
    elif 'bottle' in str(path):
        label = 32
    elif 'lamp' in str(path):
        label = 23
    elif 'piano' in str(path):
        label = 5
    elif 'stool' in str(path):
        label = 16
    elif 'bench' in str(path):
        label = 28
    elif 'chair' in str(path):
        label = 37
    elif 'bathtub' in str(path):
        label = 10
    elif 'vase' in str(path):
        label = 33
    elif 'flower' in str(path):
        label = 31
    elif 'plant' in str(path):
        label = 34
    elif 'keyboard' in str(path):
        label = 3
    elif 'night' in str(path):
        label = 4
    elif 'sofa' in str(path):
        label = 25
    elif 'glass' in str(path):
        label = 17
    elif 'cup' in str(path):
        label = 18
    elif 'person' in str(path):
        label = 22
    elif 'range' in str(path):
        label = 35
    elif 'desk' in str(path):
        label = 24
    elif 'bed' in str(path):
        label = 11
    elif 'toilet' in str(path):
        label = 14
    elif 'laptop' in str(path):
        label = 19
    elif 'mantel' in str(path):
        label = 0
    elif 'xbox' in str(path):
        label = 1
    elif 'monitor' in str(path):
        label = 8
    elif 'stairs' in str(path):
        label = 6
    elif 'table' in str(path):
        label = 12
    elif 'car' in str(path):
        label = 36
    elif 'bowl' in str(path):
        label = 29
    elif 'wardrobe' in str(path):
        label = 2
    elif 'tv' in str(path):
        label = 39
    elif 'cone' in str(path):
        label = 20
    elif 'sink' in str(path):
        label = 27
    elif 'airplane' in str(path):
        label = 26

    mesh = trimesh.load_mesh(path, process=False)

    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)
        if method == 'deformation':
            mesh = mesh_deformation(mesh)

    F = mesh.faces
    V = mesh.vertices

    Fs = mesh.faces.shape[0]
    face_coordinate = V[F.flatten()].reshape(-1, 9)

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])

    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)
    patch_num = Fs // 4 // 4 // 4
    allindex = np.array(list(range(0, Fs)))
    indices = allindex.reshape(-1, patch_num).transpose(1, 0)

    feats_patch = feats[:, indices]
    center_patch = face_center[indices]
    cordinates_patch = face_coordinate[indices]
    faces_patch = mesh.faces[indices]

    feats_patch = feats_patch
    center_patch = center_patch
    cordinates_patch = cordinates_patch
    faces_patch = faces_patch
    feats_patcha = np.concatenate((feats_patch, np.zeros((10, 256 - patch_num, 64), dtype=np.float32)), 1)
    center_patcha = np.concatenate((center_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.float32)), 0)
    cordinates_patcha = np.concatenate((cordinates_patch, np.zeros((256 - patch_num, 64, 9), dtype=np.float32)), 0)
    faces_patcha = np.concatenate((faces_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.int)), 0)

    Fs_patcha = np.array(Fs)

    return feats_patcha, center_patcha, cordinates_patcha, faces_patcha, Fs_patcha, label

def load_mesh_seg(path, normalize=True,augments=[], request=[], seed=None):
    mesh = trimesh.load_mesh(path, process=False)
    label_path = Path(str(path).split('.')[0] + '.json')
    with open(label_path) as f:
        segment = json.load(f)

    sub_labels = np.array(segment['sub_labels']) - 1

    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)
        if method == 'deformation':
            mesh = mesh_deformation(mesh)
    if normalize:
        mesh = mesh_normalize(mesh)
    F = mesh.faces
    V = mesh.vertices
    Fs = mesh.faces.shape[0]
    face_coordinate = V[F.flatten()].reshape(-1, 9)

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])

    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)
    patch_num = Fs // 4 // 4 // 4
    if patch_num != 256:
        print(path)
    allindex = np.array(list(range(0, Fs)))
    indices = allindex.reshape(-1, patch_num).transpose(1, 0)

    feats_patch = feats[:, indices]
    center_patch = face_center[indices]
    cordinates_patch = face_coordinate[indices]
    faces_patch = mesh.faces[indices]
    label_patch = sub_labels[indices]
    label_patcha = np.concatenate((label_patch, np.zeros((256 - patch_num, 64), dtype=np.float32)), 0)
    label_patcha = np.expand_dims(label_patcha, axis=2)
    feats_patch = feats_patch
    center_patch = center_patch
    cordinates_patch = cordinates_patch
    faces_patch = faces_patch
    feats_patcha = np.concatenate((feats_patch, np.zeros((10, 256 - patch_num, 64), dtype=np.float32)), 1)
    center_patcha = np.concatenate((center_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.float32)), 0)
    cordinates_patcha = np.concatenate((cordinates_patch, np.zeros((256 - patch_num, 64, 9), dtype=np.float32)), 0)
    faces_patcha = np.concatenate((faces_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.int)), 0)
    feats_patcha = feats_patcha.transpose(1, 2, 0)
    Fs_patcha = np.array(Fs)
    Fs_patcha = Fs_patcha.repeat(256 * 64).reshape(256, 64, 1)

    return faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha

def load_mesh_shape(path, augments=[], request=[], seed=None):

    mesh = trimesh.load_mesh(path, process=False)

    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)
        if method == 'deformation':
            mesh = mesh_deformation(mesh)

    F = mesh.faces
    V = mesh.vertices

    Fs = mesh.faces.shape[0]
    face_coordinate = V[F.flatten()].reshape(-1, 9)

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])

    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)
    patch_num = Fs // 4 // 4 // 4
    allindex = np.array(list(range(0, Fs)))
    indices = allindex.reshape(-1, patch_num).transpose(1, 0)

    feats_patch = feats[:, indices]
    center_patch = face_center[indices]
    cordinates_patch = face_coordinate[indices]
    faces_patch = mesh.faces[indices]

    feats_patch = feats_patch
    center_patch = center_patch
    cordinates_patch = cordinates_patch
    faces_patch = faces_patch

    feats_patcha = np.concatenate((feats_patch, np.zeros((10, 256 - patch_num, 64), dtype=np.float32)), 1)
    center_patcha = np.concatenate((center_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.float32)), 0)
    cordinates_patcha = np.concatenate((cordinates_patch, np.zeros((256 - patch_num, 64, 9), dtype=np.float32)), 0)
    faces_patcha = np.concatenate((faces_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.int)), 0)

    Fs_patcha = np.array(Fs)

    return feats_patcha, center_patcha, cordinates_patcha, faces_patcha, Fs_patcha

class ClassificationDataset(data.Dataset):
    def __init__(self, dataroot, train=True, augment=None):
        super().__init__()

        self.dataroot = Path(dataroot)
        self.augments = []
        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs', 'normal']

        self.mesh_paths = []
        self.labels = []
        self.browse_dataroot()
        if train and augment:
            self.augments = augment

    def browse_dataroot(self):
        self.shape_classes = [x.name for x in self.dataroot.iterdir() if x.is_dir()]

        for obj_class in self.dataroot.iterdir():
            if obj_class.is_dir():
                label = self.shape_classes.index(obj_class.name)
                for obj_path in (obj_class / self.mode).iterdir():
                    if obj_path.is_file():
                        self.mesh_paths.append(obj_path)
                        self.labels.append(label)

        self.mesh_paths = np.array(self.mesh_paths)
        self.labels = np.array(self.labels)

    def __getitem__(self, idx):

        # label = self.labels[idx]

        if self.mode == 'train':

            feats, center, cordinates, faces, Fs, label = load_mesh(self.mesh_paths[idx], augments=self.augments,
                                                                    request=self.feats)

            return feats, center, cordinates, faces, Fs, label, str(self.mesh_paths[idx])
        else:

            feats, center, cordinates, faces, Fs, label = load_mesh(self.mesh_paths[idx],
                                                                    augments=self.augments,
                                                                    request=self.feats)
            return feats, center, cordinates, faces, Fs, label, str(self.mesh_paths[idx])

    def __len__(self):
        return len(self.mesh_paths)


class SegmentationDataset(data.Dataset):
    def __init__(self, dataroot, train=True, augments=None):
        super().__init__()

        self.dataroot = dataroot

        self.augments = []
        # if train and augments:
        # self.augments = augments
        self.augments = augments
        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs', 'normal']

        self.mesh_paths = []
        self.raw_paths = []
        self.seg_paths = []
        self.browse_dataroot()

    # self.set_attrs(total_len=len(self.mesh_paths))

    def browse_dataroot(self):
        for dataset in (Path(self.dataroot) / self.mode).iterdir():
            if dataset.is_dir():
                for obj_path in dataset.iterdir():
                    if obj_path.suffix == '.obj':
                        obj_name = obj_path.stem
                        seg_path = obj_path.parent / (obj_name + '.json')

                        self.mesh_paths.append(str(obj_path))

        self.mesh_paths = np.array(self.mesh_paths)

    def __getitem__(self, idx):

        if self.mode == 'train':

            faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha = load_mesh_seg(
                self.mesh_paths[idx],
                normalize=True,
                augments=self.augments,
                request=self.feats)

            return faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha
        else:
            faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha = load_mesh_seg(
                self.mesh_paths[idx],
                normalize=True,
                request=self.feats)

            return faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha

    def __len__(self):
        return len(self.mesh_paths)


class ShapeNetDataset(data.Dataset):
    def __init__(self, dataroot, train=True, augment=None):
        super().__init__()

        self.dataroot = Path(dataroot)
        self.augments = []
        self.feats = ['area', 'face_angles', 'curvs', 'normal']
        self.mesh_paths = []

        self.browse_dataroot()
        if train and augment:
            self.augments = augment

    def browse_dataroot(self):
        self.shape_classes = [x.name for x in self.dataroot.iterdir() if x.is_dir()]

        for obj_class in self.dataroot.iterdir():
            if obj_class.is_dir():
                for obj_path in (obj_class).iterdir():
                    if obj_path.is_file():
                        self.mesh_paths.append(obj_path)

        self.mesh_paths = np.array(self.mesh_paths)

    def __getitem__(self, idx):
        label = 0
        feats, center, cordinates, faces, Fs = load_mesh_shape(self.mesh_paths[idx], augments=self.augments,
                                                             request=self.feats)

        return   feats, center, cordinates, faces, Fs, label, str(self.mesh_paths[idx])



    def __len__(self):
        return len(self.mesh_paths)

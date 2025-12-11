# /maploc/data/standalone_dataset.py

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json
from pathlib import Path

class StandaloneWireframeDataset(Dataset):
    def __init__(self, root_dir, split_path, scene, image_size=(512, 512)):
        """
        一个专门用于独立训练 FeatExtNet 的简洁数据集。
        :param root_dir: 数据集的根目录 (例如: .../UAVD4L-LoD/)
        :param split_path: 指向 .json 分割文件的路径
        :param scene: 要加载的场景名 (例如: 'Synthesis' 或 'inTraj')
        :param image_size: 图像要调整到的大小
        """
        super().__init__()
        
        self.root = Path(root_dir)
        self.scene = scene
        self.image_size = image_size

        # 定义图像和真值蒙版的目录
        self.image_dir = self.root / scene / "Query_image"
        self.mask_dir = self.root / scene / "Line_mask" # <-- 使用预渲染的真值图

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # 从 JSON 文件加载图像列表
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        
        # 兼容两种JSON格式 ('train'/'val' 在顶层或内层)
        if 'train' in split_data:
            self.image_names = split_data['train'].get(scene, [])
        else:
            self.image_names = split_data.get(scene, [])

        if not self.image_names:
            raise ValueError(f"No images found for scene '{scene}' in split file '{split_path}'")
            
        print(f"Dataset initialized for scene '{scene}'. Found {len(self.image_names)} images.")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # 加载图像
        img_path = self.image_dir / img_name
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1) # HWC -> CHW

        # 加载对应的线框真值图
        mask_name = os.path.splitext(img_name)[0] + ".png" # 真值图通常是png格式
        mask_path = self.mask_dir / mask_name
        wireframe_gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if wireframe_gt is None:
             raise FileNotFoundError(f"Could not read mask: {mask_path}. Make sure it exists.")
        wireframe_gt = cv2.resize(wireframe_gt, self.image_size, interpolation=cv2.INTER_NEAREST)
        wireframe_gt = (wireframe_gt > 128).astype(np.float32) # 二值化
        wireframe_gt = torch.from_numpy(wireframe_gt).unsqueeze(0) # H,W -> 1,H,W
        
        return {
            'image': image,
            'wireframe_gt': wireframe_gt
        }
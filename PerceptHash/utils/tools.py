import os
import numpy as np
import torch
import multiprocessing
from PIL import Image, ImageFile
from tqdm import tqdm
from torchvision import transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from sklearn.metrics import roc_curve, auc
except ImportError:
    logger.error("scikit-learn not found. ROC/EER will be disabled.")
    roc_curve, auc = lambda *a, **k: (None, None, None), lambda *a, **k: 0.5

# --- Global Settings ---
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Helper Functions for Pre-loading ---
def _load_and_resize_image(path_and_size):
    """
    Worker function that loads an image from a path and immediately resizes it.
    This avoids storing the full-resolution image in the main process memory.
    """
    path, target_size = path_and_size
    try:
        # Load image
        img = Image.open(path).convert('RGB')
        
        # Resize image (using cv2 is generally faster)
        try:
            import cv2
            # Convert PIL to NumPy array and resize
            return cv2.resize(np.array(img), (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # Fallback to PIL resize
            return np.array(img.resize((target_size, target_size)))
            
    except Exception as e:
        logger.warning(f"Failed to load/resize {path}: {e}. Returning placeholder.")
        # Return a black placeholder image of the correct size and type
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

# --- Dataset Classes ---
class SharedMemoryPIHDTripletDataset(Dataset):
    """[Slow Start/Fast Train] PIHD Triplet Dataset using pre-loaded images."""
    def __init__(self, images_np, transform, group_size=97):
        self.images_np = images_np
        self.transform = transform
        self.group_size = group_size
        self.num_groups = len(self.images_np) // self.group_size
        logger.info(f"Initialized SharedMemoryPIHDTripletDataset with {self.num_groups} groups.")

    def __len__(self): return self.num_groups

    def __getitem__(self, index):
        start_idx = index * self.group_size
        pos_offset = np.random.randint(1, 49)
        neg_offset = np.random.randint(49, 97)
        
        anchor_img = self.transform(image=self.images_np[start_idx].copy())['image']
        positive_img = self.transform(image=self.images_np[start_idx + pos_offset].copy())['image']
        negative_img = self.transform(image=self.images_np[start_idx + neg_offset].copy())['image']
        
        return anchor_img, positive_img, negative_img

class ImageList(Dataset):
    def __init__(self, image_list_lines, transform):
        self.imgs = [(line.split()[0], np.array([int(la) for la in line.split()[1:]])) for line in image_list_lines]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.transform(Image.open(path).convert('RGB'))
        return img, target, index

    def __len__(self): return len(self.imgs)

class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.transform(Image.fromarray(img))
        return img, np.eye(10, dtype=np.int8)[target], index

class PIHDDataset(Dataset):
    """
    Dataset for PIHD.
    - 'train' mode: returns triplets (anchor, positive, negative).
    - 'test' mode: returns all distorted/dissimilar images as queries.
    """
    def __init__(self, data_path, transform=None, group_size=97, num_groups=None, mode='train'):
        self.data_path = data_path
        self.transform = transform
        self.group_size = group_size
        self.mode = mode
        
        self.image_files = sorted(os.listdir(data_path))
        
        max_groups = len(self.image_files) // self.group_size
        self.num_groups = min(num_groups, max_groups) if num_groups is not None else max_groups

        if self.mode == 'test':
            # In test mode, we build a list of all query images (both positive and negative)
            self.test_items = []
            for i in range(self.num_groups):
                start_idx = i * self.group_size
                
                # The query images are all images from index 1 to 96 in a group
                # This includes 48 positive samples and 48 negative samples
                for j in range(1, self.group_size):
                    query_path = os.path.join(self.data_path, self.image_files[start_idx + j])
                    # The label is the group_id 'i'
                    self.test_items.append((query_path, i))

    def __len__(self):
        return self.num_groups if self.mode == 'train' else len(self.test_items)

    def _load_and_transform(self, path):
        """Helper to load image and apply transform correctly for Albumentations."""
        img = Image.open(path).convert('RGB')
        if self.transform:
            img_np = np.array(img)
            return self.transform(image=img_np)['image']
        return img

    def __getitem__(self, index):
        if self.mode == 'train':
            start_idx = index * self.group_size
            anchor_path = os.path.join(self.data_path, self.image_files[start_idx])
            pos_path = os.path.join(self.data_path, self.image_files[start_idx + 1 + np.random.randint(48)])
            neg_path = os.path.join(self.data_path, self.image_files[start_idx + 49 + np.random.randint(48)])

            anchor_img = self._load_and_transform(anchor_path)
            positive_img = self._load_and_transform(pos_path)
            negative_img = self._load_and_transform(neg_path)
            return anchor_img, positive_img, negative_img
        
        else: # test mode
            query_path, label = self.test_items[index]
            query_img = self._load_and_transform(query_path)
            return query_img, label, index

class PIHDDatabaseDataset(Dataset):
    """
    Dataset for the PIHD database/gallery (original, non-distorted images).
    """
    def __init__(self, data_path, transform=None, group_size=97, num_groups=None):
        self.data_path = data_path
        self.transform = transform
        
        image_files = sorted(os.listdir(data_path))
        max_groups = len(image_files) // group_size
        self.num_groups = min(num_groups, max_groups) if num_groups is not None else max_groups
            
        self.db_items = [
            (os.path.join(self.data_path, image_files[i * group_size]), i)
            for i in range(self.num_groups)
        ]

    def __len__(self):
        return len(self.db_items)

    def __getitem__(self, index):
        path, label = self.db_items[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img_np = np.array(img)
            img = self.transform(image=img_np)['image']
        return img, label, index

# --- Data Transforms ---
def get_pihd_transforms(config):
    crop_size = config["crop_size"]
    train_transform = A.Compose([
        A.Resize(crop_size, crop_size), A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.6), A.ToGray(p=0.1), A.GaussianBlur(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(crop_size, crop_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, val_transform

def image_transform(resize_size, crop_size, data_set):
    step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)] if data_set == "train_set" else [transforms.CenterCrop(crop_size)]
    return transforms.Compose([
        transforms.Resize(resize_size), *step,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Main Data Loading Functions ---
def get_data(config):
    if "cifar" in config["dataset"]: return cifar_dataset(config)
    if config["dataset"] == "pihd": return pihd_dataset(config)
    
    dset_loaders, dsets_dict = {}, {}
    for phase in ["train_set", "test", "database"]:
        data_cfg = config["data"][phase]
        transform = image_transform(config["resize_size"], config["crop_size"], phase)
        dataset = ImageList(open(data_cfg["list_path"]).readlines(), transform=transform)
        dsets_dict[phase] = dataset
        dset_loaders[phase] = DataLoader(dataset, batch_size=data_cfg["batch_size"], 
                                         shuffle=(phase == "train_set"), num_workers=config.get("num_work", 4))
        logger.info(f"{phase} size: {len(dataset)}")
        
    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets_dict["train_set"]), len(dsets_dict["test"]), len(dsets_dict["database"])

def pihd_dataset(config):
    
    train_transform, val_transform = get_pihd_transforms(config)
    data_path, batch_size, num_workers = config["data_path"], config["batch_size"], config.get("num_work", 4)

    if config.get("preload_data", False):
            logger.info("PIHD: Pre-loading data (Optimized Memory Usage).")
            img_dir = os.path.join(data_path, "train/train_class")
            paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
            
            # --- 【优化点】 ---
            # 创建 (路径, 尺寸) 的任务元组列表
            tasks = [(p, config["resize_size"]) for p in paths]
            
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                # 直接让每个worker加载并缩放图片，只返回缩放后的小尺寸np数组
                images_np = list(tqdm(pool.imap(_load_and_resize_image, tasks, chunksize=100), total=len(tasks), desc="Loading and Resizing"))
            
            # images_np 现在是一个包含所有 uint8 numpy 数组的列表
            # 它的理论内存占用约为 15GB (256x256)，而不是 140GB+
            train_dataset = SharedMemoryPIHDTripletDataset(images_np, train_transform, config.get("group_size", 97))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True, persistent_workers=bool(num_workers > 0))
    else:
        logger.info("PIHD: Loading data from disk (fast start, slow train).")
        train_dataset = PIHDDataset(os.path.join(data_path, "train/train_class"), train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = PIHDDataset(os.path.join(data_path, "val/val_class"), val_transform, mode='test')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    db_dataset = PIHDDatabaseDataset(os.path.join(data_path, "val/val_class"), val_transform)
    db_loader = DataLoader(db_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, db_loader, len(train_dataset), len(val_dataset), len(db_dataset)
def cifar_dataset(config):
    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_root = './datasets/'
    train_set = MyCIFAR10(root=cifar_root, train=True, transform=transform, download=True)
    test_set = MyCIFAR10(root=cifar_root, train=False, transform=transform, download=True)

    X = np.concatenate((train_set.data, test_set.data))
    L = np.concatenate((np.array(train_set.targets), np.array(test_set.targets)))

    # Split dataset
    test_size = 1000 if config["dataset"] == "cifar10-2" else 100
    train_size = 5000 if config["dataset"] == "cifar10-2" else 500
    
    test_idx, train_idx, db_idx = [], [], []
    for label in range(10):
        idx = np.where(L == label)[0]
        perm = np.random.permutation(idx.shape[0])
        idx = idx[perm]
        test_idx.append(idx[:test_size])
        train_idx.append(idx[test_size : train_size + test_size])
        db_idx.append(idx[train_size + test_size:])
    
    test_index, train_index, database_index = np.concatenate(test_idx), np.concatenate(train_idx), np.concatenate(db_idx)

    if config["dataset"] == "cifar10-1": database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2": database_index = train_index

    for dset, index in [(train_set, train_index), (test_set, test_index)]:
        dset.data, dset.targets = X[index], L[index]
    
    db_set = MyCIFAR10(root=cifar_root, train=False, transform=transform)
    db_set.data, db_set.targets = X[database_index], L[database_index]
    
    logger.info(f"CIFAR | Train: {len(train_set)} | Test: {len(test_set)} | DB: {len(db_set)}")

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=3)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=3)
    db_loader = DataLoader(db_set, batch_size=config["batch_size"], shuffle=False, num_workers=3)
    
    return train_loader, test_loader, db_loader, len(train_set), len(test_set), len(db_set)


# --- Evaluation & Metrics ---

def Calc_ROC_EER(qB, qL, qI, rB, rL, group_size=97):
    """计算ROC-AUC和EER指标 - 修复版本"""
    try:
        pos_dists, neg_dists = [], []
        ref_map = {label.item(): code for code, label in zip(rB, rL.squeeze())}
        queries_per_group, pos_per_group = group_size - 1, (group_size - 1) // 2

        for i in tqdm(range(len(qB)), desc="Building ROC pairs", ncols=80):
            query_code, query_label, query_idx = qB[i], qL[i].item(), qI[i].item()
            if query_label in ref_map:
                ref_code = ref_map[query_label]
                hamm_dist = 0.5 * (query_code.shape[0] - torch.dot(query_code, ref_code))
                (pos_dists if query_idx % queries_per_group < pos_per_group else neg_dists).append(hamm_dist.item())
        
        if not pos_dists or not neg_dists: 
            logger.warning("No positive or negative distances found. Returning default values.")
            return 0.5, 0.5  # ROC-AUC, EER

        y_true = np.concatenate([np.ones(len(pos_dists)), np.zeros(len(neg_dists))])
        y_scores = qB.shape[1] - np.array(pos_dists + neg_dists)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        if fpr is None: 
            logger.warning("ROC curve calculation failed. Returning default values.")
            return 0.5, 0.5

        roc_auc_score = auc(fpr, tpr)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer_score = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
        
        # 返回 ROC-AUC, EER（2个值）
        return roc_auc_score, eer_score
        
    except Exception as e:
        logger.error(f"Error in Calc_ROC_EER calculation: {e}")
        return 0.5, 0.5  # ROC-AUC, EER

def validate_pihd(config, bit, best_metric, test_loader, db_loader, net, epoch):
    """PIHD验证函数 - 确保返回正确的值 (mAP, ROC-AUC, EER). 不计算 mAP，只返回 0.0 作为占位。"""
    device = config["device"]
    
    # 确保模型在评估模式
    net.eval()
    
    with torch.no_grad():
        # 获取查询集的哈希码和标签
        query_codes = []
        query_labels = []
        query_indices = []
        
        for batch in test_loader:
            imgs, labels, indices = [x.to(device) for x in batch]
            codes = net(imgs)
            # 将哈希码转换为二进制码
            binary_codes = torch.sign(codes)
            
            query_codes.append(binary_codes.cpu())
            query_labels.append(labels.cpu())
            query_indices.append(indices.cpu())
        
        # 获取数据库的哈希码和标签
        db_codes = []
        db_labels = []
        db_indices = []
        
        for batch in db_loader:
            imgs, labels, indices = [x.to(device) for x in batch]
            codes = net(imgs)
            # 将哈希码转换为二进制码
            binary_codes = torch.sign(codes)
            
            db_codes.append(binary_codes.cpu())
            db_labels.append(labels.cpu())
            db_indices.append(indices.cpu())
    
    # 合并所有批次的结果
    qB = torch.cat(query_codes, dim=0)
    qL = torch.cat(query_labels, dim=0)
    qI = torch.cat(query_indices, dim=0)
    
    rB = torch.cat(db_codes, dim=0)
    rL = torch.cat(db_labels, dim=0)
    rI = torch.cat(db_indices, dim=0)
    
    # 计算ROC和EER
    try:
        # 使用配置中的group_size，而不是访问模型的dataset属性
        group_size = config.get("group_size", 97)
        
        # 现在只计算 ROC-AUC 和 EER，mAP 固定为 0.0（不计算）
        roc_auc, eer = Calc_ROC_EER(qB, qL, qI, rB, rL, group_size)
        # 转换为 Python 原生类型以方便写入及后续比较
        try:
            roc_auc_val = float(roc_auc)
        except Exception:
            roc_auc_val = 0.0
        try:
            eer_val = float(eer)
        except Exception:
            eer_val = 1.0

        mAP_val = 0.0  # mAP 不计算，返回占位
        return mAP_val, roc_auc_val, eer_val
         
    except Exception as e:
        logger.error(f"Error in ROC/EER calculation: {e}")
        # 确保返回 3 个值 (mAP, ROC-AUC, EER)
        return 0.0, 0.0, 1.0

# --- Configuration ---
def config_dataset(config):
    """Configure dataset-specific parameters."""
    dataset_configs = {
        "cifar": {"num_train": 5000, "num_query": 1000, "topK": -1, "n_class": 10},
        "nuswide_21": {"num_train": 10500, "num_query": 2100, "topK": 5000, "n_class": 21, "data_path": "datasets/nuswide_21"},
        "nuswide_21_m": {"num_train": 10500, "num_query": 2100, "topK": 5000, "n_class": 21, "data_path": "datasets/nuswide_81"},
        "nuswide_81_m": {"num_train": 10000, "num_query": 5000, "topK": 5000, "n_class": 81, "data_path": "datasets/nuswide_81"},
        "coco": {"num_train": 10000, "num_query": 5000, "topK": 5000, "n_class": 80, "data_path": "datasets/coco"},
        "imagenet": {"num_train": 13000, "num_query": 5000, "topK": 1000, "n_class": 100, "data_path": "datasets/imagenet"},
        "pihd": {"n_class": -1, "data_path": "datasets/PIHD"},
    }
    for name, params in dataset_configs.items():
        if name in config["dataset"]:
            config.update(params)
            break
            
    if config["dataset"] != "pihd":
        path = config['data_path']
        config["data"] = {
            "train_set": {"list_path": f"{path}/train.txt", "batch_size": config["batch_size"]},
            "database": {"list_path": f"{path}/database.txt", "batch_size": config["batch_size"]},
            "test": {"list_path": f"{path}/test.txt", "batch_size": config["batch_size"]}
        }
    return config
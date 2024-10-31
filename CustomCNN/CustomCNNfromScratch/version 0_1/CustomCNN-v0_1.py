# 2024-10-25
# author: zhl

import argparse
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset


# # 设置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


class MicrobiomeImageDataset(Dataset):
    """
    A custom dataset class for handling microbiome images with age classification.

    This dataset is designed to load and process images of microbiome samples,
    along with their corresponding age labels. It supports age grouping based on
    microbiome developmental characteristics.

    Attributes:
        image_dir (str): Directory containing the image files
        transform (Optional[callable]): Optional transform to be applied to images
        phase (str): Dataset phase ('train' or 'val')
        data (pd.DataFrame): DataFrame containing sample information
        age_bins (List[Tuple[int, int]]): Age range definitions for classification
        class_counts (Dict[int, int]): Distribution of samples across age groups
    """

    def __init__(
            self,
            csv_path: str,
            image_dir: str,
            transform: Optional[callable] = None,
            phase: str = 'train'
    ) -> None:
        """
        Initialize the MicrobiomeImageDataset.

        Args:
            csv_path (str): Path to the CSV file containing sample information
            image_dir (str): Directory containing the image files
            transform (Optional[callable]): Transformations to apply to images
            phase (str): Dataset phase ('train' or 'val')

        Raises:
            ValueError: If CSV file is missing required columns or contains invalid data
            FileNotFoundError: If image files are missing
        """
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.phase = phase

        # Load the CSV data
        self.data = pd.read_csv(csv_path)

        # Define age bins based on microbiome developmental stages
        self.age_bins: List[Tuple[int, int]] = [
            (0, 2),  # Infancy
            (3, 6),  # Early childhood
            (7, 12),  # Childhood
            (13, 17),  # Adolescence
            (18, 35),  # Young adulthood
            (36, 55),  # Middle adulthood
            (56, 70),  # Early elderly
            (71, 110)  # Late elderly
            # (0,18),
            # (19,56),
            # (57,110)
        ]

        # Validate data integrity
        self._validate_data()

        # Calculate class distribution
        self.class_counts: Dict[int, int] = self._calculate_class_distribution()

    def _validate_data(
            self
    ) -> None:
        """
        Validate the integrity and format of the input data.

        Checks:
        1. Presence of required columns in CSV
        2. Validity of age values
        3. Existence of image files

        Raises:
            ValueError: If data validation fails
            FileNotFoundError: If image files are missing
        """
        # Check required columns
        required_columns = ['sampleid', 'age']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")

            # Validate age ranges
        invalid_ages = self.data[~self.data['age'].between(0, 110)]
        if not invalid_ages.empty:
            raise ValueError(f"Invalid age values detected: {invalid_ages['age'].tolist()}")

            # Verify image files existence
        for idx, row in self.data.iterrows():
            img_path = os.path.join(self.image_dir, f"{row['sampleid']}.jpg")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")

    def _calculate_class_distribution(
            self
    ) -> Dict[int, int]:
        """
        Calculate the distribution of samples across age groups.

        Returns:
            Dict[int, int]: Dictionary mapping class indices to sample counts
        """
        class_counts = {i: 0 for i in range(len(self.age_bins))}
        for age in self.data['age']:
            class_idx = self.age_to_class(age)
            class_counts[class_idx] += 1
        return class_counts

    def age_to_class(
            self,
            age: int
    ) -> int:
        """
        Convert age to corresponding class index based on age bins.

        Args:
            age (int): Age value to convert

        Returns:
            int: Class index corresponding to the age
        """
        for i, (low, high) in enumerate(self.age_bins):
            if low <= age <= high:
                return i
        return len(self.age_bins) - 1  # Return last class if age exceeds all bins

    def __len__(
            self
    ) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.data)

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            Tuple[torch.Tensor, int]: Tuple containing:
                - Transformed image tensor
                - Age class label

        Raises:
            IndexError: If index is out of range
            IOError: If image cannot be read
            RuntimeError: If image transformation fails
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

            # Get sample information
        sample_id = self.data.iloc[idx]['sampleid']
        age = self.data.iloc[idx]['age']

        # Construct image path
        img_path = os.path.join(self.image_dir, f"{sample_id}.jpg")

        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Failed to load image {img_path}: {str(e)}")

            # Apply transformations
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                raise RuntimeError(f"Image transformation failed for {img_path}: {str(e)}")

                # Get age class
        age_class = self.age_to_class(age)

        return image, age_class


class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=1.0, alpha_type='dynamic'):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha_type = alpha_type
        self.alpha = None

    def forward(self, inputs, targets):
        """
        inputs: [B, C] where C is the number of classes
        targets: [B] where each value is the target class index
        """
        # 确保输入的类别数量正确
        if inputs.size(1) != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes in input, got {inputs.size(1)}")

            # 计算动态权重
        if self.alpha_type == 'dynamic':
            # 计算当前batch中每个类别的样本数量
            unique_classes, class_counts = torch.unique(targets, return_counts=True)
            total_samples = targets.size(0)

            # 初始化alpha为zeros
            self.alpha = torch.zeros(self.num_classes, device=inputs.device)

            # 只为出现的类别计算权重
            for cls, count in zip(unique_classes, class_counts):
                self.alpha[cls] = 1.0 - (count / total_samples)

                # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算概率
        pt = torch.exp(-ce_loss)

        # 应用focal loss公式
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class DenseBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            growth_rate,
            num_layers
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate))

    def _make_dense_layer(
            self,
            in_channels,
            growth_rate
    ):
        return nn.Sequential(
            nn.GroupNorm(get_groups(in_channels), in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False),
            nn.GroupNorm(get_groups(4 * growth_rate), 4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)
        )

    def forward(
            self,
            x
    ):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(get_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(get_groups(out_channels), out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(get_groups(out_channels), out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(
            self,
            x
    ):
        # 主路径
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        # 短路连接
        identity = self.shortcut(x)

        # 添加残差连接
        out += identity
        out = self.relu(out)

        return out


class MultiScaleAttention(nn.Module):
    def __init__(
            self,
            channels
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Channel Attention
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(),
            nn.Linear(channels // 16, channels)
        )

        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(
            self,
            x
    ):
        # Channel Attention
        avg_pool = self.avg_pool(x).view(x.size(0), -1)
        max_pool = self.max_pool(x).view(x.size(0), -1)

        channel_att = torch.sigmoid(
            self.channel_mlp(avg_pool) + self.channel_mlp(max_pool)
        ).unsqueeze(2).unsqueeze(3)

        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial))

        return x * spatial_att


class HybridNet(nn.Module):
    def __init__(
            self,
            num_classes=8,
            input_channels=3,
            dropout_rate=0.3
    ):
        super().__init__()

        self.dropout_rate = dropout_rate

        self.channels = {
            'initial': 64,
            'after_dense': 64 + 6 * 32,
            'stage1': 128,
            'stage2': 256,
            'stage3': 512
        }

        # Initial convolution block - 修改第一个 GroupNorm
        self.conv1 = nn.Sequential(
            # 对于 input_channels=3，使用 1 组
            nn.GroupNorm(1, input_channels),  # 修改这里
            nn.Conv2d(input_channels, self.channels['initial'],
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(get_groups(self.channels['initial']), self.channels['initial']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense block
        self.dense1 = DenseBlock(self.channels['initial'], growth_rate=32, num_layers=6)

        # Transition layer
        self.trans1 = nn.Sequential(
            nn.GroupNorm(get_groups(self.channels['after_dense']), self.channels['after_dense']),
            nn.Conv2d(self.channels['after_dense'], self.channels['stage1'], 1),
            nn.AvgPool2d(2)
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.channels['stage1'], self.channels['stage1']),
            ResidualBlock(self.channels['stage1'], self.channels['stage2'], stride=2),
            ResidualBlock(self.channels['stage2'], self.channels['stage3'], stride=2)
        ])

        # Multi-scale attention
        self.attention = MultiScaleAttention(self.channels['stage3'])

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # FPN
        fpn_out_channels = 256
        self.fpn = nn.ModuleDict({
            'lateral_1': nn.Conv2d(self.channels['stage1'], fpn_out_channels, 1),
            'lateral_2': nn.Conv2d(self.channels['stage2'], fpn_out_channels, 1),
            'lateral_3': nn.Conv2d(self.channels['stage3'], fpn_out_channels, 1),
            'output_1': nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1),
            'output_2': nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1),
            'output_3': nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)
        })

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.channels['stage3'], 512),
            nn.GroupNorm(get_groups(512), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.GroupNorm(get_groups(256), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def forward(
            self,
            x
    ):
        features = {}

        # Initial convolution
        x = self.conv1(x)
        features['conv1'] = x

        # Dense block
        x = self.dense1(x)
        features['dense'] = x
        x = self.trans1(x)
        features['trans1'] = x

        # Residual blocks
        for i, block in enumerate(self.res_blocks):
            x = block(x)
            features[f'res_{i + 1}'] = x

            # FPN processing
        c1, c2, c3 = features['trans1'], features['res_2'], features['res_3']

        # Lateral connections
        lateral_3 = self.fpn['lateral_3'](c3)
        lateral_2 = self.fpn['lateral_2'](c2)
        lateral_1 = self.fpn['lateral_1'](c1)

        # Top-down pathway
        p3 = self.fpn['output_3'](lateral_3)
        p2 = self.fpn['output_2'](lateral_2 + self._upsample_add(lateral_3, lateral_2))
        p1 = self.fpn['output_1'](lateral_1 + self._upsample_add(p2, lateral_1))

        # Attention mechanism
        x = self.attention(x)

        # Global pooling and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _upsample_add(
            self,
            x,
            y
    ):
        return F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False) + y

    def _initialize_weights(
            self
    ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):  # 更新初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def get_groups(
        channels: int
) -> int:
    """
    根据通道数动态确定 GroupNorm 的组数，确保通道数能被组数整除
    """
    if channels <= 8:
        return 1  # 当通道数很少时，使用1组
    elif channels <= 16:
        return 2
    elif channels <= 32:
        return 4
    elif channels <= 64:
        return 8
    elif channels <= 128:
        return 16
    else:
        return 32


class VisualizationUtils:
    def __init__(
            self,
            save_dir
    ):
        self.save_dir = save_dir
        # 确保保存目录存在
        os.makedirs(os.path.join(save_dir, 'confusion_matrices'), exist_ok=True)

    def plot_confusion_matrix(
            self,
            y_true,
            y_pred,
            epoch,
            phase='val'
    ):
        """
        绘制并保存混淆矩阵
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        # 计算归一化的混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 创建图形
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')

        # 设置标签
        age_groups = ['0-2', '3-6', '7-12', '13-17', '18-35', '36-55', '56-70', '71-100']
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Normalized Confusion Matrix - {phase} (Epoch {epoch})')

        # 保存图形
        save_path = os.path.join(
            self.save_dir,
            'confusion_matrices',
            f'confusion_matrix_{phase}_epoch_{epoch}.png'
        )
        plt.savefig(save_path)
        plt.close()

        return cm_normalized


class CurriculumTrainer:
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
            num_epochs,
            focal_loss_params=None
    ):
        # 基本属性初始化
        if focal_loss_params is None:
            focal_loss_params = {'gamma': 1.0, 'alpha_type': 'dynamic'}
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.focal_loss_params = focal_loss_params

        # 初始化课程学习阶段
        self.curriculum_stages = [
            {'focus_groups': [0, 7], 'epochs': 50},
            {'focus_groups': [0, 1, 6, 7], 'epochs': 70},
            {'focus_groups': [0, 1, 2, 5, 6, 7], 'epochs': 90},
            {'focus_groups': [0, 1, 2, 3, 4, 5, 6, 7], 'epochs': 190},
        ]

        # 设置保存目录和日志
        self.save_dir = os.path.join('checkpoints', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_dir = os.path.join(self.save_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self._setup_logging()

        # 初始化指标记录
        self.metrics = {
            'train_loss': [], 'train_acc': [], 'train_recall': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_recall': [], 'val_f1': [],
            'learning_rates': []
        }

        # 更新模型
        self.update_model_for_stage(self.curriculum_stages[0]['focus_groups'])

    def update_model_for_stage(self, focus_groups):
        """更新模型输出层和相关组件"""
        try:
            num_classes = len(focus_groups)
            self.logger.info(f"Updating model for {num_classes} classes: {focus_groups}")

            if not hasattr(self.model, 'classifier'):
                raise AttributeError("Model does not have 'classifier' layer")

            # 获取分类器结构
            classifier_layers = list(self.model.classifier)

            # 找到最后一个Linear层
            last_linear_idx = None
            in_features = None
            for i, layer in reversed(list(enumerate(classifier_layers))):
                if isinstance(layer, nn.Linear):
                    last_linear_idx = i
                    in_features = layer.in_features
                    break

            if last_linear_idx is None:
                raise ValueError("No Linear layer found in classifier")

            # 保存当前权重状态
            if hasattr(self, 'previous_classifier_state'):
                old_linear = classifier_layers[last_linear_idx]
                old_weights = old_linear.weight.data.clone()
                old_bias = old_linear.bias.data.clone()
                old_num_classes = old_weights.size(0)

            # 创建新的输出层
            new_final_layer = nn.Linear(in_features, num_classes).to(self.device)

            # 如果存在之前的权重，尝试迁移相关权重
            if hasattr(self, 'previous_classifier_state'):
                with torch.no_grad():
                    for new_idx, old_class in enumerate(focus_groups):
                        if old_class < old_num_classes:
                            new_final_layer.weight.data[new_idx] = old_weights[old_class]
                            new_final_layer.bias.data[new_idx] = old_bias[old_class]

            # 保存当前状态
            self.previous_classifier_state = {
                'weight': new_final_layer.weight.data.clone(),
                'bias': new_final_layer.bias.data.clone()
            }

            # 更新分类器
            classifier_layers[last_linear_idx] = new_final_layer
            self.model.classifier = nn.Sequential(*classifier_layers)

            # 更新优化器
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.optimizer.param_groups[0]['lr'],
                weight_decay=self.optimizer.param_groups[0]['weight_decay']
            )

            # 更新损失函数
            self.criterion = FocalLoss(
                num_classes=num_classes,
                gamma=self.focal_loss_params['gamma'],
                alpha_type=self.focal_loss_params['alpha_type']
            ).to(self.device)

            # 验证模型输出
            self.model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                output = self.model(dummy_input)

                if output.size(1) != num_classes:
                    raise ValueError(
                        f"Model output dimension ({output.size(1)}) "
                        f"does not match expected number of classes ({num_classes})"
                    )

            # 更新标签映射
            self.current_focus_groups = focus_groups
            self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(focus_groups)}

            self.logger.info(f"Successfully updated model for {num_classes} classes")
            self.logger.info(f"Label mapping: {self.label_mapping}")

        except Exception as e:
            self.logger.error(f"Error in update_model_for_stage: {str(e)}")
            self.logger.error(f"Focus groups: {focus_groups}")
            self.logger.error(f"Model structure: {self.model}")
            raise e

    def _setup_logging(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('CurriculumTrainer')
        self.logger.setLevel(logging.INFO)

        # 确保处理器不重复添加
        if not self.logger.handlers:
            # 文件处理器
            fh = logging.FileHandler(os.path.join(self.log_dir, 'training.log'))
            fh.setLevel(logging.INFO)

            # 控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # 格式化器
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            # 添加处理器
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def map_labels(self, labels):
        """将原始标签映射到当前阶段的标签"""
        return torch.tensor([self.label_mapping[label.item()]
                             if label.item() in self.label_mapping
                             else -1
                             for label in labels]).to(self.device)

    def train(self):
        """训练模型的主循环"""
        try:
            current_stage = 0
            total_epochs = 0

            for stage in self.curriculum_stages:
                self.logger.info(f"\nStarting curriculum stage {current_stage + 1}")
                self.logger.info(f"Focus groups: {stage['focus_groups']}")

                # 更新模型以适应当前阶段
                self.update_model_for_stage(stage['focus_groups'])

                # 训练当前阶段
                for epoch in range(stage['epochs']):
                    total_epochs += 1
                    self.logger.info(f"\nEpoch {total_epochs}/{self.num_epochs}")

                    # 训练和验证
                    train_metrics = self._train_epoch()
                    val_metrics = self._validate()

                    # 更新学习率
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # 记录指标
                    self._update_metrics(train_metrics, val_metrics)

                    # 保存检查点
                    self._save_checkpoint(total_epochs, val_metrics)

                current_stage += 1

            self.logger.info("Training completed!")

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise e

    def _train_epoch(self):
        """执行一个训练周期"""
        self.model.train()
        metrics = {
            'loss': 0.0,
            'correct': 0,
            'total': 0,
            'predictions': [],
            'true_labels': []
        }

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            try:
                # 将数据移到设备上
                data = data.to(self.device)
                mapped_labels = self.map_labels(labels)

                # 过滤掉不在当前阶段的样本
                valid_mask = mapped_labels != -1
                if valid_mask.sum() == 0:
                    continue

                data = data[valid_mask]
                mapped_labels = mapped_labels[valid_mask]

                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, mapped_labels)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                metrics['total'] += mapped_labels.size(0)
                metrics['correct'] += (predicted == mapped_labels).sum().item()
                metrics['loss'] += loss.item()

                # 收集预测结果用于计算其他指标
                metrics['predictions'].extend(predicted.cpu().numpy())
                metrics['true_labels'].extend(mapped_labels.cpu().numpy())

                # 打印进度
                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f'Train Batch [{batch_idx + 1}/{len(self.train_loader)}] '
                                     f'Loss: {loss.item():.4f} '
                                     f'Acc: {100.0 * metrics["correct"] / metrics["total"]:.2f}%')

            except Exception as e:
                self.logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                raise e

                # 计算平均指标
        metrics['loss'] /= len(self.train_loader)
        metrics['accuracy'] = 100.0 * metrics['correct'] / metrics['total']

        # 计算其他指标
        predictions = np.array(metrics['predictions'])
        true_labels = np.array(metrics['true_labels'])
        metrics['recall'] = recall_score(true_labels, predictions, average='macro')
        metrics['f1'] = f1_score(true_labels, predictions, average='macro')

        self.logger.info(f"\nTraining Epoch Summary:")
        self.logger.info(f"Average Loss: {metrics['loss']:.4f}")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.2f}%")
        self.logger.info(f"Recall: {metrics['recall']:.4f}")
        self.logger.info(f"F1 Score: {metrics['f1']:.4f}")

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, stage_num, phase='validation'):
        """
        绘制并保存混淆矩阵

        参数:
        - y_true: 真实标签
        - y_pred: 预测标签
        - stage_num: 当前课程学习阶段
        - phase: 训练阶段('training' 或 'validation')
        """
        try:
            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred)

            # 计算百分比
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            # 创建图形
            plt.figure(figsize=(10, 8))

            # 使用seaborn绘制热力图
            sns.heatmap(
                cm_percent,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                square=True,
                xticklabels=self.current_focus_groups,
                yticklabels=self.current_focus_groups
            )

            # 设置标题和标签
            plt.title(f'Confusion Matrix - Stage {stage_num} ({phase})\n'
                      f'Classes: {self.current_focus_groups}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # 保存图形
            save_path = os.path.join(
                self.save_dir,
                f'confusion_matrix_stage{stage_num}_{phase}.png'
            )
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

            # 记录日志
            self.logger.info(f"Confusion matrix saved to {save_path}")

            # 计算每个类别的指标
            class_metrics = {}
            for i, class_label in enumerate(self.current_focus_groups):
                metrics = {
                    'Precision': cm[i, i] / cm[:, i].sum() * 100 if cm[:, i].sum() != 0 else 0,
                    'Recall': cm[i, i] / cm[i, :].sum() * 100 if cm[i, :].sum() != 0 else 0
                }
                metrics['F1'] = 2 * (metrics['Precision'] * metrics['Recall']) / \
                                (metrics['Precision'] + metrics['Recall']) \
                    if (metrics['Precision'] + metrics['Recall']) != 0 else 0
                class_metrics[class_label] = metrics

            # 记录每个类别的指标
            self.logger.info(f"\nClass-wise metrics for Stage {stage_num} ({phase}):")
            for class_label, metrics in class_metrics.items():
                self.logger.info(f"\nClass {class_label}:")
                for metric_name, value in metrics.items():
                    self.logger.info(f"{metric_name}: {value:.2f}%")

            return cm, class_metrics

        except Exception as e:
            self.logger.error(f"Error in plot_confusion_matrix: {str(e)}")
            raise e

    def _validate(self):
        """执行验证"""
        self.model.eval()
        metrics = {
            'loss': 0.0,
            'correct': 0,
            'total': 0,
            'predictions': [],
            'true_labels': []
        }

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.val_loader):
                try:
                    # 将数据移到设备上
                    data = data.to(self.device)
                    mapped_labels = self.map_labels(labels)

                    # 过滤掉不在当前阶段的样本
                    valid_mask = mapped_labels != -1
                    if valid_mask.sum() == 0:
                        continue

                    data = data[valid_mask]
                    mapped_labels = mapped_labels[valid_mask]

                    # 前向传播
                    outputs = self.model(data)
                    loss = self.criterion(outputs, mapped_labels)

                    # 计算准确率
                    _, predicted = torch.max(outputs.data, 1)
                    metrics['total'] += mapped_labels.size(0)
                    metrics['correct'] += (predicted == mapped_labels).sum().item()
                    metrics['loss'] += loss.item()

                    # 收集预测结果
                    metrics['predictions'].extend(predicted.cpu().numpy())
                    metrics['true_labels'].extend(mapped_labels.cpu().numpy())

                except Exception as e:
                    self.logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    raise e

                    # 计算平均指标
        metrics['loss'] /= len(self.val_loader)
        metrics['accuracy'] = 100.0 * metrics['correct'] / metrics['total']

        # 计算其他指标
        predictions = np.array(metrics['predictions'])
        true_labels = np.array(metrics['true_labels'])
        metrics['recall'] = recall_score(true_labels, predictions, average='macro')
        metrics['f1'] = f1_score(true_labels, predictions, average='macro')

        # 绘制混淆矩阵
        try:
            # 计算混淆矩阵
            cm = confusion_matrix(true_labels, predictions)

            # 计算百分比
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            # 创建图形
            plt.figure(figsize=(10, 8))

            # 使用seaborn绘制热力图
            sns.heatmap(
                cm_percent,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                square=True,
                xticklabels=self.current_focus_groups,
                yticklabels=self.current_focus_groups
            )

            # 设置标题和标签
            current_epoch = len(self.metrics['train_loss'])
            plt.title(f'Confusion Matrix - Epoch {current_epoch}\n'
                      f'Classes: {self.current_focus_groups}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # 保存图形
            save_path = os.path.join(
                self.save_dir,
                f'confusion_matrix_epoch{current_epoch}.png'
            )
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

            # 计算每个类别的指标
            class_metrics = {}
            for i, class_label in enumerate(self.current_focus_groups):
                metrics[f'class_{class_label}'] = {
                    'precision': cm[i, i] / cm[:, i].sum() * 100 if cm[:, i].sum() != 0 else 0,
                    'recall': cm[i, i] / cm[i, :].sum() * 100 if cm[i, :].sum() != 0 else 0
                }
                metrics[f'class_{class_label}']['f1'] = 2 * (
                        metrics[f'class_{class_label}']['precision'] *
                        metrics[f'class_{class_label}']['recall']
                ) / (
                                                                metrics[f'class_{class_label}']['precision'] +
                                                                metrics[f'class_{class_label}']['recall']
                                                        ) if (
                                                                     metrics[f'class_{class_label}']['precision'] +
                                                                     metrics[f'class_{class_label}']['recall']
                                                             ) != 0 else 0

            # 记录日志
            self.logger.info(f"\nValidation Summary:")
            self.logger.info(f"Average Loss: {metrics['loss']:.4f}")
            self.logger.info(f"Accuracy: {metrics['accuracy']:.2f}%")
            self.logger.info(f"Recall: {metrics['recall']:.4f}")
            self.logger.info(f"F1 Score: {metrics['f1']:.4f}")

            # 记录每个类别的指标
            self.logger.info("\nPer-class metrics:")
            for class_label in self.current_focus_groups:
                self.logger.info(f"\nClass {class_label}:")
                self.logger.info(f"Precision: {metrics[f'class_{class_label}']['precision']:.2f}%")
                self.logger.info(f"Recall: {metrics[f'class_{class_label}']['recall']:.2f}%")
                self.logger.info(f"F1: {metrics[f'class_{class_label}']['f1']:.2f}%")

        except Exception as e:
            self.logger.error(f"Error in confusion matrix plotting: {str(e)}")

        return metrics

    def _update_metrics(self, train_metrics, val_metrics):
        """更新指标记录"""
        self.metrics['train_loss'].append(train_metrics['loss'])
        self.metrics['train_acc'].append(train_metrics['accuracy'])
        self.metrics['train_recall'].append(train_metrics['recall'])
        self.metrics['train_f1'].append(train_metrics['f1'])

        self.metrics['val_loss'].append(val_metrics['loss'])
        self.metrics['val_acc'].append(val_metrics['accuracy'])
        self.metrics['val_recall'].append(val_metrics['recall'])
        self.metrics['val_f1'].append(val_metrics['f1'])

        if self.scheduler is not None:
            self.metrics['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )

    def _save_checkpoint(self, epoch, val_metrics):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': self.metrics,
            'val_metrics': val_metrics,
            'current_focus_groups': self.current_focus_groups,
            'label_mapping': self.label_mapping
        }

        # 保存最新的检查点
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

        # 如果是最佳验证准确率，保存最佳检查点
        if not hasattr(self, 'best_val_acc') or val_metrics['accuracy'] > self.best_val_acc:
            self.best_val_acc = val_metrics['accuracy']
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved new best checkpoint with validation accuracy: {self.best_val_acc:.2f}%")

    def _plot_training_curves(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        ax1.plot(self.metrics['train_loss'], label='Train Loss')
        ax1.plot(self.metrics['val_loss'], label='Val Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # 准确率曲线
        ax2.plot(self.metrics['train_acc'], label='Train Acc')
        ax2.plot(self.metrics['val_acc'], label='Val Acc')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()

        # 召回率曲线
        ax3.plot(self.metrics['train_recall'], label='Train Recall')
        ax3.plot(self.metrics['val_recall'], label='Val Recall')
        ax3.set_title('Recall Curves')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Recall (%)')
        ax3.legend()

        # F1分数曲线
        ax4.plot(self.metrics['train_f1'], label='Train F1')
        ax4.plot(self.metrics['val_f1'], label='Val F1')
        ax4.set_title('F1 Score Curves')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1 Score (%)')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        plt.close()


def get_transforms(phase):
    """
    获取图像转换函数
    Args:
        phase: 'train' 或 'val'
    Returns:
        transforms.Compose: 图像转换pipeline
    """
    # 基础转换，训练和验证都需要
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]

    if phase == 'train':
        # 训练时的数据增强
        train_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            )
        ]
        return transforms.Compose(train_transforms + base_transforms)

    return transforms.Compose(base_transforms)


def get_training_config(model, dataset, args):
    """
    获取训练配置
    Args:
        model: 模型实例
        dataset: 训练数据集
        args: 训练参数
    Returns:
        dict: 包含训练所需的criterion、optimizer和scheduler
    """
    # 计算类别权重
    class_counts = dataset.class_counts
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)

    # 计算每个类别的权重
    class_weights = torch.FloatTensor([
        total_samples / (num_classes * count)
        for count in class_counts.values()
    ]).to(args.device)

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=True
    )

    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.t0,
        T_mult=args.t_mult,
        eta_min=args.min_lr
    )

    return {
        'optimizer': optimizer,
        'scheduler': scheduler,
        'focal_loss_params': {
            'gamma': 1.0,
            'alpha_type': 'dynamic'
        }
    }


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train age prediction model')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to the CSV file containing data')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    parser.add_argument('--t0', type=int, default=10,
                        help='T0 parameter for cosine annealing')
    parser.add_argument('--t_mult', type=int, default=2,
                        help='T_mult parameter for cosine annealing')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置设备
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")

    # Create training dataset
    # This dataset will be used for model training with data augmentation
    train_dataset = MicrobiomeImageDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        transform=get_transforms('train'),  # Apply training-specific transformations
        phase='train'
    )

    # Create validation dataset
    # This dataset will be used for model evaluation without data augmentation
    val_dataset = MicrobiomeImageDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        transform=get_transforms('val'),  # Apply validation-specific transformations
        phase='val'
    )

    # Handle class imbalance using weighted sampling
    # Get the count of samples for each class
    class_counts: Dict[int, int] = train_dataset.class_counts

    # Calculate weights for each sample
    # Weight = 1 / (number of samples in the class)
    # This gives higher weights to underrepresented classes
    weights: List[float] = [
        1.0 / class_counts[train_dataset.age_to_class(age)]
        for age in train_dataset.data['age']
    ]

    # Create a weighted sampler for balanced training
    # num_samples = len(weights) ensures we use all training samples in each epoch
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True  # Sample with replacement to ensure balanced classes
    )

    # Create training DataLoader with balanced sampling
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Use weighted sampler for balanced class distribution
        num_workers=args.num_workers,  # Number of subprocesses for data loading
        pin_memory=True,  # Enables faster data transfer to CUDA devices
        drop_last=True  # Drop the last incomplete batch to ensure consistent batch sizes
    )

    # Create validation DataLoader with original distribution
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No shuffling for validation to maintain reproducibility
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # Drop the last incomplete batch for consistency
    )

    # 初始化模型
    model = HybridNet(num_classes=8).to(args.device)

    # 获取训练配置
    training_config = get_training_config(model, train_dataset, args)

    # 创建训练器并开始训练
    trainer = CurriculumTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        # criterion=training_config['criterion'],
        optimizer=training_config['optimizer'],
        scheduler=training_config['scheduler'],
        device=args.device,
        num_epochs=args.num_epochs,
        focal_loss_params=training_config['focal_loss_params']
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        # 保存最终模型和训练状态
        trainer._save_model(epoch=-1, val_acc=0, final=True)
        trainer._save_metrics()


if __name__ == '__main__':
    main()

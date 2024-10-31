# Customized CNN

## Version 0.1 
### Model Architecture
Dense Block + Residual Block + Attention
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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
```
### Results
finish at 2024-10-31
1. 400 epochs
   1. 3 stages. [0,7] -> [0,1,6,7] -> [0,1,2,5,6,7] -> [0,1,2,3,4,5,6,7]
2. Batch Size = 128, Learning Rate = 1e-4, Gamma = 1.0 (Focal Loss)
3. Performances:
   1. Training Epoch Summary:
      1. 2024-10-31 11:28:41,157 - CurriculumTrainer - INFO - Average Loss: 0.5352
      2. 2024-10-31 11:28:41,157 - CurriculumTrainer - INFO - Accuracy: 65.58%
      3. 2024-10-31 11:28:41,157 - CurriculumTrainer - INFO - Recall: 0.6556
      4. 2024-10-31 11:28:41,157 - CurriculumTrainer - INFO - F1 Score: 0.6416
       
   2. Validation Summary:
      1. 2024-10-31 11:30:03,025 - CurriculumTrainer - INFO - Average Loss: 0.5447
      2. 2024-10-31 11:30:03,025 - CurriculumTrainer - INFO - Accuracy: 52.23%
      3. 2024-10-31 11:30:03,025 - CurriculumTrainer - INFO - Recall: 0.7065
      4. 2024-10-31 11:30:03,025 - CurriculumTrainer - INFO - F1 Score: 0.6195
   3. Class 0:
      1. 2024-10-31 11:30:03,025 - CurriculumTrainer - INFO - Precision: 73.85%
      2. 2024-10-31 11:30:03,025 - CurriculumTrainer - INFO - Recall: 96.62%
      3. 2024-10-31 11:30:03,025 - CurriculumTrainer - INFO - F1: 83.72%
      
   4. Class 1:
      1. 2024-10-31 11:30:03,025 - CurriculumTrainer - INFO - Precision: 68.35%
      2. 2024-10-31 11:30:03,025 - CurriculumTrainer - INFO - Recall: 84.31%
      3. 2024-10-31 11:30:03,025 - CurriculumTrainer - INFO - F1: 75.50% 
   5. Class 2:
      1. 2024-10-31 11:30:03,025 - CurriculumTrainer - INFO - Precision: 64.42%
      2. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - Recall: 90.82%
      3. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - F1: 75.37%
   6. Class 3:
      1. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - Precision: 69.42%
      2. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - Recall: 98.44%
      3. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - F1: 81.42%
   7. Class 4:
      1. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - Precision: 43.82%
      2. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - Recall: 45.84%
      3. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - F1: 44.81%
   8. Class 5:
      1. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - Precision: 60.44%
      2. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - Recall: 36.32%
      3. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - F1: 45.37%
   9. Class 6:
      1. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - Precision: 36.86%
      2. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - Recall: 37.46%
      3. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - F1: 37.16%
   10. Class 7:
       1. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - Precision: 39.97%
       2. 2024-10-31 11:30:03,026 - CurriculumTrainer - INFO - Recall: 75.37%
       3. 2024-10-31 11:30:03,027 - CurriculumTrainer - INFO - F1: 52.23%

### Analysis 
1. The performances of class 4,5,6 are very bad. However, we have a lot of data of class 4,5,6 in the whole dataset, so I doubt that the weighted sampler has reduced the weight of data of class 4,5,6 because the weighted sampler calculates the weight of each class by the reciprocal of their frequency.
2. The learning process is actually very slow, but the good new is that we are less much probable having overfitting on validation dataset.
3. The curriculum learning is not easy determining the right stages to learn and process.
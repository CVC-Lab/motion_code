import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConvolutionalNetwork(nn.Module):
    def __init__(self, input_channels):
        super(FullyConvolutionalNetwork, self).__init__()
        # Apply 1x1 convolutions without sharing information between channels
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=1, groups=input_channels)
        # self.n1 = nn.BatchNorm2d(input_channels)
        self.n1 = nn.GroupNorm(num_groups=input_channels, num_channels=input_channels)  # GroupNorm after first conv layer
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=1, groups=input_channels)
        # self.n2 = nn.BatchNorm2d(input_channels)
        self.n2 = nn.GroupNorm(num_groups=input_channels, num_channels=input_channels)  # GroupNorm after second conv layer
        self.relu2 = nn.ReLU()
        # Note: No need for bias if followed by BatchNorm or similar. Consider your specific needs.

    def forward(self, x):
        x = self.conv1(x)
        x = self.n1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.n2(x)
        x = self.relu2(x)
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # Flatten to get BxC
        x = torch.flatten(x, start_dim=1)
        return x

class TemporalFusionFCN(nn.Module):
    def __init__(self, input_channels, num_classes=5):
        super(TemporalFusionFCN, self).__init__()
        self.fcn = FullyConvolutionalNetwork(input_channels)
        # The fully connected layer for temporal fusion and final prediction
        self.fc = nn.Linear(input_channels, num_classes)
        # self.fc1 = nn.Linear(input_channels, 128)  # First FC layer
        # self.relu1 = nn.ReLU()  # ReLU activation
        # self.fc2 = nn.Linear(128, 64)  # Second FC layer
        # self.relu2 = nn.ReLU()  # ReLU activation
        # self.fc3 = nn.Linear(64, num_classes)  # Final FC layer to predict class scores

        

    def forward(self, x):
        # Assuming x is a list of tensors for each time step
        # batch_size = x[0].size(0)
        # Process each time step
        # temporal_features = [self.fcn(time_step) for time_step in x]
        # # Aggregate features from all time steps
        # # Here we simply concatenate, but consider more complex fusion techniques
        # aggregated = torch.cat(temporal_features, dim=1)
        temporal_features = self.fcn(x)
        # x = self.relu1(self.fc1(temporal_features))
        # x = self.relu2(self.fc2(x))
        # scores = self.fc3(x)
        # Predict class scores
        scores = self.fc(temporal_features)
        return scores

# # Example usage
# input_channels = 3  # Adjust based on your needs
# model = TemporalFusionFCN(input_channels)

# # Example input tensor for a single time step
# # Replace with actual data
# T = 10
# B, H, W, C = 4, 176, 54, 3  # Example dimensions
# # x = [torch.randn(B, C, H, W) for _ in range(T)]  # Assuming T time steps
# x = torch.randn(B, C, H, W)
# output = model(x)
# print(output.shape)  # Should be [B, 5] for the class scores
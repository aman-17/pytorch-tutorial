"""
Question 2: Dynamic Neural Network with Variable Depth (Hard)

Implement a neural network that can dynamically change its depth during forward pass based on
the input data. The network should have a "confidence threshold" - if the prediction confidence
is above the threshold, it exits early; otherwise, it continues through more layers.

Requirements:
1. Start with 3 layers minimum, can go up to 6 layers maximum
2. After each layer (starting from layer 3), check if max probability > threshold
3. If yes, return prediction; if no, continue to next layer
4. Track which layer made the final prediction

Example usage should produce different exit layers for different inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDepthNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, confidence_threshold=0.8):
        super(DynamicDepthNetwork, self).__init__()
        
        # Initialize layers (6 layers total)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.layer6 = nn.Linear(hidden_size, hidden_size)
        
        # Output classifiers for each potential exit point
        self.classifier3 = nn.Linear(hidden_size, num_classes)
        self.classifier4 = nn.Linear(hidden_size, num_classes)
        self.classifier5 = nn.Linear(hidden_size, num_classes)
        self.classifier6 = nn.Linear(hidden_size, num_classes)
        
        # Store confidence threshold
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
    
    def forward(self, x, return_exit_layer=False):
        # Pass through first 3 layers (mandatory)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        
        # Check confidence after layer 3
        logits3 = self.classifier3(x)
        confidence3 = self.get_confidence(logits3)
        
        if confidence3 > self.confidence_threshold:
            if return_exit_layer:
                return logits3, 3
            return logits3
        
        # Continue to layer 4
        x = F.relu(self.layer4(x))
        logits4 = self.classifier4(x)
        confidence4 = self.get_confidence(logits4)
        
        if confidence4 > self.confidence_threshold:
            if return_exit_layer:
                return logits4, 4
            return logits4
        
        # Continue to layer 5
        x = F.relu(self.layer5(x))
        logits5 = self.classifier5(x)
        confidence5 = self.get_confidence(logits5)
        
        if confidence5 > self.confidence_threshold:
            if return_exit_layer:
                return logits5, 5
            return logits5
        
        # Final layer 6 (always return result)
        x = F.relu(self.layer6(x))
        logits6 = self.classifier6(x)
        
        if return_exit_layer:
            return logits6, 6
        return logits6
    
    def get_confidence(self, logits):
        # Calculate confidence as max probability after softmax
        probs = F.softmax(logits, dim=1)
        max_confidence = torch.max(probs, dim=1)[0]
        return torch.max(max_confidence).item()

# Test your implementation
if __name__ == "__main__":
    # Create network
    net = DynamicDepthNetwork(input_size=784, hidden_size=128, num_classes=10, confidence_threshold=0.9)
    
    # Test with different inputs
    batch_size = 4
    x1 = torch.randn(batch_size, 784)  # Random input
    x2 = torch.ones(batch_size, 784) * 0.5  # More predictable input
    
    with torch.no_grad():
        output1, exit_layer1 = net(x1, return_exit_layer=True)
        output2, exit_layer2 = net(x2, return_exit_layer=True)
        
        print(f"Input 1 - Exit layer: {exit_layer1}, Max confidence: {torch.max(F.softmax(output1, dim=1)):.3f}")
        print(f"Input 2 - Exit layer: {exit_layer2}, Max confidence: {torch.max(F.softmax(output2, dim=1)):.3f}")
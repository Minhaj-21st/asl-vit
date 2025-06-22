import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast

# Device configuration: Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations with data augmentation
# - Resize images to 224x224
# - Apply random horizontal flips and random rotations for augmentation
# - Normalize the images with mean and std suitable for the pretrained model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalization as per model requirement
])

# Load the dataset from a directory with images organized in subfolders by class
data_dir = '/lustre/fs1/home/cap5415.student28/ASL'  # replace with your dataset path
train_data = datasets.ImageFolder(data_dir, transform=transform)

# Split the dataset into training and validation subsets (80% training, 20% validation)
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Shuffle for better training
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Retrieve class names and the number of classes
class_names = train_data.classes
num_classes = len(class_names)
print("Classes:", class_names)

# Compute class weights to handle class imbalance
# Class weights are calculated based on the dataset distribution and used in the loss function
class_weights = compute_class_weight(class_weight='balanced', classes=range(num_classes), y=train_data.targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Load a pretrained Vision Transformer model from Hugging Face
# Modify the model for the classification task with the given number of classes
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=num_classes)
model = model.to(device)

# Freeze the transformer encoder layers initially for fine-tuning only the classifier head
for param in model.vit.encoder.parameters():
    param.requires_grad = False

# Define the loss function (CrossEntropyLoss with class weights) and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

# Learning rate scheduler: Reduce LR by half every 2 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# Mixed precision scaler for efficient training
scaler = GradScaler()

# Training function
def train(model, train_loader):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Forward pass with mixed precision
        with autocast():
            outputs = model(images).logits
            loss = criterion(outputs, labels)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)  # Accumulate loss

    return running_loss / len(train_loader.dataset)  # Return average loss

# Validation function
def evaluate(model, val_loader):
    model.eval()  # Set model to evaluation mode
    y_true, y_pred = [], []

    with torch.no_grad():  # No gradient calculation during validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, preds = torch.max(outputs, 1)  # Get the class with max probability
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred

# Train the model for a specified number of epochs
num_epochs = 7
for epoch in range(num_epochs):
    train_loss = train(model, train_loader)  # Perform training for one epoch
    scheduler.step()  # Update learning rate

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")

    # Unfreeze all layers after 5 epochs to fine-tune the entire model
    if epoch == 4:
        for param in model.vit.encoder.parameters():
            param.requires_grad = True

# Evaluate the model on the validation set
y_true, y_pred = evaluate(model, val_loader)

# Calculate and print evaluation metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average='macro'))
print("Recall:", recall_score(y_true, y_pred, average='macro'))
print("F1 Score:", f1_score(y_true, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

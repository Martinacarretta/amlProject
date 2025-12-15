import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights

import seaborn as sns
from sklearn.metrics import confusion_matrix

''' 
############################################################
############################################################
############################################################
CONFIGURATION
############################################################
############################################################
############################################################
'''
projection_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
BATCH_SIZE = 64

model_path = "hierarchical_sports_model_20pct_penal_1p0.pth"

output_dir = "Testing_results"
os.makedirs(output_dir, exist_ok=True)
print(f"Plots will be saved to: {os.path.abspath(output_dir)}")

superclassDict = {
    'precision & target sports': 0,
    'water': 1,
    'field & team ball sports': 2,
    'court ball sports': 3,
    'combat & strength sports': 4,
    'equestrian & animal sports': 5,
    'ice & snow sports': 6,
    'motor & wheel racing': 7,
    'gymnastics': 8,
    'aerial': 9,
    'track & field athletics': 10
}

def get_class_dict():
    # Merging all dicts into one block to save space
    return {
        "air hockey": 1001, "archery": 1002, "axe throwing": 1003, "billiards": 1004, "bowling": 1005, "croquet": 1006, "curling": 1007, "disc golf": 1008, "golf": 1009, "horseshoe pitching": 1010, "shuffleboard": 1011,
        "canoe slamon": 2001, "fly fishing": 2002, "hydroplane racing": 2003, "log rolling": 2004, "rowing": 2005, "sailboat racing": 2006, "surfing": 2007, "swimming": 2008, "water cycling": 2009, "water polo": 2010,
        "ampute football": 3001, "baseball": 3002, "cricket": 3003, "field hockey": 3004, "football": 3005, "frisbee": 3006, "gaga": 3007, "lacrosse": 3008, "roller derby": 3009, "rugby": 3010, "ultimate": 3011,
        "basketball": 4001, "jai alai": 4002, "tennis": 4003, "volleyball": 4004, "wheelchair basketball": 4005, "table tennis": 4006,
        "arm wrestling": 5001, "boxing": 5002, "fencing": 5003, "jousting": 5004, "judo": 5005, "olympic wrestling": 5006, "rock climbing": 5007, "steer wrestling": 5008, "sumo wrestling": 5009, "tug of war": 5010, "weightlifting": 5011,
        "barell racing": 6001, "bull riding": 6002, "chuckwagon racing": 6003, "harness racing": 6004, "horse jumping": 6005, "horse racing": 6006, "polo": 6007,
        "bobsled": 7001, "figure skating men": 7002, "figure skating women": 7003, "figure skating pairs": 7004, "giant slalom": 7005, "hockey": 7006, "ice climbing": 7007, "ice yachting": 7008, "luge": 7009, "mushing": 7010, "ski jumping": 7011, "snow boarding": 7012, "snowmobile racing": 7013, "speed skating": 7014,
        "bike polo": 8001, "bmx": 8002, "formula 1 racing": 8003, "motorcycle racing": 8004, "nascar racing": 8005, "rollerblade racing": 8006, "sidecar racing": 8007, "track bicycle": 8008, "wheelchair racing": 8009,
        "balance beam": 9001, "baton twirling": 9002, "cheerleading": 9003, "parallel bar": 9004, "pole dancing": 9005, "pommel horse": 9006, "rings": 9007, "trapeze": 9008, "uneven bars": 9009,
        "bungee jumping": 10001, "hang gliding": 10002, "sky surfing": 10003, "skydiving": 10004, "wingsuit flying": 10005,
        "hammer throw": 11001, "high jump": 11002, "hurdles": 11003, "javelin": 11004, "pole climbing": 11005, "pole vault": 11006, "shot put": 11007
    }
    
classDict = get_class_dict()
INVERSE_classDict = {v: k for k, v in classDict.items()}

''' 
############################################################
############################################################
############################################################
MODEL DEFINITION & DATA LOADERS
############################################################
############################################################
############################################################
'''

class HierarchicalModel(nn.Module): # CAMBIO!
    def __init__(self, backbone, projection, head):
        super().__init__()
        self.backbone = backbone
        self.projection = projection
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        feats = self.projection(x)       # Shape: [Batch, 128]
        super_logits = self.head(feats)  # Shape: [Batch, 11]
        return feats, super_logits

class ClassifierHead(nn.Module):
    """single-layer linear task head for few-shot classification"""
    def __init__(self,
                 num_classes: int, # number of output neurons for classification
                 input_size: int = projection_size, # size of the feature vector from the backbone
                 dropout: bool = True, # important to avoid overfitting on tiny classification tasks
                ):
        super().__init__()

        self.fc = nn.Linear(input_size, num_classes)
        self.relu = nn.ReLU()
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = nn.Identity()

        self.to(device)

    def forward(self, x):
        # assume x is already unactivated feature logits, so we activate it first
        x = self.fc(self.relu(self.dropout(x)))

        return x
    
class ClassifierHeadSmall(nn.Module):
    def __init__(self, num_classes, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.to(device)

    def forward(self, x):
        return self.fc(x)  # no ReLU, no dropout

class SportsTestDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # Map strings to IDs immediately
        self.data["super_idx"] = self.data["superclass"].map(superclassDict)
        self.data["class_idx"] = self.data["class"].map(classDict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        image = self.transform(image)
        return image, row["super_idx"], row["class_idx"]

def get_test_loader(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SportsTestDataset(path, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
''' 
############################################################
############################################################
############################################################
TESTING SCRIPT
############################################################
############################################################
############################################################
'''

test_loader = get_test_loader(r"test.csv")


prev = 0
count = 1
heads = []
subClassIdx = {}

for idx, label in enumerate(list(classDict.values())):
    if prev != label // 1000:
        if prev != 0:
            heads.append(ClassifierHeadSmall(count, input_size = projection_size)) # CAMBIO! de 11 a projection size
            # heads.append(ClassifierHead(count, input_size = 11, dropout=False))
            count = 1
        prev = label // 1000
    else:
        count+=1
    subClassIdx[label] = idx

heads.append(ClassifierHeadSmall(count, input_size = projection_size)) # CAMBIO! lo mismo

backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # CAMBIO!
backbone.fc = nn.Identity()

projection_size = 128
num_classes = 11

projection_layer = nn.Sequential(nn.Linear(512, projection_size), nn.ReLU())
head = ClassifierHead(num_classes=num_classes, input_size=projection_size) # CAMBIO! de input_seize = 11 a projection size

model = HierarchicalModel(backbone, projection_layer, head).to(device)

# Load the checkpoint
checkpoint = torch.load(model_path, map_location=device)
# Load weights
model.load_state_dict(checkpoint['backbone_state_dict'])

if len(heads) != len(checkpoint['heads_state_dict']):
    print(f"WARNING: Mismatch in heads count! Created: {len(heads)}, Saved: {len(checkpoint['heads_state_dict'])}")
else:
    print(f"Loading weights for {len(heads)} subclass heads...")
    for i, h in enumerate(heads):
        h.load_state_dict(checkpoint['heads_state_dict'][i])
        h.to(device)
        h.eval()

model.eval()
print("Model AND subclass heads loaded successfully!")

def test(model, val_loader, heads):
    model.eval()

    ce_loss = nn.CrossEntropyLoss()

    total_super_loss = 0.0
    total_sub_loss = 0.0

    total_super_correct = 0
    total_sub_correct = 0

    total_samples = 0

    # Initialize lists to store images, true labels, and predicted labels
    all_images = []
    all_true_labels = [] # Will store original class codes (e.g., 1001)
    all_predicted_labels = [] # Will store global indices (e.g., 0, 1, ...)


    with torch.no_grad():
        for i, (x_lab, y_super_lab, y_lab) in enumerate(val_loader):
            if i%5 == 0:
                print(f"Batch {i}/{len(val_loader)}")

            x_lab = x_lab.to(device)
            y_super_lab = y_super_lab.to(device)
            y_lab = y_lab.to(device)

            # --- forward ---
            feats, y_super_pred = model(x_lab)
            y_sub_pred = forward_head(feats, y_super_pred.argmax(dim=1), heads)

            # convertir labels de subclase al índice global
            y_lab_np = y_lab.cpu().numpy()
            y_lab_global_idx = torch.tensor([subClassIdx[int(g)] for g in y_lab_np], dtype=torch.long, device=device)

            # --- losses ---
            loss_super = ce_loss(y_super_pred, y_super_lab)
            loss_sub = ce_loss(y_sub_pred, y_lab_global_idx)

            total_super_loss += loss_super.item() * x_lab.size(0)
            total_sub_loss += loss_sub.item() * x_lab.size(0)

            # --- accuracies ---
            super_pred_labels = y_super_pred.argmax(dim=1)
            sub_pred_labels = y_sub_pred.argmax(dim=1)

            total_super_correct += (super_pred_labels == y_super_lab).sum().item()
            total_sub_correct += (sub_pred_labels == y_lab_global_idx).sum().item()

            total_samples += x_lab.size(0)

            # Store images, true labels, and predicted labels
            all_images.append(x_lab.cpu())
            all_true_labels.append(y_lab.cpu()) # Original class codes
            all_predicted_labels.append(sub_pred_labels.cpu()) # Global indices

    avg_super_loss = total_super_loss / total_samples
    avg_sub_loss = total_sub_loss / total_samples

    acc_super = total_super_correct / total_samples
    acc_sub = total_sub_correct / total_samples

    print({"super_loss": avg_super_loss, "sub_loss": avg_sub_loss, "super_acc": acc_super, "sub_acc": acc_sub})

    return {
        "super_loss": avg_super_loss,
        "sub_loss": avg_sub_loss,
        "super_acc": acc_super,
        "sub_acc": acc_sub
    }, all_images, all_true_labels, all_predicted_labels

def forward_head(features, y, heads):
    device = features.device # CAMBIO!
    B = features.size(0) # CAMBIO!

    num_global_classes = max(subClassIdx.values()) + 1

    # inicializamos con -inf para que CE/KL funcione
    full_logits = torch.full((B, num_global_classes), -50.0, device=device)

    for super_idx, head in enumerate(heads):

        mask = (y == super_idx)
        if mask.sum() == 0:
            continue

        batch_idxs = mask.nonzero(as_tuple=False).squeeze(1)  # (b_i,)

        feats = features[batch_idxs] # CAMBIO!
        local_logits = head(feats) # CAMBIO!

        C_local = local_logits.size(1)

        # local class ids = 1..C_local
        local_class_ids = torch.arange(1, C_local + 1, device=device)

        # códigos: 1000*(super_idx+1) + local
        codes = (super_idx + 1) * 1000 + local_class_ids

        # Convertimos a global ids
        global_ids = torch.tensor(
            [subClassIdx[int(c)] for c in codes.cpu().numpy()],
            device=device
        )   # (C_local,)

        full_logits[
            batch_idxs[:, None],   # (b_i, 1)
            global_ids[None, :]    # (1, C_local)
        ] = local_logits           # (b_i, C_local)

    return full_logits

inverse_classDict = {v: k for k, v in classDict.items()}
inverse_subClassIdx = {v: k for k, v in subClassIdx.items()}

metrics, all_images_batches, all_true_labels_batches, all_predicted_labels_batches = test(model, test_loader, heads)
print(metrics)

''' 
############################################################
############################################################
############################################################
PLOTTING SCRIPT
############################################################
############################################################
############################################################
'''


all_images = torch.cat(all_images_batches)
all_true_labels = torch.cat(all_true_labels_batches)
all_predicted_labels = torch.cat(all_predicted_labels_batches)

# Function to denormalize and display image
def imshow(img_tensor, title=None):
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    
num_samples_to_display = 20 # Display first 20 samples
cols = 4  # Number of columns in the grid
rows = math.ceil(num_samples_to_display / cols)

for plot_id in range(1, 3):
    start_idx = (plot_id - 1) * num_samples_to_display
    end_idx = start_idx + num_samples_to_display
        
    plt.figure(figsize=(20, 5 * rows))
    print(f"\n--- Displaying {num_samples_to_display} Samples in a {rows}x{cols} Grid ---")

    for i in range(num_samples_to_display):
        current_idx = start_idx + i
        image_tensor = all_images[current_idx]
        true_label_code = int(all_true_labels[current_idx])
        predicted_label_global_idx = int(all_predicted_labels[current_idx])

        # Map true label code to name
        true_label_name = inverse_classDict.get(true_label_code, "Unknown True Label")

        # Map predicted global index to class code, then to name
        predicted_label_code = inverse_subClassIdx.get(predicted_label_global_idx, None)
        if predicted_label_code is not None:
            predicted_label_name = inverse_classDict.get(predicted_label_code, "Unknown Predicted Label")
        else:
            predicted_label_name = "Unknown Predicted Label Index"

        is_correct = (true_label_code == predicted_label_code)
        title_color = 'green' if is_correct else 'red'

        plt.subplot(rows, cols, i+1)
        imshow(image_tensor)
        plt.title(f"T: {true_label_name}\nP: {predicted_label_name}", 
                color=title_color, 
                fontsize=10, 
                fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"sample_predictions_{plot_id}.png")
    plt.savefig(save_path)
    plt.close() # Close memory to free RAM
    print(f"Saved grid to {save_path}")

''' 
############################################################
############################################################
############################################################
############################################################
CONFUSION MATRIX SCRIPT
############################################################
############################################################
############################################################
'''

# Ensure tensors are on CPU and numpy
y_true = all_true_labels.cpu().numpy().astype(int)
y_pred_global = all_predicted_labels.cpu().numpy().astype(int)

# Convert Predicted Global Indices -> Predicted Subclass Codes
inverse_subClassIdx = {v: k for k, v in subClassIdx.items()}
y_pred = np.array([inverse_subClassIdx[idx] for idx in y_pred_global])

def get_super_idx(code):
    return (code // 1000) - 1

y_true_super = np.array([get_super_idx(c) for c in y_true])
y_pred_super = np.array([get_super_idx(c) for c in y_pred])

# We sort superclassDict by value to get names in order [0, 1, 2...]
sorted_super_names = sorted(superclassDict, key=superclassDict.get)
inverse_classDict = {v: k for k, v in classDict.items()}

def plot_cm(true, pred, calc_labels, display_labels, title, filename, figsize=(10, 8), color_map='Blues', rotation = 45, pad_title=20):

    cm = confusion_matrix(true, pred, labels=calc_labels)

    plt.figure(figsize=figsize)

    ax = sns.heatmap(cm, 
                     annot=True, fmt='d', 
                     cmap=color_map, 
                     xticklabels=display_labels, yticklabels=display_labels)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')

    plt.title(title, pad=pad_title, fontsize=16, fontweight='bold')

    plt.xticks(rotation=rotation, ha='left')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")


# ==========================================
# PART A: Superclass Confusion Matrix
# ==========================================
print("Generating Superclass Confusion Matrix...")
plot_cm(y_true_super, y_pred_super, 
        calc_labels=range(len(sorted_super_names)), # 0, 1, 2...
        display_labels=sorted_super_names, # names
        title="Superclass Confusion Matrix (Aggregated)",
        filename=os.path.join(output_dir, "cm_superclass.png"),
        figsize=(12, 10))


# ==========================================
# PART B: Subclass Confusion Matrices
# ==========================================
print("\nGenerating Subclass Matrices per Superclass...")

inverse_classDict = {v: k for k, v in classDict.items()}

for super_id in range(11):
    super_name = sorted_super_names[super_id]
    
    # Get only samples belonging to this TRUE Superclass
    mask = (y_true_super == super_id)
    
    if np.sum(mask) == 0:
        print(f"Skipping {super_name} (No samples found in validation set)")
        continue

    # Filtered True and Pred codes
    local_y_true = y_true[mask]
    local_y_pred = y_pred[mask]
    
    # Get the list of sport names belonging to this superclass
    # We filter classDict for codes starting with the correct digit
    local_sports_codes = [c for c in classDict.values() if get_super_idx(c) == super_id]
    local_sports_names = [inverse_classDict[c] for c in sorted(local_sports_codes)]
    local_sports_codes_sorted = sorted(local_sports_codes)
        
    clean_name = super_name.replace(" ", "_").replace("&", "and")
    save_path = os.path.join(output_dir, f"cm_subclass_{clean_name}.png")
    
    plot_cm(local_y_true, local_y_pred, 
            calc_labels=local_sports_codes_sorted,
            display_labels=local_sports_names,
            title=f"Confusion Matrix: {super_name}",
            filename=save_path,
            figsize=(10, 8),
            color_map='Greens',
            pad_title=40)
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from torchvision.transforms import v2  # For RandAugment / newer API
from torchvision.models import ResNet18_Weights
from torchinfo import summary

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

projection_size = 128 

BATCH_SIZE = 64

# SUBCLASS CLASSIFICATION HEADS
class ClassifierHeadSmall(nn.Module):
    def __init__(self, num_classes, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.to(device)

    def forward(self, x):
        return self.fc(x)  # no ReLU, no dropout

# SUPERCLASS CLASSIFICATION HEAD (11 classes)
class ClassifierHead(nn.Module):
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

def loaders(csv_path, ratio_labelled = 0.80, ratio_unlabelled = 0.20):
    # ============================
    #   DICCIONARIOS DE SUBCLASES
    # ============================
    superclass_1 = {
        "air hockey": 1001, "archery": 1002, "axe throwing": 1003, "billiards": 1004,
        "bowling": 1005, "croquet": 1006, "curling": 1007, "disc golf": 1008,
        "golf": 1009, "horseshoe pitching": 1010, "shuffleboard": 1011
    }
    superclass_2 = {
        "canoe slamon": 2001, "fly fishing": 2002, "hydroplane racing": 2003, "log rolling": 2004,
        "rowing": 2005, "sailboat racing": 2006, "surfing": 2007, "swimming": 2008,
        "water cycling": 2009, "water polo": 2010
    }
    superclass_3 = {
        "ampute football": 3001, "baseball": 3002, "cricket": 3003, "field hockey": 3004,
        "football": 3005, "frisbee": 3006, "gaga": 3007, "lacrosse": 3008,
        "roller derby": 3009, "rugby": 3010, "ultimate": 3011
    }
    superclass_4 = {
        "basketball": 4001, "jai alai": 4002, "tennis": 4003, "volleyball": 4004,
        "wheelchair basketball": 4005, "table tennis": 4006
    }
    superclass_5 = {
        "arm wrestling": 5001, "boxing": 5002, "fencing": 5003, "jousting": 5004,
        "judo": 5005, "olympic wrestling": 5006, "rock climbing": 5007,
        "steer wrestling": 5008, "sumo wrestling": 5009, "tug of war": 5010,
        "weightlifting": 5011
    }
    superclass_6 = {
        "barell racing": 6001, "bull riding": 6002, "chuckwagon racing": 6003,
        "harness racing": 6004, "horse jumping": 6005, "horse racing": 6006,
        "polo": 6007
    }
    superclass_7 = {
        "bobsled": 7001, "figure skating men": 7002, "figure skating women": 7003,
        "figure skating pairs": 7004, "giant slalom": 7005, "hockey": 7006,
        "ice climbing": 7007, "ice yachting": 7008, "luge": 7009,
        "mushing": 7010, "ski jumping": 7011, "snow boarding": 7012,
        "snowmobile racing": 7013, "speed skating": 7014
    }
    superclass_8 = {
        "bike polo": 8001, "bmx": 8002, "formula 1 racing": 8003,
        "motorcycle racing": 8004, "nascar racing": 8005, "rollerblade racing": 8006,
        "sidecar racing": 8007, "track bicycle": 8008, "wheelchair racing": 8009
    }
    superclass_9 = {
        "balance beam": 9001, "baton twirling": 9002, "cheerleading": 9003,
        "parallel bar": 9004, "pole dancing": 9005, "pommel horse": 9006,
        "rings": 9007, "trapeze": 9008, "uneven bars": 9009
    }
    superclass_10 = {
        "bungee jumping": 10001, "hang gliding": 10002, "sky surfing": 10003,
        "skydiving": 10004, "wingsuit flying": 10005
    }
    superclass_11 = {
        "hammer throw": 11001, "high jump": 11002, "hurdles": 11003, "javelin": 11004,
        "pole climbing": 11005, "pole vault": 11006, "shot put": 11007
    }

    # Combinamos todos en un solo dict para classDictF
    classDict = {}
    for d in [superclass_1, superclass_2, superclass_3, superclass_4,
            superclass_5, superclass_6, superclass_7, superclass_8,
            superclass_9, superclass_10, superclass_11]:
        classDict.update(d)

    # Ejemplo de superclases 
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

    # ============================
    #   SportsDataset
    # ============================
    class SportsDataset(Dataset):
        def __init__(self, csv_file, mode="general", transform=None,
                    superclassDict=None, classDict=None):
            self.data = pd.read_csv(csv_file)
            self.mode = mode
            self.transform = transform
            self.superclassDict = superclassDict
            self.classDict = classDict
            # Mapeamos las superclases a índices
            self.data["super_idx"] = self.data["superclass"].apply(lambda x: superclassDict[x])

            if mode == "labelled":
                self.data = self.data[self.data["class"] != "unknown"].reset_index(drop=True)
            elif mode == "unlabelled":
                self.data = self.data[self.data["class"] == "unknown"].reset_index(drop=True)
            # subclass indices
            if mode == "labelled":
                self.data["class_idx"] = self.data["class"].apply(lambda x: classDict[x])

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            image = Image.open(row["filepath"]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            super_idx = row["super_idx"]
            # for warmup and unlabelled, return only image and super_idx
            if self.mode in ["general", "unlabelled"]:
                return image, super_idx
            # for labelled, return image, super_idx and class_idx
            else:
                class_idx = row["class_idx"]
                return image, super_idx, class_idx

    # ============================
    #   TRANSFORMACIONES
    # ============================
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # CAMBIO!
    ])
    

    # ============================
    #   CREACIÓN DE DATASETS
    # ============================
    general_dataset = SportsDataset(
        csv_file=csv_path,
        mode="general",
        transform=transform,
        superclassDict=superclassDict,
        classDict=classDict
    )

    labelled_dataset = SportsDataset(
        csv_file=csv_path,
        mode="labelled",
        transform=transform,
        superclassDict=superclassDict,
        classDict=classDict
    )

    if ratio_unlabelled > 0.0:
        unlabelled_dataset = SportsDataset(
            csv_file=csv_path,
            mode="unlabelled",
            transform=transform,
            superclassDict=superclassDict,
            classDict=classDict
        )

    # ============================
    #   BALANCED BATCH RATIO
    # ============================
    labelled_batch_size = max(1, int(BATCH_SIZE * ratio_labelled))
    unlabelled_batch_size = max(1, int(BATCH_SIZE * ratio_unlabelled))

    # ============================
    #   CREACIÓN DE DATALOADERS
    # ============================
    general_loader = DataLoader(
        general_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    labelled_loader = DataLoader(
        labelled_dataset,
        batch_size=labelled_batch_size,
        shuffle=True,
        drop_last=True
    )

    if ratio_unlabelled > 0.0:
        unlabelled_loader = DataLoader(
            unlabelled_dataset,
            batch_size=unlabelled_batch_size,
            shuffle=True,
            drop_last=True
        )
    else: 
        unlabelled_loader = None

    return general_loader, labelled_loader, unlabelled_loader, classDict

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

def train_semi_supervised(
        model: nn.Module,
        general_loader: torch.utils.data.DataLoader,
        labelled_loader: torch.utils.data.DataLoader,
        unlabelled_loader: torch.utils.data.DataLoader,
        heads: list,
        num_epochs=10,
        warmup_epochs=5,
        lr=0.01,
        penal_weight=1.0,
        optimizer_type='SGD',
        momentum=0.9
    ):

    # ====================
    # Optimizer
    # ====================
    params = list(model.parameters())
    for h in heads:
        params.extend(list(h.parameters()))
        
    if optimizer_type.upper() == 'SGD':
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum)
    else:
        opt = torch.optim.Adam(params, lr=lr)

    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

    torch.manual_seed(42)

    losses_super = []
    losses_sub = []

    accs_super = []
    accs_sub = []

    if warmup_epochs == 0.0:
        warmup_step = 0
    for epoch in range(num_epochs + warmup_epochs):
        model.train()
        loss_super_history = []
        loss_sub_history = []

        acc_super_history = []
        acc_sub_history = []

        if epoch < warmup_epochs:
            # ====================
            # WARMUP: solo general_loader
            # ====================
            for batch_id, (x, y_super) in enumerate(general_loader):
                x, y_super = x.to(device), y_super.to(device)

                opt.zero_grad()
                _, y_super_pred = model.forward(x)

                loss = ce_loss(y_super_pred, y_super)
                loss.backward()
                opt.step()

                warmup_step = epoch * len(general_loader) + batch_id

                wandb.log({
                    "WARMUP superclass loss": loss.item(),
                    "WARMUP superclass accuracy": (y_super_pred.argmax(dim=1) == y_super).sum().item() / x.size(0)
                }, step=warmup_step)

                loss_super_history.append(loss.item())

                acc_super_history.append((y_super_pred.argmax(dim=1) == y_super).sum().item() / x.size(0))

                if (batch_id+1) % 50 == 0:
                    print(f"Warmup Epoch {epoch+1}/{warmup_epochs} | Batch {batch_id+1}/{len(general_loader)} | Loss: {np.mean(loss_super_history[-50:]):.4f} | Acc: {np.mean(acc_super_history[-50:]):.4f}")

            losses_super.append(np.mean(loss_super_history))

            accs_super.append(np.mean(acc_super_history))

            print(f"Epoch {epoch+1} completed | Avg Loss: {losses_super[-1]:.4f} | Avg Super Acc: {accs_super[-1]:.4f}")

        else:
            # ====================
            # Semi-supervised training
            # ====================
            steps = min(len(labelled_loader), len(unlabelled_loader))
            # iterate over labelled and unlabelled loaders
            for batch_id, ((x_lab, y_super_lab, y_lab), (x_unlab, y_super_unlab)) in enumerate(zip(labelled_loader, unlabelled_loader)):

                x_lab, y_super_lab, y_lab = x_lab.to(device), y_super_lab.to(device), y_lab.to(device)
                x_unlab, y_super_unlab = x_unlab.to(device), y_super_unlab.to(device)

                opt.zero_grad()

                # ---- Forward labelled ----
                feats_lab, y_super_pred_lab = model(x_lab) # CAMBIO!
                ### gt superclass routing
                y_sub_pred_lab = forward_head(feats_lab, y_super_lab, heads) # CAMBIO!

                y_lab_np = y_lab.cpu().numpy()
                y_lab_global_idx = torch.tensor([subClassIdx[int(g)] for g in y_lab_np], dtype=torch.long, device=device)

                loss_super = ce_loss(y_super_pred_lab, y_super_lab)
                loss_labelled = ce_loss(y_sub_pred_lab, y_lab_global_idx)

                # ---- Forward unlabelled ----
                feats_unlab, y_super_pred_unlab = model(x_unlab) # CAMBIO!
                y_sub_pred_unlab = forward_head(feats_unlab, y_super_unlab, heads) # CAMBIO!
                # Augmented unlabelled for UDA
                x_unlab_aug = torch.stack([randAugment(T.ToPILImage()(img.cpu())) for img in x_unlab]).to(device)
                feats_aug,y_super_pred_aug = model(x_unlab_aug) # CAMBIO!
                y_sub_pred_aug = forward_head(feats_aug, y_super_unlab, heads)

                loss_unlabelled = kl_loss(F.log_softmax(y_sub_pred_unlab, dim=1), F.log_softmax(y_sub_pred_aug, dim=1))

                # penality if superclass prediction is wrong
                penal = penal_weight * loss_super * ((y_super_pred_lab.argmax(dim=1) != y_super_lab).float()).mean()
                loss = loss_super + loss_labelled + loss_unlabelled + penal
                loss.backward()
                opt.step()

                loss_super_history.append(loss_super.item())
                loss_sub_history.append(loss_labelled.item() + loss_unlabelled.item())

                acc_super_history.append(((y_super_pred_lab.argmax(dim=1) == y_super_lab).sum().item() + (y_super_pred_unlab.argmax(dim=1) == y_super_unlab).sum().item()) / (x_lab.size(0) + x_unlab.size(0)))

                acc_sub_history.append((y_sub_pred_lab.argmax(dim=1) == y_lab_global_idx).sum().item() / x_lab.size(0))

                if (batch_id + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs+warmup_epochs} | Batch {batch_id+1}/{steps} | Super Loss: {np.mean(loss_super_history[-50:]):.4f} | Sub Loss: {np.mean(loss_sub_history[-50:]):.4f} | Super Acc: {np.mean(acc_super_history[-50:]):.4f} | Sub Acc: {np.mean(acc_sub_history[-50:]):.4f}")
                
                wandb.log({
                    "epoch": epoch, 
                    "batch": batch_id,
                    "loss superclass": loss_super.item(),
                    "loss labelled": loss_labelled.item(),
                    "loss unlabelled": loss_unlabelled.item(),
                    "penalization": penal.item(),
                    "total loss": loss.item(), 
                    "accuracy superclass": acc_super_history[-1],
                    "accuracy subclass": acc_sub_history[-1]
                }, step=warmup_step + epoch * steps + batch_id)
            
            losses_super.append(np.mean(loss_super_history))
            losses_sub.append(np.mean(loss_sub_history))

            accs_super.append(np.mean(acc_super_history))
            accs_sub.append(np.mean(acc_sub_history))

            wandb.log({
                "EPOCH loss superclass ": losses_super[-1],
                "EPOCH loss subclass ": losses_sub[-1],
                "EPOCH accuracy superclass ": accs_super[-1],
                "EPOCH accuracy subclass ": accs_sub[-1]
            }, step=warmup_epochs + (epoch+1) * steps)

            print(f"Epoch {epoch+1} completed | Avg Super Loss: {losses_super[-1]:.4f} | Avg Sub Loss: {losses_sub[-1]:.4f} | Avg Super Acc: {accs_super[-1]:.4f} | Avg Sub Acc: {accs_sub[-1]:.4f}")
        
    return losses_super, losses_sub, accs_super, accs_sub

def eval(model, val_loader, heads):
    model.eval()

    ce_loss = nn.CrossEntropyLoss()

    total_super_loss = 0.0
    total_sub_loss = 0.0

    total_super_correct = 0
    total_sub_correct = 0

    total_samples = 0

    with torch.no_grad():
        for x_lab, y_super_lab, y_lab in val_loader:

            x_lab = x_lab.to(device)
            y_super_lab = y_super_lab.to(device)
            y_lab = y_lab.to(device)

            # --- forward ---
            feats, y_super_pred = model(x_lab) # CAMBIO!
            ### predicted superclass head used to forward to subclass heads
            y_sub_pred = forward_head(feats, y_super_pred.argmax(dim=1), heads) # CAMBIO!

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

    avg_super_loss = total_super_loss / total_samples
    avg_sub_loss = total_sub_loss / total_samples

    acc_super = total_super_correct / total_samples
    acc_sub = total_sub_correct / total_samples

    print({"super_loss": avg_super_loss, "sub_loss": avg_sub_loss, "super_acc": acc_super, "sub_acc": acc_sub})

    wandb.log({
        "VAL loss superclass": avg_super_loss,
        "VAL loss subclass": avg_sub_loss,
        "VAL accuracy superclass": acc_super,
        "VAL accuracy subclass": acc_sub
    }, step=wandb.run.step)

    return {
        "super_loss": avg_super_loss,
        "sub_loss": avg_sub_loss,
        "super_acc": acc_super,
        "sub_acc": acc_sub
    }

# PRETRAINED ResNet18 backbone or not
backbone = models.resnet18(weights=None)# ResNet18_Weights.IMAGENET1K_V1) # CAMBIO!
backbone.fc = nn.Identity()

num_classes = 11 # superclasses
ratio = 20 #unlabelled ratio in percentage
path_csv = f"unlabelled_train_{ratio}pct.csv"
ratio_unlabelled = ratio /100.0
ratio_labelled = 1.0 - ratio_unlabelled
print(f"unlabelled ratio: {ratio_unlabelled}, labelled ratio: {ratio_labelled}")

num_epochs = 50
warmup_epochs = 0
lr = 0.001 #0.0003
penalisation_weight = 0.0

projection_layer = nn.Sequential(nn.Linear(512, projection_size), nn.ReLU())

head = ClassifierHead(num_classes=num_classes, input_size=projection_size) # CAMBIO! de input_seize = 11 a projection size

model_components = nn.Sequential(backbone, projection_layer, head)
model = HierarchicalModel(backbone, projection_layer, head).to(device) # CAMBIO!

general_loader, labelled_loader, unlabelled_loader, classDict = loaders(path_csv, ratio_labelled = ratio_labelled, ratio_unlabelled = ratio_unlabelled)

_, val_loader, _, _ = loaders(r"val.csv", ratio_labelled = 1.0, ratio_unlabelled = 0.0)
_, test_loader, _, _ = loaders(r"test.csv", ratio_labelled = 1.0, ratio_unlabelled = 0.0)

prev = 0
count = 1
heads = []
subClassIdx = {}

for idx, label in enumerate(list(classDict.values())):
    if prev != label // 1000:
        if prev != 0:
            # TODO:
            heads.append(ClassifierHeadSmall(count, input_size = projection_size)) # CAMBIO! de 11 a projection size
            # heads.append(ClassifierHead(count, input_size = 11, dropout=False))
            count = 1
        prev = label // 1000
    
    else:
        count+=1

    subClassIdx[label] = idx

# TODO: classifier head small enlloc de classifier head
heads.append(ClassifierHeadSmall(count, input_size = projection_size)) # CAMBIO! lo mismo
# heads.append(ClassifierHead(count, input_size = 11, dropout=False))

# RandAugment transform for UDA
randAugment = v2.Compose([
            v2.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.5),
            v2.RandomApply([v2.GaussianBlur(kernel_size=int(0.1 * 224 + 1)),], p=0.5),
            v2.RandomGrayscale(p=0.2),
            v2.ToTensor(), 
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # CAMBIO!
        ])


run = wandb.init(
    project="AML sports classification",
    config={
        "ratio_unlabelled": ratio_unlabelled,
        "projection size": projection_size, 
        "batch size": BATCH_SIZE,
        "num classes": num_classes,
        "epochs": num_epochs, 
        "warmup epochs": warmup_epochs, 
        "learning rate": lr,
        "penalisation weight": penalisation_weight
    },
    sync_tensorboard=True,
    save_code=True,
)

losses_super, losses_sub, accs_super, accs_sub = train_semi_supervised(model, general_loader, labelled_loader, unlabelled_loader, heads, num_epochs=num_epochs, warmup_epochs=warmup_epochs, lr = lr, penal_weight=penalisation_weight)

metrics = eval(model, val_loader, heads)
print(metrics)


# ============================
#   SAVE MODEL
# ============================
str_weight = str(penalisation_weight).replace('.', 'p')

path_to_save = f"Baseline.pth"

print("Saving model...")
torch.save({
    'backbone_state_dict': model.state_dict(),           # Saves ResNet + Projection + SuperHead
    'heads_state_dict': [h.state_dict() for h in heads], # Saves all the sub-class heads
    'class_dict': classDict                              # Useful to remember the label mappings
}, path_to_save)

print(f"Model saved to {path_to_save}")

wandb.save(path_to_save)
wandb.finish()
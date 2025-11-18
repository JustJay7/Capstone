import os
import yaml
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from ultralytics import YOLO
from utils.dataset_utils import CrackDataset
from models.gcc3 import GCC3
from models.sgam import SGAM
from models.coord_attn import CoordAttention
from ultralytics.nn.modules import C2f
from ultralytics.nn.tasks import v8DetectionLoss

# --- Wrapper for Custom Blocks ---
class WrappedModule(nn.Module):
    def __init__(self, module, i, f):
        super().__init__()
        self.module = module
        self.i = i
        self.f = f

    def forward(self, x):
        return self.module(x)

# --- Config loader ---
def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# --- Dataset Collate ---
def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, labels

# --- Model Patcher ---
def inject_custom_blocks(model):
    backbone = model.model
    print("🔧 Injecting GCC3, SGAM, and CoordAttention blocks (optimized for cracks)...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inject_indices = [6, 8]
    spatial_indices = [6]
    attention_indices = [4]

    for idx in inject_indices:
        if isinstance(backbone[idx], C2f):
            ch_in = backbone[idx].cv1.conv.in_channels
            backbone[idx] = WrappedModule(GCC3(ch_in).to(device), i=idx, f=backbone[idx].f)

    for idx in spatial_indices:
        if isinstance(backbone[idx], C2f):
            ch_in = backbone[idx].cv1.conv.in_channels
            backbone[idx] = WrappedModule(SGAM(ch_in).to(device), i=idx, f=backbone[idx].f)

    for idx in attention_indices:
        if isinstance(backbone[idx], C2f):
            ch_in = backbone[idx].cv1.conv.in_channels
            backbone[idx] = WrappedModule(CoordAttention(ch_in, ch_in).to(device), i=idx, f=backbone[idx].f)

    print("✅ Injection completed!")

# --- Main Training Function ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    save_dir = os.path.join("checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    resume_checkpoint = None
    checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_ckpt = os.path.join(save_dir, checkpoints[-1])
        print(f"🔄 Found checkpoint: {latest_ckpt}")
        start_epoch = int(latest_ckpt.split('_')[-1].split('.')[0])
        resume_checkpoint = latest_ckpt
    else:
        print("🆕 No checkpoint found. Starting fresh training.")
        start_epoch = 0

    # Load configs
    logging_cfg = load_yaml_config("configs/training_config.yaml")['logging']
    model_cfg = load_yaml_config("configs/model_config.yaml")['model']
    train_cfg = load_yaml_config("configs/training_config.yaml")['training']
    dataset_cfg = load_yaml_config("configs/training_config.yaml")['dataset']

    # Load Dataset
    train_dataset = CrackDataset(dataset_cfg['train_path'], img_size=train_cfg['image_size'])
    val_dataset = CrackDataset(dataset_cfg['val_path'], img_size=train_cfg['image_size'])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, collate_fn=collate_fn)

    # Load YOLOv8 model
    model = YOLO("yolov8s.pt")
    model = model.model.to(device)

    # Inject crack detection modules
    inject_custom_blocks(model)

    # Load checkpoint if exists
    if resume_checkpoint:
        print(f"🔄 Loading model weights from {resume_checkpoint}")
        state_dict = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        # After model.load_state_dict(state_dict)
    if start_epoch > train_cfg['warmup_epochs']:
        print("🧩 Automatically unfreezing model because resumed after warmup epochs...")
        for param in model.parameters():
            param.requires_grad = True


    # Freeze early layers
    for idx, block in enumerate(model.model):
        if idx <= 4:
            for param in block.parameters():
                param.requires_grad = False
    print("🧊 Backbone partially frozen")

    # Optimizer & Scheduler
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg['epochs'], eta_min=1e-6)

    # Loss Function
    class EasyDict(dict):
        def __getattr__(self, name):
            return self[name]
        def __setattr__(self, name, value):
            self[name] = value

    loss_fn = v8DetectionLoss(model)
    loss_fn.hyp = EasyDict({
        "box": 7.5,
        "cls": 0.3,
        "obj": 1.0,
        "label_smoothing": 0.0,
        "fl_gamma": 0.5,
        "iou_type": "ciou",
        "anchor_t": 4.0,
        "dfl": 1.5
    })

    print("📚 Starting training loop...")

    for epoch in range(start_epoch, train_cfg['epochs']):
        if epoch == train_cfg['warmup_epochs']:
            print("🔥 Unfreezing full backbone...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
            scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg['epochs'], eta_min=1e-6)

        model.train()
        total_loss = 0

        for step, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)

            optimizer.zero_grad()

            batch_targets = {
                "bboxes": [],
                "cls": [],
                "batch_idx": []
            }
            for i, label in enumerate(targets):
                if label.shape[0] == 0:
                    continue
                batch_targets["bboxes"].append(label[:, :4])
                batch_targets["cls"].append(label[:, 4])
                batch_targets["batch_idx"].append(torch.full((label.shape[0], 1), i, dtype=torch.float32, device=device))

            if batch_targets["bboxes"]:
                batch_targets["bboxes"] = torch.cat(batch_targets["bboxes"]).to(device)
                batch_targets["cls"] = torch.cat(batch_targets["cls"]).to(device)
                batch_targets["batch_idx"] = torch.cat(batch_targets["batch_idx"]).to(device)
            else:
                batch_targets["bboxes"] = torch.zeros((0, 4), device=device)
                batch_targets["cls"] = torch.zeros((0,), device=device)
                batch_targets["batch_idx"] = torch.zeros((0, 1), device=device)

            preds = model(imgs)
            loss, _ = loss_fn(preds, batch_targets)

            loss = loss.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % logging_cfg['log_interval'] == 0:
                print(f"   🔄 Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"🔁 Epoch {epoch+1}/{train_cfg['epochs']} completed | 🧮 Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % train_cfg['save_interval'] == 0:
            ckpt_path = os.path.join(save_dir, f"yolov8_crack_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Model checkpoint saved to: {ckpt_path}")

    print("🎯 Training completed successfully!")

if __name__ == "__main__":
    main()

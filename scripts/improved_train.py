# MFM/scripts/improved_train.py
# Example usage:
# python MFM/scripts/improved_train.py --dataset-path /path/to/dataset.hdf5 --vla-path /path/to/vla --qwen-path /path/to/qwen --checkpoint-dir /path/to/checkpoints --batch-size 4 --gradient-accumulation-steps 8 --learning-rate 5e-5 --num-epochs 40 --finetune-vla-lora --save-interval 5

import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from PIL import Image
import os
import json
import gc
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- Environment and Path Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 🔥 Import the new command-injecting model
from MFM.models.improved_unified_model import TrueEndToEndDualViewVLA

# Matplotlib style setup for plots
plt.rcParams.update({
    'font.size': 14, 'figure.figsize': (12, 8), 'figure.dpi': 150,
    'axes.grid': True, 'grid.linestyle': ':', 'legend.fontsize': 'medium'
})

def set_random_seeds(seed):
    """Sets random seeds for reproducibility."""
    print(f"🎲 Setting random seeds to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True # Enable for performance

class TrueDualViewDataset(Dataset):
    """
    Dataset class for loading dual-view (agent-view and eye-in-hand) data.
    This version is simplified for clarity and focuses on batch processing.
    """
    def __init__(self, hdf5_path: str, image_size: int = 224):
        print(f"📂 Loading dual-view dataset: {hdf5_path}")
        self.hdf5_path = hdf5_path
        self.image_size = image_size
        self.file = None # To be opened in __getitem__ for multi-worker compatibility
        
        with h5py.File(self.hdf5_path, 'r') as f:
            self.trajectory_keys = [k for k in f.keys() if 'images' in f[k] and 'agentview' in f[k]['images'] and 'robot0_eye_in_hand' in f[k]['images']]
            print(f"Found {len(self.trajectory_keys)} trajectories with dual-view data.")
            if self.trajectory_keys:
                self.has_failure_context = 'failure_context' in f[self.trajectory_keys[0]]
            else:
                self.has_failure_context = False

        self.indices = []
        for traj_idx, key in enumerate(tqdm(self.trajectory_keys, desc="Indexing data")):
            with h5py.File(self.hdf5_path, 'r') as f:
                traj_len = len(f[key]['actions'])
                if traj_len > 0:
                    # We only use the current frame, so history_len is not a constraint
                    for step_idx in range(traj_len):
                        self.indices.append((traj_idx, step_idx))

        print(f"✅ Dataset ready. Total samples: {len(self.indices)}")
        if len(self.indices) == 0:
            raise ValueError("❌ No valid samples found!")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r', libver='latest', swmr=True)
        
        traj_idx, step_idx = self.indices[idx]
        key = self.trajectory_keys[traj_idx]
        traj_group = self.file[key]

        agentview_img = Image.fromarray(traj_group['images']['agentview'][step_idx]).resize((self.image_size, self.image_size))
        robot_eye_img = Image.fromarray(traj_group['images']['robot0_eye_in_hand'][step_idx]).resize((self.image_size, self.image_size))
        task_prompt = str(traj_group.attrs.get('task_description', ''))
        expert_action = torch.tensor(traj_group['actions'][step_idx], dtype=torch.float32)

        failure_context = "No failure context available."
        if self.has_failure_context and 'failure_context' in traj_group:
            try:
                context_prompts = traj_group['failure_context']['prompts']
                if step_idx < len(context_prompts):
                    raw_prompt = context_prompts[step_idx]
                    failure_context = raw_prompt.decode('utf-8') if isinstance(raw_prompt, bytes) else str(raw_prompt)
            except Exception: pass
        
        return {
            'current_agentview_image': agentview_img,
            'current_robot_eye_image': robot_eye_img,
            'task_prompt': task_prompt,
            'expert_action': expert_action,
            'failure_context': failure_context
        }

def dual_view_collate_fn(batch):
    """
    🔥 New collate function that supports true batching for dual-view data.
    """
    if not batch: return {}
    
    return {
        'current_agentview_images': [item['current_agentview_image'] for item in batch],
        'current_robot_eye_images': [item['current_robot_eye_image'] for item in batch],
        'task_prompts': [item['task_prompt'] for item in batch],
        'expert_actions': torch.stack([item['expert_action'] for item in batch]),
        'failure_contexts': [item['failure_context'] for item in batch],
        'total_batch_size': len(batch)
    }

class TrainingMonitor:
    """🔥 Enhanced training monitor with plotting capabilities."""
    def __init__(self, save_dir):
        self.losses = []
        self.learning_rates = []
        self.start_time = time.time()
        self.save_dir = Path(save_dir) / "plots"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def log_step(self, loss, lr):
        """Logs metrics for a single training step."""
        self.losses.append(loss)
        self.learning_rates.append(lr)
    
    def plot_and_save(self, epoch):
        """Generates and saves a plot of training metrics."""
        if not self.losses: return
        
        fig, ax1 = plt.subplots()
        
        # Plot Loss on the primary y-axis
        color = 'tab:red'
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(self.losses, color=color, alpha=0.8, label='Training Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('log') # Log scale is often better for viewing loss trends
        
        # Plot Learning Rate on the secondary y-axis
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Learning Rate', color=color)
        ax2.plot(self.learning_rates, color=color, linestyle='--', alpha=0.7, label='Learning Rate')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title(f'Training Metrics after Epoch {epoch+1}')
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        
        save_path = self.save_dir / f"training_metrics_epoch_{epoch+1}.png"
        plt.savefig(save_path)
        print(f"📊 Saved training plot to {save_path}")
        plt.close(fig)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_dir):
    """Saves a model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving checkpoint to {save_dir}")
    model.save_lora_weights(str(save_dir))
    
    training_state = {
        'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(), 'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(training_state, save_dir / "training_state.pt")
    print(f"✅ Checkpoint saved")

def train_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, grad_accum_steps, monitor):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(pbar):
        if not batch: continue
        
        expert_actions = batch['expert_actions'].to(device)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # 🔥 The model now accepts the entire batch dictionary
            predicted_actions = model(batch, device)
            loss = criterion(predicted_actions, expert_actions)
        
        loss_scaled = loss / grad_accum_steps
        
        if torch.isnan(loss_scaled) or torch.isinf(loss_scaled):
            print(f"⚠️ Skipping batch {batch_idx} due to NaN/Inf loss.")
            optimizer.zero_grad()
            continue

        scaler.scale(loss_scaled).backward()
        
        # Log loss and learning rate for each step
        monitor.log_step(loss.item(), optimizer.param_groups[0]['lr'])

        # Gradient accumulation step
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}", 'avg_loss': f"{avg_loss:.4f}",
            'mem': f"{torch.cuda.memory_allocated()/1024**3:.2f}GB"
        })
    
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Command-Injecting Dual-View Training")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--vla-path", type=str, default="/root/autodl-tmp/openvla/weights/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--qwen-path", type=str, default="/root/autodl-tmp/openvla/weights/MFM")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints_cmd_inject")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4) # 🔥 Reduced real batch size
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8) # 🔥 Increased grad accum
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune-vla-lora", action='store_true', help="Enable LoRA fine-tuning for VLA.")
    parser.add_argument("--save-interval", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seeds(args.seed)
    
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    
    print("\n" + "="*70)
    print("🚀 Command-Injecting Dual-View Training 🚀")
    print(f"  - Architecture: CoT guides VLA via instruction injection")
    print(f"  - Real Batch Size: {args.batch_size}")
    print(f"  - Grad Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"  - Effective Batch Size: {effective_batch_size}")
    print("="*70 + "\n")

    dataset = TrueDualViewDataset(args.dataset_path, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dual_view_collate_fn, num_workers=4, pin_memory=True)
    
    print("🤖 Initializing new command-injecting model...")
    model = TrueEndToEndDualViewVLA(vla_path=args.vla_path, qwen_path=args.qwen_path, use_lora=True, finetune_vla_lora=args.finetune_vla_lora).to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(dataloader))
    
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.MSELoss()
    monitor = TrainingMonitor(save_dir=args.checkpoint_dir)
    
    print("🚀 Starting training...")
    best_loss = float('inf')
    
    try:
        for epoch in range(args.num_epochs):
            print(f"\n{'='*30} Epoch {epoch+1}/{args.num_epochs} {'='*30}")
            avg_loss = train_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, args.gradient_accumulation_steps, monitor)
            scheduler.step()
            
            print(f"\n📊 Epoch {epoch+1} Summary: Average Loss = {avg_loss:.6f}")
            monitor.plot_and_save(epoch)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"🏆 New best loss: {best_loss:.6f}. Saving best model...")
                save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, Path(args.checkpoint_dir) / "best_model")
            
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, Path(args.checkpoint_dir) / f"epoch_{epoch+1}")
            
            gc.collect(); torch.cuda.empty_cache()
                        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
    finally:
        print("\n📋 Final Training Summary:")
        print(f"  - Best recorded loss: {best_loss:.6f}")
        print(f"  - Saving final model and plots...")
        save_checkpoint(model, optimizer, scheduler, "final", best_loss, Path(args.checkpoint_dir) / "final_model")
        monitor.plot_and_save("final")
        print("🧹 Training complete!")

if __name__ == "__main__":
    main()
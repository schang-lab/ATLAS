import os
import torch
from src.training import create_directory


def save_training_checkpoint(
    dit_model,
    optimizer,
    global_step,
    epoch,
    best_val_loss,
    training_dir,
    accelerator,
    args,
    wandb_run_id=None,
    autoencoder_state_dict=None,
):
    """
    Save complete training checkpoint with all necessary state for resuming
    """
    checkpoint = {
        'global_step': global_step,
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'model_state_dict': dit_model.module.state_dict() if accelerator.num_processes > 1 else dit_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),  # Save all training arguments
        'wandb_run_id': wandb_run_id,
        'accelerator_state': {
            'num_processes': accelerator.num_processes,
            'device': str(accelerator.device),
        }
    }

    if autoencoder_state_dict is not None:
        checkpoint['autoencoder_state_dict'] = autoencoder_state_dict
    
    with training_dir():
        checkpoint_path = f"training_checkpoint_step_{global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f'Saved complete training checkpoint: {checkpoint_path}')
        
        # Also save as latest checkpoint for easy resuming
        latest_path = "training_checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        print(f'Saved latest training checkpoint: {latest_path}')
    
    return checkpoint_path


def load_training_checkpoint(checkpoint_path, dit_model, optimizer, accelerator, args, autoencoder=None):
    """
    Load complete training checkpoint and restore training state
    """
    print(f"\n=== Loading Complete Training Checkpoint ===")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
    
    # Restore arguments from checkpoint to prevent weight jumping
    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        print("Restoring training arguments from checkpoint:")
        
        # Critical parameters that affect training stability
        critical_params = [
            # Basic training parameters
            'BATCH_SIZE', 'gradient_accumulation_steps', 'OPTIM_LR',
            'EPOCHS', 'TIMESTEPS', 'TIMESTAMP', 'prediction_type', 'timestep_sampling',
            
            # Autoencoder and model parameters
            'autoencoder_path', 'training_phase', 'ablation_mode', 'no_compression',
            'dit_checkpoint_path', 'config',
            
            # Loss parameters
            'use_anchor_loss', 'anchor_loss_weight',
            
            # Data parameters
            'data_dir', 'data_type', 'conditional_dropout',
            'poi_coordinates_csv', 'enable_length_condition', 'length_vocab_size',
            
            # Validation and logging
            'enable_validation', 'eval_samples', 'use_wandb',
            'wandb_project', 'wandb_run_name', 'wandb_id', 'wandb_api_key',
            
            # Step-based training
            'log_steps', 'save_steps', 'eval_steps', 'max_steps', 'warmup_steps',
            
            # System parameters
            'force_cpu', 'NUM_WORKERS'
        ]
        
        for param in critical_params:
            if param in saved_args and hasattr(args, param):
                old_value = getattr(args, param)
                new_value = saved_args[param]
                if old_value != new_value:
                    setattr(args, param, new_value)
                    print(f"  {param}: {old_value} → {new_value}")
                else:
                    print(f"  {param}: {new_value} (unchanged)")
    
    # Load model state
    missing_keys, unexpected_keys = dit_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in model checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in model checkpoint: {unexpected_keys}")

    # Load autoencoder state if available
    if autoencoder is not None and 'autoencoder_state_dict' in checkpoint:
        try:
            autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
            print("Successfully loaded autoencoder state")
        except Exception as e:
            print(f"Warning: Could not load autoencoder state: {e}")

    # Load optimizer state
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Successfully loaded optimizer state")
    except Exception as e:
        print(f"Warning: Could not load optimizer state: {e}")
        print("Continuing with fresh optimizer state")
    
    # Extract training state
    global_step = checkpoint.get('global_step', 0)
    epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', torch.tensor(9999999))
    wandb_run_id = checkpoint.get('wandb_run_id', None)
    
    print(f"Resuming training from:")
    print(f"  Global step: {global_step}")
    print(f"  Epoch: {epoch}")
    print(f"  Best validation loss: {best_val_loss}")
    print(f"  Wandb run ID: {wandb_run_id}")
    
    return global_step, epoch, best_val_loss, wandb_run_id

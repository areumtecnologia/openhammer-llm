#!/usr/bin/env python3
"""
OpenHammer LLM Studio - Command Line Interface
For systems without GUI support
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.model import (
    ModelConfig, TrainingConfig, HardwareProfile, GPTModel,
    Trainer, Tokenizer, DatasetManager, ExperimentResult
)


def print_header():
    """Print application header"""
    print("\n" + "="*70)
    print("  OpenHammer LLM Studio - Low-Cost Language Model Creator")
    print("  CLI Edition")
    print("="*70 + "\n")


def print_hardware_info(hp: HardwareProfile):
    """Print hardware information"""
    print("\n📊 Hardware Profile:")
    print(f"   RAM: {hp.ram_gb} GB")
    print(f"   VRAM: {hp.vram_gb} GB" if hp.has_gpu else "   VRAM: N/A")
    print(f"   CPU Cores: {hp.cpu_cores}")
    print(f"   GPU: {hp.gpu_type.upper()}" if hp.has_gpu else "   GPU: None")
    print(f"   Recommendation: Max {hp.recommend_max_model()} model, {hp.recommend_quantization()} quantization")
    print()


def get_menu_choice(options: list, prompt: str = "Choice") -> int:
    """Get user menu choice"""
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    while True:
        try:
            choice = input(f"\n{prompt} (1-{len(options)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
            print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def configure_model() -> ModelConfig:
    """Interactive model configuration"""
    print("\n🔧 Model Configuration")
    print("-" * 40)
    
    print("\nQuick presets:")
    presets = [
        ("Tiny (1 layer, 16 embd)", 1, 16, 16, 2),
        ("Small (2 layers, 32 embd)", 2, 32, 32, 4),
        ("Medium (4 layers, 64 embd)", 4, 64, 64, 8),
        ("Custom", 0, 0, 0, 0)
    ]
    
    preset_idx = get_menu_choice([p[0] for p in presets], "Select preset")
    
    if preset_idx < 3:
        _, n_layer, n_embd, block_size, n_head = presets[preset_idx]
    else:
        try:
            n_layer = int(input("Number of layers [2]: ") or "2")
            n_embd = int(input("Embedding dimension [32]: ") or "32")
            block_size = int(input("Block size [32]: ") or "32")
            n_head = int(input("Attention heads [4]: ") or "4")
        except ValueError:
            print("Invalid input, using defaults")
            n_layer, n_embd, block_size, n_head = 2, 32, 32, 4
    
    vocab_size = 256  # Default for character-level
    
    try:
        config = ModelConfig(
            n_layer=n_layer,
            n_embd=n_embd,
            block_size=block_size,
            n_head=n_head,
            vocab_size=vocab_size
        )
        print(f"\n✓ Model configured: {config.num_params:,} parameters")
        return config
    except Exception as e:
        print(f"✗ Error: {e}")
        return configure_model()


def configure_training() -> TrainingConfig:
    """Interactive training configuration"""
    print("\n🎯 Training Configuration")
    print("-" * 40)
    
    try:
        lr = float(input("Learning rate [0.01]: ") or "0.01")
        steps = int(input("Training steps [1000]: ") or "1000")
        beta1 = float(input("Beta1 [0.85]: ") or "0.85")
        beta2 = float(input("Beta2 [0.99]: ") or "0.99")
        temp = float(input("Temperature [0.5]: ") or "0.5")
        
        return TrainingConfig(
            learning_rate=lr,
            num_steps=steps,
            beta1=beta1,
            beta2=beta2,
            temperature=temp
        )
    except ValueError as e:
        print(f"Invalid input: {e}")
        return configure_training()


def load_dataset(dm: DatasetManager):
    """Load dataset interactively"""
    print("\n📊 Dataset Selection")
    print("-" * 40)
    
    sources = [
        "Sample Dataset (Names)",
        "Sample Dataset (Code)",
        "Local File",
        "Exit"
    ]
    
    idx = get_menu_choice(sources, "Select source")
    
    if idx == 0:
        return dm.create_sample_dataset("names")
    elif idx == 1:
        return dm.create_sample_dataset("code")
    elif idx == 2:
        path = input("Enter file path: ").strip()
        if os.path.exists(path):
            return dm.load_local(path)
        else:
            print(f"File not found: {path}")
            return load_dataset(dm)
    else:
        return None


def main():
    """Main CLI entry point"""
    print_header()
    
    # Detect hardware
    hp = HardwareProfile.detect_system()
    print_hardware_info(hp)
    
    # Initialize managers
    dm = DatasetManager()
    
    # Main menu loop
    model = None
    tokenizer = None
    trainer = None
    
    while True:
        print("\n" + "="*40)
        print("  Main Menu")
        print("="*40)
        
        options = [
            "Configure Model",
            "Load Dataset",
            "Train Model",
            "Generate Text",
            "Save/Load Model",
            "View Hardware Info",
            "Exit"
        ]
        
        choice = get_menu_choice(options, "Select option")
        
        if choice == 0:  # Configure Model
            config = configure_model()
            model = GPTModel(config)
            print(f"\n✓ Model initialized with {config.num_params:,} parameters")
        
        elif choice == 1:  # Load Dataset
            docs = load_dataset(dm)
            if docs:
                tokenizer = Tokenizer(docs)
                print(f"\n✓ Dataset loaded: {len(docs)} documents")
                print(f"  Vocabulary: {tokenizer.vocab_size} tokens")
                print(f"  Sample: {docs[0][:50]}...")
        
        elif choice == 2:  # Train Model
            if not model or not tokenizer:
                print("\n✗ Please configure model and load dataset first!")
                continue
            
            train_config = configure_training()
            trainer = Trainer(model, tokenizer, train_config)
            
            print(f"\n🚀 Starting training for {train_config.num_steps} steps...")
            print("-" * 60)
            
            def callback(step, loss):
                if step % 50 == 0:
                    print(f"Step {step:4d}/{train_config.num_steps:4d} | Loss: {loss:.4f}")
            
            try:
                loss_history = trainer.train(docs, callback=callback)
                print("-" * 60)
                print(f"✓ Training complete! Final loss: {loss_history[-1]:.4f}")
                
                # Generate samples
                print("\n📝 Generated samples:")
                for i in range(5):
                    sample = model.generate(tokenizer, max_tokens=30)
                    print(f"  {i+1}. {sample}")
            except KeyboardInterrupt:
                print("\n\n⚠ Training interrupted by user")
        
        elif choice == 3:  # Generate Text
            if not model or not tokenizer:
                print("\n✗ Please load a model first!")
                continue
            
            try:
                temp = float(input("Temperature [0.5]: ") or "0.5")
                max_tokens = int(input("Max tokens [50]: ") or "50")
                
                print("\n✨ Generating text...")
                text = model.generate(tokenizer, max_tokens=max_tokens, temperature=temp)
                print(f"\nGenerated:\n{text}\n")
            except ValueError as e:
                print(f"Invalid input: {e}")
        
        elif choice == 4:  # Save/Load Model
            sub_options = ["Save Model", "Load Model", "Back"]
            sub_choice = get_menu_choice(sub_options, "Select")
            
            if sub_choice == 0:  # Save
                if not model:
                    print("\n✗ No model to save!")
                    continue
                
                path = input("Enter filename [model.pkl]: ").strip() or "model.pkl"
                try:
                    model.save(path)
                    print(f"✓ Model saved to {path}")
                except Exception as e:
                    print(f"✗ Error saving: {e}")
            
            elif sub_choice == 1:  # Load
                path = input("Enter filename [model.pkl]: ").strip() or "model.pkl"
                try:
                    model = GPTModel.load(path)
                    print(f"✓ Model loaded from {path}")
                    print(f"  Config: {model.config.n_layer} layers, {model.config.n_embd} embd")
                except Exception as e:
                    print(f"✗ Error loading: {e}")
        
        elif choice == 5:  # Hardware Info
            print_hardware_info(hp)
        
        elif choice == 6:  # Exit
            print("\n👋 Goodbye!\n")
            break
        
        else:
            print("\n✗ Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
        sys.exit(0)

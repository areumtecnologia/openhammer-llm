"""
Model Manager - Handles model saving, loading, checkpoints, and quality assessment
"""

import os
import json
import pickle
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib


@dataclass
class ModelMetadata:
    """Metadata for saved models"""
    model_id: str
    name: str
    description: str
    created_at: str
    model_config: dict
    training_config: dict
    final_loss: float
    total_steps: int
    use_case: str
    dataset_hash: str
    checkpoint_path: str
    is_checkpoint: bool = False
    parent_model_id: str = None
    quality_score: float = 0.0
    test_results: dict = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ModelMetadata':
        return cls(**data)


class QualityAssessor:
    """
    Automatic quality assessment for trained models.
    
    Evaluates model performance on multiple criteria:
    - Loss convergence
    - Text coherence
    - Diversity of outputs
    - Response to test prompts
    """
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.test_prompts = [
            "Hello",
            "The meaning of life is",
            "Once upon a time",
            "In conclusion",
            "To be or not to be",
        ]
    
    def assess(self, num_samples: int = 5, max_tokens: int = 30) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment.
        
        Returns:
            Dictionary with quality metrics and scores
        """
        results = {
            'coherence_score': 0.0,
            'diversity_score': 0.0,
            'completion_rate': 0.0,
            'avg_length': 0.0,
            'test_outputs': [],
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # Generate test samples
        outputs = []
        for prompt in self.test_prompts[:num_samples]:
            try:
                if hasattr(self.model, 'generate_text'):
                    # Torch model
                    output = self.model.generate_text(
                        self.tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=0.5,
                        device=getattr(self.model, 'device', None)
                    )
                else:
                    # Pure Python model
                    output = self.model.generate(
                        self.tokenizer,
                        max_tokens=max_tokens,
                        temperature=0.5
                    )
                outputs.append(output)
            except Exception as e:
                outputs.append(f"Error: {e}")
        
        results['test_outputs'] = outputs
        
        # Calculate metrics
        valid_outputs = [o for o in outputs if not o.startswith("Error")]
        
        if valid_outputs:
            # Completion rate
            results['completion_rate'] = len(valid_outputs) / num_samples
            
            # Average length
            lengths = [len(o.split()) for o in valid_outputs]
            results['avg_length'] = sum(lengths) / len(lengths) if lengths else 0
            
            # Diversity score (unique words / total words)
            all_words = []
            for output in valid_outputs:
                words = output.lower().split()
                all_words.extend(words)
            
            if all_words:
                unique_ratio = len(set(all_words)) / len(all_words)
                results['diversity_score'] = min(1.0, unique_ratio * 2)
            
            # Coherence score (heuristic: presence of common patterns)
            coherence_indicators = ['the', 'a', 'is', 'to', 'and', 'of', 'in']
            coherence_count = sum(
                1 for word in all_words if word in coherence_indicators
            )
            results['coherence_score'] = min(1.0, coherence_count / len(all_words) * 5) if all_words else 0
            
            # Overall score (weighted average)
            results['overall_score'] = (
                results['completion_rate'] * 0.3 +
                results['diversity_score'] * 0.25 +
                results['coherence_score'] * 0.25 +
                min(1.0, results['avg_length'] / 10) * 0.2
            )
        
        # Generate recommendations
        if results['overall_score'] < 0.3:
            results['recommendations'].append("Model needs more training steps")
        if results['diversity_score'] < 0.3:
            results['recommendations'].append("Increase temperature or training data diversity")
        if results['coherence_score'] < 0.3:
            results['recommendations'].append("Model may need more training on coherent text")
        if results['completion_rate'] < 0.8:
            results['recommendations'].append("Check for numerical stability issues")
        
        return results


class ModelManager:
    """
    Manages model lifecycle: saving, loading, checkpoints, and versioning.
    """
    
    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_dir / "model_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, ModelMetadata]:
        """Load model registry from disk"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {k: ModelMetadata.from_dict(v) for k, v in data.items()}
        return {}
    
    def _save_registry(self):
        """Save model registry to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(
                {k: v.to_dict() for k, v in self.registry.items()},
                f, indent=2
            )
    
    def _compute_dataset_hash(self, docs: List[str]) -> str:
        """Compute hash of dataset for tracking"""
        combined = ''.join(sorted(docs))
        return hashlib.md5(combined.encode()).hexdigest()
    
    def save_model(
        self,
        model,
        tokenizer,
        model_config,
        training_config,
        final_loss: float,
        total_steps: int,
        use_case: str,
        docs: List[str],
        name: str = None,
        description: str = "",
        is_checkpoint: bool = False,
        parent_model_id: str = None
    ) -> str:
        """
        Save model to disk with metadata.
        
        Args:
            model: Trained model instance
            tokenizer: Tokenizer instance
            model_config: ModelConfig instance
            training_config: TrainingConfig instance
            final_loss: Final training loss
            total_steps: Total training steps completed
            use_case: Model use case (completion, chat, etc.)
            docs: Training documents
            name: Optional model name
            description: Optional description
            is_checkpoint: Whether this is a checkpoint
            parent_model_id: ID of parent model if this is a fine-tune
        
        Returns:
            Model ID
        """
        # Generate model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"model_{timestamp}"
        
        # Create directory for this model
        model_dir = self.base_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model weights
        model_path = model_dir / "model.pt"
        if hasattr(model, 'state_dict'):
            # Torch model
            torch.save(model.state_dict(), model_path)
        else:
            # Pure Python model - save state_dict
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model.state_dict, f)
        
        # Save tokenizer
        tokenizer_path = model_dir / "tokenizer.pkl"
        with open(tokenizer_path, 'wb') as f:
            pickle.dump({
                'uchars': tokenizer.uchars,
                'BOS': tokenizer.BOS,
                'EOS': tokenizer.EOS,
                'char_to_idx': tokenizer.char_to_idx,
                'idx_to_char': tokenizer.idx_to_char
            }, f)
        
        # Save configs
        config_path = model_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'model_config': model_config.to_dict(),
                'training_config': training_config.to_dict(),
                'use_case': use_case if hasattr(use_case, 'mode') else str(use_case)
            }, f, indent=2)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name or f"Model {timestamp}",
            description=description,
            created_at=datetime.now().isoformat(),
            model_config=model_config.to_dict(),
            training_config=training_config.to_dict(),
            final_loss=final_loss,
            total_steps=total_steps,
            use_case=use_case.mode if hasattr(use_case, 'mode') else str(use_case),
            dataset_hash=self._compute_dataset_hash(docs),
            checkpoint_path=str(model_path),
            is_checkpoint=is_checkpoint,
            parent_model_id=parent_model_id
        )
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update registry
        self.registry[model_id] = metadata
        self._save_registry()
        
        return model_id
    
    def save_checkpoint(
        self,
        model,
        tokenizer,
        model_config,
        training_config,
        current_loss: float,
        current_step: int,
        total_steps: int,
        use_case: str,
        docs: List[str],
        parent_model_id: str = None
    ) -> str:
        """Save training checkpoint"""
        return self.save_model(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            training_config=training_config,
            final_loss=current_loss,
            total_steps=current_step,
            use_case=use_case,
            docs=docs,
            name=f"Checkpoint step {current_step}/{total_steps}",
            description=f"Checkpoint at step {current_step} with loss {current_loss:.4f}",
            is_checkpoint=True,
            parent_model_id=parent_model_id
        )
    
    def load_model(self, model_id: str):
        """
        Load model from disk.
        
        Returns:
            Tuple of (model_instance, tokenizer, model_config, training_config, use_case)
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.registry[model_id]
        model_dir = Path(metadata.checkpoint_path).parent
        
        # Load configs
        config_path = model_dir / "config.json"
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        from core.model import ModelConfig, TrainingConfig, ModelUseCase
        model_config = ModelConfig.from_dict(configs['model_config'])
        training_config = TrainingConfig.from_dict(configs['training_config'])
        
        # Determine use case
        use_case_str = configs.get('use_case', 'completion')
        if isinstance(use_case_str, dict):
            use_case = ModelUseCase(**use_case_str)
        else:
            use_case = ModelUseCase(mode=use_case_str)
        
        # Load tokenizer
        tokenizer_path = model_dir / "tokenizer.pkl"
        with open(tokenizer_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        # Reconstruct tokenizer
        from core.model import Tokenizer
        class SimpleTokenizer:
            def __init__(self, data):
                self.uchars = data['uchars']
                self.BOS = data['BOS']
                self.EOS = data['EOS']
                self.char_to_idx = data['char_to_idx']
                self.idx_to_char = data['idx_to_char']
                self.vocab_size = len(self.uchars) + 2
            
            def encode(self, text: str) -> List[int]:
                return [self.BOS] + [self.char_to_idx.get(ch, self.BOS) for ch in text] + [self.EOS]
            
            def decode(self, tokens: List[int]) -> str:
                return ''.join(self.idx_to_char.get(t, '?') for t in tokens if t not in [self.BOS, self.EOS])
        
        tokenizer = SimpleTokenizer(tokenizer_data)
        
        # Load model weights
        model_path = model_dir / "model.pt"
        pkl_path = model_dir / "model.pkl"
        
        # Check if PyTorch model
        try:
            import torch
            from core.model import TorchGPTModel
            
            if model_path.exists():
                # Load as Torch model
                model = TorchGPTModel(model_config, use_case)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                return model, tokenizer, model_config, training_config, use_case
        except ImportError:
            pass
        
        # Load as pure Python model
        if pkl_path.exists():
            from core.model import GPTModel
            model = GPTModel(model_config, use_case)
            with open(pkl_path, 'rb') as f:
                state_dict = pickle.load(f)
            # Restore state dict (simplified - would need proper weight restoration)
            model.state_dict = state_dict
            return model, tokenizer, model_config, training_config, use_case
        
        raise FileNotFoundError(f"Model weights not found in {model_dir}")
    
    def list_models(self, include_checkpoints: bool = False) -> List[ModelMetadata]:
        """List all saved models"""
        models = list(self.registry.values())
        if not include_checkpoints:
            models = [m for m in models if not m.is_checkpoint]
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    def delete_model(self, model_id: str):
        """Delete model from disk and registry"""
        if model_id not in self.registry:
            return
        
        metadata = self.registry[model_id]
        model_dir = Path(metadata.checkpoint_path).parent
        
        # Remove files
        import shutil
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del self.registry[model_id]
        self._save_registry()
    
    def get_model_info(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model"""
        return self.registry.get(model_id)
    
    def update_quality_score(self, model_id: str, quality_score: float, test_results: dict):
        """Update quality assessment for a model"""
        if model_id in self.registry:
            self.registry[model_id].quality_score = quality_score
            self.registry[model_id].test_results = test_results
            self._save_registry()

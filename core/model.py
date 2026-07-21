"""
Core module for LLM Desktop App - Pure Python implementation
Essência algorítmica para criação de Modelos de Linguagem.

Este módulo fornece a base para treinar modelos desde zero (estilo 'micro-llm' puro em Python/Numpy)
até configurações avançadas com aceleradores GPU para casos de uso complexos:
- Chat e conversação
- Function Calling e ferramentas
- Agentes autônomos
- Completion de código e texto estruturado

Princípios:
1. Transparência: O algoritmo de treinamento (Autograd, Adam, Attention) é explícito
2. Escalabilidade: Comece simples em CPU, escale para GPU quando necessário
3. Versatilidade: Suporte a múltiplos casos de uso (Chat, Tools, Agents)
4. Minimalismo: Cada linha de código tem propósito educacional
"""

import os
import math
import random
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

random.seed(42)


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    n_layer: int = 2
    n_embd: int = 32
    block_size: int = 32
    n_head: int = 4
    vocab_size: int = 256
    
    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
    
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head
    
    @property
    def num_params(self) -> int:
        """Estimate total parameters"""
        params = 0
        # Embeddings
        params += self.vocab_size * self.n_embd  # wte
        params += self.block_size * self.n_embd  # wpe
        # Layers
        for _ in range(self.n_layer):
            params += 4 * (self.n_embd * self.n_embd)  # attention weights
            params += 2 * (self.n_embd * 4 * self.n_embd)  # MLP
        # Output
        params += self.vocab_size * self.n_embd  # lm_head
        return params
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ModelConfig':
        return cls(**data)


@dataclass
class TrainingConfig:
    """Configuration for training"""
    learning_rate: float = 0.01
    beta1: float = 0.85
    beta2: float = 0.99
    eps_adam: float = 1e-8
    num_steps: int = 1000
    batch_size: int = 1
    temperature: float = 0.5
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingConfig':
        return cls(**data)


@dataclass
class ModelUseCase:
    """Configuration for model use case and capabilities"""
    mode: str = "completion"  # completion, chat, function_calling, agent
    supports_tools: bool = False
    supports_json_output: bool = False
    system_prompt: str = ""
    tool_definitions: List[Dict] = None
    
    def __post_init__(self):
        if self.tool_definitions is None:
            self.tool_definitions = []
    
    @property
    def description(self) -> str:
        descriptions = {
            "completion": "Text completion and generation",
            "chat": "Conversational AI with context",
            "function_calling": "Tool/API integration",
            "agent": "Autonomous task execution"
        }
        return descriptions.get(self.mode, "Custom mode")
    
    @classmethod
    def create_chat_model(cls) -> 'ModelUseCase':
        """Create configuration for chat/conversation"""
        return cls(
            mode="chat",
            system_prompt="You are a helpful assistant."
        )
    
    @classmethod
    def create_function_calling_model(cls, tools: List[Dict] = None) -> 'ModelUseCase':
        """Create configuration for function calling"""
        return cls(
            mode="function_calling",
            supports_tools=True,
            supports_json_output=True,
            tool_definitions=tools or []
        )
    
    @classmethod
    def create_agent_model(cls, tools: List[Dict] = None) -> 'ModelUseCase':
        """Create configuration for autonomous agents"""
        return cls(
            mode="agent",
            supports_tools=True,
            supports_json_output=True,
            system_prompt="You are an autonomous agent. Think step by step.",
            tool_definitions=tools or []
        )


@dataclass
class TrainingStack:
    """Training stack configuration"""
    use_gpu: bool = False
    backend: str = "cpu"  # cpu, cuda, mps, rocm
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = [] if not self.use_gpu else ["torch", "numpy"]
    
    @property
    def description(self) -> str:
        if self.use_gpu:
            return f"GPU Accelerated ({self.backend}) - Requires: {', '.join(self.dependencies)}"
        return "CPU Only - No additional dependencies"
    
    @classmethod
    def detect_available(cls) -> List['TrainingStack']:
        """Detect available training stacks on current system"""
        stacks = [cls(use_gpu=False, backend="cpu", dependencies=[])]
        
        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                stacks.append(cls(use_gpu=True, backend="cuda", dependencies=["torch"]))
            else:
                # CUDA installed but not available - likely missing NVIDIA Container Toolkit
                print("[WARNING] PyTorch has CUDA support but no GPU detected.")
                print("          If using Docker/Podman, run with: --gpus all")
        except ImportError:
            pass
        
        # Check for MPS (Apple Silicon)
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                stacks.append(cls(use_gpu=True, backend="mps", dependencies=["torch"]))
        except ImportError:
            pass
        
        return stacks


@dataclass
class HardwareProfile:
    """Hardware capability profile"""
    ram_gb: float = 8.0
    vram_gb: float = 0.0
    cpu_cores: int = 4
    has_gpu: bool = False
    gpu_type: str = "none"  # none, cuda, rocm, metal, vulkan
    supports_avx2: bool = True
    supports_avx512: bool = False
    recommended_stack: str = "cpu"
    
    def recommend_quantization(self) -> str:
        """Recommend quantization level based on hardware"""
        if self.vram_gb >= 16 or self.ram_gb >= 32:
            return "Q8_0"
        elif self.vram_gb >= 8 or self.ram_gb >= 16:
            return "Q4_K_M"
        else:
            return "Q4_0"
    
    def recommend_max_model(self) -> str:
        """Recommend maximum model size"""
        total_mem = self.vram_gb if self.vram_gb > 0 else self.ram_gb
        if total_mem >= 64:
            return "70B"
        elif total_mem >= 32:
            return "30B"
        elif total_mem >= 16:
            return "13B"
        elif total_mem >= 8:
            return "7B"
        else:
            return "3B"
    
    @classmethod
    def detect_system(cls) -> 'HardwareProfile':
        """Detect current system capabilities"""
        import psutil
        
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count(logical=False) or 4
        
        # Detect GPU and recommend stack
        has_gpu = False
        gpu_type = "none"
        vram_gb = 0.0
        recommended_stack = "cpu"
        
        try:
            import torch
            if torch.cuda.is_available():
                has_gpu = True
                gpu_type = "cuda"
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                recommended_stack = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                has_gpu = True
                gpu_type = "metal"
                recommended_stack = "mps"
            else:
                # PyTorch has CUDA but GPU not accessible - common in containers without --gpus all
                print("[INFO] PyTorch CUDA support detected but GPU not accessible.")
                print("       This usually means:")
                print("       1. Running in Docker/Podman without '--gpus all' flag")
                print("       2. NVIDIA Container Toolkit not installed/configured")
                print("       3. User lacks permissions to access /dev/nvidia* devices")
                print("")
                print("       To fix in Docker/Podman:")
                print("       docker run --gpus all <image>")
                print("       podman run --device nvidia.com/gpu=all <image>")
        except ImportError:
            pass
        
        return cls(
            ram_gb=round(ram_gb, 1),
            vram_gb=round(vram_gb, 1),
            cpu_cores=cpu_cores,
            has_gpu=has_gpu,
            gpu_type=gpu_type,
            recommended_stack=recommended_stack
        )


class Value:
    """Autograd scalar value for computational graph"""
    __slots__ = ('data', 'grad', '_children', '_local_grads', '_op')
    
    def __init__(self, data, children=(), local_grads=(), op=''):
        self.data = data
        self.grad = 0
        self._children = tuple(children)
        self._local_grads = tuple(local_grads)
        self._op = op
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1), '+')
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data), '*')
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Value(self.data**other, (self,), (other * self.data**(other-1),), f'**{other}')
        raise NotImplementedError
    
    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,), 'log')
    
    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),), 'exp')
    
    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),), 'relu')
    
    def tanh(self):
        return Value(math.tanh(self.data), (self,), (1 - self.data**2,), 'tanh')
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def backward(self):
        """Compute gradients using backpropagation"""
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


class Matrix:
    """Simple matrix operations for neural network"""
    
    @staticmethod
    def create(nout: int, nin: int, std: float = 0.08) -> List[List[Value]]:
        """Initialize matrix with Gaussian values"""
        return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    
    @staticmethod
    def zeros(nout: int, nin: int) -> List[List[Value]]:
        """Create zero matrix"""
        return [[Value(0) for _ in range(nin)] for _ in range(nout)]
    
    @staticmethod
    def linear(x: List[Value], w: List[List[Value]]) -> List[Value]:
        """Linear transformation: y = Wx"""
        return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
    
    @staticmethod
    def softmax(logits: List[Value]) -> List[Value]:
        """Numerically stable softmax"""
        max_val = max(val.data for val in logits)
        exps = [(val - max_val).exp() for val in logits]
        total = sum(exps)
        return [e / total for e in exps]
    
    @staticmethod
    def rmsnorm(x: List[Value], eps: float = 1e-5) -> List[Value]:
        """RMSNorm normalization"""
        ms = sum(xi * xi for xi in x) / len(x)
        scale = (ms + eps) ** -0.5
        return [xi * scale for xi in x]


class Tokenizer:
    """Character-level tokenizer with special tokens"""
    
    def __init__(self, docs: List[str]):
        self.docs = docs
        self.uchars = sorted(set(''.join(docs)))
        self.BOS = len(self.uchars)
        self.EOS = len(self.uchars) + 1
        self.vocab_size = len(self.uchars) + 2
        self.char_to_idx = {ch: i for i, ch in enumerate(self.uchars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.idx_to_char[self.BOS] = '<BOS>'
        self.idx_to_char[self.EOS] = '<EOS>'
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        return [self.BOS] + [self.char_to_idx.get(ch, self.BOS) for ch in text] + [self.EOS]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text"""
        return ''.join(self.idx_to_char.get(t, '?') for t in tokens if t not in [self.BOS, self.EOS])
    
    @property
    def vocab(self) -> List[str]:
        """Get vocabulary list"""
        return list(self.uchars) + ['<BOS>', '<EOS>']


class GPTModel:
    """Minimal GPT model implementation with multi-use case support"""
    
    def __init__(self, config: ModelConfig, use_case: ModelUseCase = None):
        self.config = config
        self.use_case = use_case or ModelUseCase()
        self.state_dict = self._initialize_weights()
        self.keys_cache = None
        self.values_cache = None
        self.conversation_history = []  # For chat mode
    
    def _initialize_weights(self) -> Dict[str, List[List[Value]]]:
        """Initialize all model parameters"""
        cfg = self.config
        state_dict = {
            'wte': Matrix.create(cfg.vocab_size, cfg.n_embd),
            'wpe': Matrix.create(cfg.block_size, cfg.n_embd),
            'lm_head': Matrix.create(cfg.vocab_size, cfg.n_embd)
        }
        
        for i in range(cfg.n_layer):
            state_dict[f'layer{i}.attn_wq'] = Matrix.create(cfg.n_embd, cfg.n_embd)
            state_dict[f'layer{i}.attn_wk'] = Matrix.create(cfg.n_embd, cfg.n_embd)
            state_dict[f'layer{i}.attn_wv'] = Matrix.create(cfg.n_embd, cfg.n_embd)
            state_dict[f'layer{i}.attn_wo'] = Matrix.create(cfg.n_embd, cfg.n_embd)
            state_dict[f'layer{i}.mlp_fc1'] = Matrix.create(4 * cfg.n_embd, cfg.n_embd)
            state_dict[f'layer{i}.mlp_fc2'] = Matrix.create(cfg.n_embd, 4 * cfg.n_embd)
        
        return state_dict
    
    def get_params(self) -> List[Value]:
        """Get all parameters as flat list"""
        params = []
        for mat in self.state_dict.values():
            for row in mat:
                for p in row:
                    params.append(p)
        return params
    
    def forward(self, token_id: int, pos_id: int, keys: List, values: List) -> List[Value]:
        """Forward pass for single token"""
        cfg = self.config
        
        # Token and position embeddings
        tok_emb = self.state_dict['wte'][token_id]
        pos_emb = self.state_dict['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = Matrix.rmsnorm(x)
        
        # Transformer layers
        for li in range(cfg.n_layer):
            # Multi-head attention
            x_residual = x
            x = Matrix.rmsnorm(x)
            
            q = Matrix.linear(x, self.state_dict[f'layer{li}.attn_wq'])
            k = Matrix.linear(x, self.state_dict[f'layer{li}.attn_wk'])
            v = Matrix.linear(x, self.state_dict[f'layer{li}.attn_wv'])
            
            keys[li].append(k)
            values[li].append(v)
            
            x_attn = []
            for h in range(cfg.n_head):
                hs = h * cfg.head_dim
                q_h = q[hs:hs+cfg.head_dim]
                k_h = [ki[hs:hs+cfg.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+cfg.head_dim] for vi in values[li]]
                
                # Attention scores
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(cfg.head_dim)) / (cfg.head_dim ** 0.5)
                    for t in range(len(k_h))
                ]
                attn_weights = Matrix.softmax(attn_logits)
                
                # Weighted sum
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(cfg.head_dim)
                ]
                x_attn.extend(head_out)
            
            x = Matrix.linear(x_attn, self.state_dict[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]
            
            # MLP
            x_residual = x
            x = Matrix.rmsnorm(x)
            x = Matrix.linear(x, self.state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = Matrix.linear(x, self.state_dict[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]
        
        # Output projection
        logits = Matrix.linear(x, self.state_dict['lm_head'])
        return logits
    
    def generate(self, tokenizer: Tokenizer, max_tokens: int = 50, 
                 temperature: float = 0.5, prompt: str = None) -> str:
        """Generate text autoregressively with use case support"""
        keys = [[] for _ in range(self.config.n_layer)]
        values = [[] for _ in range(self.config.n_layer)]
        
        # Handle different use cases
        if self.use_case.mode == "chat" and prompt:
            # Format as conversation turn
            formatted_prompt = f"{self.use_case.system_prompt}\n\nUser: {prompt}\nAssistant:"
        elif self.use_case.mode == "function_calling" and prompt:
            # Format for JSON output
            formatted_prompt = f"{self.use_case.system_prompt}\n\nQuery: {prompt}\nResponse (JSON):"
        elif self.use_case.mode == "agent" and prompt:
            # Format for agent reasoning
            formatted_prompt = f"{self.use_case.system_prompt}\n\nTask: {prompt}\nThoughts:"
        else:
            formatted_prompt = prompt or ""
        
        token_id = tokenizer.BOS
        generated = []
        
        for pos_id in range(max_tokens):
            logits = self.forward(token_id, pos_id, keys, values)
            probs = Matrix.softmax([l / temperature for l in logits])
            
            # Sample from distribution
            weights = [max(0, p.data) for p in probs]
            total = sum(weights)
            if total == 0:
                break
            weights = [w/total for w in weights]
            
            # Ensure weights length matches vocab size
            while len(weights) < tokenizer.vocab_size:
                weights.append(0.0)
            weights = weights[:tokenizer.vocab_size]
            
            token_id = random.choices(range(tokenizer.vocab_size), weights=weights)[0]
            
            if token_id == tokenizer.EOS or token_id == tokenizer.BOS:
                break
            
            generated.append(token_id)
        
        return tokenizer.decode(generated)
    
    def chat(self, tokenizer: Tokenizer, user_message: str, 
             max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Chat mode with conversation history"""
        if self.use_case.mode != "chat":
            # Switch to chat mode
            self.use_case = ModelUseCase.create_chat_model()
        
        # Build context from conversation history
        context = "\n".join(self.conversation_history[-5:])  # Last 5 turns
        full_prompt = f"{context}\nUser: {user_message}\nAssistant:" if context else f"User: {user_message}\nAssistant:"
        
        response = self.generate(tokenizer, max_tokens=max_tokens, temperature=temperature, prompt=full_prompt)
        
        # Update conversation history
        self.conversation_history.append(f"User: {user_message}")
        self.conversation_history.append(f"Assistant: {response}")
        
        return response
    
    def call_function(self, tokenizer: Tokenizer, query: str,
                      max_tokens: int = 200, temperature: float = 0.3) -> Dict:
        """Function calling mode - returns structured JSON"""
        if self.use_case.mode != "function_calling":
            self.use_case = ModelUseCase.create_function_calling_model(self.use_case.tool_definitions)
        
        tools_context = ""
        if self.use_case.tool_definitions:
            tools_context = "Available tools:\n" + "\n".join(
                f"- {t.get('name', 'tool')}: {t.get('description', '')}" 
                for t in self.use_case.tool_definitions
            ) + "\n\n"
        
        prompt = f"{tools_context}Query: {query}\nRespond with JSON:"
        response_text = self.generate(tokenizer, max_tokens=max_tokens, temperature=temperature, prompt=prompt)
        
        # Try to parse as JSON
        import json
        try:
            # Extract JSON from response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response_text[start:end])
        except:
            pass
        
        return {"response": response_text, "raw": True}
    
    def save(self, path: str):
        """Save model weights"""
        data = {
            'config': self.config.to_dict(),
            'weights': [[p.data for p in row] for mat in self.state_dict.values() for row in mat]
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'GPTModel':
        """Load model from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        config = ModelConfig.from_dict(data['config'])
        model = cls(config)
        
        weight_iter = iter(data['weights'])
        for mat_name, mat in model.state_dict.items():
            for i, row in enumerate(mat):
                for j, _ in enumerate(row):
                    model.state_dict[mat_name][i][j].data = next(weight_iter)
        
        return model


class Trainer:
    """Training loop with Adam optimizer"""
    
    def __init__(self, model: GPTModel, tokenizer: Tokenizer, 
                 train_config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = train_config
        self.params = model.get_params()
        
        # Adam buffers
        self.m = [0.0] * len(self.params)
        self.v = [0.0] * len(self.params)
        
        # Training history
        self.loss_history = []
        self.step = 0
    
    def train_step(self, docs: List[str]) -> float:
        """Single training step"""
        doc = docs[self.step % len(docs)]
        tokens = self.tokenizer.encode(doc)
        n = min(self.model.config.block_size, len(tokens) - 1)
        
        if n == 0:
            return 0.0
        
        # Forward pass
        keys = [[] for _ in range(self.model.config.n_layer)]
        values = [[] for _ in range(self.model.config.n_layer)]
        losses = []
        
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = self.model.forward(token_id, pos_id, keys, values)
            probs = Matrix.softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        
        loss = (1 / n) * sum(losses)
        
        # Backward pass
        loss.backward()
        
        # Adam update
        lr_t = self.config.learning_rate * (1 - self.step / self.config.num_steps)
        
        for i, p in enumerate(self.params):
            self.m[i] = self.config.beta1 * self.m[i] + (1 - self.config.beta1) * p.grad
            self.v[i] = self.config.beta2 * self.v[i] + (1 - self.config.beta2) * p.grad ** 2
            
            m_hat = self.m[i] / (1 - self.config.beta1 ** (self.step + 1))
            v_hat = self.v[i] / (1 - self.config.beta2 ** (self.step + 1))
            
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + self.config.eps_adam)
            p.grad = 0
        
        self.loss_history.append(loss.data)
        self.step += 1
        
        return loss.data
    
    def train(self, docs: List[str], callback=None) -> List[float]:
        """Full training loop"""
        for step in range(self.config.num_steps):
            loss = self.train_step(docs)
            
            if callback and step % 10 == 0:
                callback(step, loss)
        
        return self.loss_history


class DatasetManager:
    """Manage datasets from various sources"""
    
    def __init__(self, cache_dir: str = "./datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_local(self, path: str) -> List[str]:
        """Load dataset from local file"""
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def load_from_huggingface(self, dataset_name: str, split: str = "train",
                             column: str = "text") -> List[str]:
        """Load dataset from Hugging Face Hub"""
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_name, split=split)
            return [str(item[column]) for item in dataset if column in item]
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    
    def download_url(self, url: str, filename: str = None) -> List[str]:
        """Download dataset from URL"""
        import urllib.request
        
        if filename is None:
            filename = url.split('/')[-1]
        
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            urllib.request.urlretrieve(url, filepath)
        
        return self.load_local(str(filepath))
    
    def create_sample_dataset(self, name: str = "names") -> List[str]:
        """Create sample dataset for testing"""
        if name == "names":
            return [
                "emma", "olivia", "ava", "isabella", "sophia",
                "charlotte", "mia", "amelia", "harper", "evelyn",
                "abigail", "emily", "elizabeth", "mila", "ella",
                "avery", "sofia", "camila", "aria", "scarlett"
            ]
        elif name == "code":
            return [
                "def hello(): print('world')",
                "class Foo: pass",
                "if x > 0: return True",
                "for i in range(10): print(i)"
            ]
        else:
            return self.create_sample_dataset("names")


@dataclass
class ExperimentResult:
    """Results from a training experiment"""
    model_config: ModelConfig
    training_config: TrainingConfig
    final_loss: float
    loss_history: List[float]
    samples: List[str]
    training_time: float
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            'model_config': self.model_config.to_dict(),
            'training_config': self.training_config.to_dict(),
            'final_loss': self.final_loss,
            'loss_history': self.loss_history,
            'samples': self.samples,
            'training_time': self.training_time,
            'timestamp': self.timestamp
        }
    
    def save(self, path: str):
        """Save results to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


print("✓ Core module loaded successfully")

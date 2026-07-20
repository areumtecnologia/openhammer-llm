# LLM Studio - Low-Cost Language Model Creator

## 🎯 Visão Geral

LLM Studio é um aplicativo desktop para criar e treinar modelos de linguagem em hardware de baixo custo. Baseado na implementação minimalista do GPT de @karpathy, este projeto democratiza o acesso ao desenvolvimento de IA.

## ✨ Funcionalidades

### Desktop App (GUI)
- **Interface Gráfica Intuitiva** - PySide6/Qt nativo
- **Configuração Visual de Modelo** - Ajuste camadas, embedding, attention heads
- **Treinamento com Progresso** - Barra de progresso e monitoramento de loss em tempo real
- **Geração de Texto** - Controle de temperatura e comprimento
- **Suporte a Datasets** - Hugging Face, arquivos locais, URLs
- **Salvar/Carregar Modelos** - Persistência de pesos treinados
- **Histórico de Experimentos** - Exportação em JSON

### CLI (Terminal)
- **Sem Dependências Gráficas** - Ideal para servidores e SSH
- **Menu Interativo** - Navegação simples por texto
- **Mesmas Funcionalidades** - Todo o poder do app desktop

## 🚀 Instalação

### Requisitos Mínimos
- Python 3.8+
- 4GB RAM (8GB recomendado)
- Sem necessidade de GPU

### Instalação Rápida

```bash
# Clone ou navegue até o diretório
cd /workspace

# Instale dependências opcionais
pip install PySide6  # Para GUI
pip install psutil   # Para detecção de hardware
pip install datasets # Para Hugging Face

# Execute o app desktop
python ui/app.py

# Ou use a versão CLI
python llm_studio_cli.py
```

### Script de Instalação Automática

```bash
chmod +x install.sh
./install.sh
```

## 📖 Uso Básico

### Via Interface Gráfica

1. **Configure o Modelo**
   - Acesse a aba "Model Config"
   - Ajuste parâmetros (camadas, embedding, etc.)
   - Clique em "Initialize Model"

2. **Carregue Dataset**
   - Vá para "Dataset" tab
   - Escolha fonte (sample, arquivo local, URL)
   - Clique em "Load Dataset"

3. **Treine**
   - Acesse "Training" tab
   - Configure hiperparâmetros
   - Clique em "Start Training"

4. **Gere Texto**
   - Vá para "Inference" tab
   - Ajuste temperatura
   - Clique em "Generate Text"

### Via CLI

```bash
python llm_studio_cli.py

# Menu interativo aparecerá:
# 1. Configure Model
# 2. Load Dataset
# 3. Train Model
# 4. Generate Text
# ...
```

## 🔧 Configurações Recomendadas

### Hardware Muito Limitado (< 8GB RAM)
```
Layers: 1-2
Embedding: 16-32
Block Size: 16-32
Heads: 2-4
Steps: 500-1000
```

### Hardware Moderado (8-16GB RAM)
```
Layers: 2-4
Embedding: 32-64
Block Size: 32-64
Heads: 4-8
Steps: 1000-2000
```

### Hardware Bom (> 16GB RAM)
```
Layers: 4-8
Embedding: 64-128
Block Size: 64-128
Heads: 8-16
Steps: 2000-5000
```

## 📁 Estrutura do Projeto

```
/workspace/
├── core/
│   └── model.py          # Núcleo: modelo, treino, tokenizer
├── ui/
│   └── app.py            # Interface gráfica (PySide6)
├── backends/             # Backends futuros (ONNX, llama.cpp)
├── config/               # Configurações salvas
├── datasets/             # Cache de datasets
├── assets/               # Recursos visuais
├── llm_studio_cli.py     # Interface CLI
├── install.sh            # Script de instalação
└── README_APP.md         # Esta documentação
```

## 🧪 Exemplo de Código

```python
from core.model import ModelConfig, GPTModel, Trainer, Tokenizer, DatasetManager

# Configurar modelo
config = ModelConfig(
    n_layer=2,
    n_embd=32,
    block_size=32,
    n_head=4
)

model = GPTModel(config)

# Carregar dataset
dm = DatasetManager()
docs = dm.create_sample_dataset("names")
tokenizer = Tokenizer(docs)

# Treinar
from core.model import TrainingConfig
train_config = TrainingConfig(num_steps=1000)
trainer = Trainer(model, tokenizer, train_config)
trainer.train(docs)

# Gerar texto
text = model.generate(tokenizer, max_tokens=50)
print(text)
```

## 🎯 Casos de Uso

### Educacional
- Aprenda como transformers funcionam
- Experimente com arquiteturas diferentes
- Entenda o processo de treinamento

### Prototipagem
- Teste ideias rapidamente
- Valide conceitos antes de escalar
- Desenvolva modelos customizados

### Hardware Limitado
- Raspberry Pi
- Laptops antigos
- Sistemas sem GPU

## ⚙️ Integrações Futuras

### Planejadas
- [ ] **ONNX Runtime** - Aceleração agnóstica
- [ ] **llama.cpp** - Inferência otimizada em CPU
- [ ] **Hugging Face Hub** - Download direto de modelos
- [ ] **QLoRA** - Fine-tuning eficiente
- [ ] **Quantização** - Redução de memória (INT8, INT4)
- [ ] **Exportação GGUF** - Compatibilidade com llama.cpp

### Em Pesquisa
- [ ] Suporte a NPUs (Intel, AMD)
- [ ] Offloading de KV cache
- [ ] Model routing dinâmico
- [ ] Composição de adapters PEFT

## 🐛 Solução de Problemas

### Erro: "PySide6 not installed"
```bash
pip install PySide6
# Ou use a versão CLI
python llm_studio_cli.py
```

### Erro: "Memory Error"
- Reduza o tamanho do modelo
- Diminua block_size e embedding
- Use menos camadas

### Treinamento Lento
- Reduza num_steps para teste
- Use modelo menor
- Considere menos camadas

## 📊 Benchmarks Esperados

| Hardware | Modelo | Velocidade | Uso RAM |
|----------|--------|------------|---------|
| Raspberry Pi 4 | Tiny (1K params) | ~50 tokens/s | ~100MB |
| Laptop i5 (8GB) | Small (10K params) | ~20 tokens/s | ~500MB |
| Desktop (16GB) | Medium (100K params) | ~5 tokens/s | ~2GB |

*Nota: Implementação pura Python, sem otimizações C/C++*

## 🤝 Contribuindo

Contribuições são bem-vindas! Áreas de interesse:
- Otimizações de desempenho
- Novos backends (ONNX, llama.cpp)
- Melhorias de UI/UX
- Documentação
- Testes

## 📄 Licença

Baseado no código original de @karpathy (MIT).
Este projeto segue licença MIT.

## 🔗 Links Úteis

- [Código Original do Karpathy](https://github.com/karpathy/makemore)
- [Documentação PySide6](https://doc.qt.io/qtforpython/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [ONNX Runtime](https://onnxruntime.ai/)

## 💡 Dicas

1. **Comece pequeno** - Modele tiny para testar
2. **Monitore o loss** - Deve diminuir com treino
3. **Ajuste temperatura** - Baixa (= previsível), Alta (= criativo)
4. **Salve checkpoints** - Não perca progresso
5. **Use samples** - Dataset内置 para testes rápidos

---

**Criado com ❤️ para democratizar IA**

*"The most atomic way to train and run inference for a GPT in pure, dependency-free Python. Everything else is just efficiency."* - @karpathy

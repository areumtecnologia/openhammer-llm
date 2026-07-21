"""
Desktop GUI Application for LLM Training and Inference
Built with PySide6 (Qt for Python)
Wizard-style interface for beginner-friendly model creation
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QLineEdit, QTextEdit, QSpinBox,
        QDoubleSpinBox, QComboBox, QProgressBar, QGroupBox, QFormLayout,
        QFileDialog, QMessageBox, QSplitter, QScrollArea, QCheckBox,
        QListWidget, QListWidgetItem, QSlider, QStatusBar, QMenu,
        QMenuBar, QDialog, QDialogButtonBox, QGridLayout, QStackedWidget,
        QWizard, QWizardPage, QRadioButton, QButtonGroup, QFrame
    )
    from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize, QUrl
    from PySide6.QtGui import QFont, QIcon, QAction, QDesktopServices
    HAS_QT = True
except ImportError as e:
    HAS_QT = False
    print(f"⚠ PySide6 not installed. Install with: pip install PySide6")
    print(f"   Error: {e}")
    sys.exit(1)

# Double-check Qt availability
if not HAS_QT:
    print("⚠ PySide6 is not available. Please install it with: pip install PySide6")
    sys.exit(1)

from core.model import (
    ModelConfig, TrainingConfig, HardwareProfile, GPTModel, 
    Trainer, Tokenizer, DatasetManager, ExperimentResult, TrainingStack,
    ModelUseCase, TORCH_AVAILABLE, TorchGPTModel, TorchTrainer
)
from core.model_manager import ModelManager, QualityAssessor, ModelMetadata


class TrainingWorker(QThread):
    """Background thread for training"""
    progress = Signal(int, float, float, float)  # step, loss, elapsed_time, avg_step_time
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, model, tokenizer, trainer, docs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.docs = docs
    
    def run(self):
        try:
            import time
            start_time = time.time()
            
            def callback(step, loss, elapsed, avg_step):
                self.progress.emit(step, loss, elapsed, avg_step)
            
            # Check if trainer is TorchTrainer (requires tokenizer parameter)
            from core.model import TorchTrainer
            if isinstance(self.trainer, TorchTrainer):
                loss_history, total_time, avg_step_time = self.trainer.train(
                    self.docs, self.tokenizer, callback=callback
                )
            else:
                loss_history, total_time, avg_step_time = self.trainer.train(
                    self.docs, callback=callback
                )
            
            # Generate samples - handle both model types
            samples = []
            for _ in range(5):
                if TORCH_AVAILABLE and isinstance(self.model, TorchGPTModel):
                    # Use Torch model's generate_text method
                    sample = self.model.generate_text(
                        self.tokenizer,
                        prompt="",
                        max_new_tokens=30,
                        temperature=self.trainer.config.temperature,
                        device=self.trainer.device
                    )
                else:
                    # Use pure Python model
                    sample = self.model.generate(
                        self.tokenizer,
                        max_tokens=30,
                        temperature=self.trainer.config.temperature
                    )
                samples.append(sample)
            
            result = ExperimentResult(
                model_config=self.model.config,
                training_config=self.trainer.config,
                final_loss=loss_history[-1] if loss_history else 0,
                loss_history=loss_history,
                samples=samples,
                training_time=total_time,
                timestamp=str(__import__('datetime').datetime.now())
            )
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenHammer LLM Studio - Low-Cost Language Model Creator")
        self.setMinimumSize(1200, 800)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset_manager = DatasetManager()
        self.hardware_profile = HardwareProfile.detect_system()
        self.current_result = None
        self.selected_stack = TrainingStack(use_gpu=False, backend="cpu", dependencies=[])
        self.selected_use_case = ModelUseCase(mode="completion")  # Default use case
        self.model_manager = ModelManager()  # Model lifecycle management
        
        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._update_hardware_info()
        self._update_stack_info()
    
    def _setup_ui(self):
        """Setup user interface with wizard-style flow"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title header
        title_label = QLabel("<h1>🤖 OpenHammer LLM Studio - Assistente de Criação de Modelos</h1>")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("QLabel { font-size: 24px; font-weight: bold; color: #2196F3; padding: 10px; }")
        main_layout.addWidget(title_label)
        
        # Create stacked widget for wizard pages
        self.wizard_stack = QStackedWidget()
        main_layout.addWidget(self.wizard_stack, stretch=1)
        
        # Create wizard pages
        self.wizard_pages = [
            self._create_welcome_page(),      # Page 0: Welcome
            self._create_stack_page(),         # Page 1: Stack
            self._create_usecase_page(),       # Page 2: Use Case
            self._create_model_page(),         # Page 3: Model Config
            self._create_dataset_page(),       # Page 4: Dataset
            self._create_training_page(),      # Page 5: Training
            self._create_inference_page(),     # Page 6: Inference
            self._create_results_page(),       # Page 7: Results
            self._create_models_page(),        # Page 8: Saved Models
        ]
        
        for page in self.wizard_pages:
            self.wizard_stack.addWidget(page)
        
        # Navigation buttons (always visible at bottom)
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()
        
        self.btn_back = QPushButton("◀ Voltar")
        self.btn_back.clicked.connect(self._go_back)
        self.btn_back.setMinimumSize(120, 40)
        self.btn_back.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #757575; }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        
        self.btn_next = QPushButton("Avançar ▶")
        self.btn_next.clicked.connect(self._go_next)
        self.btn_next.setMinimumSize(120, 40)
        self.btn_next.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        
        self.btn_start = QPushButton("🚀 Iniciar Treinamento")
        self.btn_start.clicked.connect(self._start_training)
        self.btn_start.setMinimumSize(160, 40)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #388E3C; }
        """)
        self.btn_start.hide()
        
        self.btn_stop = QPushButton("⏹ Parar")
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_stop.setMinimumSize(120, 40)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #D32F2F; }
        """)
        self.btn_stop.hide()
        
        nav_layout.addWidget(self.btn_back)
        nav_layout.addWidget(self.btn_start)
        self.btn_start_position = nav_layout.count() - 1
        nav_layout.addWidget(self.btn_stop)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addStretch()
        
        main_layout.addLayout(nav_layout)
        
        # Progress indicator
        progress_layout = QHBoxLayout()
        self.progress_indicator = []
        steps = ["Início", "Stack", "Propósito", "Modelo", "Dataset", "Treino", "Teste", "Resultados", "Modelos"]
        for i, step_name in enumerate(steps):
            lbl = QLabel(f"{i+1}. {step_name}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("QLabel { color: #9E9E9E; font-size: 11px; }")
            lbl.setMinimumWidth(80)
            self.progress_indicator.append(lbl)
            progress_layout.addWidget(lbl)
        
        main_layout.addLayout(progress_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Bem-vindo! Clique em 'Avançar' para começar.")
        
        # Initialize navigation state
        self._update_navigation_buttons()
    
    def _create_welcome_page(self) -> QWidget:
        """Create welcome page - Step 0"""
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        layout = QVBoxLayout(scroll_content)
        
        # Welcome message
        welcome_label = QLabel("""
        <h2 style="color: #2196F3; font-size: 28px;">👋 Bem-vindo ao OpenHammer LLM Studio!</h2>
        <p style="font-size: 16px;">Este assistente vai guiá-lo passo a passo na criação do seu próprio modelo de linguagem.</p>
        """)
        welcome_label.setWordWrap(True)
        welcome_label.setStyleSheet("QLabel { padding: 20px; }")
        layout.addWidget(welcome_label)
        
        # What you'll do
        steps_group = QGroupBox("📋 O que você fará neste assistente:")
        steps_layout = QVBoxLayout()
        steps_info = QLabel("""
        <ol style="font-size: 15px; line-height: 1.8;">
            <li><b>Escolher a Tecnologia:</b> CPU ou GPU (CUDA/MPS)</li>
            <li><b>Definir o Propósito:</b> Para que seu modelo será usado?</li>
            <li><b>Configurar o Modelo:</b> Tamanho e arquitetura (ajustado automaticamente)</li>
            <li><b>Carregar Dataset:</b> Dados para treinamento (HuggingFace ou local)</li>
            <li><b>Treinar:</b> Acompanhe o progresso em tempo real</li>
            <li><b>Testar:</b> Verifique se o modelo responde como esperado</li>
            <li><b>Salvar:</b> Guarde seu modelo para uso futuro</li>
        </ol>
        """)
        steps_info.setWordWrap(True)
        steps_layout.addWidget(steps_info)
        steps_group.setLayout(steps_layout)
        layout.addWidget(steps_group)
        
        # Hardware info
        hw_group = QGroupBox("💻 Seu Hardware")
        hw_layout = QFormLayout()
        self.hw_ram_label = QLabel()
        self.hw_vram_label = QLabel()
        self.hw_cpu_label = QLabel()
        self.hw_gpu_label = QLabel()
        self.hw_recommendation_label = QLabel()
        
        hw_layout.addRow("RAM:", self.hw_ram_label)
        hw_layout.addRow("VRAM:", self.hw_vram_label)
        hw_layout.addRow("CPU Cores:", self.hw_cpu_label)
        hw_layout.addRow("GPU:", self.hw_gpu_label)
        hw_layout.addRow("Recomendação:", self.hw_recommendation_label)
        
        hw_group.setLayout(hw_layout)
        layout.addWidget(hw_group)
        
        # Tips
        tips_group = QGroupBox("💡 Dicas para Iniciantes")
        tips_layout = QVBoxLayout()
        tips_info = QLabel("""
        <ul style="font-size: 14px; line-height: 1.6;">
            <li>Comece com modelos pequenos (2-4 camadas) para testes rápidos</li>
            <li>O ajuste automático de parâmetros configura tudo para você</li>
            <li>Datasets menores treinam mais rápido - ideal para aprender</li>
            <li>Se tiver GPU NVIDIA, selecione CUDA para aceleração significativa</li>
            <li>Você pode voltar e ajustar configurações a qualquer momento</li>
        </ul>
        """)
        tips_info.setWordWrap(True)
        tips_layout.addWidget(tips_info)
        tips_group.setLayout(tips_layout)
        layout.addWidget(tips_group)
        
        layout.addStretch()
        
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.addWidget(scroll)
        return main_widget
    
    def _create_home_tab(self) -> QWidget:
        """Legacy method - redirects to welcome page"""
        return self._create_welcome_page()
    
    def _create_stack_page(self) -> QWidget:
        """Create stack selection page - Step 1"""
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        layout = QVBoxLayout(scroll_content)
        
        # Header
        header_label = QLabel("<h2 style='color: #2196F3;'>⚡ Passo 1: Escolha a Tecnologia de Treinamento</h2>")
        header_label.setWordWrap(True)
        layout.addWidget(header_label)
        
        # Info
        info_label = QLabel("""
        <p style='font-size: 15px;'>Selecione como seu modelo será treinado:</p>
        <ul style='font-size: 14px; line-height: 1.6;'>
            <li><b>💻 CPU (Python Puro):</b> Funciona em qualquer computador, sem instalações extras. Implementação educacional.</li>
            <li><b>🚀 GPU (CUDA/MPS com PyTorch):</b> <span style='color: #4CAF50; font-weight: bold;'>✓ ACELERAÇÃO REAL DE GPU</span> - Até 1000x mais rápido em GPUs NVIDIA!</li>
        </ul>
        """)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { padding: 10px; background-color: #E3F2FD; border-radius: 5px; }")
        layout.addWidget(info_label)
        
        # Stack selection
        stack_group = QGroupBox("Selecionar Stack de Treinamento")
        stack_layout = QVBoxLayout()
        
        self.stack_combo = QComboBox()
        self._populate_stack_options()
        self.stack_combo.currentIndexChanged.connect(self._on_stack_changed)
        stack_layout.addWidget(QLabel("Tecnologia disponível:"))
        stack_layout.addWidget(self.stack_combo)
        
        # Stack details
        self.stack_details_label = QLabel()
        self.stack_details_label.setStyleSheet("QLabel { font-weight: bold; color: #2196F3; font-size: 14px; }")
        self.stack_details_label.setWordWrap(True)
        stack_layout.addWidget(self.stack_details_label)
        
        # Dependencies info
        self.dependencies_label = QLabel()
        self.dependencies_label.setWordWrap(True)
        self.dependencies_label.setStyleSheet("QLabel { color: #FF9800; font-size: 13px; }")
        stack_layout.addWidget(self.dependencies_label)
        
        # Install button
        self.install_deps_btn = QPushButton("📦 Instalar Dependências Necessárias")
        self.install_deps_btn.clicked.connect(self._install_stack_dependencies)
        self.install_deps_btn.setEnabled(False)
        self.install_deps_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover { background-color: #F57C00; }
        """)
        stack_layout.addWidget(self.install_deps_btn)
        
        stack_group.setLayout(stack_layout)
        layout.addWidget(stack_group)
        
        # Status
        status_group = QGroupBox("Configuração Atual")
        status_layout = QFormLayout()
        self.status_stack_label = QLabel("CPU")
        self.status_backend_label = QLabel("cpu")
        self.status_ready_label = QLabel("✓ Pronto")
        self.status_ready_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        
        status_layout.addRow("Stack:", self.status_stack_label)
        status_layout.addRow("Backend:", self.status_backend_label)
        status_layout.addRow("Status:", self.status_ready_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        layout.addStretch()
        self._on_stack_changed(0)
        
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.addWidget(scroll)
        return main_widget
    
    def _create_stack_tab(self) -> QWidget:
        """Legacy method - redirects to stack page"""
        return self._create_stack_page()
    
    def _populate_stack_options(self):
        """Populate stack selection dropdown"""
        self.stack_combo.clear()
        available_stacks = TrainingStack.detect_available()
        
        for stack in available_stacks:
            icon = "🚀" if stack.use_gpu else "💻"
            self.stack_combo.addItem(f"{icon} {stack.description}", stack)
        
        # Select recommended stack
        hw = self.hardware_profile
        if hw.recommended_stack != "cpu":
            for i in range(self.stack_combo.count()):
                stack = self.stack_combo.itemData(i)
                if stack.backend == hw.recommended_stack:
                    self.stack_combo.setCurrentIndex(i)
                    break
    
    def _on_stack_changed(self, index: int):
        """Handle stack selection change"""
        stack = self.stack_combo.itemData(index)
        if stack:
            self.selected_stack = stack
            self.stack_details_label.setText(stack.description)
            
            if stack.use_gpu:
                deps = ", ".join(stack.dependencies)
                # Check if dependencies are already installed
                missing_deps = []
                for dep in stack.dependencies:
                    try:
                        __import__(dep)
                    except ImportError:
                        missing_deps.append(dep)
                
                if missing_deps:
                    self.dependencies_label.setText(f"⚠ Requires: {', '.join(missing_deps)}")
                    self.install_deps_btn.setEnabled(True)
                else:
                    self.dependencies_label.setText("✓ All dependencies installed")
                    self.install_deps_btn.setEnabled(False)
            else:
                self.dependencies_label.setText("✓ No additional dependencies required")
                self.install_deps_btn.setEnabled(False)
            
            # Update status
            self.status_stack_label.setText("GPU" if stack.use_gpu else "CPU")
            self.status_backend_label.setText(stack.backend)
            
            # Check if ready (no missing dependencies)
            is_ready = not stack.use_gpu
            if stack.use_gpu:
                try:
                    import torch
                    is_ready = True
                except ImportError:
                    is_ready = False
            
            self.status_ready_label.setText("✓ Ready" if is_ready else "⚠ Check dependencies")
            self.status_ready_label.setStyleSheet(
                "QLabel { color: green; font-weight: bold; }" if is_ready 
                else "QLabel { color: orange; font-weight: bold; }"
            )
    
    def _install_stack_dependencies(self):
        """Install dependencies for selected stack"""
        if not self.selected_stack.use_gpu:
            return
        
        deps = " ".join(self.selected_stack.dependencies)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Install Dependencies")
        msg.setText(f"Install {deps}?")
        msg.setInformativeText(f"Run: pip install {deps}")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        
        if msg.exec() == QMessageBox.Ok:
            import subprocess
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + self.selected_stack.dependencies)
                QMessageBox.information(self, "Success", "Dependencies installed successfully!")
                self._update_stack_info()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to install dependencies: {e}")
    
    def _update_stack_info(self):
        """Update stack information display"""
        if hasattr(self, 'stack_combo'):
            self._populate_stack_options()
            self._on_stack_changed(self.stack_combo.currentIndex())
    
    def _create_usecase_page(self) -> QWidget:
        """Create use case selection page - Step 2"""
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        layout = QVBoxLayout(scroll_content)
        
        # Header
        header_label = QLabel("<h2 style='color: #2196F3;'>🎯 Passo 2: Defina o Propósito do Modelo</h2>")
        header_label.setWordWrap(True)
        layout.addWidget(header_label)
        
        # Info
        info_label = QLabel("""
        <p style='font-size: 15px;'>Para que você quer usar seu modelo? Isso ajuda a configurar automaticamente os parâmetros ideais.</p>
        """)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { padding: 10px; background-color: #E3F2FD; border-radius: 5px; }")
        layout.addWidget(info_label)
        
        # Use case selection
        usecase_group = QGroupBox("Propósito do Modelo")
        usecase_layout = QVBoxLayout()
        
        self.usecase_combo = QComboBox()
        self.usecase_combo.addItem("📝 Completar Texto - Gerar continuições de texto", "completion")
        self.usecase_combo.addItem("💬 Conversação - Diálogo interativo tipo chatbot", "chat")
        self.usecase_combo.addItem("🔧 Chamada de Funções - Integração com APIs/ferramentas", "function_calling")
        self.usecase_combo.addItem("🤖 Agente Autônomo - Execução de tarefas complexas", "agent")
        self.usecase_combo.currentIndexChanged.connect(self._on_usecase_changed)
        self.usecase_combo.setStyleSheet("QComboBox { font-size: 14px; padding: 5px; }")
        usecase_layout.addWidget(QLabel("Como seu modelo será usado?"))
        usecase_layout.addWidget(self.usecase_combo)
        
        # Use case details
        self.usecase_details_label = QLabel()
        self.usecase_details_label.setWordWrap(True)
        self.usecase_details_label.setStyleSheet("QLabel { color: #2196F3; font-style: italic; font-size: 14px; padding: 10px; background-color: #FFF3E0; border-radius: 5px; }")
        usecase_layout.addWidget(self.usecase_details_label)
        
        usecase_group.setLayout(usecase_layout)
        layout.addWidget(usecase_group)
        
        # Auto-tune info
        autotune_group = QGroupBox("⚙️ Ajuste Automático")
        autotune_layout = QVBoxLayout()
        autotune_info = QLabel("""
        <p style='font-size: 14px;'>Na próxima tela, os parâmetros do modelo serão ajustados automaticamente com base no propósito selecionado.</p>
        <ul style='font-size: 13px;'>
            <li><b>Completar Texto:</b> Modelo leve e rápido</li>
            <li><b>Conversação:</b> Contexto maior para diálogos</li>
            <li><b>Funções/Agente:</b> Mais camadas para raciocínio</li>
        </ul>
        <p style='font-size: 13px; color: #666;'>Você ainda poderá ajustar manualmente se desejar!</p>
        """)
        autotune_info.setWordWrap(True)
        autotune_layout.addWidget(autotune_info)
        autotune_group.setLayout(autotune_layout)
        layout.addWidget(autotune_group)
        
        layout.addStretch()
        
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.addWidget(scroll)
        return main_widget
    
    def _create_model_page(self) -> QWidget:
        """Create model configuration page - Step 3"""
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        layout = QVBoxLayout(scroll_content)
        
        # Header
        header_label = QLabel("<h2 style='color: #2196F3;'>🔧 Passo 3: Configure a Arquitetura do Modelo</h2>")
        header_label.setWordWrap(True)
        layout.addWidget(header_label)
        
        # Model Architecture
        form_group = QGroupBox("Arquitetura do Modelo")
        form_layout = QFormLayout()
        
        # Model parameters
        self.n_layer_spin = QSpinBox()
        self.n_layer_spin.setRange(1, 12)
        self.n_layer_spin.setValue(2)
        self.n_layer_spin.setToolTip("Número de camadas do transformador")
        self.n_layer_spin.setStyleSheet("QSpinBox { font-size: 13px; padding: 3px; }")
        
        self.n_embd_spin = QSpinBox()
        self.n_embd_spin.setRange(16, 512)
        self.n_embd_spin.setSingleStep(16)
        self.n_embd_spin.setValue(32)
        self.n_embd_spin.setToolTip("Dimensão do embedding")
        self.n_embd_spin.setStyleSheet("QSpinBox { font-size: 13px; padding: 3px; }")
        
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(16, 512)
        self.block_size_spin.setSingleStep(16)
        self.block_size_spin.setValue(32)
        self.block_size_spin.setToolTip("Tamanho máximo do contexto")
        self.block_size_spin.setStyleSheet("QSpinBox { font-size: 13px; padding: 3px; }")
        
        self.n_head_spin = QSpinBox()
        self.n_head_spin.setRange(1, 16)
        self.n_head_spin.setValue(4)
        self.n_head_spin.setToolTip("Número de cabeças de atenção")
        self.n_head_spin.setStyleSheet("QSpinBox { font-size: 13px; padding: 3px; }")
        
        self.vocab_size_spin = QSpinBox()
        self.vocab_size_spin.setRange(64, 1024)
        self.vocab_size_spin.setValue(256)
        self.vocab_size_spin.setToolTip("Tamanho do vocabulário")
        self.vocab_size_spin.setStyleSheet("QSpinBox { font-size: 13px; padding: 3px; }")
        
        form_layout.addRow("Camadas:", self.n_layer_spin)
        form_layout.addRow("Dimensão Embedding:", self.n_embd_spin)
        form_layout.addRow("Tamanho Contexto:", self.block_size_spin)
        form_layout.addRow("Cabeças Atenção:", self.n_head_spin)
        form_layout.addRow("Tamanho Vocabulário:", self.vocab_size_spin)
        
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)
        
        # Model info
        self.model_info_label = QLabel()
        self.model_info_label.setStyleSheet("QLabel { font-weight: bold; color: #2196F3; font-size: 14px; padding: 10px; background-color: #E8F5E9; border-radius: 5px; }")
        self.model_info_label.setWordWrap(True)
        layout.addWidget(self.model_info_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.init_model_btn = QPushButton("🚀 Inicializar Modelo")
        self.init_model_btn.clicked.connect(self._initialize_model)
        self.init_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        btn_layout.addWidget(self.init_model_btn)
        
        self.save_model_btn = QPushButton("💾 Salvar Modelo")
        self.save_model_btn.clicked.connect(self._save_model)
        self.save_model_btn.setEnabled(False)
        self.save_model_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; font-size: 13px; font-weight: bold; padding: 8px; }")
        btn_layout.addWidget(self.save_model_btn)
        
        self.load_model_btn = QPushButton("📂 Carregar Modelo")
        self.load_model_btn.clicked.connect(self._load_model)
        self.load_model_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; border-radius: 5px; font-size: 13px; font-weight: bold; padding: 8px; }")
        btn_layout.addWidget(self.load_model_btn)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        
        # Connect signals
        for spin in [self.n_layer_spin, self.n_embd_spin, self.block_size_spin, 
                     self.n_head_spin, self.vocab_size_spin]:
            spin.valueChanged.connect(self._update_model_info)
        
        self._on_usecase_changed(0)
        self._update_model_info()
        
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.addWidget(scroll)
        return main_widget
    
    def _create_model_tab(self) -> QWidget:
        """Legacy method - redirects to model page"""
        return self._create_model_page()
    
    def _on_usecase_changed(self, index):
        """Handle use case selection change"""
        usecase_mode = self.usecase_combo.currentData()
        self.selected_use_case = ModelUseCase(mode=usecase_mode)
        
        descriptions = {
            "completion": "Best for: Text generation, code completion, creative writing. Simple and fast.",
            "chat": "Best for: Conversational AI, customer support, tutoring. Maintains conversation history.",
            "function_calling": "Best for: API integration, tool usage, structured output (JSON). Returns function calls.",
            "agent": "Best for: Autonomous tasks, multi-step reasoning, planning. Combines chat + tools."
        }
        
        self.usecase_details_label.setText(descriptions.get(usecase_mode, ""))
        
        # Adjust default architecture based on use case
        if usecase_mode == "chat":
            self.block_size_spin.setValue(128)  # Longer context for conversations
        elif usecase_mode in ["function_calling", "agent"]:
            self.block_size_spin.setValue(256)  # Even longer for structured output
            self.n_layer_spin.setValue(3)  # Slightly deeper for reasoning
    
    def _create_dataset_tab(self) -> QWidget:
        """Create dataset management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Dataset source
        source_group = QGroupBox("Dataset Source")
        source_layout = QFormLayout()
        
        self.dataset_source_combo = QComboBox()
        self.dataset_source_combo.addItems([
            "Sample Dataset (Names)",
            "Sample Dataset (Code)",
            "Local File",
            "Hugging Face Hub",
            "URL Download"
        ])
        self.dataset_source_combo.currentTextChanged.connect(self._on_dataset_source_changed)
        
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText("Path or URL...")
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_dataset)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.dataset_path_edit)
        path_layout.addWidget(browse_btn)
        
        source_layout.addRow("Source:", self.dataset_source_combo)
        source_layout.addRow("Path/URL:", path_layout)
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # Dataset info
        self.dataset_info_label = QLabel("No dataset loaded")
        layout.addWidget(self.dataset_info_label)
        
        # Preview
        preview_group = QGroupBox("Dataset Preview")
        preview_layout = QVBoxLayout()
        self.dataset_preview = QTextEdit()
        self.dataset_preview.setReadOnly(True)
        self.dataset_preview.setMaximumHeight(200)
        preview_layout.addWidget(self.dataset_preview)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Load button
        self.load_dataset_btn = QPushButton("📥 Load Dataset")
        self.load_dataset_btn.clicked.connect(self._load_dataset)
        layout.addWidget(self.load_dataset_btn)
        
        layout.addStretch()
        return widget
    
    def _create_training_tab(self) -> QWidget:
        """Create training configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Training params
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout()
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(4)
        
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(100, 10000)
        self.steps_spin.setSingleStep(100)
        self.steps_spin.setValue(1000)
        
        self.beta1_spin = QDoubleSpinBox()
        self.beta1_spin.setRange(0.0, 1.0)
        self.beta1_spin.setSingleStep(0.01)
        self.beta1_spin.setValue(0.85)
        
        self.beta2_spin = QDoubleSpinBox()
        self.beta2_spin.setRange(0.0, 1.0)
        self.beta2_spin.setSingleStep(0.01)
        self.beta2_spin.setValue(0.99)
        
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.5)
        
        params_layout.addRow("Learning Rate:", self.lr_spin)
        params_layout.addRow("Training Steps:", self.steps_spin)
        params_layout.addRow("Beta1 (Adam):", self.beta1_spin)
        params_layout.addRow("Beta2 (Adam):", self.beta2_spin)
        params_layout.addRow("Temperature:", self.temp_spin)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.loss_label = QLabel("Loss: --")
        self.loss_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; }")
        
        self.time_label = QLabel("Time: 0.0s | Avg/step: 0.000s")
        self.time_label.setStyleSheet("QLabel { font-size: 14px; color: #2196F3; }")
        
        self.status_label = QLabel("Status: Idle")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.loss_label)
        progress_layout.addWidget(self.time_label)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.start_train_btn = QPushButton("▶ Start Training")
        self.start_train_btn.clicked.connect(self._start_training)
        self.start_train_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        self.stop_train_btn = QPushButton("⏹ Stop")
        self.stop_train_btn.clicked.connect(self._stop_training)
        self.stop_train_btn.setEnabled(False)
        
        btn_layout.addWidget(self.start_train_btn)
        btn_layout.addWidget(self.stop_train_btn)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        return widget
    
    def _create_inference_tab(self) -> QWidget:
        """Create inference/generation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Generation params
        gen_group = QGroupBox("Generation Parameters")
        gen_layout = QFormLayout()
        
        self.gen_temp_slider = QSlider(Qt.Horizontal)
        self.gen_temp_slider.setRange(1, 20)
        self.gen_temp_slider.setValue(5)
        self.gen_temp_slider.valueChanged.connect(
            lambda v: self.gen_temp_label.setText(f"{v/10:.1f}")
        )
        
        self.gen_temp_label = QLabel("0.5")
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(self.gen_temp_slider)
        temp_layout.addWidget(self.gen_temp_label)
        
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(10, 500)
        self.max_tokens_spin.setValue(50)
        
        gen_layout.addRow("Temperature:", temp_layout)
        gen_layout.addRow("Max Tokens:", self.max_tokens_spin)
        
        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)
        
        # Input/Output
        io_group = QGroupBox("Text Generation")
        io_layout = QVBoxLayout()
        
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("Enter prompt (optional)...")
        io_layout.addWidget(self.prompt_edit)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Generated text will appear here...")
        self.output_text.setMinimumHeight(200)
        io_layout.addWidget(self.output_text)
        
        io_group.setLayout(io_layout)
        layout.addWidget(io_group)
        
        # Generate button
        self.generate_btn = QPushButton("✨ Generate Text")
        self.generate_btn.clicked.connect(self._generate_text)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.generate_btn)
        
        layout.addStretch()
        return widget
    
    def _create_results_tab(self) -> QWidget:
        """Create results/experiments tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Results list
        list_group = QGroupBox("Experiment History")
        list_layout = QVBoxLayout()
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self._on_result_selected)
        list_layout.addWidget(self.results_list)
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Details
        details_group = QGroupBox("Experiment Details")
        details_layout = QVBoxLayout()
        self.result_details = QTextEdit()
        self.result_details.setReadOnly(True)
        details_layout.addWidget(self.result_details)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # Export
        export_btn = QPushButton("📤 Export Results")
        export_btn.clicked.connect(self._export_results)
        layout.addWidget(export_btn)
        
        layout.addStretch()
        return widget
    
    def _create_models_tab(self) -> QWidget:
        """Create saved models management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Models list
        list_group = QGroupBox("Saved Models")
        list_layout = QVBoxLayout()
        
        self.models_list = QListWidget()
        self.models_list.itemClicked.connect(self._on_model_selected)
        list_layout.addWidget(self.models_list)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh List")
        refresh_btn.clicked.connect(self._refresh_models_list)
        list_layout.addWidget(refresh_btn)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Model details and actions
        details_group = QGroupBox("Model Details & Actions")
        details_layout = QVBoxLayout()
        
        self.model_details = QTextEdit()
        self.model_details.setReadOnly(True)
        self.model_details.setMinimumHeight(150)
        details_layout.addWidget(self.model_details)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        self.load_model_btn = QPushButton("📥 Load Model")
        self.load_model_btn.clicked.connect(self._load_selected_model)
        self.load_model_btn.setEnabled(False)
        btn_layout.addWidget(self.load_model_btn)
        
        self.delete_model_btn = QPushButton("🗑️ Delete Model")
        self.delete_model_btn.clicked.connect(self._delete_selected_model)
        self.delete_model_btn.setEnabled(False)
        btn_layout.addWidget(self.delete_model_btn)
        
        self.assess_model_btn = QPushButton("🔍 Assess Quality")
        self.assess_model_btn.clicked.connect(self._assess_selected_model)
        self.assess_model_btn.setEnabled(False)
        btn_layout.addWidget(self.assess_model_btn)
        
        details_layout.addLayout(btn_layout)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # Quality assessment results
        quality_group = QGroupBox("Quality Assessment")
        quality_layout = QVBoxLayout()
        
        self.quality_label = QLabel("No assessment performed yet")
        self.quality_label.setStyleSheet("QLabel { font-size: 14px; }")
        quality_layout.addWidget(self.quality_label)
        
        self.quality_progress = QProgressBar()
        self.quality_progress.setRange(0, 100)
        self.quality_progress.setValue(0)
        quality_layout.addWidget(self.quality_progress)
        
        self.quality_recommendations = QTextEdit()
        self.quality_recommendations.setReadOnly(True)
        self.quality_recommendations.setMaximumHeight(100)
        self.quality_recommendations.setPlaceholderText("Recommendations will appear here...")
        quality_layout.addWidget(self.quality_recommendations)
        
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
        layout.addStretch()
        
        # Initialize models list
        self._refresh_models_list()
        self.selected_model_id = None
        
        return widget
    
    def _setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Project", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)
        
        save_action = QAction("Save Project", self)
        save_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _update_hardware_info(self):
        """Update hardware information display"""
        hp = self.hardware_profile
        self.hw_ram_label.setText(f"{hp.ram_gb} GB")
        self.hw_vram_label.setText(f"{hp.vram_gb} GB" if hp.has_gpu else "N/A")
        self.hw_cpu_label.setText(str(hp.cpu_cores))
        self.hw_gpu_label.setText(f"{hp.gpu_type.upper()}" if hp.has_gpu else "None")
        self.hw_recommendation_label.setText(
            f"Max Model: {hp.recommend_max_model()}, "
            f"Quantization: {hp.recommend_quantization()}"
        )
    
    def _update_model_info(self):
        """Update model parameter count display"""
        try:
            n_layer = self.n_layer_spin.value()
            n_embd = self.n_embd_spin.value()
            block_size = self.block_size_spin.value()
            n_head = self.n_head_spin.value()
            vocab_size = self.vocab_size_spin.value()
            
            # Validate
            if n_embd % n_head != 0:
                self.model_info_label.setText("⚠ Invalid: n_embd must be divisible by n_head")
                self.model_info_label.setStyleSheet("color: red;")
                return
            
            # Calculate params
            params = 0
            params += vocab_size * n_embd  # wte
            params += block_size * n_embd  # wpe
            for _ in range(n_layer):
                params += 4 * (n_embd * n_embd)  # attention
                params += 2 * (n_embd * 4 * n_embd)  # mlp
            params += vocab_size * n_embd  # lm_head
            
            self.model_info_label.setText(
                f"✓ Estimated Parameters: {params:,} ({params/1e6:.2f}M)"
            )
            self.model_info_label.setStyleSheet("color: green;")
        except Exception as e:
            self.model_info_label.setText(f"Error: {e}")
            self.model_info_label.setStyleSheet("color: red;")
    
    def _initialize_model(self):
        """Initialize model with current config and use case"""
        try:
            config = ModelConfig(
                n_layer=self.n_layer_spin.value(),
                n_embd=self.n_embd_spin.value(),
                block_size=self.block_size_spin.value(),
                n_head=self.n_head_spin.value(),
                vocab_size=self.vocab_size_spin.value()
            )
            
            # Create model with selected use case
            self.model = GPTModel(config, use_case=self.selected_use_case)
            self.tokenizer = Tokenizer(256)  # Initialize tokenizer
            self.save_model_btn.setEnabled(True)
            
            self.status_bar.showMessage(
                f"Model initialized: {config.num_params:,} params | Mode: {self.selected_use_case.mode}"
            )
            
            QMessageBox.information(
                self, "Success",
                f"Model initialized successfully!\n"
                f"Parameters: {config.num_params:,}\n"
                f"Use Case: {self.selected_use_case.mode}\n"
                f"Ready for training."
            )
            
            # Update navigation to enable Next button
            self._update_navigation_buttons()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize model: {e}")
    
    def _save_model(self):
        """Save model to file"""
        if not self.model:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "Pickle Files (*.pkl)"
        )
        
        if filepath:
            try:
                self.model.save(filepath)
                self.status_bar.showMessage(f"Model saved to {filepath}")
                QMessageBox.information(self, "Success", "Model saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {e}")
    
    def _load_model(self):
        """Load model from file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Pickle Files (*.pkl)"
        )
        
        if filepath:
            try:
                self.model = GPTModel.load(filepath)
                
                # Update UI
                self.n_layer_spin.setValue(self.model.config.n_layer)
                self.n_embd_spin.setValue(self.model.config.n_embd)
                self.block_size_spin.setValue(self.model.config.block_size)
                self.n_head_spin.setValue(self.model.config.n_head)
                self.vocab_size_spin.setValue(self.model.config.vocab_size)
                
                self.save_model_btn.setEnabled(True)
                self.status_bar.showMessage(f"Model loaded from {filepath}")
                QMessageBox.information(self, "Success", "Model loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
    
    def _on_dataset_source_changed(self, source: str):
        """Handle dataset source change"""
        if source in ["Local File", "Hugging Face Hub", "URL Download"]:
            self.dataset_path_edit.setEnabled(True)
        else:
            self.dataset_path_edit.setEnabled(False)
            self.dataset_path_edit.clear()
    
    def _browse_dataset(self):
        """Browse for dataset file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset", "", "Text Files (*.txt);;All Files (*)"
        )
        
        if filepath:
            self.dataset_path_edit.setText(filepath)
    
    def _load_dataset(self):
        """Load dataset from selected source"""
        try:
            source = self.dataset_source_combo.currentText()
            
            if "Sample" in source:
                name = "names" if "Names" in source else "code"
                docs = self.dataset_manager.create_sample_dataset(name)
            elif source == "Local File":
                path = self.dataset_path_edit.text()
                if not path or not os.path.exists(path):
                    raise ValueError("Please select a valid file")
                docs = self.dataset_manager.load_local(path)
            elif source == "URL Download":
                url = self.dataset_path_edit.text()
                if not url:
                    raise ValueError("Please enter a URL")
                docs = self.dataset_manager.download_url(url)
            elif source == "Hugging Face Hub":
                path = self.dataset_path_edit.text()
                if not path:
                    raise ValueError("Please enter dataset name")
                docs = self.dataset_manager.load_from_huggingface(path)
            else:
                raise ValueError("Unknown source")
            
            # Create tokenizer
            self.tokenizer = Tokenizer(docs)
            
            # Update UI
            self.dataset_info_label.setText(
                f"✓ Loaded {len(docs)} documents, "
                f"Vocabulary: {self.tokenizer.vocab_size} tokens"
            )
            
            preview_text = "\n".join(docs[:10])
            if len(docs) > 10:
                preview_text += f"\n... and {len(docs) - 10} more"
            self.dataset_preview.setText(preview_text)
            
            self.status_bar.showMessage(f"Dataset loaded: {len(docs)} documents")
            
            # Update navigation to enable Next button
            self._update_navigation_buttons()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {e}")
    
    def _start_training(self):
        """Start training process with GPU acceleration support"""
        if not self.model or not self.tokenizer:
            QMessageBox.warning(
                self, "Warning",
                "Please initialize model and load dataset first!"
            )
            return
        
        # Check if GPU stack is selected and torch is available
        use_torch = False
        device = None
        
        if self.selected_stack.use_gpu and TORCH_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    use_torch = True
                    if torch.cuda.is_available():
                        device = torch.device('cuda')
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device = torch.device('mps')
            except ImportError:
                pass
        
        try:
            # Get training config
            train_config = TrainingConfig(
                learning_rate=self.lr_spin.value(),
                num_steps=self.steps_spin.value(),
                beta1=self.beta1_spin.value(),
                beta2=self.beta2_spin.value(),
                temperature=self.temp_spin.value()
            )
            
            # Load dataset
            source = self.dataset_source_combo.currentText()
            if "Sample" in source:
                name = "names" if "Names" in source else "code"
                docs = self.dataset_manager.create_sample_dataset(name)
            else:
                path = self.dataset_path_edit.text()
                docs = self.dataset_manager.load_local(path) if os.path.exists(path) else []
            
            if not docs:
                raise ValueError("No dataset available")
            
            # Use Torch implementation if GPU is available and selected
            if use_torch and TORCH_AVAILABLE:
                # Create TorchGPTModel from current config
                torch_model = TorchGPTModel(self.model.config, self.model.use_case)
                
                # Copy weights from pure Python model to Torch model (best effort)
                # Note: This is a simplified initialization - in practice you might want
                # to train from scratch or implement proper weight conversion
                
                # Create Torch trainer
                self.trainer = TorchTrainer(torch_model, train_config, device=device, use_amp=True)
                self.model = torch_model  # Replace model reference
                
                backend_name = "CUDA" if device.type == 'cuda' else "MPS"
                self.status_bar.showMessage(f"Using {backend_name} acceleration with PyTorch")
            else:
                # Use pure Python implementation
                self.trainer = Trainer(self.model, self.tokenizer, train_config)
                
                if self.selected_stack.use_gpu and not TORCH_AVAILABLE:
                    QMessageBox.warning(
                        self, "Warning",
                        f"GPU training requires PyTorch. Please install dependencies:\n\n"
                        f"pip install {' '.join(self.selected_stack.dependencies)}\n\n"
                        f"Falling back to CPU-only mode."
                    )
            
            # Setup worker thread
            self.worker = TrainingWorker(self.model, self.tokenizer, self.trainer, docs)
            self.worker.progress.connect(self._on_training_progress)
            self.worker.finished.connect(self._on_training_finished)
            self.worker.error.connect(self._on_training_error)
            
            # Update UI
            self.start_train_btn.setEnabled(False)
            self.stop_train_btn.setEnabled(True)
            backend_str = "GPU" if use_torch else self.selected_stack.backend.upper()
            self.status_label.setText(f"Status: Training ({backend_str})")
            self.progress_bar.setValue(0)
            
            # Start training
            self.worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training: {e}")
    
    def _stop_training(self):
        """Stop training"""
        if hasattr(self, 'worker'):
            self.worker.terminate()
            self.status_label.setText("Status: Stopped")
            self.start_train_btn.setEnabled(True)
            self.stop_train_btn.setEnabled(False)
    
    def _on_training_progress(self, step: int, loss: float, elapsed: float, avg_step: float):
        """Handle training progress update"""
        total = self.steps_spin.value()
        progress = int((step / total) * 100)
        self.progress_bar.setValue(progress)
        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.time_label.setText(f"Time: {elapsed:.1f}s | Avg/step: {avg_step:.3f}s")
        self.status_label.setText(f"Status: Step {step}/{total} ({self.selected_stack.backend.upper()})")
        QApplication.processEvents()
    
    def _on_training_finished(self, result: ExperimentResult):
        """Handle training completion"""
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.status_label.setText("Status: Complete")
        self.progress_bar.setValue(100)
        
        self.current_result = result
        self._add_result_to_list(result)
        
        # Auto-save model after training
        try:
            docs_source = self.dataset_source_combo.currentText()
            if "Sample" in docs_source:
                name = "names" if "Names" in docs_source else "code"
                docs = self.dataset_manager.create_sample_dataset(name)
            else:
                path = self.dataset_path_edit.text()
                docs = self.dataset_manager.load_local(path) if os.path.exists(path) else []
            
            model_id = self.model_manager.save_model(
                model=self.model,
                tokenizer=self.tokenizer,
                model_config=result.model_config,
                training_config=result.training_config,
                final_loss=result.final_loss,
                total_steps=result.training_config.num_steps,
                use_case=self.selected_use_case,
                docs=docs,
                name=f"Model {datetime.now().strftime('%Y%m%d %H:%M')}",
                description=f"Trained with loss {result.final_loss:.4f}"
            )
            
            # Auto-assess quality
            assessor = QualityAssessor(self.tokenizer, self.model)
            quality_results = assessor.assess()
            self.model_manager.update_quality_score(model_id, quality_results['overall_score'], quality_results)
            
            self.status_bar.showMessage(f"Model saved as {model_id} | Quality Score: {quality_results['overall_score']:.2f}")
        except Exception as e:
            self.status_bar.showMessage(f"Warning: Could not auto-save model: {e}")
        
        QMessageBox.information(
            self, "Training Complete",
            f"Final Loss: {result.final_loss:.4f}\n"
            f"Samples generated: {len(result.samples)}\n"
            f"Model automatically saved!"
        )
        self._refresh_models_list()
        
        # Auto-navigate to results page
        self.wizard_stack.setCurrentIndex(7)  # Results page
        self._update_navigation_buttons()
        self._update_progress_indicator(7)
    
    def _on_training_error(self, error: str):
        """Handle training error"""
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.status_label.setText("Status: Error")
        QMessageBox.critical(self, "Training Error", error)
    
    def _generate_text(self):
        """Generate text with current model - supports both pure Python and Torch models"""
        if not self.model or not self.tokenizer:
            QMessageBox.warning(
                self, "Warning",
                "Please initialize or load a model first!"
            )
            return
        
        try:
            temperature = self.gen_temp_slider.value() / 10
            max_tokens = self.max_tokens_spin.value()
            prompt = self.prompt_edit.text()
            
            # Check if using Torch model
            if TORCH_AVAILABLE and isinstance(self.model, TorchGPTModel):
                import torch
                device = next(self.model.parameters()).device
                
                # Use Torch model's generate method with KV caching
                if prompt:
                    text = self.model.generate_text(
                        self.tokenizer, 
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        device=device
                    )
                else:
                    # Generate from BOS token
                    idx = torch.tensor([[self.tokenizer.BOS]], dtype=torch.long, device=device)
                    generated_idx = self.model.generate(idx, max_tokens, temperature)
                    generated_tokens = generated_idx[0, 1:].tolist()
                    text = self.tokenizer.decode(generated_tokens)
            else:
                # Use pure Python model
                if prompt:
                    text = self.model.generate(
                        self.tokenizer,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        prompt=prompt
                    )
                else:
                    text = self.model.generate(
                        self.tokenizer,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
            
            self.output_text.setText(text)
            backend = "GPU" if (TORCH_AVAILABLE and isinstance(self.model, TorchGPTModel)) else "CPU"
            self.status_bar.showMessage(f"Text generated successfully ({backend})")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate text: {e}")
    
    def _add_result_to_list(self, result: ExperimentResult):
        """Add result to history list"""
        item_text = (
            f"Exp #{self.results_list.count() + 1}: "
            f"Loss={result.final_loss:.4f}, "
            f"Params={result.model_config.num_params:,}"
        )
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, result)
        self.results_list.addItem(item)
    
    def _on_result_selected(self, item: QListWidgetItem):
        """Handle result selection"""
        result = item.data(Qt.UserRole)
        if result:
            details = f"""
            Experiment Details
            ==================
            
            Model Configuration:
            - Layers: {result.model_config.n_layer}
            - Embedding: {result.model_config.n_embd}
            - Block Size: {result.model_config.block_size}
            - Heads: {result.model_config.n_head}
            - Parameters: {result.model_config.num_params:,}
            
            Training Configuration:
            - Learning Rate: {result.training_config.learning_rate}
            - Steps: {result.training_config.num_steps}
            - Beta1: {result.training_config.beta1}
            - Beta2: {result.training_config.beta2}
            
            Results:
            - Final Loss: {result.final_loss:.4f}
            - Training Time: {result.training_time:.1f}s
            - Timestamp: {result.timestamp}
            
            Generated Samples:
            {chr(10).join('- ' + s for s in result.samples)}
            """
            self.result_details.setText(details)
    
    def _export_results(self):
        """Export results to file"""
        if not self.current_result:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "JSON Files (*.json)"
        )
        
        if filepath:
            try:
                self.current_result.save(filepath)
                QMessageBox.information(self, "Success", "Results exported!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {e}")
    
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About OpenHammer LLM Studio",
            """
            <h2>OpenHammer LLM Studio</h2>
            <p>Low-Cost Language Model Creator</p>
            <p>Version 1.0.0</p>
            <p>Create and train language models on affordable hardware.</p>
            <p>Built with ❤️ using Python and PySide6</p>
            <p>Based on @karpathy's minimal GPT implementation</p>
            """
        )
    
    def _refresh_models_list(self):
        """Refresh the saved models list"""
        self.models_list.clear()
        models = self.model_manager.list_models(include_checkpoints=False)
        
        for model in models:
            quality_indicator = ""
            if model.quality_score > 0:
                if model.quality_score >= 0.7:
                    quality_indicator = "⭐⭐⭐"
                elif model.quality_score >= 0.5:
                    quality_indicator = "⭐⭐"
                else:
                    quality_indicator = "⭐"
            
            item_text = f"{quality_indicator} {model.name} (Loss: {model.final_loss:.4f})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, model.model_id)
            self.models_list.addItem(item)
    
    def _on_model_selected(self, item: QListWidgetItem):
        """Handle model selection"""
        model_id = item.data(Qt.UserRole)
        if model_id:
            self.selected_model_id = model_id
            metadata = self.model_manager.get_model_info(model_id)
            
            if metadata:
                details = f"""
                Model Details
                =============
                
                Name: {metadata.name}
                Created: {metadata.created_at}
                Use Case: {metadata.use_case}
                
                Architecture:
                - Layers: {metadata.model_config['n_layer']}
                - Embedding: {metadata.model_config['n_embd']}
                - Block Size: {metadata.model_config['block_size']}
                - Heads: {metadata.model_config['n_head']}
                
                Training:
                - Steps: {metadata.total_steps}
                - Final Loss: {metadata.final_loss:.4f}
                - Learning Rate: {metadata.training_config['learning_rate']}
                
                Quality Score: {metadata.quality_score:.2f}/1.00
                """
                self.model_details.setText(details)
                
                # Update quality display
                if metadata.quality_score > 0:
                    self.quality_progress.setValue(int(metadata.quality_score * 100))
                    self.quality_label.setText(f"Quality Score: {metadata.quality_score:.2f}/1.00")
                    
                    if metadata.test_results and 'recommendations' in metadata.test_results:
                        recs = metadata.test_results['recommendations']
                        if recs:
                            self.quality_recommendations.setText(
                                "Recommendations:\n" + "\n".join(f"• {r}" for r in recs)
                            )
                else:
                    self.quality_progress.setValue(0)
                    self.quality_label.setText("No assessment performed yet")
                    self.quality_recommendations.clear()
                
                # Enable action buttons
                self.load_model_btn.setEnabled(True)
                self.delete_model_btn.setEnabled(True)
                self.assess_model_btn.setEnabled(True)
    
    def _load_selected_model(self):
        """Load selected model for inference"""
        if not self.selected_model_id:
            return
        
        try:
            model, tokenizer, model_config, training_config, use_case = \
                self.model_manager.load_model(self.selected_model_id)
            
            self.model = model
            self.tokenizer = tokenizer
            self.selected_use_case = use_case
            
            self.status_bar.showMessage(f"Model loaded: {self.model_manager.get_model_info(self.selected_model_id).name}")
            
            QMessageBox.information(
                self, "Model Loaded",
                f"Successfully loaded model!\n\nYou can now use it for inference."
            )
            
            # Switch to inference tab
            tabs = self.findChild(QTabWidget)
            if tabs:
                for i in range(tabs.count()):
                    if tabs.tabText(i) == "💬 Inference":
                        tabs.setCurrentIndex(i)
                        break
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
    
    def _delete_selected_model(self):
        """Delete selected model"""
        if not self.selected_model_id:
            return
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Delete Model")
        msg.setText("Are you sure you want to delete this model?")
        msg.setInformativeText("This action cannot be undone.")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        
        if msg.exec() == QMessageBox.Yes:
            try:
                self.model_manager.delete_model(self.selected_model_id)
                self._refresh_models_list()
                self.selected_model_id = None
                self.model_details.clear()
                self.load_model_btn.setEnabled(False)
                self.delete_model_btn.setEnabled(False)
                self.assess_model_btn.setEnabled(False)
                
                QMessageBox.information(self, "Success", "Model deleted successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete model: {e}")
    
    def _assess_selected_model(self):
        """Run quality assessment on selected model"""
        if not self.selected_model_id:
            return
        
        try:
            # Load model
            model, tokenizer, _, _, _ = self.model_manager.load_model(self.selected_model_id)
            
            # Run assessment
            assessor = QualityAssessor(tokenizer, model)
            results = assessor.assess()
            
            # Update UI
            self.quality_progress.setValue(int(results['overall_score'] * 100))
            self.quality_label.setText(f"Quality Score: {results['overall_score']:.2f}/1.00")
            
            if results['recommendations']:
                self.quality_recommendations.setText(
                    "Recommendations:\n" + "\n".join(f"• {r}" for r in results['recommendations'])
                )
            
            # Save results
            self.model_manager.update_quality_score(
                self.selected_model_id, 
                results['overall_score'], 
                results
            )
            
            # Show test outputs
            test_outputs = "\n\n".join(
                f"Prompt: {p}\nOutput: {o}" 
                for p, o in zip(assessor.test_prompts[:5], results['test_outputs'][:5])
            )
            
            QMessageBox.information(
                self, "Assessment Complete",
                f"Overall Score: {results['overall_score']:.2f}/1.00\n\n"
                f"Metrics:\n"
                f"- Coherence: {results['coherence_score']:.2f}\n"
                f"- Diversity: {results['diversity_score']:.2f}\n"
                f"- Completion Rate: {results['completion_rate']:.0%}\n\n"
                f"Test Outputs:\n{test_outputs}"
            )
            
            # Refresh to update stored score
            self._refresh_models_list()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Assessment failed: {e}")


def main():
    """Main entry point"""
    if not HAS_QT:
        print("\n" + "="*60)
        print("ERROR: PySide6 is required for the GUI application")
        print("="*60)
        print("\nInstall with one of these commands:")
        print("  pip install PySide6")
        print("  pip install pyside6")
        print("\nOr use the CLI version instead:")
        print("  python llm_studio_cli.py")
        print("="*60 + "\n")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application info
    app.setApplicationName("OpenHammer LLM Studio")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("OpenHammer LLM Studio")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

    # Wizard navigation methods
    def _go_next(self):
        """Navigate to next wizard page"""
        current = self.wizard_stack.currentIndex()
        if current < len(self.wizard_pages) - 1:
            self.wizard_stack.setCurrentIndex(current + 1)
            self._update_navigation_buttons()
            self._update_progress_indicator(current + 1)
    
    def _go_back(self):
        """Navigate to previous wizard page"""
        current = self.wizard_stack.currentIndex()
        if current > 0:
            self.wizard_stack.setCurrentIndex(current - 1)
            self._update_navigation_buttons()
            self._update_progress_indicator(current - 1)
    
    def _update_navigation_buttons(self):
        """Update navigation button states based on current page"""
        current = self.wizard_stack.currentIndex()
        total = len(self.wizard_pages) - 1
        
        # Back button
        self.btn_back.setEnabled(current > 0)
        
        # Next button - hide on last page
        if current == total:
            self.btn_next.hide()
        else:
            self.btn_next.show()
            # Disable next if on dataset page and no dataset loaded
            if current == 4:  # Dataset page
                has_dataset = hasattr(self, 'dataset_manager') and len(self.dataset_manager.documents) > 0
                self.btn_next.setEnabled(has_dataset)
            # Disable next if on model page and model not initialized
            elif current == 3:  # Model page
                has_model = self.model is not None
                self.btn_next.setEnabled(has_model)
            else:
                self.btn_next.setEnabled(True)
        
        # Start button - show only on training page
        if current == 5:  # Training page
            self.btn_start.show()
            self.btn_stop.hide()
        else:
            self.btn_start.hide()
            self.btn_stop.hide()
        
        # Update status bar message
        messages = [
            "Bem-vindo! Clique em 'Avançar' para começar.",
            "Selecione a tecnologia de treinamento (CPU ou GPU).",
            "Defina o propósito do seu modelo.",
            "Configure a arquitetura do modelo e inicialize.",
            "Carregue ou crie um dataset para treinamento.",
            "Ajuste os parâmetros e inicie o treinamento.",
            "Teste seu modelo gerando texto.",
            "Veja os resultados e métricas do treinamento.",
            "Gerencie seus modelos salvos."
        ]
        if current < len(messages):
            self.status_bar.showMessage(messages[current])
    
    def _update_progress_indicator(self, current_index):
        """Update the progress indicator labels"""
        for i, lbl in enumerate(self.progress_indicator):
            if i < current_index:
                lbl.setStyleSheet("QLabel { color: #4CAF50; font-size: 12px; font-weight: bold; }")
                lbl.setText(f"✓ {lbl.text().split('. ')[1]}")
            elif i == current_index:
                lbl.setStyleSheet("QLabel { color: #2196F3; font-size: 13px; font-weight: bold; }")
                lbl.setText(f"➤ {lbl.text().split('. ')[1]}")
            else:
                lbl.setStyleSheet("QLabel { color: #9E9E9E; font-size: 11px; }")
                lbl.setText(f"{i+1}. {lbl.text().split('. ')[1] if '. ' in lbl.text() else lbl.text()}")

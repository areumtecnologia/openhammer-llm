"""
Wizard-style GUI for LLM Training - Beginner Friendly
Step-by-step assistant for creating language models
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
        QLabel, QPushButton, QLineEdit, QTextEdit, QSpinBox,
        QDoubleSpinBox, QComboBox, QProgressBar, QGroupBox, QFormLayout,
        QFileDialog, QMessageBox, QScrollArea, QCheckBox,
        QListWidget, QListWidgetItem, QSlider, QStatusBar, 
        QStackedWidget, QFrame, QRadioButton, QButtonGroup,
        QDialog, QGridLayout, QSplitter
    )
    from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QSize
    from PySide6.QtGui import QFont, QIcon, QAction, QDesktopServices
    HAS_QT = True
except ImportError as e:
    HAS_QT = False
    print(f"⚠ PySide6 not installed. Install with: pip install PySide6")
    sys.exit(1)

if not HAS_QT:
    print("⚠ PySide6 is not available. Please install it with: pip install PySide6")
    sys.exit(1)

from core.model import (
    ModelConfig, TrainingConfig, HardwareProfile, GPTModel, 
    Trainer, Tokenizer, DatasetManager, ExperimentResult, TrainingStack,
    ModelUseCase, TORCH_AVAILABLE, TorchGPTModel, TorchTrainer
)
from core.model_manager import ModelManager, QualityAssessor, ModelMetadata


class DatasetLoadWorker(QThread):
    """Background thread for loading datasets with progress"""
    progress = Signal(str, int)  # message, percentage
    finished = Signal(object, object)  # docs, tokenizer
    error = Signal(str)
    
    def __init__(self, source_type, path=None):
        super().__init__()
        self.source_type = source_type
        self.path = path
        self.dataset_manager = DatasetManager()
    
    def run(self):
        try:
            self.progress.emit("Preparing dataset...", 10)
            
            if self.source_type == "sample_names":
                self.progress.emit("Loading sample names dataset...", 30)
                docs = self.dataset_manager.create_sample_dataset("names")
            elif self.source_type == "sample_code":
                self.progress.emit("Loading sample code dataset...", 30)
                docs = self.dataset_manager.create_sample_dataset("code")
            elif self.source_type == "local":
                self.progress.emit("Loading local file...", 30)
                if not self.path or not os.path.exists(self.path):
                    raise ValueError("File not found")
                docs = self.dataset_manager.load_local(self.path)
            elif self.source_type == "huggingface":
                self.progress.emit("Connecting to Hugging Face Hub...", 20)
                if not self.path:
                    raise ValueError("Dataset name required")
                docs = self.dataset_manager.load_from_huggingface(self.path)
                self.progress.emit("Downloading dataset...", 50)
            elif self.source_type == "url":
                self.progress.emit("Downloading from URL...", 30)
                if not self.path:
                    raise ValueError("URL required")
                docs = self.dataset_manager.download_url(self.path)
            else:
                raise ValueError("Unknown source type")
            
            self.progress.emit("Building tokenizer...", 80)
            tokenizer = Tokenizer(docs)
            self.progress.emit("Dataset ready!", 100)
            
            self.finished.emit(docs, tokenizer)
        except Exception as e:
            self.error.emit(f"Failed to load dataset: {e}")


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
            if isinstance(self.trainer, TorchTrainer):
                loss_history, total_time, avg_step_time = self.trainer.train(
                    self.docs, self.tokenizer, callback=callback
                )
            else:
                loss_history, total_time, avg_step_time = self.trainer.train(
                    self.docs, callback=callback
                )
            
            # Generate samples
            samples = []
            for _ in range(5):
                if TORCH_AVAILABLE and isinstance(self.model, TorchGPTModel):
                    sample = self.model.generate_text(
                        self.tokenizer,
                        prompt="",
                        max_new_tokens=30,
                        temperature=self.trainer.config.temperature,
                        device=self.trainer.device
                    )
                else:
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
                timestamp=str(datetime.now())
            )
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class WizardWindow(QMainWindow):
    """Main wizard window with step-by-step flow"""
    
    STEPS = [
        ("Welcome", "🏠"),
        ("Purpose", "🎯"),
        ("Architecture", "🔧"),
        ("Dataset", "📊"),
        ("Training", "⚡"),
        ("Review", "📋"),
        ("Train", "🚀"),
        ("Results", "🏆"),
        ("Inference", "💬")
    ]
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenHammer LLM Studio - Model Creation Wizard")
        self.setMinimumSize(1000, 700)
        
        # State
        self.current_step = 0
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.docs = None
        self.hardware_profile = HardwareProfile.detect_system()
        self.selected_stack = TrainingStack.detect_available()[0]
        self.selected_use_case = ModelUseCase(mode="completion")
        self.model_manager = ModelManager()
        self.training_result = None
        self.dataset_worker = None
        self.training_worker = None
        
        # Setup UI
        self._setup_ui()
        self._update_hardware_info()
        self._show_welcome_message()
    
    def _setup_ui(self):
        """Setup main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header with progress
        header_frame = QFrame()
        header_frame.setStyleSheet("QFrame { background-color: #f5f5f5; padding: 10px; }")
        header_layout = QHBoxLayout(header_frame)
        
        # Step indicator
        self.step_label = QLabel("Step 1/9: Welcome")
        self.step_label.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; color: #2196F3; }")
        header_layout.addWidget(self.step_label)
        
        header_layout.addStretch()
        
        # Progress bar
        self.wizard_progress = QProgressBar()
        self.wizard_progress.setRange(0, len(self.STEPS) - 1)
        self.wizard_progress.setValue(0)
        self.wizard_progress.setMaximumWidth(300)
        header_layout.addWidget(self.wizard_progress)
        
        main_layout.addWidget(header_frame)
        
        # Content area with stacked widget
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("QStackedWidget { background-color: white; padding: 20px; }")
        
        # Create all step pages
        self.pages = []
        self.pages.append(self._create_welcome_page())
        self.pages.append(self._create_purpose_page())
        self.pages.append(self._create_architecture_page())
        self.pages.append(self._create_dataset_page())
        self.pages.append(self._create_training_params_page())
        self.pages.append(self._create_review_page())
        self.pages.append(self._create_training_page())
        self.pages.append(self._create_results_page())
        self.pages.append(self._create_inference_page())
        
        for page in self.pages:
            self.stack.addWidget(page)
        
        main_layout.addWidget(self.stack, 1)
        
        # Navigation buttons
        nav_frame = QFrame()
        nav_frame.setStyleSheet("QFrame { background-color: #f5f5f5; padding: 10px; }")
        nav_layout = QHBoxLayout(nav_frame)
        
        self.back_btn = QPushButton("◀ Back")
        self.back_btn.clicked.connect(self._go_back)
        self.back_btn.setEnabled(False)
        self.back_btn.setMinimumWidth(100)
        
        nav_layout.addWidget(self.back_btn)
        nav_layout.addStretch()
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._go_next)
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.next_btn.setMinimumWidth(100)
        nav_layout.addWidget(self.next_btn)
        
        self.start_btn = QPushButton("🚀 Start Training")
        self.start_btn.clicked.connect(self._start_training_from_wizard)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.start_btn.setMinimumWidth(150)
        self.start_btn.setVisible(False)
        nav_layout.addWidget(self.start_btn)
        
        main_layout.addWidget(nav_frame)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _create_welcome_page(self) -> QWidget:
        """Welcome page"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel("🎉 Welcome to OpenHammer LLM Studio!")
        title.setStyleSheet("QLabel { font-size: 28px; font-weight: bold; color: #2196F3; }")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Create Your Own Language Model - Step by Step")
        subtitle.setStyleSheet("QLabel { font-size: 18px; color: #666; }")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addSpacing(30)
        
        info_box = QGroupBox("What You'll Do")
        info_layout = QVBoxLayout()
        steps_text = """
        <ol style='font-size: 16px; line-height: 2;'>
            <li><b>Choose Purpose:</b> Select what your model will do (chat, code, text completion)</li>
            <li><b>Configure Architecture:</b> We'll auto-suggest optimal settings for your purpose</li>
            <li><b>Load Dataset:</b> Use sample data or your own text files</li>
            <li><b>Review & Train:</b> Verify settings and start training</li>
            <li><b>Test & Use:</b> Generate text and save your model</li>
        </ol>
        """
        info_label = QLabel(steps_text)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        info_box.setLayout(info_layout)
        layout.addWidget(info_box)
        
        layout.addSpacing(20)
        
        # Hardware info
        hw_box = QGroupBox("Your Hardware")
        hw_layout = QFormLayout()
        self.hw_ram_label = QLabel()
        self.hw_gpu_label = QLabel()
        self.hw_recommendation_label = QLabel()
        hw_layout.addRow("RAM:", self.hw_ram_label)
        hw_layout.addRow("GPU:", self.hw_gpu_label)
        hw_layout.addRow("Recommendation:", self.hw_recommendation_label)
        hw_box.setLayout(hw_layout)
        layout.addWidget(hw_box)
        
        layout.addStretch()
        return widget
    
    def _create_purpose_page(self) -> QWidget:
        """Model purpose selection page"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("🎯 What Will Your Model Do?")
        title.setStyleSheet("QLabel { font-size: 24px; font-weight: bold; }")
        layout.addWidget(title)
        
        desc = QLabel("Select the primary use case. This will automatically configure optimal settings.")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        layout.addSpacing(20)
        
        # Use case options
        self.usecase_group = QButtonGroup()
        usecases = [
            ("📝 Text Completion", "completion", 
             "Generate text continuations, complete sentences, creative writing"),
            ("💬 Chatbot", "chat", 
             "Conversational AI, customer support, interactive dialogue"),
            ("🔧 Code Assistant", "code", 
             "Code completion, function generation, programming help"),
            ("📄 Document Summarizer", "summarization", 
             "Condense long texts into shorter summaries"),
            ("🎭 Creative Writing", "creative", 
             "Poetry, stories, artistic text generation")
        ]
        
        for label, mode, description in usecases:
            radio = QRadioButton(f"{label}")
            radio.setStyleSheet("QRadioButton { font-size: 16px; padding: 5px; }")
            self.usecase_group.addButton(radio)
            layout.addWidget(radio)
            
            desc_label = QLabel(f"   {description}")
            desc_label.setStyleSheet("QLabel { color: #666; font-size: 14px; }")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)
        
        # Set default
        self.usecase_group.buttons()[0].setChecked(True)
        
        layout.addStretch()
        return widget
    
    def _create_architecture_page(self) -> QWidget:
        """Model architecture configuration page"""
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        
        title = QLabel("🔧 Model Architecture")
        title.setStyleSheet("QLabel { font-size: 24px; font-weight: bold; }")
        layout.addWidget(title)
        
        desc = QLabel("We've pre-configured optimal settings based on your purpose. Feel free to adjust for advanced control.")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        layout.addSpacing(20)
        
        # Auto-tune button
        auto_btn = QPushButton("⚡ Auto-Configure for Selected Purpose")
        auto_btn.clicked.connect(self._auto_configure_architecture)
        auto_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 10px; font-weight: bold; }")
        layout.addWidget(auto_btn)
        
        layout.addSpacing(20)
        
        # Architecture parameters
        form = QFormLayout()
        
        self.n_layer_spin = QSpinBox()
        self.n_layer_spin.setRange(1, 12)
        self.n_layer_spin.setValue(2)
        self.n_layer_spin.setToolTip("Number of transformer layers (depth)")
        form.addRow("Layers (Depth):", self.n_layer_spin)
        
        self.n_embd_spin = QSpinBox()
        self.n_embd_spin.setRange(16, 512)
        self.n_embd_spin.setSingleStep(16)
        self.n_embd_spin.setValue(32)
        self.n_embd_spin.setToolTip("Embedding dimension (width)")
        form.addRow("Embedding Size:", self.n_embd_spin)
        
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(16, 512)
        self.block_size_spin.setSingleStep(16)
        self.block_size_spin.setValue(32)
        self.block_size_spin.setToolTip("Maximum context length")
        form.addRow("Context Length:", self.block_size_spin)
        
        self.n_head_spin = QSpinBox()
        self.n_head_spin.setRange(1, 16)
        self.n_head_spin.setValue(4)
        self.n_head_spin.setToolTip("Number of attention heads")
        form.addRow("Attention Heads:", self.n_head_spin)
        
        self.vocab_size_spin = QSpinBox()
        self.vocab_size_spin.setRange(64, 1024)
        self.vocab_size_spin.setValue(256)
        self.vocab_size_spin.setToolTip("Vocabulary size")
        form.addRow("Vocabulary Size:", self.vocab_size_spin)
        
        layout.addLayout(form)
        
        layout.addSpacing(20)
        
        # Model info
        self.model_info_label = QLabel()
        self.model_info_label.setStyleSheet("QLabel { font-weight: bold; color: #2196F3; font-size: 16px; }")
        self.model_info_label.setWordWrap(True)
        layout.addWidget(self.model_info_label)
        
        layout.addStretch()
        scroll.setWidget(content)
        
        # Connect signals
        for spin in [self.n_layer_spin, self.n_embd_spin, self.block_size_spin, 
                     self.n_head_spin, self.vocab_size_spin]:
            spin.valueChanged.connect(self._update_model_info)
        
        self._update_model_info()
        return widget
    
    def _create_dataset_page(self) -> QWidget:
        """Dataset loading page"""
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        
        title = QLabel("📊 Load Training Data")
        title.setStyleSheet("QLabel { font-size: 24px; font-weight: bold; }")
        layout.addWidget(title)
        
        desc = QLabel("Your model learns from examples. Choose a dataset source:")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        layout.addSpacing(20)
        
        # Dataset source
        self.dataset_source_combo = QComboBox()
        self.dataset_source_combo.addItems([
            "📝 Sample: Names (Good for testing)",
            "💻 Sample: Code (Programming examples)",
            "📁 Local Text File",
            "🌐 Hugging Face Hub",
            "🔗 URL Download"
        ])
        self.dataset_source_combo.currentTextChanged.connect(self._on_dataset_source_changed)
        layout.addWidget(QLabel("Dataset Source:"))
        layout.addWidget(self.dataset_source_combo)
        
        # Path input
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText("Enter path or URL...")
        self.dataset_path_edit.setEnabled(False)
        layout.addWidget(QLabel("Path/URL/Dataset Name:"))
        layout.addWidget(self.dataset_path_edit)
        
        # Browse button for local files
        browse_btn = QPushButton("📂 Browse Files...")
        browse_btn.clicked.connect(self._browse_dataset)
        browse_btn.setEnabled(False)
        layout.addWidget(browse_btn)
        
        layout.addSpacing(20)
        
        # Progress indicator
        self.dataset_progress_label = QLabel()
        self.dataset_progress_label.setStyleSheet("QLabel { color: #2196F3; font-weight: bold; }")
        self.dataset_progress_label.setWordWrap(True)
        layout.addWidget(self.dataset_progress_label)
        
        self.dataset_progress_bar = QProgressBar()
        self.dataset_progress_bar.setRange(0, 100)
        self.dataset_progress_bar.setValue(0)
        self.dataset_progress_bar.setVisible(False)
        layout.addWidget(self.dataset_progress_bar)
        
        # Dataset info
        self.dataset_info_label = QLabel("No dataset loaded yet")
        self.dataset_info_label.setStyleSheet("QLabel { color: #666; }")
        self.dataset_info_label.setWordWrap(True)
        layout.addWidget(self.dataset_info_label)
        
        # Preview
        preview_group = QGroupBox("Preview (First 10 items)")
        preview_layout = QVBoxLayout()
        self.dataset_preview = QTextEdit()
        self.dataset_preview.setReadOnly(True)
        self.dataset_preview.setMaximumHeight(150)
        preview_layout.addWidget(self.dataset_preview)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Load button
        self.load_dataset_btn = QPushButton("📥 Load Dataset")
        self.load_dataset_btn.clicked.connect(self._load_dataset_async)
        self.load_dataset_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.load_dataset_btn)
        
        layout.addStretch()
        scroll.setWidget(content)
        return widget
    
    def _create_training_params_page(self) -> QWidget:
        """Training parameters page"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("⚡ Training Configuration")
        title.setStyleSheet("QLabel { font-size: 24px; font-weight: bold; }")
        layout.addWidget(title)
        
        desc = QLabel("Configure how your model learns. Auto-configured based on your purpose.")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        layout.addSpacing(20)
        
        # Auto-tune button
        auto_btn = QPushButton("⚡ Auto-Configure Training Parameters")
        auto_btn.clicked.connect(self._auto_configure_training)
        auto_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; padding: 10px; font-weight: bold; }")
        layout.addWidget(auto_btn)
        
        layout.addSpacing(20)
        
        # Training params
        form = QFormLayout()
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(4)
        form.addRow("Learning Rate:", self.lr_spin)
        
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(100, 10000)
        self.steps_spin.setSingleStep(100)
        self.steps_spin.setValue(1000)
        form.addRow("Training Steps:", self.steps_spin)
        
        self.beta1_spin = QDoubleSpinBox()
        self.beta1_spin.setRange(0.0, 1.0)
        self.beta1_spin.setSingleStep(0.01)
        self.beta1_spin.setValue(0.85)
        form.addRow("Beta1 (Adam):", self.beta1_spin)
        
        self.beta2_spin = QDoubleSpinBox()
        self.beta2_spin.setRange(0.0, 1.0)
        self.beta2_spin.setSingleStep(0.01)
        self.beta2_spin.setValue(0.99)
        form.addRow("Beta2 (Adam):", self.beta2_spin)
        
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.5)
        form.addRow("Temperature:", self.temp_spin)
        
        layout.addLayout(form)
        
        layout.addSpacing(20)
        
        # Stack selection
        stack_group = QGroupBox("Training Backend")
        stack_layout = QVBoxLayout()
        
        self.stack_combo = QComboBox()
        self._populate_stack_options()
        self.stack_combo.currentIndexChanged.connect(self._on_stack_changed)
        stack_layout.addWidget(QLabel("Select Backend:"))
        stack_layout.addWidget(self.stack_combo)
        
        self.stack_details_label = QLabel()
        self.stack_details_label.setStyleSheet("QLabel { color: #2196F3; font-weight: bold; }")
        stack_layout.addWidget(self.stack_details_label)
        
        stack_group.setLayout(stack_layout)
        layout.addWidget(stack_group)
        
        layout.addStretch()
        return widget
    
    def _create_review_page(self) -> QWidget:
        """Review configuration page"""
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        
        title = QLabel("📋 Review Configuration")
        title.setStyleSheet("QLabel { font-size: 24px; font-weight: bold; }")
        layout.addWidget(title)
        
        desc = QLabel("Verify all settings before starting training")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        layout.addSpacing(20)
        
        # Configuration summary
        self.review_text = QTextEdit()
        self.review_text.setReadOnly(True)
        self.review_text.setMinimumHeight(400)
        layout.addWidget(self.review_text)
        
        layout.addStretch()
        scroll.setWidget(content)
        return widget
    
    def _create_training_page(self) -> QWidget:
        """Training progress page"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("🚀 Training in Progress")
        title.setStyleSheet("QLabel { font-size: 24px; font-weight: bold; color: #4CAF50; }")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        layout.addSpacing(20)
        
        # Progress
        self.train_progress_bar = QProgressBar()
        self.train_progress_bar.setRange(0, 100)
        self.train_progress_bar.setValue(0)
        self.train_progress_bar.setMinimumHeight(30)
        layout.addWidget(self.train_progress_bar)
        
        # Stats
        stats_layout = QGridLayout()
        
        self.loss_label = QLabel("Loss: --")
        self.loss_label.setStyleSheet("QLabel { font-size: 20px; font-weight: bold; }")
        stats_layout.addWidget(self.loss_label, 0, 0)
        
        self.time_label = QLabel("Time: 0.0s")
        self.time_label.setStyleSheet("QLabel { font-size: 16px; color: #2196F3; }")
        stats_layout.addWidget(self.time_label, 0, 1)
        
        self.step_time_label = QLabel("Avg/step: 0.000s")
        self.step_time_label.setStyleSheet("QLabel { font-size: 16px; color: #2196F3; }")
        stats_layout.addWidget(self.step_time_label, 1, 0)
        
        self.status_label = QLabel("Status: Waiting...")
        self.status_label.setStyleSheet("QLabel { font-size: 16px; }")
        stats_layout.addWidget(self.status_label, 1, 1)
        
        layout.addLayout(stats_layout)
        
        layout.addSpacing(30)
        
        # Live samples
        samples_group = QGroupBox("Live Samples (Generated During Training)")
        samples_layout = QVBoxLayout()
        self.live_samples = QTextEdit()
        self.live_samples.setReadOnly(True)
        self.live_samples.setMinimumHeight(150)
        self.live_samples.setPlaceholderText("Sample outputs will appear here...")
        samples_layout.addWidget(self.live_samples)
        samples_group.setLayout(samples_layout)
        layout.addWidget(samples_group)
        
        # Stop button
        self.stop_train_btn = QPushButton("⏹ Stop Training")
        self.stop_train_btn.clicked.connect(self._stop_training)
        self.stop_train_btn.setEnabled(False)
        self.stop_train_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.stop_train_btn)
        
        layout.addStretch()
        return widget
    
    def _create_results_page(self) -> QWidget:
        """Results page"""
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        
        title = QLabel("🏆 Training Complete!")
        title.setStyleSheet("QLabel { font-size: 28px; font-weight: bold; color: #4CAF50; }")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        layout.addSpacing(20)
        
        # Results summary
        self.results_summary = QTextEdit()
        self.results_summary.setReadOnly(True)
        self.results_summary.setMinimumHeight(200)
        layout.addWidget(self.results_summary)
        
        layout.addSpacing(20)
        
        # Quality assessment
        quality_group = QGroupBox("Quality Assessment")
        quality_layout = QVBoxLayout()
        
        self.quality_score_label = QLabel("Overall Score: --/1.00")
        self.quality_score_label.setStyleSheet("QLabel { font-size: 20px; font-weight: bold; }")
        quality_layout.addWidget(self.quality_score_label)
        
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
        
        layout.addSpacing(20)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        save_btn = QPushButton("💾 Save Model")
        save_btn.clicked.connect(self._save_model_from_wizard)
        btn_layout.addWidget(save_btn)
        
        test_btn = QPushButton("🧪 Run Auto-Test")
        test_btn.clicked.connect(self._run_auto_test)
        btn_layout.addWidget(test_btn)
        
        infer_btn = QPushButton("💬 Go to Inference")
        infer_btn.clicked.connect(lambda: self.stack.setCurrentIndex(8))
        btn_layout.addWidget(infer_btn)
        
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        scroll.setWidget(content)
        return widget
    
    def _create_inference_page(self) -> QWidget:
        """Inference page"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("💬 Test Your Model")
        title.setStyleSheet("QLabel { font-size: 24px; font-weight: bold; }")
        layout.addWidget(title)
        
        # Generation params
        gen_group = QGroupBox("Generation Settings")
        gen_layout = QFormLayout()
        
        self.gen_temp_slider = QSlider(Qt.Horizontal)
        self.gen_temp_slider.setRange(1, 20)
        self.gen_temp_slider.setValue(5)
        self.gen_temp_label = QLabel("0.5")
        
        temp_widget = QWidget()
        temp_layout = QHBoxLayout(temp_widget)
        temp_layout.setContentsMargins(0, 0, 0, 0)
        temp_layout.addWidget(self.gen_temp_slider)
        temp_layout.addWidget(self.gen_temp_label)
        
        self.gen_temp_slider.valueChanged.connect(
            lambda v: self.gen_temp_label.setText(f"{v/10:.1f}")
        )
        
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(10, 500)
        self.max_tokens_spin.setValue(50)
        
        gen_layout.addRow("Temperature:", temp_widget)
        gen_layout.addRow("Max Tokens:", self.max_tokens_spin)
        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)
        
        layout.addSpacing(20)
        
        # Input/Output
        layout.addWidget(QLabel("Enter your prompt:"))
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("Type something to generate text...")
        layout.addWidget(self.prompt_edit)
        
        generate_btn = QPushButton("✨ Generate")
        generate_btn.clicked.connect(self._generate_text)
        generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        layout.addWidget(generate_btn)
        
        layout.addWidget(QLabel("Generated Output:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(200)
        self.output_text.setPlaceholderText("Generated text will appear here...")
        layout.addWidget(self.output_text)
        
        layout.addStretch()
        return widget
    
    # Navigation methods
    def _go_back(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.stack.setCurrentIndex(self.current_step)
            self._update_navigation()
    
    def _go_next(self):
        """Go to next step"""
        # Validation before proceeding
        if not self._validate_current_step():
            return
        
        if self.current_step < len(self.STEPS) - 1:
            self.current_step += 1
            self.stack.setCurrentIndex(self.current_step)
            self._update_navigation()
            
            # Update review when reaching review page
            if self.current_step == 5:
                self._update_review()
    
    def _update_navigation(self):
        """Update navigation buttons and labels"""
        # Update step label
        name, icon = self.STEPS[self.current_step]
        self.step_label.setText(f"Step {self.current_step + 1}/{len(self.STEPS)}: {icon} {name}")
        
        # Update progress
        self.wizard_progress.setValue(self.current_step)
        
        # Update buttons
        self.back_btn.setEnabled(self.current_step > 0)
        
        # Show/hide next button based on step
        if self.current_step == len(self.STEPS) - 2:  # Before training
            self.next_btn.setVisible(False)
            self.start_btn.setVisible(True)
        elif self.current_step == len(self.STEPS) - 1:  # Results
            self.next_btn.setVisible(False)
            self.start_btn.setVisible(False)
        else:
            self.next_btn.setVisible(True)
            self.start_btn.setVisible(False)
    
    def _validate_current_step(self) -> bool:
        """Validate current step before proceeding"""
        step = self.current_step
        
        if step == 1:  # Purpose
            # Always valid - default selection exists
            return True
        
        elif step == 2:  # Architecture
            # Validate architecture
            n_embd = self.n_embd_spin.value()
            n_head = self.n_head_spin.value()
            if n_embd % n_head != 0:
                QMessageBox.warning(
                    self, "Invalid Configuration",
                    "Embedding size must be divisible by number of attention heads."
                )
                return False
            return True
        
        elif step == 3:  # Dataset
            if not self.docs or not self.tokenizer:
                QMessageBox.warning(
                    self, "Dataset Required",
                    "Please load a dataset before proceeding."
                )
                return False
            return True
        
        elif step == 4:  # Training params
            # Basic validation
            if self.steps_spin.value() < 100:
                QMessageBox.warning(
                    self, "Invalid Steps",
                    "Training steps must be at least 100."
                )
                return False
            return True
        
        return True
    
    # Auto-configuration methods
    def _auto_configure_architecture(self):
        """Auto-configure architecture based on selected purpose"""
        use_case = self._get_selected_use_case()
        
        configs = {
            "completion": {"layers": 2, "embd": 64, "block": 64, "heads": 4, "vocab": 256},
            "chat": {"layers": 3, "embd": 96, "block": 128, "heads": 6, "vocab": 512},
            "code": {"layers": 4, "embd": 128, "block": 256, "heads": 8, "vocab": 512},
            "summarization": {"layers": 3, "embd": 96, "block": 256, "heads": 6, "vocab": 512},
            "creative": {"layers": 3, "embd": 96, "block": 128, "heads": 6, "vocab": 512}
        }
        
        config = configs.get(use_case.mode, configs["completion"])
        
        self.n_layer_spin.setValue(config["layers"])
        self.n_embd_spin.setValue(config["embd"])
        self.block_size_spin.setValue(config["block"])
        self.n_head_spin.setValue(config["heads"])
        self.vocab_size_spin.setValue(config["vocab"])
        
        self._update_model_info()
        
        QMessageBox.information(
            self, "Auto-Configured",
            f"Architecture optimized for {use_case.mode}!\n\n"
            f"The settings balance quality and training speed for your use case."
        )
    
    def _auto_configure_training(self):
        """Auto-configure training parameters based on purpose"""
        use_case = self._get_selected_use_case()
        
        configs = {
            "completion": {"lr": 0.01, "steps": 1000, "beta1": 0.85, "beta2": 0.99},
            "chat": {"lr": 0.005, "steps": 2000, "beta1": 0.9, "beta2": 0.995},
            "code": {"lr": 0.003, "steps": 3000, "beta1": 0.9, "beta2": 0.999},
            "summarization": {"lr": 0.005, "steps": 2000, "beta1": 0.9, "beta2": 0.995},
            "creative": {"lr": 0.008, "steps": 1500, "beta1": 0.85, "beta2": 0.99}
        }
        
        config = configs.get(use_case.mode, configs["completion"])
        
        self.lr_spin.setValue(config["lr"])
        self.steps_spin.setValue(config["steps"])
        self.beta1_spin.setValue(config["beta1"])
        self.beta2_spin.setValue(config["beta2"])
        
        QMessageBox.information(
            self, "Auto-Configured",
            f"Training parameters optimized for {use_case.mode}!"
        )
    
    # Helper methods
    def _get_selected_use_case(self) -> ModelUseCase:
        """Get currently selected use case"""
        for i, btn in enumerate(self.usecase_group.buttons()):
            if btn.isChecked():
                usecases = ["completion", "chat", "code", "summarization", "creative"]
                return ModelUseCase(mode=usecases[i])
        return ModelUseCase(mode="completion")
    
    def _update_model_info(self):
        """Update model parameter count display"""
        try:
            n_layer = self.n_layer_spin.value()
            n_embd = self.n_embd_spin.value()
            block_size = self.block_size_spin.value()
            n_head = self.n_head_spin.value()
            vocab_size = self.vocab_size_spin.value()
            
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
                f"✓ Estimated Parameters: {params:,} ({params/1e6:.2f}M)\n"
                f"Recommended for: Low-cost hardware | Training time: ~{(params/1e6)*0.5:.1f}s per step on CPU"
            )
            self.model_info_label.setStyleSheet("color: green;")
        except Exception as e:
            self.model_info_label.setText(f"Error: {e}")
            self.model_info_label.setStyleSheet("color: red;")
    
    def _update_hardware_info(self):
        """Update hardware information"""
        hp = self.hardware_profile
        self.hw_ram_label.setText(f"{hp.ram_gb} GB")
        self.hw_gpu_label.setText(f"{hp.gpu_type.upper()}" if hp.has_gpu else "None detected")
        self.hw_recommendation_label.setText(
            f"Max Model: {hp.recommend_max_model()}, "
            f"Quantization: {hp.recommend_quantization()}"
        )
    
    def _show_welcome_message(self):
        """Show welcome message"""
        self.status_bar.showMessage("Welcome! Follow the wizard to create your model.")
    
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
        
        self._on_stack_changed(0)
    
    def _on_stack_changed(self, index: int):
        """Handle stack selection change"""
        stack = self.stack_combo.itemData(index)
        if stack:
            self.selected_stack = stack
            self.stack_details_label.setText(
                f"{stack.description} | GPU: {'Yes' if stack.use_gpu else 'No'}"
            )
    
    def _on_dataset_source_changed(self, source: str):
        """Handle dataset source change"""
        if "Local" in source or "Hugging Face" in source or "URL" in source:
            self.dataset_path_edit.setEnabled(True)
            self.dataset_path_edit.setPlaceholderText(
                "File path" if "Local" in source else
                "Dataset name (e.g., 'glue/mrpc')" if "Hugging Face" in source else
                "https://..."
            )
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
    
    def _load_dataset_async(self):
        """Load dataset in background thread"""
        source = self.dataset_source_combo.currentText()
        
        if "Sample: Names" in source:
            source_type = "sample_names"
            path = None
        elif "Sample: Code" in source:
            source_type = "sample_code"
            path = None
        elif "Local" in source:
            source_type = "local"
            path = self.dataset_path_edit.text()
        elif "Hugging Face" in source:
            source_type = "huggingface"
            path = self.dataset_path_edit.text()
        elif "URL" in source:
            source_type = "url"
            path = self.dataset_path_edit.text()
        else:
            return
        
        # Disable button during load
        self.load_dataset_btn.setEnabled(False)
        self.dataset_progress_bar.setVisible(True)
        self.dataset_progress_label.setText("Loading dataset...")
        
        # Start worker
        self.dataset_worker = DatasetLoadWorker(source_type, path)
        self.dataset_worker.progress.connect(self._on_dataset_progress)
        self.dataset_worker.finished.connect(self._on_dataset_loaded)
        self.dataset_worker.error.connect(self._on_dataset_error)
        self.dataset_worker.start()
    
    def _on_dataset_progress(self, message: str, percent: int):
        """Handle dataset loading progress"""
        self.dataset_progress_label.setText(message)
        self.dataset_progress_bar.setValue(percent)
    
    def _on_dataset_loaded(self, docs, tokenizer):
        """Handle successful dataset load"""
        self.docs = docs
        self.tokenizer = tokenizer
        
        self.dataset_info_label.setText(
            f"✓ Loaded {len(docs)} documents | Vocabulary: {tokenizer.vocab_size} tokens"
        )
        
        preview_text = "\n".join(docs[:10])
        if len(docs) > 10:
            preview_text += f"\n... and {len(docs) - 10} more"
        self.dataset_preview.setText(preview_text)
        
        self.load_dataset_btn.setEnabled(True)
        self.dataset_progress_bar.setVisible(False)
        self.status_bar.showMessage(f"Dataset loaded: {len(docs)} documents")
        
        QMessageBox.information(
            self, "Success",
            f"Dataset loaded successfully!\n\n"
            f"- Documents: {len(docs)}\n"
            f"- Vocabulary: {tokenizer.vocab_size} tokens\n\n"
            "You can now proceed to the next step."
        )
    
    def _on_dataset_error(self, error: str):
        """Handle dataset loading error"""
        self.load_dataset_btn.setEnabled(True)
        self.dataset_progress_bar.setVisible(False)
        self.dataset_progress_label.setText("")
        
        QMessageBox.critical(self, "Error", error)
        self.status_bar.showMessage("Dataset loading failed")
    
    def _update_review(self):
        """Update review page with current configuration"""
        use_case = self._get_selected_use_case()
        stack = self.selected_stack
        
        review = f"""
        📋 CONFIGURATION SUMMARY
        ========================
        
        🎯 MODEL PURPOSE
        - Use Case: {use_case.mode}
        
        🔧 ARCHITECTURE
        - Layers: {self.n_layer_spin.value()}
        - Embedding Size: {self.n_embd_spin.value()}
        - Context Length: {self.block_size_spin.value()}
        - Attention Heads: {self.n_head_spin.value()}
        - Vocabulary Size: {self.vocab_size_spin.value()}
        
        📊 DATASET
        - Documents: {len(self.docs) if self.docs else 0}
        - Vocabulary: {self.tokenizer.vocab_size if self.tokenizer else 0}
        
        ⚡ TRAINING PARAMETERS
        - Learning Rate: {self.lr_spin.value()}
        - Steps: {self.steps_spin.value()}
        - Beta1: {self.beta1_spin.value()}
        - Beta2: {self.beta2_spin.value()}
        - Temperature: {self.temp_spin.value()}
        
        🖥️ BACKEND
        - Stack: {stack.description}
        - GPU Acceleration: {'Yes' if stack.use_gpu else 'No'}
        
        ========================
        Ready to train! Click "Start Training" to begin.
        """
        
        self.review_text.setText(review)
    
    def _start_training_from_wizard(self):
        """Start training from wizard"""
        # Initialize model
        try:
            config = ModelConfig(
                n_layer=self.n_layer_spin.value(),
                n_embd=self.n_embd_spin.value(),
                block_size=self.block_size_spin.value(),
                n_head=self.n_head_spin.value(),
                vocab_size=self.vocab_size_spin.value()
            )
            
            use_case = self._get_selected_use_case()
            
            # Determine if using torch
            use_torch = False
            device = None
            
            if self.selected_stack.use_gpu and TORCH_AVAILABLE:
                import torch
                if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    use_torch = True
                    if torch.cuda.is_available():
                        device = torch.device('cuda')
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device = torch.device('mps')
            
            if use_torch:
                self.model = TorchGPTModel(config, use_case)
                train_config = TrainingConfig(
                    learning_rate=self.lr_spin.value(),
                    num_steps=self.steps_spin.value(),
                    beta1=self.beta1_spin.value(),
                    beta2=self.beta2_spin.value(),
                    temperature=self.temp_spin.value()
                )
                self.trainer = TorchTrainer(self.model, train_config, device=device, use_amp=True)
            else:
                self.model = GPTModel(config, use_case)
                train_config = TrainingConfig(
                    learning_rate=self.lr_spin.value(),
                    num_steps=self.steps_spin.value(),
                    beta1=self.beta1_spin.value(),
                    beta2=self.beta2_spin.value(),
                    temperature=self.temp_spin.value()
                )
                self.trainer = Trainer(self.model, self.tokenizer, train_config)
            
            # Move to training page
            self.current_step = 6
            self.stack.setCurrentIndex(6)
            self._update_navigation()
            
            # Start training
            self._start_training()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize: {e}")
    
    def _start_training(self):
        """Start training process"""
        self.start_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self.status_label.setText("Status: Training...")
        
        self.training_worker = TrainingWorker(self.model, self.tokenizer, self.trainer, self.docs)
        self.training_worker.progress.connect(self._on_training_progress)
        self.training_worker.finished.connect(self._on_training_finished)
        self.training_worker.error.connect(self._on_training_error)
        self.training_worker.start()
    
    def _stop_training(self):
        """Stop training"""
        if self.training_worker:
            self.training_worker.terminate()
            self.status_label.setText("Status: Stopped")
            self.stop_train_btn.setEnabled(False)
    
    def _on_training_progress(self, step: int, loss: float, elapsed: float, avg_step: float):
        """Handle training progress"""
        total = self.steps_spin.value()
        progress = int((step / total) * 100)
        self.train_progress_bar.setValue(progress)
        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.time_label.setText(f"Time: {elapsed:.1f}s")
        self.step_time_label.setText(f"Avg/step: {avg_step:.3f}s")
        self.status_label.setText(f"Status: Step {step}/{total}")
        QApplication.processEvents()
    
    def _on_training_finished(self, result: ExperimentResult):
        """Handle training completion"""
        self.training_result = result
        self.stop_train_btn.setEnabled(False)
        self.status_label.setText("Status: Complete ✓")
        self.train_progress_bar.setValue(100)
        
        # Auto-save model
        try:
            use_case = self._get_selected_use_case()
            model_id = self.model_manager.save_model(
                model=self.model,
                tokenizer=self.tokenizer,
                model_config=result.model_config,
                training_config=result.training_config,
                final_loss=result.final_loss,
                total_steps=result.training_config.num_steps,
                use_case=use_case,
                docs=self.docs,
                name=f"Model {datetime.now().strftime('%Y%m%d %H:%M')}",
                description=f"Trained with loss {result.final_loss:.4f}"
            )
            
            # Auto-assess
            assessor = QualityAssessor(self.tokenizer, self.model)
            quality_results = assessor.assess()
            self.model_manager.update_quality_score(model_id, quality_results['overall_score'], quality_results)
            
            self.status_bar.showMessage(f"Model saved: {model_id} | Quality: {quality_results['overall_score']:.2f}")
        except Exception as e:
            self.status_bar.showMessage(f"Warning: Could not save model: {e}")
        
        # Move to results page
        self.current_step = 7
        self.stack.setCurrentIndex(7)
        self._update_navigation()
        
        # Display results
        self._display_results(result)
    
    def _on_training_error(self, error: str):
        """Handle training error"""
        self.stop_train_btn.setEnabled(False)
        self.status_label.setText("Status: Error ✗")
        QMessageBox.critical(self, "Training Error", error)
    
    def _display_results(self, result: ExperimentResult):
        """Display training results"""
        summary = f"""
        ✅ TRAINING COMPLETE
        
        📊 RESULTS
        - Final Loss: {result.final_loss:.4f}
        - Training Time: {result.training_time:.1f}s
        - Steps Completed: {result.training_config.num_steps}
        
        📈 LOSS HISTORY
        - Initial Loss: {result.loss_history[0]:.4f}
        - Final Loss: {result.final_loss:.4f}
        - Improvement: {((result.loss_history[0] - result.final_loss) / result.loss_history[0] * 100):.1f}%
        
        🎨 SAMPLE OUTPUTS
        """
        
        for i, sample in enumerate(result.samples, 1):
            summary += f"\n{i}. {sample[:100]}{'...' if len(sample) > 100 else ''}"
        
        self.results_summary.setText(summary)
        
        # Auto-assessment
        try:
            assessor = QualityAssessor(self.tokenizer, self.model)
            results = assessor.assess()
            
            self.quality_score_label.setText(f"Overall Score: {results['overall_score']:.2f}/1.00")
            self.quality_progress.setValue(int(results['overall_score'] * 100))
            
            if results['recommendations']:
                self.quality_recommendations.setText(
                    "Recommendations:\n" + "\n".join(f"• {r}" for r in results['recommendations'])
                )
        except Exception as e:
            self.quality_score_label.setText("Assessment failed")
    
    def _save_model_from_wizard(self):
        """Save model from wizard"""
        if not self.training_result:
            return
        
        try:
            use_case = self._get_selected_use_case()
            model_id = self.model_manager.save_model(
                model=self.model,
                tokenizer=self.tokenizer,
                model_config=self.training_result.model_config,
                training_config=self.training_result.training_config,
                final_loss=self.training_result.final_loss,
                total_steps=self.training_result.training_config.num_steps,
                use_case=use_case,
                docs=self.docs,
                name=f"Model {datetime.now().strftime('%Y%m%d %H:%M')}",
                description=f"Trained with loss {self.training_result.final_loss:.4f}"
            )
            
            QMessageBox.information(
                self, "Model Saved",
                f"Model saved successfully!\n\nModel ID: {model_id}\n\n"
                f"You can load it later from the Models tab."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")
    
    def _run_auto_test(self):
        """Run automatic quality test"""
        if not self.model or not self.tokenizer:
            return
        
        try:
            assessor = QualityAssessor(self.tokenizer, self.model)
            results = assessor.assess()
            
            self.quality_score_label.setText(f"Overall Score: {results['overall_score']:.2f}/1.00")
            self.quality_progress.setValue(int(results['overall_score'] * 100))
            
            if results['recommendations']:
                self.quality_recommendations.setText(
                    "Recommendations:\n" + "\n".join(f"• {r}" for r in results['recommendations'])
                )
            
            QMessageBox.information(
                self, "Auto-Test Complete",
                f"Quality Score: {results['overall_score']:.2f}/1.00\n\n"
                f"Metrics:\n"
                f"- Coherence: {results['coherence_score']:.2f}\n"
                f"- Diversity: {results['diversity_score']:.2f}\n"
                f"- Completion Rate: {results['completion_rate']:.0%}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Test failed: {e}")
    
    def _generate_text(self):
        """Generate text from prompt"""
        if not self.model or not self.tokenizer:
            QMessageBox.warning(self, "Warning", "No model loaded!")
            return
        
        try:
            temperature = self.gen_temp_slider.value() / 10
            max_tokens = self.max_tokens_spin.value()
            prompt = self.prompt_edit.text()
            
            if TORCH_AVAILABLE and isinstance(self.model, TorchGPTModel):
                import torch
                device = next(self.model.parameters()).device
                
                if prompt:
                    text = self.model.generate_text(
                        self.tokenizer, prompt=prompt,
                        max_new_tokens=max_tokens, temperature=temperature, device=device
                    )
                else:
                    idx = torch.tensor([[self.tokenizer.BOS]], dtype=torch.long, device=device)
                    generated_idx = self.model.generate(idx, max_tokens, temperature)
                    text = self.tokenizer.decode(generated_idx[0, 1:].tolist())
            else:
                if prompt:
                    text = self.model.generate(
                        self.tokenizer, max_tokens=max_tokens,
                        temperature=temperature, prompt=prompt
                    )
                else:
                    text = self.model.generate(
                        self.tokenizer, max_tokens=max_tokens, temperature=temperature
                    )
            
            self.output_text.setText(text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Generation failed: {e}")


def main():
    """Main entry point"""
    if not HAS_QT:
        print("\n" + "="*60)
        print("ERROR: PySide6 is required for the GUI application")
        print("="*60)
        print("\nInstall with: pip install PySide6")
        print("\nOr use the CLI version instead:")
        print("  python llm_studio_cli.py")
        print("="*60 + "\n")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application info
    app.setApplicationName("OpenHammer LLM Studio")
    app.setApplicationVersion("2.0.0 Wizard Edition")
    
    window = WizardWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

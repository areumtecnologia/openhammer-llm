"""
Desktop GUI Application for LLM Training and Inference
Built with PySide6 (Qt for Python)
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QLineEdit, QTextEdit, QSpinBox,
        QDoubleSpinBox, QComboBox, QProgressBar, QGroupBox, QFormLayout,
        QFileDialog, QMessageBox, QSplitter, QScrollArea, QCheckBox,
        QListWidget, QListWidgetItem, QSlider, QStatusBar, QMenu,
        QAction, QMenuBar, QDialog, QDialogButtonBox, QGridLayout
    )
    from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
    from PySide6.QtGui import QFont, QIcon, QAction
    HAS_QT = True
except ImportError:
    HAS_QT = False
    print("⚠ PySide6 not installed. Install with: pip install PySide6")
    sys.exit(1)

from core.model import (
    ModelConfig, TrainingConfig, HardwareProfile, GPTModel, 
    Trainer, Tokenizer, DatasetManager, ExperimentResult
)


class TrainingWorker(QThread):
    """Background thread for training"""
    progress = Signal(int, float)
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
            def callback(step, loss):
                self.progress.emit(step, loss)
            
            self.trainer.train(self.docs, callback=callback)
            
            # Generate samples
            samples = []
            for _ in range(5):
                sample = self.model.generate(self.tokenizer, max_tokens=30)
                samples.append(sample)
            
            result = ExperimentResult(
                model_config=self.model.config,
                training_config=self.trainer.config,
                final_loss=self.trainer.loss_history[-1] if self.trainer.loss_history else 0,
                loss_history=self.trainer.loss_history,
                samples=samples,
                training_time=0,
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
        
        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._update_hardware_info()
    
    def _setup_ui(self):
        """Setup user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Create tabs
        tabs.addTab(self._create_home_tab(), "🏠 Home")
        tabs.addTab(self._create_model_tab(), "🔧 Model Config")
        tabs.addTab(self._create_dataset_tab(), "📊 Dataset")
        tabs.addTab(self._create_training_tab(), "🎯 Training")
        tabs.addTab(self._create_inference_tab(), "💬 Inference")
        tabs.addTab(self._create_results_tab(), "📈 Results")
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _create_home_tab(self) -> QWidget:
        """Create home/welcome tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Hardware info
        hw_group = QGroupBox("Hardware Profile")
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
        hw_layout.addRow("Recommendation:", self.hw_recommendation_label)
        
        hw_group.setLayout(hw_layout)
        layout.addWidget(hw_group)
        
        # Quick start
        quick_group = QGroupBox("Quick Start")
        quick_layout = QVBoxLayout()
        quick_label = QLabel("""
        <h3>Welcome to OpenHammer LLM Studio!</h3>
        <p>Create and train language models on low-cost hardware.</p>
        <ol>
            <li>Configure your model architecture</li>
            <li>Load or create a dataset</li>
            <li>Train your model</li>
            <li>Generate text with inference</li>
        </ol>
        <p><b>Tip:</b> Start with small models (2 layers, 32 embd) for testing.</p>
        """)
        quick_label.setWordWrap(True)
        quick_layout.addWidget(quick_label)
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
        layout.addStretch()
        return widget
    
    def _create_model_tab(self) -> QWidget:
        """Create model configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        form_group = QGroupBox("Model Architecture")
        form_layout = QFormLayout()
        
        # Model parameters
        self.n_layer_spin = QSpinBox()
        self.n_layer_spin.setRange(1, 12)
        self.n_layer_spin.setValue(2)
        self.n_layer_spin.setToolTip("Number of transformer layers")
        
        self.n_embd_spin = QSpinBox()
        self.n_embd_spin.setRange(16, 512)
        self.n_embd_spin.setSingleStep(16)
        self.n_embd_spin.setValue(32)
        self.n_embd_spin.setToolTip("Embedding dimension")
        
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(16, 512)
        self.block_size_spin.setSingleStep(16)
        self.block_size_spin.setValue(32)
        self.block_size_spin.setToolTip("Maximum context length")
        
        self.n_head_spin = QSpinBox()
        self.n_head_spin.setRange(1, 16)
        self.n_head_spin.setValue(4)
        self.n_head_spin.setToolTip("Number of attention heads")
        
        self.vocab_size_spin = QSpinBox()
        self.vocab_size_spin.setRange(64, 1024)
        self.vocab_size_spin.setValue(256)
        self.vocab_size_spin.setToolTip("Vocabulary size")
        
        form_layout.addRow("Layers:", self.n_layer_spin)
        form_layout.addRow("Embedding Dim:", self.n_embd_spin)
        form_layout.addRow("Block Size:", self.block_size_spin)
        form_layout.addRow("Attention Heads:", self.n_head_spin)
        form_layout.addRow("Vocab Size:", self.vocab_size_spin)
        
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)
        
        # Model info
        self.model_info_label = QLabel()
        self.model_info_label.setStyleSheet("QLabel { font-weight: bold; color: #2196F3; }")
        layout.addWidget(self.model_info_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.init_model_btn = QPushButton("🚀 Initialize Model")
        self.init_model_btn.clicked.connect(self._initialize_model)
        btn_layout.addWidget(self.init_model_btn)
        
        self.save_model_btn = QPushButton("💾 Save Model")
        self.save_model_btn.clicked.connect(self._save_model)
        self.save_model_btn.setEnabled(False)
        btn_layout.addWidget(self.save_model_btn)
        
        self.load_model_btn = QPushButton("📂 Load Model")
        self.load_model_btn.clicked.connect(self._load_model)
        btn_layout.addWidget(self.load_model_btn)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        
        # Connect signals
        for spin in [self.n_layer_spin, self.n_embd_spin, self.block_size_spin, 
                     self.n_head_spin, self.vocab_size_spin]:
            spin.valueChanged.connect(self._update_model_info)
        
        self._update_model_info()
        return widget
    
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
        
        self.status_label = QLabel("Status: Idle")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.loss_label)
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
        """Initialize model with current config"""
        try:
            config = ModelConfig(
                n_layer=self.n_layer_spin.value(),
                n_embd=self.n_embd_spin.value(),
                block_size=self.block_size_spin.value(),
                n_head=self.n_head_spin.value(),
                vocab_size=self.vocab_size_spin.value()
            )
            
            self.model = GPTModel(config)
            self.save_model_btn.setEnabled(True)
            
            self.status_bar.showMessage(
                f"Model initialized with {config.num_params:,} parameters"
            )
            
            QMessageBox.information(
                self, "Success",
                f"Model initialized successfully!\n"
                f"Parameters: {config.num_params:,}\n"
                f"Ready for training."
            )
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
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {e}")
    
    def _start_training(self):
        """Start training process"""
        if not self.model or not self.tokenizer:
            QMessageBox.warning(
                self, "Warning",
                "Please initialize model and load dataset first!"
            )
            return
        
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
            
            # Create trainer
            self.trainer = Trainer(self.model, self.tokenizer, train_config)
            
            # Setup worker thread
            self.worker = TrainingWorker(self.model, self.tokenizer, self.trainer, docs)
            self.worker.progress.connect(self._on_training_progress)
            self.worker.finished.connect(self._on_training_finished)
            self.worker.error.connect(self._on_training_error)
            
            # Update UI
            self.start_train_btn.setEnabled(False)
            self.stop_train_btn.setEnabled(True)
            self.status_label.setText("Status: Training...")
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
    
    def _on_training_progress(self, step: int, loss: float):
        """Handle training progress update"""
        total = self.steps_spin.value()
        progress = int((step / total) * 100)
        self.progress_bar.setValue(progress)
        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.status_label.setText(f"Status: Step {step}/{total}")
        QApplication.processEvents()
    
    def _on_training_finished(self, result: ExperimentResult):
        """Handle training completion"""
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.status_label.setText("Status: Complete")
        self.progress_bar.setValue(100)
        
        self.current_result = result
        self._add_result_to_list(result)
        
        QMessageBox.information(
            self, "Training Complete",
            f"Final Loss: {result.final_loss:.4f}\n"
            f"Samples generated: {len(result.samples)}"
        )
    
    def _on_training_error(self, error: str):
        """Handle training error"""
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.status_label.setText("Status: Error")
        QMessageBox.critical(self, "Training Error", error)
    
    def _generate_text(self):
        """Generate text with current model"""
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
            
            # Generate
            if prompt:
                # For now, just generate from scratch
                # TODO: Implement prompt conditioning
                pass
            
            text = self.model.generate(
                self.tokenizer,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            self.output_text.setText(text)
            self.status_bar.showMessage("Text generated successfully")
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

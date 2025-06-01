# Video Agent Repository README

This repository contains a video analysis system with capabilities for keyframe extraction, facial expression recognition, and video question answering evaluation.

## 📁 Repository Structure

```
video_agent/
├── data_process/           
│   ├── complex_question_selection.py  
│   ├── error_case_saved.py   
│   ├── error_complex.py       
│   └── separation_complex_question.py 
├── evaluation/
│   ├── complex_question_2API.ipynb    
│   └── complex_question_agent_gpt4v.ipynb   
│   └── complex_question_agent_llama.ipynb    
│   └── complex_question_agent_local.ipynb    
│   └── VQA_gpt4v.ipynb    
│   └── eval_mvbench_0.py    
│   └── eval_mvbench_1.py    
└── local_model/
    ├── Facial_Expression_Recognition/
    ├── KeyFrame_Extraction/
    ├── PaddleVideo/
    ├── VideoCaptioningTransformer/
    └── requirement.txt
```

## 🔧 Installation

Install the required dependencies:

```bash
cd local_model
pip install -r requirement.txt
```

## 📊 Data Processing

### Video Data Processing

The evaluation scripts `eval_mvbench_0.py` support multiple video formats and preprocessing:

- **Video Reading**: Uses `decord` library for efficient video loading
- **Frame Sampling**: Configurable number of segments (default: 8-16 frames)
- **Resolution**: Standardized to 224x224 pixels
- **Data Transformations**:
  - `GroupScale`: Rescales images maintaining aspect ratio
  - `GroupCenterCrop`: Center crops to target size
  - `GroupNormalize`: Normalizes pixel values
  - `ToTorchFormatTensor`: Converts to PyTorch tensor format


## 🔍 Inference

### MVBench Evaluation

Evaluate video understanding capabilities on MVBench dataset:

```python
# Run MVBench evaluation
python evaluation/eval_mvbench_1.py
```

**Evaluation Process:**
1. Load video samples with question-answer pairs
2. Process videos through the model pipeline
3. Generate predictions for multiple-choice questions
4. Calculate accuracy metrics by task type
5. Save results to JSON file


### Agent Interface

The system provides a Agent interface:

```bash
# Run ipynb file for different base models:
complex_question_agent_gpt4v.ipynb   
complex_question_agent_llama.ipynb    
complex_question_agent_local.ipynb    
VQA_gpt4v.ipynb  
```

## 📈 Performance Metrics

The evaluation system tracks:
- **Overall Accuracy**: Percentage of correct predictions
- **Task-specific Accuracy**: Performance breakdown by question type
- **Detailed Results**: Individual predictions with ground truth comparisons

Results are automatically saved in structured formats (JSON/CSV) for further analysis.

## 🎯 Key Features

- **Multi-modal Processing**: Handles video, audio, and text inputs
- **Flexible Keyframe Extraction**: Three different extraction methods
- **Comprehensive Evaluation**: Support for multiple benchmark datasets  
- **GPU Acceleration**: Optimized for CUDA-enabled systems
- **Modular Design**: Easy to extend with new models and evaluation metrics

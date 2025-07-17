# PI-GAN Training and Evaluation System Guide

## 🎯 System Overview

The PI-GAN THz metamaterial system has been completely restructured with optimized training and comprehensive evaluation capabilities. All components now support English language plotting and provide detailed loss curves and accuracy metrics.

## 🔧 Training System

### 1. Optimized Training

**Fixed Issues:**
- ✅ Generator initialization parameters corrected
- ✅ DataLoader worker count optimized (reduced to 2)
- ✅ Training curves and loss visualization added
- ✅ Enhanced loss functions with constraint penalties

**Usage:**
```bash
# Run optimized training with plotting
python core/train/optimized_trainer.py --epochs 200 --batch_size 64

# Custom parameters
python core/train/optimized_trainer.py \
    --epochs 200 \
    --batch_size 32 \
    --device cuda \
    --seed 42
```

**Training Outputs:**
- Training loss curves saved to `plots/training_curves_YYYYMMDD_HHMMSS.png`
- Model checkpoints every 50 epochs
- Real-time constraint violation monitoring

### 2. Training Features

**Loss Visualization:**
- Generator vs Discriminator loss curves
- Parameter constraint violation rate tracking
- Learning rate schedules
- Detailed loss component breakdown

**Optimization Features:**
- Pre-training of forward model (50 epochs)
- Progressive loss weight scheduling
- Constraint penalties and physics losses
- Stability regularization

## 📊 Evaluation System

### 1. Unified Evaluation (Recommended)

```bash
# Complete evaluation with all visualizations
python core/evaluate/unified_evaluator.py --num_samples 1000
```

**Generates:**
- `plots/forward_network_evaluation_YYYYMMDD_HHMMSS.png`
- `plots/pigan_evaluation_YYYYMMDD_HHMMSS.png`
- `plots/structural_prediction_evaluation_YYYYMMDD_HHMMSS.png`
- `plots/model_validation_evaluation_YYYYMMDD_HHMMSS.png`
- `plots/comprehensive_summary_YYYYMMDD_HHMMSS.png`
- `plots/unified_evaluation_report.txt`

### 2. Individual Evaluation Modules

Each evaluation module now includes comprehensive plotting:

#### Forward Network Evaluation
```bash
python core/evaluate/evaluate_fwd_model.py --num_samples 1000
```

**Plots Include:**
- R² score comparison (Spectrum vs Metrics)
- Error metrics comparison (MAE, RMSE, MAPE)
- Performance radar chart
- Spectrum reconstruction examples

#### PI-GAN Evaluation
```bash
python core/evaluate/evaluate_pigan.py --num_samples 1000
```

**Plots Include:**
- Generator performance metrics
- Discriminator accuracy analysis
- Score distribution comparison
- Parameter prediction scatter plots
- Performance summary radar chart

#### Structural Prediction Evaluation
```bash
python core/evaluate/evaluate_structural_prediction.py --num_samples 500
```

**Plots Include:**
- Parameter violation rate assessment
- Quality metrics comparison
- Performance radar chart
- Detailed statistics summary

#### Model Validation Evaluation
```bash
python core/evaluate/evaluate_model_validation.py --num_samples 500
```

**Plots Include:**
- Validation metrics overview
- Error metrics (cycle consistency, stability)
- Validation quality radar chart
- Detailed assessment summary

## 🎨 Visualization Features

### English Language Support
- All plots now use English labels and titles
- Professional styling with DejaVu Sans font
- Consistent color coding across all evaluations

### Plot Types
- **Bar Charts**: Performance comparisons with target lines
- **Radar Charts**: Multi-dimensional performance visualization
- **Scatter Plots**: Parameter prediction accuracy
- **Histograms**: Score and error distributions
- **Line Plots**: Training curves and time series

### Color Coding
- 🟢 **Green**: Excellent performance (meets targets)
- 🔵 **Blue**: Good performance (close to targets)
- 🟠 **Orange**: Moderate performance (needs improvement)
- 🔴 **Red**: Poor performance (requires optimization)

## 📈 Performance Targets

### Training Targets
- Generator Loss: Stable convergence without oscillation
- Discriminator Loss: Balanced with generator (not dominating)
- Violation Rate: < 10% (target < 5%)
- Learning Rate: Smooth decay schedule

### Evaluation Targets
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Forward Network Spectrum R² | 0.50 | > 0.85 | ❌ Needs Improvement |
| Forward Network Metrics R² | 0.80 | > 0.90 | 🟠 Close to Target |
| PI-GAN Parameter R² | 0.53 | > 0.80 | ❌ Needs Improvement |
| Discriminator Accuracy | 0.61 | > 0.80 | ❌ Needs Improvement |
| Parameter Violation Rate | 87.4% | < 10% | ❌ Critical Issue |
| Physical Plausibility | 0.13 | > 0.80 | ❌ Critical Issue |
| Cycle Consistency Error | 0.052 | < 0.01 | ❌ Needs Improvement |

## 🔍 Common Issues and Solutions

### Training Issues

**Issue: High Violation Rate**
```bash
# Solution: Use optimized trainer with stronger constraints
python core/train/optimized_trainer.py --epochs 200
```

**Issue: Training Instability**
- Check loss curves in plots/training_curves_*.png
- Reduce learning rate if oscillating
- Increase constraint loss weights

**Issue: Poor Generator Performance**
- Increase reconstruction loss weight
- Add more training epochs
- Check data quality and preprocessing

### Evaluation Issues

**Issue: Low R² Scores**
- Retrain with optimized parameters
- Check data preprocessing
- Increase model capacity

**Issue: High Parameter Violations**
- Add stronger constraint penalties
- Implement parameter clipping
- Review parameter range definitions

## 🚀 Optimization Workflow

### Step 1: Analyze Current Performance
```bash
# Run comprehensive evaluation
python core/evaluate/unified_evaluator.py --num_samples 1000
```

### Step 2: Check Training Configuration
```bash
# Review optimization settings
python config/training_optimization.py
```

### Step 3: Run Optimized Training
```bash
# Train with optimization
python core/train/optimized_trainer.py --epochs 200
```

### Step 4: Monitor Training Progress
- Check `plots/training_curves_*.png` every 10 epochs
- Monitor constraint violation rate
- Ensure stable loss convergence

### Step 5: Re-evaluate Performance
```bash
# Re-run evaluation after training
python core/evaluate/unified_evaluator.py --num_samples 1000
```

## 📁 File Structure

```
PI-GAN-THz/
├── core/
│   ├── train/
│   │   ├── optimized_trainer.py      # 🔄 Fixed & Enhanced
│   │   └── pretrain_fwd_model.py
│   └── evaluate/
│       ├── unified_evaluator.py      # 🎨 Visualization Integrated
│       ├── evaluate_fwd_model.py     # 📊 Plotting Added
│       ├── evaluate_pigan.py         # 📊 Plotting Added
│       ├── evaluate_structural_prediction.py  # 📊 Plotting Added
│       └── evaluate_model_validation.py       # 📊 Plotting Added
├── config/
│   ├── config.py
│   └── training_optimization.py      # 🎯 Optimization Settings
├── plots/                            # 📈 All Visualizations
└── logs/                            # 📝 Training Logs
```

## 🎯 Key Improvements Made

### Training System
1. **Fixed model initialization** - Corrected parameter mismatches
2. **Added training visualization** - Real-time loss and accuracy curves
3. **Optimized DataLoader** - Reduced worker count to prevent warnings
4. **Enhanced loss functions** - Added constraint and physics penalties

### Evaluation System
1. **English language support** - All plots in English
2. **Comprehensive plotting** - Each module generates detailed visualizations
3. **Performance assessment** - Color-coded ratings and target comparisons
4. **Detailed statistics** - Quantitative analysis with recommendations

### Visualization Features
1. **Professional styling** - Consistent fonts and colors
2. **Multi-plot layouts** - Comprehensive analysis in single images
3. **Interactive legends** - Clear identification of data series
4. **High-resolution output** - 300 DPI for publication quality

## 📞 Usage Examples

### Complete Training and Evaluation Pipeline
```bash
# 1. Train optimized model
python core/train/optimized_trainer.py --epochs 200

# 2. Evaluate all modules
python core/evaluate/unified_evaluator.py --num_samples 1000

# 3. Check individual modules if needed
python core/evaluate/evaluate_fwd_model.py
python core/evaluate/evaluate_pigan.py
```

### Quick Performance Check
```bash
# Fast evaluation with fewer samples
python core/evaluate/unified_evaluator.py --num_samples 500
```

### Monitoring Training Progress
```bash
# During training, monitor plots/ directory for:
# - training_curves_*.png (updated every training session)
# - Individual evaluation plots (if running evaluations)
```

## 📊 Expected Results

After running the complete pipeline, you should see:

1. **Training curves** showing convergence
2. **Evaluation plots** with performance metrics
3. **Color-coded assessments** indicating areas for improvement
4. **Detailed reports** with specific recommendations

The system is now fully equipped to provide comprehensive analysis and visualization of the PI-GAN THz metamaterial inverse design performance.
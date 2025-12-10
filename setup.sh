#!/bin/bash
# setup.sh - Automated project setup script
# Run this file to set up the entire project: bash setup.sh

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║        🌡️  WEATHER PREDICTION CAPSTONE - AUTOMATED SETUP              ║"
echo "║                                                                          ║"
echo "║  This script will:                                                       ║"
echo "║  1. Create project directories                                          ║"
echo "║  2. Install Python dependencies                                         ║"
echo "║  3. Generate sample data                                                ║"
echo "║  4. Process data pipeline                                               ║"
echo "║  5. Train ML models                                                     ║"
echo "║                                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"
echo ""

# Step 1: Create directories
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p models
mkdir -p results
echo "   ✓ Created data/ models/ results/ directories"
echo ""

# Step 2: Install dependencies
echo "📦 Installing Python dependencies..."
echo "   (This may take 5-10 minutes for TensorFlow...)"
pip install -r requirements.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Dependencies installed successfully"
else
    echo "   ⚠️  Some dependencies failed to install. Try manual: pip install -r requirements.txt"
fi
echo ""

# Step 3: Generate data
echo "📊 Generating sample weather data..."
python3 generate_sample_data.py
echo ""

# Step 4: Process pipeline
echo "🔧 Running data processing pipeline..."
python3 data_pipeline.py
echo ""

# Step 5: Train models
echo "🤖 Training ML models..."
echo "   (This will train Random Forest, XGBoost, and LSTM)"
python3 ml_training.py
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ SETUP COMPLETE!                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Generated Files:"
echo "   ✓ data/weather_data.csv - Raw weather data (8,760 records)"
echo "   ✓ data/processed_data.csv - Processed data (25 features)"
echo "   ✓ models/random_forest.pkl - Trained Random Forest model"
echo "   ✓ models/xgboost.pkl - Trained XGBoost model"
echo "   ✓ models/lstm.h5 - Trained LSTM model"
echo "   ✓ results/model_comparison.csv - Performance metrics"
echo ""
echo "🚀 Next Step: Launch Dashboard"
echo "   Run: streamlit run app.py"
echo ""
echo "📝 Then open: http://localhost:8501"
echo ""
echo "🎉 Happy forecasting!"
echo ""

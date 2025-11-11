#!/bin/bash
# Launch EP Performance Analysis Dashboard with Public URL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."  # Go to repo root

echo "=========================================="
echo "EP Performance Analysis Dashboard"
echo "=========================================="
echo ""

# Check if gradio is installed
if ! env/bin/python -c "import gradio" 2>/dev/null; then
    echo "📦 Installing gradio..."
    env/bin/pip install -q gradio
    echo "✅ Gradio installed"
    echo ""
fi

# Check if plotly is installed
if ! env/bin/python -c "import plotly" 2>/dev/null; then
    echo "📦 Installing plotly..."
    env/bin/pip install -q plotly
    echo "✅ Plotly installed"
    echo ""
fi

# Check if cache exists, if not run precompute
CACHE_FILE="$SCRIPT_DIR/.analysis_cache.pkl"
if [ ! -f "$CACHE_FILE" ]; then
    echo "⚡ No cache found. Pre-computing analysis data for instant dashboard loading..."
    echo "   This will take ~60 seconds but only needs to run once."
    echo ""
    chmod +x "$SCRIPT_DIR/precompute_analysis.py"
    env/bin/python "$SCRIPT_DIR/precompute_analysis.py"
    echo ""
fi

echo "🚀 Starting ultra-deep dashboard..."
echo ""
echo "⚠️  IMPORTANT: The public Gradio link will appear below after ~10 seconds."
echo "   Look for a line that says: 'Running on public URL: https://...gradio.live'"
echo ""
echo "   Copy that URL and open it in your local browser."
echo ""
echo "Press Ctrl+C to stop the dashboard when done."
echo ""
echo "=========================================="
echo ""

# Kill any existing dashboard
pkill -f "interactive_dashboard.py" 2>/dev/null || true
sleep 2

# Launch dashboard - output will go directly to terminal
# Using interactive_dashboard.py with cache support
env/bin/python "$SCRIPT_DIR/interactive_dashboard.py"

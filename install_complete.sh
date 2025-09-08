#!/bin/bash

# LLM Firewall API - Complete Dependency Installation Script
# Handles all edge cases and dependency conflicts for clean installation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 LLM Firewall API - Complete Installation${NC}"
echo "=============================================="
echo ""

# Function to print colored output
print_step() {
    echo -e "${BLUE}📦 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "   Found Python $python_version"

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "Not in a virtual environment"
    echo "   Recommendation: python3 -m venv .venv && source .venv/bin/activate"
    echo ""
else
    print_success "Virtual environment active: $VIRTUAL_ENV"
fi

# Function to install with error handling
install_package() {
    local package=$1
    local description=$2
    
    echo "   Installing $description..."
    if pip install $package; then
        print_success "$description installed"
    else
        print_error "Failed to install $description"
        return 1
    fi
}

# Step 1: Upgrade pip and basic tools
print_step "Upgrading pip and basic tools"
pip install --upgrade pip setuptools wheel
print_success "Basic tools upgraded"
install_package "python-dotenv>=1.0.0" "python-dotenv"
install_package "aiohttp>=3.9.0" "aiohttp"

# Step 2: Install FastAPI stack
print_step "Installing FastAPI stack"
install_package "fastapi>=0.104.0" "FastAPI"
install_package "uvicorn>=0.24.0" "Uvicorn"
install_package "pydantic>=2.4.2" "Pydantic"

# Step 3: Install LlamaFirewall (this will install torch>=2.4.1)
print_step "Installing LlamaFirewall and dependencies"
install_package "llamafirewall>=1.0.2" "LlamaFirewall"
install_package "openai>=1.3.0" "OpenAI client"
install_package "tenacity>=9.1.2" "Tenacity retry library"

# Step 4: Install LLM Guard dependencies carefully
print_step "Installing LLM Guard dependencies (handling conflicts)"

# Install LLM Guard without dependencies first
echo "   Installing llm-guard (no dependencies)..."
pip install --no-deps llm-guard>=0.3.0
print_success "llm-guard package installed"

# Install presidio for privacy protection
install_package "presidio-analyzer>=2.2.359" "Presidio Analyzer"
install_package "presidio-anonymizer>=2.2.359" "Presidio Anonymizer"

# Install security and NLP dependencies
install_package "detect-secrets" "Secret detection"
install_package "faker" "Fake data generation"
install_package "nltk" "Natural Language Toolkit"
install_package "spacy" "spaCy NLP"

# Install text processing utilities
install_package "fuzzysearch" "Fuzzy string search"
install_package "json-repair" "JSON repair utility"
install_package "structlog" "Structured logging"

# Install tokenization and ML utilities
install_package "tiktoken" "OpenAI tokenizer"
install_package "sentencepiece" "SentencePiece tokenizer"

# Install ML dependencies
install_package "span-marker" "Span marker NER"
install_package "scikit-learn" "Scikit-learn ML"
install_package "datasets" "HuggingFace datasets"
install_package "evaluate" "HuggingFace evaluate"
install_package "accelerate" "HuggingFace accelerate"

# Step 5: Verify installations
print_step "Verifying installations"

echo "   Testing FastAPI..."
python3 -c "import fastapi; print(f'FastAPI {fastapi.__version__}')" && print_success "FastAPI working"

echo "   Testing LlamaFirewall..."
python3 -c "import llamafirewall; print('LlamaFirewall working')" && print_success "LlamaFirewall working"

echo "   Testing LLM Guard..."
python3 -c "
import llm_guard
from llm_guard.input_scanners import PromptInjection, Toxicity, Secrets
print('LLM Guard core scanners working')
" && print_success "LLM Guard working"

echo "   Testing API module..."
if [ -f "api.py" ]; then
    python3 -c "
import sys
import os
sys.path.insert(0, os.getcwd())
from api import parse_llmguard_config, LLMGUARD_AVAILABLE
print(f'API module working, LLM Guard available: {LLMGUARD_AVAILABLE}')
" && print_success "API module working"
else
    print_warning "api.py not found in current directory"
fi

# Step 6: Final summary
echo ""
echo -e "${GREEN}🎉 Installation Complete!${NC}"
echo "=========================="
echo ""
echo "📋 Installed components:"
echo "   ✅ FastAPI web framework"
echo "   ✅ LlamaFirewall security scanner"
echo "   ✅ LLM Guard advanced scanning suite"
echo "   ✅ OpenAI integration"
echo "   ✅ Privacy protection (Presidio)"
echo "   ✅ Secret detection"
echo "   ✅ NLP processing tools"
echo ""
echo "🚀 Quick start:"
echo "   1. Copy .env.template to .env"
echo "   2. Configure environment variables"
echo "   3. Start server: uvicorn api:app --host 0.0.0.0 --port 8080"
echo "   4. Test: curl http://localhost:8080/health"
echo ""
echo "🔧 LLM Guard configuration:"
echo "   Set LLMGUARD_ENABLED=true in .env"
echo "   Configure scanners: LLMGUARD_INPUT_SCANNERS=[\"PromptInjection\",\"Toxicity\"]"
echo ""
print_success "Your LLM Firewall API is ready! 🛡️"

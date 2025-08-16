#!/bin/bash
# DipMaster Enhanced V4 - Claude Agents Setup Script for macOS
# Created: 2025-08-16
# Version: 4.0.0

echo "🚀 DipMaster Enhanced V4 - Claude Agents Setup"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "CLAUDE.md" ]; then
    echo -e "${RED}❌ Error: Please run this script from the DipMaster-Trading-System root directory${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Checking system requirements...${NC}"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✅ Python ${PYTHON_VERSION} found${NC}"
else
    echo -e "${RED}❌ Python 3.11+ required${NC}"
    exit 1
fi

# Check Node.js version  
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✅ Node.js ${NODE_VERSION} found${NC}"
else
    echo -e "${YELLOW}⚠️ Node.js 18+ recommended for frontend components${NC}"
fi

# Check Git
if command -v git &> /dev/null; then
    echo -e "${GREEN}✅ Git found${NC}"
else
    echo -e "${RED}❌ Git required${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}🔧 Setting up Claude Code configuration...${NC}"

# Verify Claude configuration files exist
if [ ! -f ".claude-config.json" ]; then
    echo -e "${RED}❌ .claude-config.json not found${NC}"
    exit 1
fi

if [ ! -d ".claude/agents" ]; then
    echo -e "${RED}❌ .claude/agents directory not found${NC}"
    exit 1
fi

# Count agents
AGENT_COUNT=$(ls -1 .claude/agents/*.md 2>/dev/null | wc -l)
echo -e "${GREEN}✅ Found ${AGENT_COUNT} agent configurations${NC}"

# List agents
echo -e "${BLUE}📝 Available agents:${NC}"
for agent in .claude/agents/*.md; do
    if [ -f "$agent" ]; then
        AGENT_NAME=$(basename "$agent" .md)
        echo -e "  • $AGENT_NAME"
    fi
done

echo ""
echo -e "${BLUE}📦 Setting up Python environment...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
if [ -f "dependencies/requirements.txt" ]; then
    echo -e "${YELLOW}Installing core dependencies...${NC}"
    pip install -r dependencies/requirements.txt
elif [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}Installing dependencies from requirements.txt...${NC}"
    pip install -r requirements.txt
fi

# Install ML dependencies if available
if [ -f "dependencies/requirements-ml.txt" ]; then
    echo -e "${YELLOW}Installing ML dependencies...${NC}"
    pip install -r dependencies/requirements-ml.txt
fi

echo ""
echo -e "${BLUE}🧪 Running basic system tests...${NC}"

# Test basic imports
python3 -c "
import sys
try:
    import pandas as pd
    import numpy as np
    print('✅ Core data packages: OK')
except ImportError as e:
    print(f'❌ Core data packages: {e}')
    sys.exit(1)

try:
    import sklearn
    print('✅ Scikit-learn: OK')
except ImportError:
    print('⚠️ Scikit-learn: Not installed (optional)')

try:
    import lightgbm
    import xgboost
    print('✅ ML packages: OK')
except ImportError:
    print('⚠️ ML packages: Not fully installed (will install when needed)')
"

echo ""
echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo ""
echo -e "${BLUE}📖 Next steps:${NC}"
echo -e "1. Open your project in Claude Code"
echo -e "2. Verify agents are loaded in Claude Code settings"
echo -e "3. Test an agent: Try asking 'Use the strategy-orchestrator agent to plan a new feature'"
echo -e "4. Review CLAUDE.md for detailed usage instructions"
echo ""
echo -e "${BLUE}🔍 Quick verification commands:${NC}"
echo -e "• List agents: ls .claude/agents/"
echo -e "• Check config: cat .claude-config.json"
echo -e "• Run demo: python run_enhanced_features_demo.py"
echo -e "• View docs: cat CLAUDE.md"
echo ""
echo -e "${YELLOW}💡 Pro tip: Use the agent workflow for systematic development!${NC}"
echo -e "Start with strategy-orchestrator → data-infrastructure-builder → feature-engineering-labeler..."
echo ""
echo -e "${GREEN}Happy trading! 🚀📈${NC}"
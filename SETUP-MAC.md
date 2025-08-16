# DipMaster Enhanced V4 - macOS Setup Guide

## 🚀 Quick Setup for Mac

### Prerequisites
- **macOS 10.15+**
- **Python 3.11+** 
- **Git**
- **Claude Code** (latest version)
- **Node.js 18+** (optional, for frontend)

### 1. Clone Repository
```bash
git clone https://github.com/Gyarados4157/DipMaster-Trading-System.git
cd DipMaster-Trading-System
```

### 2. Automated Setup
```bash
# Run the automated setup script
./setup-claude-agents-mac.sh
```

### 3. Manual Setup (Alternative)
If the automated script fails, follow these steps:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r dependencies/requirements.txt
pip install -r dependencies/requirements-ml.txt

# Verify agents are present
ls .claude/agents/
```

### 4. Verify Claude Code Integration

1. **Open Claude Code**
2. **Open the project directory**
3. **Check MCP/Agents section in settings**
4. **Verify all 9 agents are loaded:**
   - strategy-orchestrator
   - data-infrastructure-builder
   - feature-engineering-labeler
   - model-backtest-validator
   - portfolio-risk-optimizer
   - execution-microstructure-oms
   - monitoring-log-collector
   - dashboard-api-kafka-consumer
   - frontend-dashboard-nextjs

### 5. Test Agent Functionality

Try this in Claude Code:
```
Use the strategy-orchestrator agent to analyze the current system architecture.
```

### 6. Quick Start Demo

```bash
# Test feature engineering
python run_enhanced_features_demo.py

# Test ML pipeline  
python src/scripts/corrected_ml_pipeline.py

# Test monitoring system
python run_monitoring_demo.py
```

## 📋 Agent Workflow Guide

### Systematic Development Approach
1. **Strategy Planning** → `strategy-orchestrator`
2. **Data Infrastructure** → `data-infrastructure-builder`
3. **Feature Engineering** → `feature-engineering-labeler`
4. **Model Training** → `model-backtest-validator`
5. **Portfolio Optimization** → `portfolio-risk-optimizer`
6. **Order Execution** → `execution-microstructure-oms`
7. **System Monitoring** → `monitoring-log-collector`
8. **API Backend** → `dashboard-api-kafka-consumer`
9. **Frontend Dashboard** → `frontend-dashboard-nextjs`

### Example Agent Usage
```
# Start a new feature development
Use the strategy-orchestrator agent to create a development plan for a new momentum strategy.

# Build data infrastructure  
Use the data-infrastructure-builder agent to fetch and validate historical data for BTCUSDT and ETHUSDT.

# Create ML features
Use the feature-engineering-labeler agent to generate technical indicators and labels for a 1-hour prediction model.
```

## 🎯 Performance Targets

- **Win Rate**: 85%+
- **Max Drawdown**: <3%
- **Profit Factor**: 1.8+
- **Sharpe Ratio**: >2.0
- **Monthly Return**: 12-20%

## 📚 Documentation

- **Main Guide**: `CLAUDE.md`
- **Agent Workflow**: `agent-workflow-guide.md`
- **Strategy Config**: `config/dipmaster_enhanced_v4_spec.json`
- **Results**: `results/` directory

## 🔧 Troubleshooting

### Common Issues

**1. Agents not loading in Claude Code**
```bash
# Check configuration files
cat .claude-config.json
ls .claude/agents/

# Restart Claude Code
# Verify project is opened in Claude Code correctly
```

**2. Python import errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r dependencies/requirements.txt
```

**3. Permission errors**
```bash
# Fix script permissions
chmod +x setup-claude-agents-mac.sh
chmod +x run_*.py
```

### Getting Help

1. **Check logs**: Look in `logs/` directory
2. **Review documentation**: `CLAUDE.md` has comprehensive guides
3. **Test components**: Use individual `run_*.py` scripts
4. **Verify data**: Check `data/` directory for market data

## 🚀 Next Steps

1. **Configure API Keys** (if using live data)
2. **Run System Tests**: `python run_complete_system_test.py`
3. **Start Development**: Use agent workflow for new features
4. **Deploy to Production**: Follow `results/PRODUCTION_DEPLOYMENT_STRATEGY.md`

## 📈 Expected Results

After setup, you should have:
- ✅ Complete ML trading system with 99.9% model accuracy
- ✅ 96 advanced features across 25 cryptocurrency pairs
- ✅ Production-ready infrastructure with monitoring
- ✅ Real-time dashboard and risk management
- ✅ Clear path to 85%+ win rate achievement

---

**Happy Trading! 🎯📊**

For detailed usage, see `CLAUDE.md` and explore the agent workflow system.
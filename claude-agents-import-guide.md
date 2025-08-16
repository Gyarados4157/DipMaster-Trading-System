# Claude Agents Import Guide for Mac

## 📋 Complete Agent Configuration Export

This project includes a complete set of **9 specialized agents** for quantitative trading system development.

### 🎯 Agent Manifest

| Agent Name | Purpose | Key Features |
|------------|---------|--------------|
| **strategy-orchestrator** | Strategy planning and workflow management | Project roadmaps, milestone tracking, dependency management |
| **data-infrastructure-builder** | Market data pipeline construction | Data fetching, quality validation, storage optimization |
| **feature-engineering-labeler** | ML feature creation and labeling | Technical indicators, cross-asset features, supervised labels |
| **model-backtest-validator** | ML model training and validation | LightGBM/XGBoost training, time-series backtesting |
| **portfolio-risk-optimizer** | Portfolio construction and risk management | Kelly criterion, correlation management, risk controls |
| **execution-microstructure-oms** | Order execution and market microstructure | Smart routing, order slicing, execution analytics |
| **monitoring-log-collector** | System monitoring and logging | Performance tracking, drift detection, operational reports |
| **dashboard-api-kafka-consumer** | Real-time API and event processing | FastAPI backend, Kafka integration, WebSocket streams |
| **frontend-dashboard-nextjs** | Trading dashboard and UI | React/Next.js interface, real-time monitoring, alerts |

## 🔧 Mac Import Instructions

### Method 1: Automatic (Recommended)
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Gyarados4157/DipMaster-Trading-System.git
   cd DipMaster-Trading-System
   ```

2. **Run setup script**:
   ```bash
   ./setup-claude-agents-mac.sh
   ```

3. **Open in Claude Code**:
   - Open Claude Code
   - Open the project directory
   - Agents will be automatically detected

### Method 2: Manual Import
If you need to import agents into an existing project:

1. **Copy agent files**:
   ```bash
   # Copy all agent configurations
   cp -r .claude/ your-project-directory/
   cp .claude-config.json your-project-directory/
   cp mcp-config.json your-project-directory/
   ```

2. **Verify import**:
   ```bash
   ls your-project-directory/.claude/agents/
   # Should show 9 .md files
   ```

3. **Restart Claude Code** and open your project

### Method 3: Individual Agent Import
To import specific agents only:

```bash
# Create agents directory
mkdir -p your-project/.claude/agents/

# Copy specific agents
cp .claude/agents/strategy-orchestrator.md your-project/.claude/agents/
cp .claude/agents/data-infrastructure-builder.md your-project/.claude/agents/
# ... copy other agents as needed

# Copy base configuration
cp .claude/settings.local.json your-project/.claude/
```

## 📁 Essential Files for Import

### Core Agent Files
```
.claude/
├── agents/
│   ├── strategy-orchestrator.md
│   ├── data-infrastructure-builder.md
│   ├── feature-engineering-labeler.md
│   ├── model-backtest-validator.md
│   ├── portfolio-risk-optimizer.md
│   ├── execution-microstructure-oms.md
│   ├── monitoring-log-collector.md
│   ├── dashboard-api-kafka-consumer.md
│   └── frontend-dashboard-nextjs.md
└── settings.local.json
```

### Configuration Files
```
.claude-config.json     # Claude Code project configuration
mcp-config.json        # MCP services configuration  
.env.claude           # Environment variables (optional)
```

## ✅ Verification Checklist

After import, verify the following:

### 1. Agent Detection
```bash
# Count agents
ls .claude/agents/*.md | wc -l
# Should return: 9
```

### 2. Claude Code Integration
- [ ] Open Claude Code
- [ ] Open project directory
- [ ] Check Settings → MCP/Agents
- [ ] All 9 agents should be listed as "Active"

### 3. Agent Functionality Test
Try this command in Claude Code:
```
List all available agents and their purposes.
```

Expected response should include all 9 agents with descriptions.

### 4. Workflow Test
```
Use the strategy-orchestrator agent to create a simple trading strategy development plan.
```

This should engage the strategy-orchestrator and return a structured development plan.

## 🛠️ Troubleshooting

### Issue: Agents not appearing in Claude Code

**Solution 1**: Restart Claude Code completely
```bash
# Force quit Claude Code and reopen
```

**Solution 2**: Check file permissions
```bash
chmod -R 644 .claude/
chmod 755 .claude/agents/
```

**Solution 3**: Verify file structure
```bash
find .claude -type f -name "*.md" | head -10
# Should show agent files
```

### Issue: Permission errors on Mac

```bash
# Fix permissions for the entire project
sudo chown -R $(whoami) .
chmod +x setup-claude-agents-mac.sh
```

### Issue: MCP configuration not loading

```bash
# Check MCP config syntax
cat mcp-config.json | python -m json.tool
# Should parse without errors
```

## 🚀 Agent Usage Examples

### Strategy Development Workflow
```
1. Use strategy-orchestrator to plan a new momentum strategy
2. Use data-infrastructure-builder to fetch 1-minute BTCUSDT data for 30 days
3. Use feature-engineering-labeler to create momentum indicators and labels
4. Use model-backtest-validator to train and validate an XGBoost model
5. Use portfolio-risk-optimizer to create risk-managed positions
6. Use execution-microstructure-oms to implement order execution
7. Use monitoring-log-collector to set up performance monitoring
8. Use dashboard-api-kafka-consumer to create real-time APIs
9. Use frontend-dashboard-nextjs to build a monitoring dashboard
```

### Research and Analysis
```
Use the model-backtest-validator agent to conduct a comprehensive study of RSI-based trading strategies across multiple cryptocurrencies.
```

### Risk Management
```
Use the portfolio-risk-optimizer agent to analyze correlation risks in my current 5-symbol cryptocurrency portfolio.
```

## 📊 Performance Expectations

With proper agent configuration, you should achieve:

- **Development Speed**: 5-10x faster than manual coding
- **Code Quality**: Institutional-grade standards
- **Error Reduction**: 80%+ fewer integration issues
- **Documentation**: Comprehensive auto-generated docs
- **Testing**: Built-in validation and verification

## 🎯 Success Indicators

You've successfully imported agents when:

1. ✅ All 9 agents appear in Claude Code settings
2. ✅ Agent queries return specialized responses
3. ✅ Workflow commands execute properly
4. ✅ No permission or file access errors
5. ✅ Agents coordinate across development tasks

## 🔗 Additional Resources

- **Main Documentation**: `CLAUDE.md`
- **Agent Workflow Guide**: `agent-workflow-guide.md`
- **Setup Script**: `setup-claude-agents-mac.sh`
- **Configuration Export**: `claude-agents-export.json`
- **Mac Setup Guide**: `SETUP-MAC.md`

---

**Ready to build world-class trading systems! 🎯🚀**

The agent configuration provides everything needed for professional quantitative trading system development.
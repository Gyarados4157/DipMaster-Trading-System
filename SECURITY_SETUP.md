# 🔒 DipMaster Trading System - Security Setup Guide

## 📋 Overview

This guide walks you through setting up the enterprise-grade security system for DipMaster Trading System v4.0. The security enhancements include:

- **Encrypted API Key Storage**: AES-256-GCM encryption with PBKDF2 key derivation
- **Role-Based Access Control (RBAC)**: Fine-grained permission management
- **Comprehensive Audit Logging**: Tamper-evident security logs
- **Session Management**: Secure user sessions with timeout
- **Real-time Security Monitoring**: Automated threat detection

## 🚀 Quick Start

### 1. Run the Security Setup Script

```bash
# Run interactive setup (recommended for first-time setup)
python setup_security.py

# Non-interactive setup (for automation)
export DIPMASTER_MASTER_PASSWORD="your-secure-master-password"
export DIPMASTER_ADMIN_PASSWORD="your-admin-password"
python setup_security.py --non-interactive
```

### 2. Start the Secure System

```bash
# Interactive mode with authentication
python main_secure.py --config config/dipmaster_secure.json

# Automated mode (set environment variables)
export DIPMASTER_USER_ID="admin"
export DIPMASTER_USER_PASSWORD="your-password"
python main_secure.py --config config/dipmaster_secure.json --no-interactive
```

## 🔧 Detailed Setup Process

### Step 1: Master Password Setup

The master password encrypts all API keys and sensitive data. You can either:

**Option A: Generate Secure Password (Recommended)**
```bash
python -c "from src.security.crypto_manager import generate_master_key; print(generate_master_key())"
```

**Option B: Use Custom Password**
- Must be at least 16 characters
- Include uppercase, lowercase, numbers, and symbols
- Store securely in password manager

### Step 2: Environment Configuration

Create or update your `.env` file:

```bash
# Master password for encryption
DIPMASTER_MASTER_PASSWORD=your_secure_master_password_here

# Default admin credentials (optional)
DIPMASTER_USER_ID=admin
DIPMASTER_USER_PASSWORD=your_admin_password

# API keys (legacy support - prefer encrypted storage)
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# Optional: Slack notifications
DIPMASTER_SLACK_WEBHOOK=https://hooks.slack.com/services/...
```

### Step 3: API Key Management

#### Store New API Keys
```bash
# Interactive mode
python src/tools/key_management_tool.py interactive

# Command line mode
python src/tools/key_management_tool.py store \
  --key-id binance-production \
  --exchange binance \
  --description "Production trading API key"
```

#### List Stored Keys
```bash
python src/tools/key_management_tool.py list
```

#### Retrieve API Key
```bash
python src/tools/key_management_tool.py get --key-id binance-production
```

#### Rotate API Key
```bash
python src/tools/key_management_tool.py rotate --key-id binance-production
```

### Step 4: User Management

#### Default Users Created
- **admin**: Full system access (Role: ADMIN)
- **trader**: Trading operations only (Role: TRADER)

#### Create Additional Users
```python
from src.security.access_control import AccessController, Role

controller = AccessController()
success = controller.create_user(
    user_id='analyst',
    password='secure_password',
    roles=[Role.ANALYST],
    enabled=True
)
```

## 🛡️ Security Features

### 1. Encrypted API Key Storage

- **Algorithm**: AES-256-GCM with PBKDF2 key derivation
- **Iterations**: 480,000 (NIST recommended)
- **Salt**: 32-byte random salt per key
- **Storage**: Encrypted JSON with integrity checksums

### 2. Role-Based Access Control

| Role | Permissions |
|------|-------------|
| **ADMIN** | Full system access, user management, security admin |
| **TRADER** | Execute trades, view positions, basic dashboard |
| **ANALYST** | Read-only access to trading data and reports |
| **OPERATOR** | System monitoring, configuration management |
| **SECURITY_OFFICER** | Security admin, audit logs, key rotation |

### 3. Audit Logging

All security-sensitive operations are logged:

```bash
# View security logs
tail -f logs/security/security_audit_*.jsonl

# Generate audit report
python main_secure.py --security-audit

# Check log integrity
python src/tools/key_management_tool.py health
```

### 4. Session Management

- **Timeout**: 8 hours default (configurable)
- **IP Tracking**: Client IP logging for audit
- **Concurrent Sessions**: Multiple sessions per user supported
- **Automatic Cleanup**: Expired sessions removed automatically

## 🔍 System Monitoring

### Health Checks

```bash
# System health check
python src/tools/key_management_tool.py health

# Validate security setup
python setup_security.py --validate-only
```

### Security Metrics

The system tracks:
- Failed authentication attempts
- API key access patterns
- Session activity
- Configuration changes
- Trading operations
- System errors and anomalies

### Alerting

Configure real-time alerts in `config/dipmaster_secure.json`:

```json
{
  "notifications": {
    "security_alerts": true,
    "slack_webhook": "${DIPMASTER_SLACK_WEBHOOK}",
    "email_enabled": false
  }
}
```

## 🚨 Security Best Practices

### 1. Environment Security
- Run on dedicated secure server
- Use VPN or private network access
- Enable firewall with minimal open ports
- Keep system updated

### 2. Key Management
- Rotate API keys monthly
- Use separate keys for different environments
- Monitor key usage patterns
- Revoke unused keys immediately

### 3. Access Control
- Use strong unique passwords
- Enable IP restrictions when possible
- Regular access review and cleanup
- Monitor failed login attempts

### 4. Monitoring
- Review audit logs daily
- Set up automated alerts
- Monitor system resource usage
- Regular security assessments

## 🔧 Configuration Options

### Security Configuration (`config/dipmaster_secure.json`)

```json
{
  "security": {
    "encrypted_api_keys": true,
    "access_control_enabled": true,
    "audit_logging_enabled": true,
    "session_timeout_minutes": 480,
    "max_failed_attempts": 5,
    "lockout_duration_minutes": 30,
    "require_strong_passwords": true,
    "enable_ip_restrictions": false,
    "real_time_security_alerts": true
  }
}
```

### Access Control Configuration

```json
{
  "users": {
    "admin": {
      "roles": ["admin"],
      "enabled": true,
      "ip_restrictions": ["192.168.1.0/24"]
    }
  },
  "ip_whitelist": ["192.168.1.0/24"],
  "ip_blacklist": []
}
```

## 🆘 Troubleshooting

### Common Issues

#### 1. Authentication Fails
```bash
# Check user exists and is enabled
python -c "
from src.security.access_control import AccessController
ac = AccessController()
users = ac.users
print('Users:', list(users.keys()))
print('Admin enabled:', users.get('admin', {}).get('enabled'))
"
```

#### 2. API Key Not Found
```bash
# List available keys
python src/tools/key_management_tool.py list

# Check key manager health
python src/tools/key_management_tool.py health
```

#### 3. Configuration Errors
```bash
# Validate configuration
python -c "
from src.security.secure_config_loader import SecureConfigLoader
loader = SecureConfigLoader()
config = loader.load_config('config/dipmaster_secure.json')
print('✅ Configuration valid')
"
```

#### 4. Permission Denied
```bash
# Check file permissions
ls -la config/
chmod 600 config/encrypted_keys.json
chmod 600 config/access_control.json
```

### Emergency Recovery

#### Reset Master Password
1. Stop the system
2. Delete `config/encrypted_keys.json`
3. Run setup script again
4. Re-enter API keys

#### Reset User Passwords
```bash
# Delete access control file to recreate defaults
rm config/access_control.json
python setup_security.py --validate-only
```

#### Access Logs for Debugging
```bash
# System logs
tail -f logs/dipmaster_secure_*.log

# Security audit logs
tail -f logs/security/security_audit_*.jsonl

# Key operation logs
grep "KEY_" logs/security/security_audit_*.jsonl
```

## 📞 Support

For security-related issues:

1. **Check audit logs** for detailed error information
2. **Review permissions** on configuration files
3. **Validate environment** variables are set correctly
4. **Test components** individually using the tools provided
5. **Consult CLAUDE.md** for additional system documentation

## 🔄 Migration from v3.0

If migrating from the previous version:

1. **Backup existing configuration**:
   ```bash
   cp config/dipmaster_v3_optimized.json config/dipmaster_v3_backup.json
   ```

2. **Run security setup**:
   ```bash
   python setup_security.py
   ```

3. **Migrate API keys**:
   ```bash
   # Extract from old config and store securely
   python src/tools/key_management_tool.py store --key-id binance-main
   ```

4. **Update launch scripts**:
   ```bash
   # Old: python main.py --config config/dipmaster_v3_optimized.json
   # New: python main_secure.py --config config/dipmaster_secure.json
   ```

---

**🔒 Remember**: Security is only as strong as your operational practices. Regularly review logs, rotate keys, and keep the system updated.

**📋 Next Steps**: After setup, read the [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) for production deployment best practices.
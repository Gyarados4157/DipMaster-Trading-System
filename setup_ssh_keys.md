# SSH密钥配置指南

## 你的公钥内容
请将以下公钥内容复制到服务器：

```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDHiJwgf21DN2pl4f91Mf/QPW6IwZHL3VPbb/uXKnW/IA9c05QV0F1HzqikpCVjIukbhYpr1lfyxGNiCIRSsGZgfX4aaXzKvG+QIq+Tj9Hf0X6lxgZBBHvTaOBKQDDQdGn91zyhedeuyz5D5h3uiSZQSTo8quXPb7GQ/7nLNnZZCQMjdgF5s8kmcaEkWUJoSKOKkceI530W2OTnISJAeBM6Fyl7ubb100OKHTdq43R/o4WqlXlu7TsJ/qFFum5lvz4NH5wU+cf41GQYUhCXxZESc8sbGnFqsFVfpdoTD3Y2ZubNrtK+scvJyaZa2ueTuobmD+Q3G8sgGevo1/CD45oyegQifIRoObtVXQu6yjYhdgSG53QkfTv8fM5jY+2v/VE5KxmQPuGoM/eS2tDMCaj3SNU7skgZgjFZuiijtUnsnPOiH3V1LmqqanadM0rqjp8uYPUeqnQ0fj6FkRewwFc9N/ZUw+Sdu7ejeuGUYW9sKfHFA6wIHwKe81EtFrxkvVyHe88mQ1srN66NFEO7lCUajfReoEmv+ZzAAKmHxPI8LmSByxdwR1qUzKwLDwuJsrYYGGJwa3eKIFe+FQQHpEMpSooHauB47YjW9CdhBnOwfAfRTtNKECPp4iZQphNP6DexaILrGoAOfkweZkDuGTDSrGbJ6gX8Kq0kUWndDs46CQ== zhangxuanyang@Xuanyangs-Macbook.local
```

## 在服务器上执行的命令

### 1. 通过阿里云VNC登录服务器后执行：

```bash
# 创建SSH目录和文件
mkdir -p /root/.ssh
chmod 700 /root/.ssh
touch /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# 添加公钥（一行命令）
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDHiJwgf21DN2pl4f91Mf/QPW6IwZHL3VPbb/uXKnW/IA9c05QV0F1HzqikpCVjIukbhYpr1lfyxGNiCIRSsGZgfX4aaXzKvG+QIq+Tj9Hf0X6lxgZBBHvTaOBKQDDQdGn91zyhedeuyz5D5h3uiSZQSTo8quXPb7GQ/7nLNnZZCQMjdgF5s8kmcaEkWUJoSKOKkceI530W2OTnISJAeBM6Fyl7ubb100OKHTdq43R/o4WqlXlu7TsJ/qFFum5lvz4NH5wU+cf41GQYUhCXxZESc8sbGnFqsFVfpdoTD3Y2ZubNrtK+scvJyaZa2ueTuobmD+Q3G8sgGevo1/CD45oyegQifIRoObtVXQu6yjYhdgSG53QkfTv8fM5jY+2v/VE5KxmQPuGoM/eS2tDMCaj3SNU7skgZgjFZuiijtUnsnPOiH3V1LmqqanadM0rqjp8uYPUeqnQ0fj6FkRewwFc9N/ZUw+Sdu7ejeuGUYW9sKfHFA6wIHwKe81EtFrxkvVyHe88mQ1srN66NFEO7lCUajfReoEmv+ZzAAKmHxPI8LmSByxdwR1qUzKwLDwuJsrYYGGJwa3eKIFe+FQQHpEMpSooHauB47YjW9CdhBnOwfAfRTtNKECPp4iZQphNP6DexaILrGoAOfkweZkDuGTDSrGbJ6gX8Kq0kUWndDs46CQ== zhangxuanyang@Xuanyangs-Macbook.local" >> /root/.ssh/authorized_keys

# 确保SSH配置允许密钥认证
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/#AuthorizedKeysFile/AuthorizedKeysFile/' /etc/ssh/sshd_config

# 重启SSH服务
systemctl restart sshd

# 验证配置
ls -la /root/.ssh/
cat /root/.ssh/authorized_keys
systemctl status sshd
```

### 2. 或者使用编辑器手动添加：

```bash
# 使用nano/vi编辑器
nano /root/.ssh/authorized_keys
# 或
vi /root/.ssh/authorized_keys

# 将公钥内容粘贴进去，保存退出
```

## 配置完成后测试

在本地Mac上测试连接：

```bash
ssh dipmaster-aliyun
# 应该能直接登录，不需要密码
```

## 如果仍然无法连接

检查SSH配置：

```bash
# 在服务器上检查SSH配置
cat /etc/ssh/sshd_config | grep -E "(PubkeyAuthentication|AuthorizedKeysFile)"

# 检查SELinux状态（可能阻止SSH密钥）
getenforce
# 如果是Enforcing，临时禁用：
setenforce 0

# 检查文件权限
ls -la /root/.ssh/
```

## 备用方案：重置root密码

如果密钥认证仍有问题，可以在阿里云控制台：
1. ECS实例 → 更多 → 密码/密钥 → 重置实例密码
2. 设置新密码后重启实例
3. 用密码登录后再配置密钥
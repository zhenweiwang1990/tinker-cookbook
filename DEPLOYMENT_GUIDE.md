# CUA RL 新服务器部署指南

完整的从零开始部署流程，适用于新服务器或新同事。

## 前置要求

- Ubuntu 20.04+ 或其他 Linux 发行版
- Docker 和 Docker Compose
- Python 3.10+ （可选，用于本地开发）
- Git

## 一键部署（最简单）

```bash
# 1. Clone 代码
git clone https://github.com/your-org/tinker-cookbook.git
cd tinker-cookbook

# 2. 安装 Docker（如果没有）
cd training-monitor
./install-docker.sh  # 自动安装 Docker 和 Docker Compose

# 3. 启动所有服务（PostgreSQL + Web UI + 自动初始化数据库）
make start

# 4. 设置环境变量
export GBOX_API_KEY=your_gbox_api_key
export TINKER_API_KEY=your_tinker_api_key

# 5. 开始训练
cd ..
./train.sh
```

就这么简单！🎉

---

## 详细部署流程

### 步骤 1：系统准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基本工具
sudo apt install -y git curl wget build-essential

# 安装 Docker（如果未安装）
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 Docker Compose
sudo apt install -y docker-compose

# 添加当前用户到 docker 组（避免每次 sudo）
sudo usermod -aG docker $USER
newgrp docker  # 或者重新登录
```

### 步骤 2：Clone 代码

```bash
cd ~
git clone https://github.com/your-org/tinker-cookbook.git
cd tinker-cookbook
```

### 步骤 3：安装 Python 环境（可选但推荐）

```bash
# 安装 uv（现代 Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py

# 创建虚拟环境
cd tinker-cookbook
uv sync  # 或 python3 -m venv .venv && source .venv/bin/activate && pip install -e .
```

### 步骤 4：启动 Training Monitor

这一步会自动启动 PostgreSQL 并初始化数据库：

```bash
cd training-monitor
make start
```

**这个命令会自动：**
1. ✅ 启动 PostgreSQL（端口 5433）
2. ✅ 启动 Web UI（端口 3001）
3. ✅ 运行数据库 migrations（自动创建所有表）
4. ✅ 等待服务健康检查

验证启动成功：

```bash
# 检查服务状态
docker-compose ps

# 应该看到：
# training-monitor-postgres   Up (healthy)
# training-monitor            Up (healthy)

# 测试 API
curl http://localhost:3001/api/trainings
# 应该返回 []（空数组）

# 访问 Web UI
# 浏览器打开: http://localhost:3001
```

### 步骤 5：配置环境变量

```bash
cd ~/tinker-cookbook

# 设置 API keys
export GBOX_API_KEY=your_gbox_api_key
export TINKER_API_KEY=your_tinker_api_key

# 可选：保存到 ~/.bashrc 或 ~/.zshrc
echo 'export GBOX_API_KEY=your_gbox_api_key' >> ~/.bashrc
echo 'export TINKER_API_KEY=your_tinker_api_key' >> ~/.bashrc
source ~/.bashrc
```

### 步骤 6：运行第一次训练

```bash
# 确保在 tinker-cookbook 根目录
cd ~/tinker-cookbook

# 查看训练选项
./train.sh --help

# 开始训练（使用默认配置）
./train.sh
```

**训练脚本会自动：**
1. ✅ 连接到 PostgreSQL（127.0.0.1:5433）
2. ✅ 检查并运行必要的 migrations（如果有更新）
3. ✅ 创建训练记录
4. ✅ 开始 RL 训练

### 步骤 7：监控训练

```bash
# 查看训练日志
tail -f logs/logs.log

# 或访问 Web UI
# http://localhost:3001

# 查看数据库状态
cd training-monitor
make logs
```

---

## 给同事的快速指南

如果服务器已经部署好，同事只需要：

```bash
# 1. SSH 到服务器
ssh ubuntu@your-server-ip

# 2. 进入项目目录
cd ~/tinker-cookbook

# 3. 拉取最新代码
git pull

# 4. 确保服务运行
cd training-monitor
make start
cd ..

# 5. 开始你的实验
./train.sh --model Qwen/Qwen2.5-3B-Instruct --lr 2e-5
```

**不需要手动操作数据库！** 一切都是自动的。

---

## 常见场景

### 场景 1：首次部署

```bash
cd training-monitor
make start  # 自动初始化所有东西
```

### 场景 2：代码更新后重新部署

```bash
git pull
cd training-monitor
make restart  # 重启服务，自动运行新的 migrations
```

### 场景 3：数据库需要更新

```bash
# 方法 1：重启 training-monitor（会自动运行 migrations）
cd training-monitor
make restart

# 方法 2：手动运行 migrations
make init-db

# 方法 3：使用 Python 脚本
cd ../tinker_cookbook/recipes/cua_rl
uv run python migrate_database.py
```

### 场景 4：清空数据库重新开始

```bash
# 停止服务
cd training-monitor
make stop

# 删除 PostgreSQL 数据卷
docker-compose down -v

# 重新启动（会创建新数据库并初始化）
make start
```

---

## 故障排除

### 问题 1：端口冲突

```bash
# 如果 5433 被占用
./scripts/docker-start.sh --port 3002
```

### 问题 2：Docker 权限错误

```bash
# 添加用户到 docker 组
sudo usermod -aG docker $USER
newgrp docker
```

### 问题 3：数据库连接失败

```bash
# 检查 PostgreSQL 是否运行
docker-compose ps

# 查看日志
docker-compose logs postgres

# 重启
make restart
```

### 问题 4：表结构错误

```bash
# 手动运行 migrations
cd training-monitor
make init-db
```

### 问题 5：磁盘空间不足

```bash
# 清理 Docker 缓存
docker system prune -a

# 清理旧的训练日志
rm -rf logs/old_training_*
```

---

## 架构说明

```
tinker-cookbook/
├── training-monitor/          # Web UI + PostgreSQL
│   ├── docker-compose.yml    # 定义两个服务
│   ├── scripts/
│   │   ├── docker-start.sh   # 启动脚本（自动初始化）
│   │   └── init-database.sh  # 手动初始化脚本
│   └── Makefile              # 便捷命令
├── tinker_cookbook/recipes/cua_rl/
│   ├── database/
│   │   ├── database.py       # 数据库连接（自动运行 migrations）
│   │   └── database_models.py # 表结构定义
│   ├── alembic/              # Migration 文件
│   │   └── versions/         # 各个版本的 migration
│   └── migrate_database.py   # 手动 migration 工具
├── train.sh                   # 训练脚本
└── benchmark.sh              # 评估脚本
```

### 数据库自动化流程

1. **docker-start.sh 启动时**：
   - 启动 PostgreSQL
   - 启动 Web UI
   - 调用 `migrate_database.py` 初始化表结构

2. **train.sh / benchmark.sh 运行时**：
   - `database.py` 的 `init_database()` 自动检查版本
   - 如果有新 migration，自动运行
   - 如果失败，**中断程序**并提示

3. **手动管理**（很少需要）：
   - `make init-db` - 手动运行 migrations
   - `migrate_database.py --status` - 查看状态
   - `migrate_database.py --rebuild` - 重建数据库

---

## 最佳实践

1. **✅ 推荐**：总是先启动 `training-monitor`，然后运行训练
2. **✅ 推荐**：使用 `make start` 而不是直接 `docker-compose up`
3. **✅ 推荐**：定期 `git pull` 并 `make restart` 获取最新代码
4. **✅ 推荐**：使用 Web UI (http://localhost:3001) 监控训练
5. **⚠️ 避免**：手动修改数据库表结构
6. **⚠️ 避免**：同时运行多个 training-monitor 实例

---

## 总结

**对于新服务器/新同事**，只需要记住：

```bash
cd training-monitor && make start
```

**就完成了所有数据库初始化！** 🎉

然后直接运行 `./train.sh` 开始训练。一切都是自动的。


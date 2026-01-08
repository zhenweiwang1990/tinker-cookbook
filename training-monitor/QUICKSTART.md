# 快速开始

## 一键启动（推荐）

```bash
cd training-monitor
make start
# 或者
./scripts/docker-start.sh
```

这会：
1. 启动 PostgreSQL（端口 5433）
2. 启动 Training Monitor（端口 3001）
3. **自动初始化数据库表结构**

访问 http://localhost:3001

> **注意**：首次启动会自动运行数据库 migrations，确保表结构是最新的。

## 新服务器部署流程

在新服务器上首次部署时：

```bash
# 1. Clone 代码
git clone <repo-url>
cd tinker-cookbook

# 2. 安装依赖（如果使用 uv）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 3. 启动 training-monitor（包含 PostgreSQL + 自动初始化）
cd training-monitor
make start

# 4. 验证
curl http://localhost:3001/api/trainings

# 5. 开始训练
cd ..
export GBOX_API_KEY=your_key
export TINKER_API_KEY=your_key
./train.sh
```

**就这么简单！** 数据库会在第一次启动时自动初始化。

## 其他启动方式

### Docker Compose

```bash
docker-compose up -d
# 然后手动初始化数据库（如果 docker-start.sh 没有自动初始化）
make init-db
```

### Make 命令

```bash
make start      # 启动（自动初始化数据库）
make stop       # 停止
make logs       # 查看日志
make restart    # 重启
make build      # 构建镜像
make init-db    # 手动初始化/更新数据库
```

## 数据库管理

### 检查数据库状态

```bash
# 查看当前 migration 版本
cd ../tinker_cookbook/recipes/cua_rl
uv run python migrate_database.py --status
```

### 手动运行 migrations

通常不需要，但如果遇到问题：

```bash
# 方法 1：使用 Make
cd training-monitor
make init-db

# 方法 2：使用 Python 脚本
cd ../tinker_cookbook/recipes/cua_rl
uv run python migrate_database.py
```

## 常用命令

```bash
# 启动（自动初始化）
./scripts/docker-start.sh

# 使用不同端口
./scripts/docker-start.sh --port 3002

# 跳过构建（如果镜像已存在）
./scripts/docker-start.sh --no-build

# 查看日志
./scripts/docker-logs.sh

# 停止
./scripts/docker-stop.sh

# 初始化数据库
./scripts/init-database.sh
```

## 故障排除

### 数据库连接失败

```bash
# 检查 PostgreSQL 是否运行
docker-compose ps

# 查看 PostgreSQL 日志
docker-compose logs postgres
```

### 表结构错误（"column does not exist"）

```bash
# 运行 migrations
make init-db
```

### 端口冲突

```bash
# 使用不同端口
./scripts/docker-start.sh --port 3002
```

更多详细信息，请查看 [DOCKER.md](./DOCKER.md)

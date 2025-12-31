# Docker 部署指南

## 快速开始

### 方式 1: 使用启动脚本（推荐）

```bash
cd training-monitor
./scripts/docker-start.sh
```

脚本会自动：
- 启动 PostgreSQL 容器
- 构建 Docker 镜像
- 启动训练监控容器
- 映射端口 3000

### 方式 2: 使用 Docker Compose

```bash
cd training-monitor
docker-compose up -d
```

### 方式 3: 手动 Docker 命令

首先启动 PostgreSQL：

```bash
docker run -d \
  --name training-monitor-postgres \
  -e POSTGRES_DB=training_db \
  -e POSTGRES_USER=training_user \
  -e POSTGRES_PASSWORD=training_password \
  -p 5432:5432 \
  postgres:16-alpine
```

然后构建并运行监控容器：

```bash
# 构建镜像
docker build -t cua-rl-training-monitor:latest .

# 运行容器
docker run -d \
  --name training-monitor \
  -p 3000:3000 \
  --link training-monitor-postgres:postgres \
  -e DATABASE_URL=postgresql://training_user:training_password@postgres:5432/training_db \
  cua-rl-training-monitor:latest
```

## 启动脚本选项

```bash
./scripts/docker-start.sh [OPTIONS]

Options:
  --port PORT       Port to expose (default: 3000)
  --no-build        Skip building the image
  --help            Show help message
```

### 示例

```bash
# 使用默认设置
./scripts/docker-start.sh

# 使用不同端口
./scripts/docker-start.sh --port 3001

# 跳过构建（如果镜像已存在）
./scripts/docker-start.sh --no-build
```

## 管理容器

### 查看日志

```bash
./scripts/docker-logs.sh
# 或
docker logs -f training-monitor
```

### 停止容器

```bash
./scripts/docker-stop.sh
# 或
docker stop training-monitor
docker rm training-monitor
```

### 重启容器

```bash
docker restart training-monitor
```

### 查看容器状态

```bash
docker ps | grep training-monitor
```

## PostgreSQL 配置

### 默认配置

Docker Compose 使用以下默认配置：

- **数据库名**: `training_db`
- **用户名**: `training_user`
- **密码**: `training_password`
- **主机**: `postgres` (容器内) / `localhost` (宿主机)
- **端口**: `5432`

### 自定义配置

如果需要自定义配置，编辑 `docker-compose.yml`：

```yaml
services:
  postgres:
    environment:
      - POSTGRES_DB=your_db_name
      - POSTGRES_USER=your_user
      - POSTGRES_PASSWORD=your_password
  training-monitor:
    environment:
      - DATABASE_URL=postgresql://your_user:your_password@postgres:5432/your_db_name
```

### 连接外部 PostgreSQL

如果要连接外部 PostgreSQL 服务器，修改 `docker-compose.yml`：

```yaml
services:
  training-monitor:
    environment:
      - DATABASE_URL=postgresql://user:password@host:5432/dbname
    # 移除 depends_on postgres
```

## 端口配置

默认端口是 3000。如果端口被占用，可以修改：

```bash
./scripts/docker-start.sh --port 3001
```

或在 docker-compose.yml 中修改：

```yaml
ports:
  - "3001:3000"
```

## 开发模式

如果需要开发模式（热重载），可以挂载源代码：

```bash
# 确保 PostgreSQL 已启动
docker-compose up -d postgres

# 运行开发容器
docker run -d \
  --name training-monitor-dev \
  -p 3000:3000 \
  -v $(pwd):/app \
  -v /app/node_modules \
  --link training-monitor-postgres:postgres \
  -e DATABASE_URL=postgresql://training_user:training_password@postgres:5432/training_db \
  -e NODE_ENV=development \
  --entrypoint npm \
  cua-rl-training-monitor:latest \
  run dev
```

## 故障排除

### 容器无法启动

1. 检查 PostgreSQL 容器是否正常运行：`docker ps | grep postgres`
2. 检查数据库连接配置是否正确
3. 查看日志：`docker logs training-monitor`
4. 查看 PostgreSQL 日志：`docker logs training-monitor-postgres`

### 数据库连接失败

检查 PostgreSQL 是否可访问：

```bash
# 检查 PostgreSQL 容器状态
docker ps | grep postgres

# 测试连接
docker exec training-monitor-postgres pg_isready -U training_user -d training_db

# 查看 PostgreSQL 日志
docker logs training-monitor-postgres
```

### 表不存在

数据库表由训练代码（cua_rl）自动创建。确保：

1. 训练代码已运行并初始化数据库
2. 训练代码和监控使用相同的 PostgreSQL 数据库
3. 检查表是否存在：`docker exec -it training-monitor-postgres psql -U training_user -d training_db -c "\dt"`

### 端口冲突

使用不同的端口：

```bash
./scripts/docker-start.sh --port 3001
```

### 重新构建

如果代码有更新，需要重新构建：

```bash
docker build -t cua-rl-training-monitor:latest .
./scripts/docker-stop.sh
./scripts/docker-start.sh --no-build
```

## 生产部署

### 使用 Docker Compose（推荐）

```bash
docker-compose up -d
```

### 使用 systemd 服务

创建 `/etc/systemd/system/training-monitor.service`:

```ini
[Unit]
Description=CUA RL Training Monitor
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/path/to/training-monitor
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

然后：

```bash
sudo systemctl enable training-monitor
sudo systemctl start training-monitor
```


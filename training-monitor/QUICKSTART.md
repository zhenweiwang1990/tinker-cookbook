# 快速开始

## 一键启动（推荐）

```bash
cd training-monitor
./scripts/docker-start.sh
```

访问 http://localhost:3000

## 其他启动方式

### Docker Compose

```bash
docker-compose up -d
```

### Make 命令

```bash
make start      # 启动
make stop       # 停止
make logs       # 查看日志
make restart    # 重启
make build      # 构建镜像
```

### 手动 Docker

```bash
# 构建
docker build -t cua-rl-training-monitor:latest .

# 运行
docker run -d \
  --name training-monitor \
  -p 3000:3000 \
  -v $(pwd)/../logs:/data:ro \
  -e TRAINING_DB_PATH=/data/training.db \
  cua-rl-training-monitor:latest
```

## 常用命令

```bash
# 启动（自动查找数据库）
./scripts/docker-start.sh

# 指定数据库路径
./scripts/docker-start.sh --db-path /path/to/training.db

# 使用不同端口
./scripts/docker-start.sh --port 3001

# 跳过构建（如果镜像已存在）
./scripts/docker-start.sh --no-build

# 查看日志
./scripts/docker-logs.sh

# 停止
./scripts/docker-stop.sh
```

## 故障排除

如果遇到问题，请查看 [DOCKER.md](./DOCKER.md) 获取详细说明。

# CUA RL Training Monitor

实时监控 CUA RL 训练过程的 Next.js Web 应用。

## 功能特性

- 📊 **实时监控**: 每2秒自动刷新数据
- 📈 **三栏布局**: 
  - 左侧：训练历史列表
  - 中间：Timeline（baseline/step/eval）
  - 右侧：详细信息面板
- 🔍 **详细展示**: 
  - Rollout 列表（可折叠）
  - Turn 详情（可展开）
  - Actions 和 Observations
- 🎨 **现代化 UI**: 清晰的视觉层次和交互

## 安装

```bash
cd training-monitor
npm install
```

## 配置

### PostgreSQL 配置（Docker 模式）

使用 Docker Compose 时，PostgreSQL 会自动启动。数据库连接信息在 `docker-compose.yml` 中配置：

- 数据库名: `training_db`
- 用户名: `training_user`
- 密码: `training_password`
- 端口: `5432`

### 本地开发模式

如果本地运行（非 Docker），需要设置 PostgreSQL 连接信息：

```bash
export DATABASE_URL=postgresql://training_user:training_password@localhost:5432/training_db
```

或者在 `.env.local` 文件中：

```
DATABASE_URL=postgresql://training_user:training_password@localhost:5432/training_db
```

## 运行

### 方式 1: Docker（推荐）

一键启动：

```bash
./scripts/docker-start.sh
```

访问 http://localhost:3000

详细说明请参考 [DOCKER.md](./DOCKER.md)

### 方式 2: 本地开发

开发模式：

```bash
npm run dev
```

访问 http://localhost:3000

生产模式：

```bash
npm run build
npm start
```

## 使用说明

1. **选择训练**: 在左侧栏点击一个训练记录
2. **查看 Timeline**: 中间栏显示该训练的所有 baseline、step 和 eval
3. **查看详情**: 点击 timeline 中的项目，右侧显示详细信息
4. **查看 Rollout**: 在详情面板中展开 rollout 列表，点击 "View Full Details" 查看完整信息
5. **查看 Turn**: 在 rollout 详情中展开 turn 列表，查看每个 turn 的 actions 和 observations

## Docker 部署

### 快速启动

```bash
# 使用启动脚本（推荐）
./scripts/docker-start.sh

# 或使用 Make
make start

# 或使用 Docker Compose
docker-compose up -d
```

详细说明请参考 [DOCKER.md](./DOCKER.md)

## 技术栈

- **Next.js 14**: React 框架
- **TypeScript**: 类型安全
- **PostgreSQL**: 数据库（通过 `pg` 库）
- **CSS Modules**: 样式管理

## 注意事项

- 使用 Docker Compose 时，PostgreSQL 容器会自动启动
- 数据库表结构由训练代码（cua_rl）自动创建
- 确保训练代码和监控使用相同的 PostgreSQL 数据库
- 自动刷新间隔为 2 秒，可在组件中调整


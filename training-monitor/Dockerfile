FROM node:20-alpine AS base

# Install build dependencies (no longer needed for PostgreSQL)
RUN apk add --no-cache python3 make g++

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install all dependencies (needed for build)
# Use npm install if package-lock.json doesn't exist, otherwise use npm ci
RUN if [ -f package-lock.json ]; then npm ci && npm cache clean --force; else npm install && npm cache clean --force; fi

# Copy application files
COPY . .

# Ensure public directory exists (Next.js may need it)
RUN mkdir -p ./public

# Build the application
RUN npm run build

# Production stage
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# No runtime dependencies needed for PostgreSQL (uses pg npm package)

# Copy built application from base
COPY --from=base /app/package.json ./
COPY --from=base /app/package-lock.json* ./
COPY --from=base /app/node_modules ./node_modules
COPY --from=base /app/.next ./.next
# Copy public directory (will be empty if no files, but directory exists)
COPY --from=base /app/public ./public
COPY --from=base /app/lib ./lib
COPY --from=base /app/app ./app

# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs && \
    chown -R nextjs:nodejs /app

USER nextjs

# Expose port
EXPOSE 3000

# Set environment variables for PostgreSQL connection
ENV DATABASE_URL=postgresql://training_user:training_password@postgres:5432/training_db
ENV POSTGRES_HOST=postgres
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB=training_db
ENV POSTGRES_USER=training_user
ENV POSTGRES_PASSWORD=training_password
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Start the application
CMD ["npm", "start"]

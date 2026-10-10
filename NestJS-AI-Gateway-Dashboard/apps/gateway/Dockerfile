# Build context: repository root (the gateway depends on packages/shared).
#   docker build -f apps/gateway/Dockerfile -t ai-gateway .

FROM node:24-alpine AS build
WORKDIR /repo
# Manifests first so dependency installation is cached across source edits.
COPY package.json package-lock.json ./
COPY packages/shared/package.json packages/shared/
COPY packages/sdk/package.json packages/sdk/
COPY apps/gateway/package.json apps/gateway/
COPY apps/dashboard/package.json apps/dashboard/
RUN npm ci
COPY tsconfig.base.json ./
COPY packages/shared packages/shared
COPY apps/gateway apps/gateway
RUN npm run build -w @aigw/shared \
 && npm run prisma:generate -w @aigw/gateway \
 && npm run build -w @aigw/gateway

FROM node:24-alpine AS runtime
# Prisma's query engine links against OpenSSL.
RUN apk add --no-cache openssl
ENV NODE_ENV=production
WORKDIR /app
COPY package.json package-lock.json ./
COPY packages/shared/package.json packages/shared/
COPY packages/sdk/package.json packages/sdk/
COPY apps/gateway/package.json apps/gateway/
COPY apps/dashboard/package.json apps/dashboard/
# Production dependencies of the gateway workspace only.
RUN npm ci --omit=dev --workspace @aigw/gateway --include-workspace-root=false \
 && npm cache clean --force
COPY --from=build /repo/packages/shared/dist packages/shared/dist
COPY --from=build /repo/apps/gateway/dist apps/gateway/dist
COPY --from=build /repo/apps/gateway/prisma apps/gateway/prisma
COPY --from=build /repo/node_modules/.prisma node_modules/.prisma

USER node
WORKDIR /app/apps/gateway
EXPOSE 3000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD wget -qO- http://127.0.0.1:3000/healthz > /dev/null || exit 1
CMD ["node", "dist/main.js"]

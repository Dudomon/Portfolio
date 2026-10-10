# Build context: repository root (the dashboard depends on packages/sdk and packages/shared).
#   docker build -f apps/dashboard/Dockerfile -t ai-gateway-dashboard .

FROM node:24-alpine AS build
WORKDIR /repo
COPY package.json package-lock.json ./
COPY packages/shared/package.json packages/shared/
COPY packages/sdk/package.json packages/sdk/
COPY apps/gateway/package.json apps/gateway/
COPY apps/dashboard/package.json apps/dashboard/
RUN npm ci
COPY tsconfig.base.json ./
COPY packages packages
COPY apps/dashboard apps/dashboard
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build -w @aigw/shared \
 && npm run build -w @aigw/sdk \
 && npm run build -w @aigw/dashboard

FROM node:24-alpine AS runtime
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3001 \
    HOSTNAME=0.0.0.0
WORKDIR /app
# The standalone output carries only the traced server files and node_modules;
# static assets are served from .next/static next to the server.
COPY --from=build --chown=node:node /repo/apps/dashboard/.next/standalone ./
COPY --from=build --chown=node:node /repo/apps/dashboard/.next/static ./apps/dashboard/.next/static

USER node
EXPOSE 3001
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD wget -qO- http://127.0.0.1:3001/login > /dev/null || exit 1
CMD ["node", "apps/dashboard/server.js"]

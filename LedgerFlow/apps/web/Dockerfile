# syntax=docker/dockerfile:1.7

FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
ARG VITE_API_BASE_URL=/api/v1
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL} \
    VITE_USE_MOCKS=false
# The mock service worker is a development aid only.
RUN npm run build && rm -f dist/mockServiceWorker.js

FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime
# Document previews are fetched from presigned object-storage URLs, so that
# origin must be allowed by the CSP. Set at container start, e.g. https://s3.example.com
ENV CSP_CONNECT_SRC="" \
    API_UPSTREAM=http://api:8000
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:8080/healthz || exit 1

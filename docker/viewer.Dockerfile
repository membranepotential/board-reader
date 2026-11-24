# syntax=docker/dockerfile:1
# The viewer: the built SvelteKit app plus server/main.ts, run by Node with its types
# stripped. Two stages: `build` installs the whole lockfile and builds; the runtime
# image gets only the build output, the server modules and the production
# dependencies, on Alpine to stay small. Node, Vite and pnpm never ship.
#
# Build context is the repo root (compose.yaml); `.dockerignore` keeps the host's
# node_modules/, .svelte-kit/ and build/ out of it, so pnpm installs for the image's
# own platform from the lockfile.
ARG NODE_VERSION=24.19.0
# The pnpm that wrote packages/viewer/ui/pnpm-lock.yaml (lockfileVersion 9.0),
# installed explicitly: package.json carries no `packageManager` field for corepack.
ARG PNPM_VERSION=12.5.1

FROM node:${NODE_VERSION}-bookworm-slim AS build
ARG PNPM_VERSION
ENV CI=true
RUN npm install --global "pnpm@${PNPM_VERSION}"
WORKDIR /app
COPY packages/viewer/ui/package.json packages/viewer/ui/pnpm-lock.yaml packages/viewer/ui/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY packages/viewer/ui ./
RUN pnpm build \
    && rm server/*.test.ts \
    && pnpm prune --prod

FROM node:${NODE_VERSION}-alpine
ENV NODE_ENV=production
WORKDIR /app
COPY --from=build /app/package.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/build ./build
COPY --from=build /app/server ./server
# Root, like the other services: compose bind-mounts ./client-logs, which Docker
# creates root-owned when it is missing, and the `node` user could not write there.
# compose.yaml supplies the command (`node server/main.ts --host 0.0.0.0 --port 8080`).
CMD ["node", "server/main.ts", "--host", "0.0.0.0", "--port", "8080"]

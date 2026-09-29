FROM node:20-bookworm-slim

# kuma-bart depends on webgpu-compute-kit via a local `file:../../webgpu-compute-kit`
# package.json entry, which (from /workspace below) resolves to /webgpu-compute-kit --
# see docker-compose.yml's `additional_contexts` for how this named context is supplied.
# Must already have `dist/` built on the host (webgpu-compute-kit's own `npm run build`).
COPY --from=webgpu-compute-kit . /webgpu-compute-kit

WORKDIR /workspace

COPY package.json .
RUN npm install

COPY tsconfig.json vite.config.ts vite.demo.config.ts vitest.config.ts .
COPY src/ src/
COPY demo/ demo/
COPY test/ test/

CMD ["npm", "run", "test"]

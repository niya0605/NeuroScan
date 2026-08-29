FROM node:22-slim

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv build-essential libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN npm install -g corepack@latest && corepack pnpm install && corepack pnpm run build
RUN python3 -m pip install --break-system-packages --no-cache-dir tensorflow-cpu==2.20.0 pillow==11.3.0 numpy==2.2.6 matplotlib==3.10.6

ENV NODE_ENV=production
EXPOSE 3000
CMD ["node", "dist/index.js"]

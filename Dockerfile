FROM node:20-slim

RUN apt-get update && apt-get install -y ffmpeg fonts-dejavu-core fonts-noto-core && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev
COPY . .

EXPOSE 8080
CMD ["node", "index.js"]

FROM node:20-alpine

WORKDIR /app/backend

COPY backend/package.json ./package.json
COPY backend/package-lock.json ./package-lock.json
RUN npm install

COPY backend/ ./

EXPOSE 5000

CMD ["npm", "run", "dev"]

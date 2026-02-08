FROM node:20-alpine
WORKDIR /app
COPY ../frontend/package.json ../frontend/package-lock.json* ./
RUN npm ci
COPY ../frontend /app
EXPOSE 5173
CMD ["npm","run","dev","--","--host"]

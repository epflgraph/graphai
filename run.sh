docker compose down nginx --remove-orphans
docker compose build --no-cache nginx
docker compose up -d nginx

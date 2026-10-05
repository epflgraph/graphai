FROM alpine:latest
RUN apk add --no-cache inotify-tools docker-cli
COPY watch.sh /watch.sh
RUN chmod +x /watch.sh
CMD ["/watch.sh"]

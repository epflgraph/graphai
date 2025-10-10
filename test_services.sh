# RabbitMQ reachable?
nc -zv localhost 5672
# Redis reachable?
nc -zv localhost 6379

python - <<'PY'
from kombu import Connection
import os
url=os.environ.get("CELERY_BROKER_URL","amqp://graphai:graphai@127.0.0.1:5672/%2F")
print("Testing:", url)
with Connection(url) as conn:
    conn.connect()
    print("AMQP OK, connected and authenticated.")
PY

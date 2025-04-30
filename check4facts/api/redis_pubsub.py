import os
import redis
from dotenv import load_dotenv
from check4facts.database import task_channel_name

load_dotenv()
redis_url = os.getenv("CELERY_REDIS_URL")


def get_redis():
    return redis.Redis.from_url(redis_url)


def publish_progress(task_id: str, message: str):
    r = get_redis()
    r.set(f"progress:{task_id}", message)
    r.publish(task_channel_name(task_id), message)
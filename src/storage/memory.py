import redis
import json
import time
from typing import List, Dict, Optional
from datetime import timedelta
from src import config


class RedisMemory:
    """Redis를 사용한 대화 단기 기억 관리"""

    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # 연결 테스트
            self.redis_client.ping()
            self.ttl = timedelta(hours=config.REDIS_TTL_HOURS)
            print("[INFO] Redis connected successfully")
        except Exception as e:
            print(f"[WARN] Redis connection failed: {e}")
            self.redis_client = None

    def _get_key(self, user_id: str, conversation_id: str) -> str:
        """Redis 키 생성"""
        return f"chat:{user_id}:{conversation_id}"

    def add_message(
        self, user_id: str, conversation_id: str, role: str, content: str
    ) -> bool:
        """대화에 메시지 추가"""
        if not self.redis_client:
            return False

        try:
            key = self._get_key(user_id, conversation_id)
            message = {"role": role, "content": content, "timestamp": int(time.time())}

            # 기존 메시지 가져오기
            messages_json = self.redis_client.get(key)
            messages = json.loads(messages_json) if messages_json else []

            # 새 메시지 추가
            messages.append(message)

            # 최근 5개만 유지 (user + assistant 쌍을 고려해서 10개로 설정)
            if len(messages) > 10:
                messages = messages[-10:]

            # Redis에 저장 및 TTL 설정
            self.redis_client.setex(
                key,
                int(self.ttl.total_seconds()),
                json.dumps(messages, ensure_ascii=False),
            )
            return True

        except Exception as e:
            print(f"[WARN] Failed to add message to Redis: {e}")
            return False

    def get_conversation_history(
        self, user_id: str, conversation_id: str
    ) -> List[Dict]:
        """대화 히스토리 조회"""
        if not self.redis_client:
            return []

        try:
            key = self._get_key(user_id, conversation_id)
            messages_json = self.redis_client.get(key)

            if messages_json:
                return json.loads(messages_json)
            return []

        except Exception as e:
            print(f"[WARN] Failed to get conversation history: {e}")
            return []

    def format_history_for_llm(self, user_id: str, conversation_id: str) -> str:
        """LLM에 전달할 히스토리 형식으로 변환"""
        history = self.get_conversation_history(user_id, conversation_id)
        if not history:
            return ""

        formatted_lines = []
        for msg in history:
            role_name = "사용자" if msg["role"] == "user" else "어시스턴트"
            formatted_lines.append(f"{role_name}: {msg['content']}")

        return "\n".join(formatted_lines)

    def clear_conversation(self, user_id: str, conversation_id: str) -> bool:
        """특정 대화 삭제"""
        if not self.redis_client:
            return False

        try:
            key = self._get_key(user_id, conversation_id)
            self.redis_client.delete(key)
            return True
        except Exception as e:
            print(f"[WARN] Failed to clear conversation: {e}")
            return False


# 글로벌 인스턴스
memory = RedisMemory()

"""통합 테스트 스크립트"""

import unittest
import sys
import os

# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


class TestEngine(unittest.TestCase):
    """엔진 테스트"""

    def test_engine_import(self):
        from src.core.engine import answer_question_graph

        self.assertIsNotNone(answer_question_graph)


class TestAPI(unittest.TestCase):
    """API 테스트"""

    def test_api_import(self):
        from src.api.server import app

        self.assertIsNotNone(app)


class TestMemory(unittest.TestCase):
    """메모리 테스트"""

    def test_memory_import(self):
        from src.storage.memory import memory

        self.assertIsNotNone(memory)


if __name__ == "__main__":
    unittest.main()

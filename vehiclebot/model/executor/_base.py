
import asyncio

class ModelExecutor:
    def run(self, task : dict) -> asyncio.Task:
        raise NotImplementedError()

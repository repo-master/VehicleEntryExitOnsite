
import asyncio

class ModelExecutor:
    def run(self, task_name : str, *args, **kwargs) -> asyncio.Task:
        raise NotImplementedError()

    async def _cleanup(self):
        pass

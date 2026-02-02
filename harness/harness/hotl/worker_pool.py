"""Worker pool for parallel task execution in HOTL mode."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A task to be executed by the worker pool."""

    id: int
    func: Callable[..., Awaitable[Any]]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0  # Higher = more urgent
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""

    def __lt__(self, other: "Task") -> bool:
        """Compare by priority for priority queue ordering."""
        return self.priority > other.priority  # Higher priority first


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_id: int
    success: bool
    result: Any = None
    error: str | None = None
    duration_seconds: float = 0.0
    completed_at: datetime = field(default_factory=datetime.utcnow)


class WorkerPool:
    """
    Async worker pool for parallel task execution.

    Implements bounded parallelism with a semaphore to limit
    concurrent operations, preventing resource exhaustion.
    """

    def __init__(self, max_workers: int = 3):
        """
        Initialize the worker pool.

        Args:
            max_workers: Maximum concurrent task executions
        """
        self.max_workers = max_workers
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.results: dict[int, TaskResult] = {}
        self.workers: list[asyncio.Task] = []
        self.running = False
        self._semaphore: asyncio.Semaphore | None = None
        self._task_counter = 0

    async def start(self) -> None:
        """Start the worker pool."""
        if self.running:
            return

        self.running = True
        self._semaphore = asyncio.Semaphore(self.max_workers)

        # Start worker coroutines
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)

        logger.info(f"WorkerPool started with {self.max_workers} workers")

    async def stop(self, wait_for_completion: bool = True) -> None:
        """
        Stop the worker pool.

        Args:
            wait_for_completion: If True, wait for queued tasks to complete
        """
        self.running = False

        if wait_for_completion:
            # Wait for queue to drain
            try:
                await asyncio.wait_for(self.task_queue.join(), timeout=60.0)
            except TimeoutError:
                logger.warning("Timed out waiting for task queue to drain")

        # Cancel workers
        for worker in self.workers:
            worker.cancel()

        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        logger.info("WorkerPool stopped")

    async def _worker(self, worker_id: int) -> None:
        """
        Worker coroutine that processes tasks from the queue.

        Args:
            worker_id: Identifier for this worker
        """
        while self.running:
            try:
                # Wait for a task with timeout to allow checking running flag
                try:
                    _, task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                # Execute the task with semaphore
                async with self._semaphore:
                    start_time = datetime.utcnow()
                    try:
                        result = await task.func(*task.args, **task.kwargs)
                        self.results[task.id] = TaskResult(
                            task_id=task.id,
                            success=True,
                            result=result,
                            duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                        )
                        logger.debug(f"Worker {worker_id}: Task {task.id} completed")
                    except Exception as e:
                        self.results[task.id] = TaskResult(
                            task_id=task.id,
                            success=False,
                            error=str(e),
                            duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                        )
                        logger.error(f"Worker {worker_id}: Task {task.id} failed: {e}")

                self.task_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Worker {worker_id}: Unexpected error: {e}")

    async def submit(self, task: Task) -> int:
        """
        Submit a task for execution.

        Args:
            task: Task to execute

        Returns:
            Task ID
        """
        if not self.running:
            raise RuntimeError("WorkerPool is not running")

        # Assign task ID if not set
        if task.id == 0:
            self._task_counter += 1
            task.id = self._task_counter

        await self.task_queue.put((task.priority, task))
        logger.debug(f"Task {task.id} submitted: {task.description}")
        return task.id

    async def submit_func(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        priority: int = 0,
        description: str = "",
        **kwargs,
    ) -> int:
        """
        Submit a function for execution (convenience method).

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            priority: Task priority (higher = more urgent)
            description: Task description
            **kwargs: Keyword arguments for the function

        Returns:
            Task ID
        """
        self._task_counter += 1
        task = Task(
            id=self._task_counter,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            description=description,
        )
        return await self.submit(task)

    def get_result(self, task_id: int) -> TaskResult | None:
        """Get the result of a completed task."""
        return self.results.get(task_id)

    async def wait_for_result(self, task_id: int, timeout: float = 60.0) -> TaskResult | None:
        """
        Wait for a specific task to complete.

        Args:
            task_id: ID of the task to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            TaskResult if completed, None if timeout
        """
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            if task_id in self.results:
                return self.results[task_id]
            await asyncio.sleep(0.1)
        return None

    async def wait_all(self, timeout: float = 300.0) -> None:
        """
        Wait for all queued tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds
        """
        try:
            await asyncio.wait_for(self.task_queue.join(), timeout=timeout)
        except TimeoutError:
            raise TimeoutError(f"Tasks did not complete within {timeout}s")

    def pending_count(self) -> int:
        """Get the number of pending tasks."""
        return self.task_queue.qsize()

    def completed_count(self) -> int:
        """Get the number of completed tasks."""
        return len(self.results)

    def get_stats(self) -> dict:
        """Get worker pool statistics."""
        successful = sum(1 for r in self.results.values() if r.success)
        failed = sum(1 for r in self.results.values() if not r.success)
        avg_duration = (
            sum(r.duration_seconds for r in self.results.values()) / len(self.results)
            if self.results
            else 0.0
        )

        return {
            "running": self.running,
            "max_workers": self.max_workers,
            "pending_tasks": self.pending_count(),
            "completed_tasks": self.completed_count(),
            "successful_tasks": successful,
            "failed_tasks": failed,
            "avg_duration_seconds": avg_duration,
        }

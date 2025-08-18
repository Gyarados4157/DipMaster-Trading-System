"""
Asyncio Compatibility Module for Python 3.6+
Provides asyncio.run() functionality for older Python versions
"""
import asyncio
import sys
import functools
import logging

logger = logging.getLogger(__name__)

def asyncio_run(coro):
    """
    Python 3.6 compatible version of asyncio.run()
    
    Args:
        coro: Coroutine to run
        
    Returns:
        Result of the coroutine
    """
    try:
        # Python 3.7+ has asyncio.run()
        if hasattr(asyncio, 'run'):
            return asyncio.run(coro)
        else:
            # Python 3.6 fallback
            loop = None
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            try:
                return loop.run_until_complete(coro)
            finally:
                try:
                    # Cancel all pending tasks
                    pending = asyncio.all_tasks(loop)
                    if pending:
                        for task in pending:
                            task.cancel()
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception as e:
                    logger.warning(f"Error during cleanup: {e}")
                finally:
                    try:
                        loop.close()
                    except Exception as e:
                        logger.warning(f"Error closing loop: {e}")
                        
    except Exception as e:
        logger.error(f"Error in asyncio_run: {e}")
        raise

def get_running_loop():
    """
    Python 3.6 compatible version of asyncio.get_running_loop()
    """
    try:
        # Python 3.7+ has get_running_loop()
        if hasattr(asyncio, 'get_running_loop'):
            return asyncio.get_running_loop()
        else:
            # Python 3.6 fallback
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                raise RuntimeError("No running event loop")
            return loop
    except RuntimeError:
        # No running loop
        raise RuntimeError("No running event loop")

def create_task(coro):
    """
    Python 3.6 compatible task creation
    """
    try:
        if hasattr(asyncio, 'create_task'):
            return asyncio.create_task(coro)
        else:
            # Python 3.6 fallback
            loop = get_running_loop()
            return loop.create_task(coro)
    except Exception:
        # Last resort fallback
        return asyncio.ensure_future(coro)

# Monkey patch asyncio if needed
if not hasattr(asyncio, 'run'):
    asyncio.run = asyncio_run
    
if not hasattr(asyncio, 'get_running_loop'):
    asyncio.get_running_loop = get_running_loop
    
if not hasattr(asyncio, 'create_task'):
    asyncio.create_task = create_task

logger.info(f"Asyncio compatibility module loaded for Python {sys.version_info}")
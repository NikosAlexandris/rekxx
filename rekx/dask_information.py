import multiprocessing
from rich import print


def print_dask_status_minimal():
    """Print minimal Dask status for quick checks"""
    
    try:
        from dask.distributed import Client, get_client
        
        print("\n🚀 DASK STATUS")
        print("-" * 20)
        
        try:
            client = get_client()
            scheduler_info = client.scheduler_info()
            workers = scheduler_info.get('workers', {})
            
            n_workers = len(workers)
            total_threads = sum(w.get('nthreads', 0) for w in workers.values())
            total_memory = sum(w.get('memory_limit', 0) for w in workers.values())
            
            print(f"✅ Cluster: {n_workers} workers, {total_threads} threads, {format_bytes(total_memory)}")
            
            # Dashboard info
            dashboard_link = getattr(client, 'dashboard_link', None)
            if dashboard_link:
                print(f"📈 Dashboard: {dashboard_link}")
                
        except (ValueError, RuntimeError):
            print("⚠️  Single-threaded mode")
            
        print("-" * 20)
        
    except Exception as e:
        print(f"⚠️  Dask status unavailable: {e}")

def print_dask_configuration_clean(verbose: int = 0):
    """Print a cleaner, more digestible Dask configuration summary"""
    
    try:
        import dask
        import dask.config
        from dask.distributed import Client, get_client
        
        print("\n" + "="*60)
        print("🚀 DASK CLUSTER STATUS")
        print("="*60)
        
        # 1. System Overview - Most Important Info First
        print("\n📊 CLUSTER OVERVIEW")
        print("-" * 30)
        
        try:
            client = get_client()
            scheduler_info = client.scheduler_info()
            workers = scheduler_info.get('workers', {})
            
            n_workers = len(workers)
            total_threads = sum(w.get('nthreads', 0) for w in workers.values())
            total_memory = sum(w.get('memory_limit', 0) for w in workers.values())
            
            print(f"✅ Status: Connected to Dask cluster")
            print(f"🔗 Scheduler: {getattr(client, 'scheduler', 'N/A')}")
            print(f"👥 Workers: {n_workers}")
            print(f"🧵 Total Threads: {total_threads}")
            print(f"💾 Total Memory: {format_bytes(total_memory)}")
            
            # Dashboard info
            dashboard_link = getattr(client, 'dashboard_link', None)
            if dashboard_link:
                print(f"📈 Dashboard: {dashboard_link}")
            
        except (ValueError, RuntimeError):
            print("⚠️  No active Dask client - running in single-threaded mode")
            
        # 2. Key Configuration Settings (only the most relevant ones)
        print(f"\n⚙️  CONFIGURATION HIGHLIGHTS")
        print("-" * 30)
        
        # Get key settings
        work_stealing = dask.config.get('distributed.scheduler.work-stealing', 'Unknown')
        chunk_size = dask.config.get('array.chunk-size', 'Unknown')
        worker_memory_target = dask.config.get('distributed.worker.memory.target', 'Unknown')
        
        print(f"🔄 Work Stealing: {work_stealing}")
        print(f"📦 Array Chunk Size: {chunk_size}")
        print(f"🎯 Worker Memory Target: {worker_memory_target}")
        
        # 3. Resource Information
        print(f"\n💻 SYSTEM RESOURCES")
        print("-" * 30)
        
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        available_memory = get_memory_info()
        
        print(f"🔢 CPU Cores: {cpu_count}")
        print(f"💾 Available Memory: {available_memory}")
        print(f"📝 Current Scheduler: {dask.config.get('scheduler', 'synchronous')}")
        
        # 4. Detailed Worker Info (only if verbose > 2)
        if verbose > 2:
            print(f"\n🔍 DETAILED WORKER INFORMATION")
            print("-" * 30)
            
            try:
                client = get_client()
                scheduler_info = client.scheduler_info()
                
                for i, (worker_id, worker_info) in enumerate(scheduler_info.get('workers', {}).items(), 1):
                    print(f"\n   Worker {i}:")
                    print(f"     ID: {worker_info.get('name', 'N/A')}")
                    print(f"     Threads: {worker_info.get('nthreads', 0)}")
                    print(f"     Memory: {format_bytes(worker_info.get('memory_limit', 0))}")
                    print(f"     Status: {worker_info.get('status', 'N/A')}")
                    
                    # Current memory usage
                    current_memory = worker_info.get('metrics', {}).get('memory', 0)
                    if current_memory > 0:
                        print(f"     Current Memory Use: {format_bytes(current_memory)}")
                        
            except Exception as e:
                print(f"   Could not get detailed worker info: {e}")
        
        print("\n" + "="*60)
        
    except ImportError as e:
        print(f"⚠️  Dask not available: {e}")
    except Exception as e:
        print(f"⚠️  Error getting Dask configuration: {e}")

def print_dask_performance_summary():
    """Print performance-focused Dask summary"""
    
    try:
        from dask.distributed import Client, get_client
        
        print("\n⚡ DASK PERFORMANCE SUMMARY")
        print("-" * 35)
        
        try:
            client = get_client()
            scheduler_info = client.scheduler_info()
            workers = scheduler_info.get('workers', {})
            
            # Calculate totals
            n_workers = len(workers)
            total_threads = sum(w.get('nthreads', 0) for w in workers.values())
            total_memory_limit = sum(w.get('memory_limit', 0) for w in workers.values())
            
            # Calculate current usage
            total_memory_used = sum(w.get('metrics', {}).get('memory', 0) for w in workers.values())
            total_cpu_usage = sum(w.get('metrics', {}).get('cpu', 0) for w in workers.values())
            
            print(f"🏭 Workers: {n_workers} active")
            print(f"🧵 Threads: {total_threads} available")
            print(f"💾 Memory: {format_bytes(total_memory_used)}/{format_bytes(total_memory_limit)} " +
                  f"({100*total_memory_used/total_memory_limit:.1f}% used)")
            print(f"⚡ CPU: {total_cpu_usage:.1f}% average")
            
            # Task information if available
            task_counts = {}
            for worker_info in workers.values():
                worker_tasks = worker_info.get('metrics', {}).get('task_counts', {})
                for task_type, count in worker_tasks.items():
                    task_counts[task_type] = task_counts.get(task_type, 0) + count
            
            if task_counts:
                total_tasks = sum(task_counts.values())
                print(f"📋 Active Tasks: {total_tasks}")
                
        except Exception as e:
            print(f"⚠️  Performance info unavailable: {e}")
            
        print("-" * 35)
        
    except Exception as e:
        print(f"⚠️  Dask performance summary unavailable: {e}")

def get_memory_info():
    """Get available system memory information"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return format_bytes(mem.available)
    except ImportError:
        return "Unknown (psutil not available)"

def format_bytes(bytes_value):
    """Format bytes in human readable format"""
    if bytes_value == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    
    return f"{bytes_value:.1f} PB"

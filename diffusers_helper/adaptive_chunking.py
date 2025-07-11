import torch
import time


class AdaptiveChunkSizeManager:
    """
    Manages chunk sizes dynamically based on memory usage and performance.
    """
    def __init__(self, initial_chunk_size=1024, min_chunk_size=128, max_chunk_size=2048):
        self.current_chunk_size = initial_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.success_count = 0
        self.oom_count = 0
        self.performance_history = []
        self.aggressive_mode = False
        
    def set_aggressive_mode(self, aggressive=False, chunk_multiplier=1.0, max_chunk=2048):
        """Enable aggressive mode for higher performance."""
        self.aggressive_mode = aggressive
        if aggressive:
            self.current_chunk_size = int(self.current_chunk_size * chunk_multiplier)
            self.max_chunk_size = max_chunk
            print(f"Aggressive mode enabled: chunk_size={self.current_chunk_size}, max_chunk={self.max_chunk_size}")
        
    def get_chunk_size(self):
        return self.current_chunk_size
    
    def report_success(self, processing_time, memory_used_gb):
        """Report successful operation with timing and memory usage."""
        self.success_count += 1
        self.performance_history.append((processing_time, memory_used_gb))
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
        
        # More aggressive scaling in aggressive mode
        memory_threshold = 7.0 if self.aggressive_mode else 6.0
        success_threshold = 2 if self.aggressive_mode else 3
        scale_factor = 1.5 if self.aggressive_mode else 1.2
        
        # If we've had several successes and memory usage is low, try larger chunks
        if self.success_count >= success_threshold and memory_used_gb < memory_threshold:
            new_size = min(self.current_chunk_size * scale_factor, self.max_chunk_size)
            if new_size != self.current_chunk_size:
                print(f"Increasing chunk size from {self.current_chunk_size} to {int(new_size)} (low memory usage)")
                self.current_chunk_size = int(new_size)
                self.success_count = 0
    
    def report_oom(self):
        """Report out of memory error."""
        self.oom_count += 1
        old_size = self.current_chunk_size
        # Less aggressive reduction in aggressive mode
        reduction_factor = 0.8 if self.aggressive_mode else 0.7
        self.current_chunk_size = max(self.current_chunk_size * reduction_factor, self.min_chunk_size)
        if old_size != self.current_chunk_size:
            print(f"Reducing chunk size from {old_size} to {int(self.current_chunk_size)} due to OOM")
        self.success_count = 0
    
    def get_optimal_chunk_size_for_memory(self, available_memory_gb):
        """Calculate optimal chunk size based on available memory."""
        if self.aggressive_mode:
            # More aggressive memory usage
            if available_memory_gb > 5:
                return min(self.max_chunk_size, self.current_chunk_size * 2.0)
            elif available_memory_gb > 3:
                return min(self.max_chunk_size, self.current_chunk_size * 1.5)
            else:
                return self.current_chunk_size
        else:
            # Conservative scaling
            if available_memory_gb > 6:
                return min(self.max_chunk_size, self.current_chunk_size * 1.5)
            elif available_memory_gb > 4:
                return self.current_chunk_size
            else:
                return max(self.min_chunk_size, self.current_chunk_size * 0.8)


# Global chunk size manager
chunk_manager = AdaptiveChunkSizeManager()

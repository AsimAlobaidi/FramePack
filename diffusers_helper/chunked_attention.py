import torch
import math
import gc
import time
from .adaptive_chunking import chunk_manager


def get_optimal_chunk_size(seq_len, available_memory_gb=4.0):
    """
    Dynamically calculate optimal chunk size based on sequence length and available memory.
    """
    # Get the adaptive chunk size from our manager
    base_chunk = chunk_manager.get_optimal_chunk_size_for_memory(available_memory_gb)
    
    # In aggressive mode, allow larger chunks based on available memory
    if chunk_manager.aggressive_mode and available_memory_gb > 3:
        # Scale up more aggressively when we have memory available
        memory_scale = min(available_memory_gb / 4.0, 2.0)  # Up to 2x scaling
        base_chunk = int(base_chunk * memory_scale)
    
    # Don't exceed sequence length
    return min(base_chunk, seq_len)


def efficient_chunked_attention(q, k, v, chunk_size=None):
    """
    More efficient chunked attention that balances memory usage and performance.
    """
    B, H, N, D = q.shape
    
    # Get current memory usage
    if torch.cuda.is_available():
        memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
        memory_free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
    else:
        memory_used_gb = 0
        memory_free_gb = 4
    
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(N, memory_free_gb)
    
    start_time = time.time()
    
    try:
        # If sequence is small enough, use regular attention
        if N <= chunk_size:
            try:
                result = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                
                # Report success to adaptive manager
                processing_time = time.time() - start_time
                chunk_manager.report_success(processing_time, memory_used_gb)
                
                return result
            except torch.cuda.OutOfMemoryError:
                chunk_manager.report_oom()
                # Only fall back to CPU for very small sequences that still OOM
                if N <= 256:
                    torch.cuda.empty_cache()
                    q_cpu = q.cpu()
                    k_cpu = k.cpu() 
                    v_cpu = v.cpu()
                    result = torch.nn.functional.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
                    return result.to(q.device)
                else:
                    # Reduce chunk size and try again
                    new_chunk_size = chunk_manager.get_chunk_size()
                    return efficient_chunked_attention(q, k, v, chunk_size=new_chunk_size)
        
        # Use chunked processing for larger sequences
        output = torch.zeros_like(q)
        scale = 1.0 / math.sqrt(D)
        
        # Process queries in chunks
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            q_chunk = q[:, :, i:end_i, :]
            
            try:
                # Try to process the entire key-value sequence at once for this query chunk
                attn_weights = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(attn_weights, dim=-1)
                out_chunk = torch.matmul(attn_weights, v)
                output[:, :, i:end_i, :] = out_chunk
                
            except torch.cuda.OutOfMemoryError:
                chunk_manager.report_oom()
                # If still OOM, chunk the key-value dimension too
                torch.cuda.empty_cache()
                
                # Get updated chunk size after OOM
                kv_chunk_size = max(chunk_manager.get_chunk_size() // 2, 64)
                
                # Initialize accumulation for this query chunk
                out_chunk = torch.zeros_like(q_chunk)
                max_vals = torch.full((B, H, end_i - i, 1), float('-inf'), device=q.device)
                sum_exp = torch.zeros((B, H, end_i - i, 1), device=q.device)
                
                # Process key-value in smaller chunks
                for j in range(0, N, kv_chunk_size):
                    end_j = min(j + kv_chunk_size, N)
                    k_chunk = k[:, :, j:end_j, :]
                    v_chunk = v[:, :, j:end_j, :]
                    
                    # Compute attention scores for this chunk
                    scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale
                    
                    # Online softmax computation
                    chunk_max = torch.max(scores, dim=-1, keepdim=True)[0]
                    new_max = torch.maximum(max_vals, chunk_max)
                    
                    # Adjust previous accumulated values
                    old_scale = torch.exp(max_vals - new_max)
                    new_scale = torch.exp(chunk_max - new_max)
                    
                    # Update running sum
                    sum_exp = sum_exp * old_scale + torch.sum(torch.exp(scores - new_max), dim=-1, keepdim=True)
                    
                    # Update output
                    out_chunk = out_chunk * old_scale + torch.matmul(torch.exp(scores - new_max), v_chunk)
                    max_vals = new_max
                
                # Final normalization
                output[:, :, i:end_i, :] = out_chunk / sum_exp
        
        # Report successful completion
        processing_time = time.time() - start_time
        chunk_manager.report_success(processing_time, memory_used_gb)
        
        return output
        
    except Exception as e:
        if "out of memory" in str(e).lower():
            chunk_manager.report_oom()
        raise e


def chunked_scaled_dot_product_attention(q, k, v, chunk_size=1024, attn_mask=None):
    """
    Optimized memory-efficient chunked implementation of scaled dot product attention.
    """
    # Handle different tensor layouts
    if q.dim() == 4:
        if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            # Assume layout is [batch, seq_len, num_heads, head_dim]
            needs_transpose = True
            q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            needs_transpose = False
    else:
        needs_transpose = False
    
    # Use the more efficient chunked attention
    output = efficient_chunked_attention(q, k, v, chunk_size=chunk_size)
    
    if needs_transpose:
        output = output.transpose(1, 2)
    
    return output


def chunked_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, chunk_size=1024):
    """
    Optimized chunked version of the attention function for variable length sequences.
    """
    if cu_seqlens_q is None and cu_seqlens_kv is None and max_seqlen_q is None and max_seqlen_kv is None:
        # For standard attention, use adaptive chunk size
        try:
            return chunked_scaled_dot_product_attention(q, k, v, chunk_size=chunk_manager.get_chunk_size())
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            chunk_manager.report_oom()
            # Try with the updated (smaller) chunk size
            return chunked_scaled_dot_product_attention(q, k, v, chunk_size=chunk_manager.get_chunk_size())
    
    # For variable length sequences, be more conservative but still adaptive
    try:
        # Transpose for PyTorch attention format
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2) 
        v_t = v.transpose(1, 2)
        
        # Use adaptive chunking
        result = chunked_scaled_dot_product_attention(q_t, k_t, v_t, chunk_size=chunk_manager.get_chunk_size())
        return result.transpose(1, 2)
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        chunk_manager.report_oom()
        # Fallback with smaller chunks
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2) 
        v_t = v.transpose(1, 2)
        result = chunked_scaled_dot_product_attention(q_t, k_t, v_t, chunk_size=chunk_manager.get_chunk_size())
        return result.transpose(1, 2)

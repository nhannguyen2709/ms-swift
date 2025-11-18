# Copyright (c) Alibaba, Inc. and its affiliates.


def try_init_unsloth():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuner_backend', type=str, default='peft')
    args, _ = parser.parse_known_args()
    if args.tuner_backend == 'unsloth':
        import unsloth


def init_debugpy_if_enabled():
    import os

    """Initialize debugpy for each rank in distributed training"""
    if os.environ.get('DEBUGPY_ENABLED', '0') == '1':
        import debugpy
        
        # Get rank from Accelerate environment variables
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Calculate debug port based on rank
        debug_port = 5678 + local_rank
        
        # Check if debugpy is already listening (avoid double-init)
        if not debugpy.is_client_connected():
            try:
                debugpy.listen(("0.0.0.0", debug_port))
                print(f"[Rank {local_rank}] Debugpy listening on port {debug_port}, waiting for client...")
                debugpy.wait_for_client()
                print(f"[Rank {local_rank}] Debugger attached!")
            except Exception as e:
                print(f"[Rank {local_rank}] Failed to start debugpy: {e}")


if __name__ == '__main__':
    try_init_unsloth()
    from swift.ray import try_init_ray
    try_init_ray()
    init_debugpy_if_enabled()
    from swift.llm import sft_main
    sft_main()

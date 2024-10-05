import torch


def parallelize_model(model:torch.nn.Module, device_mesh:DeviceMesh, training:bool):
    # parallelism
    dp_mesh = None if parallel_args.dp == 1 else device_mesh['dp']
    tp_mesh = None if parallel_args.tp == 1 else device_mesh['tp']
    pp_mesh = None if parallel_args.pp == 1 else device_mesh['pp']
    # 2D parallel (tp + dp)
    if pp_mesh is None:
        pp_schedule = None
        if tp_mesh is not None:
            _ = tensor_parallelize(model, tp_mesh, training, parallel_args)
        if llama_config['activation_checkpoint_mode'] is not None:
            enable_activation_checkpoint(module, llama_config['activation_checkpoint_mode'])
        # turn on per-TransformerBlock compile after AC wrapping and before FSDP
        if llama_config['compile']:
            if model.params.norm_type == "fused_rmsnorm":
                raise NotImplementedError(
                    "fused_rmsnorm is not compatible with torch.compile yet. "
                    "Please use rmsnorm or layernorm."
                )
            enable_compile(model)
        # data parallelism
        if dp_mesh is not None:
            if model.dp_shard:
                # reference: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp
                model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
                # my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
                # model = FSDP(
                #     model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True),
                #     device_mesh=dp_mesh, use_orig_params=True
                # )
            else:
                model = DDP(model, device_ids=[device])
    # 3D parallel (pp + tp + dp)
    else:
        pp_schedule, modules = pipeline_parallelize(model, pp_mesh, training)
        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for module in modules:
            # apply SPMD-style PT-D techniques
            if tp_mesh is not None:
                _ = tensor_parallelize(module, tp_mesh, training, parallel_args)
            if llama_config['activation_checkpoint_mode'] is not None:
                enable_activation_checkpoint(module, llama_config['activation_checkpoint_mode'])
            # turn on per-TransformerBlock compile after AC wrapping and before FSDP
            if llama_config['compile']:
                if model.params.norm_type == "fused_rmsnorm":
                    raise NotImplementedError(
                        "fused_rmsnorm is not compatible with torch.compile yet. "
                        "Please use rmsnorm or layernorm."
                    )
                enable_compile(module)
            if dp_mesh is not None:
                _ = data_parallelize(module, dp_mesh, training, parallel_args)
            module.train()

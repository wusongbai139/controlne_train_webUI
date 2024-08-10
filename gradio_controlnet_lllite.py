import gradio as gr
import subprocess

def launch_training(pretrained_model_name_or_path, vae, no_half_vae, train_data_dir, conditioning_data_dir, resolution, cache_latents, vae_batch_size, 
                    cache_latents_to_disk,

                    caption_extension, max_token_length, keep_tokens, caption_dropout_rate, caption_dropout_every_n_epochs, caption_tag_dropout_rate, clip_skip,
                    debiased_estimation_loss, weighted_captions, cache_text_encoder_outputs, cache_text_encoder_outputs_to_disk, 

                    train_batch_size, dataset_repeats, max_train_epochs, max_train_steps, 

                    save_precision, save_every_n_epochs, save_every_n_steps, save_last_n_epochs, 
                    save_last_n_epochs_state, save_last_n_steps, save_last_n_steps_state, save_state, save_state_on_train_end, 
                    output_dir, output_name, 

                    optimizer_type, learning_rate, max_grad_norm, lr_scheduler, lr_warmup_steps, 
                    lr_scheduler_num_cycles, lr_scheduler_power, cond_emb_dim, network_dim, 

                    console_log_level, logging_dir, log_with, log_prefix, 
                    log_tracker_name, wandb_run_name, log_tracker_config, wandb_api_key,
                    
                    resume, mem_eff_attn, torch_compile, xformers, 
                    
                    max_data_loader_n_workers, persistent_data_loader_workers, 
                    seed, gradient_checkpointing, gradient_accumulation_steps, mixed_precision, full_fp16, full_bf16, fp8_base, 
                    
                    lowram, highvram
                    ):
    
    command = [
        "accelerate", "launch", "./sdxl_train_control_net_lllite.py",
        "--pretrained_model_name_or_path", pretrained_model_name_or_path,
        "--vae", vae,
        "--no_half_vae", str(no_half_vae),
        "--train_data_dir", train_data_dir,
        "--conditioning_data_dir", conditioning_data_dir,
        "--resolution", str(resolution),
        "--cache_latents", str(cache_latents),
        "--vae_batch_size", str(vae_batch_size),
        "--cache_latents_to_disk", str(cache_latents_to_disk),

        "--caption_extension", caption_extension,
        "--max_token_length", str(max_token_length),
        "--keep_tokens", str(keep_tokens),
        "--caption_dropout_rate", str(caption_dropout_rate),
        "--caption_dropout_every_n_epochs", str(caption_dropout_every_n_epochs),
        "--caption_tag_dropout_rate", str(caption_tag_dropout_rate),
        "--clip_skip", str(clip_skip),
        "--debiased_estimation_loss", str(debiased_estimation_loss),
        "--weighted_captions", str(weighted_captions),
        "--cache_text_encoder_outputs", str(cache_text_encoder_outputs),
        "--cache_text_encoder_outputs_to_disk", str(cache_text_encoder_outputs_to_disk),

        "--train_batch_size", str(train_batch_size),
        "--dataset_repeats", str(dataset_repeats),
        "--max_train_epochs", str(max_train_epochs),
        "--max_train_steps", str(max_train_steps),
        
        "--save_precision", save_precision,
        "--save_every_n_epochs", str(save_every_n_epochs),
        "--save_every_n_steps", str(save_every_n_steps),
        "--save_last_n_epochs", str(save_last_n_epochs),
        "--save_last_n_epochs_state", str(save_last_n_epochs_state),
        "--save_last_n_steps", str(save_last_n_steps),
        "--save_last_n_steps_state", str(save_last_n_steps_state),
        "--save_state", str(save_state),
        "--save_state_on_train_end", str(save_state_on_train_end),
        "--output_dir", output_dir,
        "--output_name", output_name,

        "--optimizer_type", optimizer_type,
        "--learning_rate", str(learning_rate),
        "--max_grad_norm", str(max_grad_norm),
        "--lr_scheduler", lr_scheduler,
        "--lr_warmup_steps", str(lr_warmup_steps),
        "--lr_scheduler_num_cycles", str(lr_scheduler_num_cycles),
        "--lr_scheduler_power", str(lr_scheduler_power),
        "--cond_emb_dim", str(cond_emb_dim),
        "--network_dim", str(network_dim),

        "--console_log_level", console_log_level,
        "--logging_dir", logging_dir,
        "--log_with", log_with,
        "--log_prefix", log_prefix,
        "--log_tracker_name", log_tracker_name,
        "--wandb_run_name", wandb_run_name,
        "--log_tracker_config", log_tracker_config,
        "--wandb_api_key", wandb_api_key,
        
        "--resume", resume,
        "--mem_eff_attn", str(mem_eff_attn),
        "--torch_compile", str(torch_compile),
        "--xformers", str(xformers),

        "--max_data_loader_n_workers", str(max_data_loader_n_workers),
        "--persistent_data_loader_workers", str(persistent_data_loader_workers),
        "--seed", str(seed),
        "--gradient_checkpointing", str(gradient_checkpointing),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--mixed_precision", mixed_precision,
        "--full_fp16", str(full_fp16),
        "--full_bf16", str(full_bf16),
        "--fp8_base", str(fp8_base),

        "--lowram", str(lowram),
        "--highvram", str(highvram),
    ]
    command = [arg for arg in command if arg != '']  
    subprocess.run(command, check=True)
    return "训练已启动，参数已经传递。"

cnlite = gr.Interface(
    fn=launch_training,
    inputs = [
        gr.Textbox(label="底模路径 Pretrained Model Path"),
        gr.Textbox(label="VAE Path"),
        gr.Checkbox(label="禁用VAE的半精度 Disable Half VAE"),
        gr.Textbox(label="将目标图片和对应的提示词文档所在文件路径放在此处 Training Data Directory"),
        gr.Textbox(label="将条件图片单独放在一个文件夹里 Conditioning Data Directory"),
        gr.Number(label="分辨率 Resolution", value=1024),
        gr.Checkbox(label="启用潜在变量的缓存 Enable Cache Latents"),
        gr.Number(label="VAE处理的批次大小 VAE Batch Size", value=1),
        gr.Checkbox(label="将潜在变量缓存到磁盘 Cache Latents to Disk"),

        gr.Textbox(label="提示词文件格式 Caption Extension", value="txt"),
        gr.Dropdown(choices=["None", "150", "225"], label="最大Token长度 Max Token Length", value="225"),
        gr.Number(label="保留的标记数 Keep Tokens", value=0),
        gr.Number(label="标题的丢弃率 Caption Dropout Rate", value=0),
        gr.Number(label=" 每`n`个epoch丢弃标题 Caption Dropout Every N Epochs", value=0),
        gr.Number(label="标题内标签的丢弃率 Caption Tag Dropout Rate", value=0),
        gr.Number(label="Clip Skip", value=1),
        gr.Checkbox(label="启用去偏估计损失 Debiased Estimation Loss"),
        gr.Checkbox(label="启用加权标题 Weighted Captions"),
        gr.Checkbox(label="启用文本编码器输出缓存 Cache Text Encoder Outputs"),
        gr.Checkbox(label="将文本编码器输出缓存到磁盘 Cache Text Encoder Outputs to Disk"),

        gr.Number(label="训练的批次大小 Train Batch Size", value=1),
        gr.Number(label="每份训练集重复训练次数 Dataset Repeats", value=1),
        gr.Number(label="最大训练轮数 Max Train Epochs", value=10),
        gr.Number(label="最大训练步数 Max Train Steps", value=0),

        gr.Dropdown(choices=["None", "float", "fp16", "bf16"], label="保存精度 Save Precision", value="float"),
        gr.Number(label="每`n`个epoch保存一次模型 Save Every N Epochs", value=1),
        gr.Number(label="每`n`步保存一次模型 Save Every N Steps", value=0),
        gr.Number(label="保存最近`n`个epoch的模型 Save Last N Epochs", value=1),
        gr.Number(label="保存最近`n`个epoch的模型状态 Save Last N Epochs State", value=0),
        gr.Number(label="保存最近`n`步的模型 Save Last N Steps", value=0),
        gr.Number(label="保存最近`n`步的模型状态 Save Last N Steps State", value=0),
        gr.Checkbox(label="启用保存模型状态 Save State"),
        gr.Checkbox(label="在训练结束时保存模型状态 Save State on Train End"),
        gr.Textbox(label="保存目录 Output Directory"),
        gr.Textbox(label="模型名字 Output Name", value="cnModel"),

        gr.Dropdown(choices=["AdamW", "Lion"], label="优化器 Optimizer Type", value="AdamW"),
        gr.Number(label="学习率 Learning Rate", value=0.0001),
        gr.Number(label="梯度剪裁的最大范数 Max Grad Norm", value=1),
        gr.Dropdown(choices=["constant_with_warmup", "cosine_with_restarts", "cosine"], label="学习率调度器 LR Scheduler", value="constant_with_warmup"),
        gr.Number(label="学习率预热步数 LR Warmup Steps", value=0),
        gr.Number(label="学习率重启轮数 LR Scheduler Num Cycles", value=1),
        gr.Number(label="LR Scheduler Power", value=1),

        gr.Number(label="cond_emb_dim", value=128),
        gr.Number(label="network_dim", value=64),

        gr.Dropdown(choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], label="日志信息在控制台的输出级别 Console Log Level", value="INFO"),
        gr.Textbox(label="日志文件夹 Logging Directory", value="./logs"),
        gr.Dropdown(choices=["tensorboard", "wandb"], label="日志模块 Log with", value="tensorboard"),
        gr.Textbox(label="日志前缀 Log Prefix"),
        gr.Textbox(label="日志跟踪器名称 Log Tracker Name"),
        gr.Textbox(label="WandB运行名称 WandB Run Name"),
        gr.Textbox(label="日志跟踪器的配置 Log Tracker Config"),
        gr.Textbox(label="WandB的API密钥 WandB API Key"),
        
        gr.Textbox(label="从指定的checkpoint恢复训练 Resume"),
        gr.Checkbox(label="启用内存高效的注意力机制 Memory Efficient Attention"),
        gr.Checkbox(label="使用Torch的编译功能 Torch Compile"),
        gr.Checkbox(label="启用xformers加速 Enable xformers"),
        gr.Number(label="数据加载器的最大工作线程数 Max Data Loader Workers", value=0),
        gr.Checkbox(label="启用持久的数据加载器工作线程 Persistent Data Loader Workers"),
        gr.Number(label="指定随机种子 Seed", value=20240724),
        gr.Checkbox(label="启用梯度检查点 Enable Gradient Checkpointing"),
        gr.Number(label="梯度累积的步数 Gradient Accumulation Steps", value=1),
        gr.Dropdown(choices=["no", "fp16", "bf16"], label="启用混合精度训练 Mixed Precision"),
        gr.Checkbox(label="启用完全fp16精度训练 Enable Full FP16"),
        gr.Checkbox(label="启用完全bf16精度训练 Enable Full BF16"),
        gr.Checkbox(label="启用fp8训练 Enable FP8 Base"),
        gr.Checkbox(label="启用低内存模式 Enable LowRAM"),
        gr.Checkbox(label="启用高显存模式 Enable HighVRAM"),
    ],
    outputs="text",
    title="controlnet_lllite模型训练（SDXL）",
    submit_btn="开始训练",
    clear_btn="清空参数",
    allow_flagging='never',  # 禁用 Flag 按钮
    description="输入各个参数训练属于你的controlnet_lllite模型"
)

## 数据
待处理文件应以因子名为文件名,以股票代码为列名,以交易日期为行名。示例如下: 

```
           000001.SZ  000002.SZ  000004.SZ  ...  873726.BJ  873806.BJ  873833.BJ                                                                   
20130104  -0.276020        NaN   0.053705   ...   NaN        NaN        NaN      
20130107  -0.500000        NaN  -0.500000   ...   NaN        NaN        NaN      
20130108   0.030570        NaN  -0.500000   ...   NaN        NaN        NaN      
20130109   0.163462        NaN  -0.155100   ...   NaN        NaN        NaN      
20130110   0.314939        NaN   0.068611   ...   NaN        NaN        NaN      
...        ...             ...   ...        ...   ...        ...        ...      
20240517  -0.500000       -0.5   0.036791   ...   0.120670  -0.500000  -0.033233      
20240520  -0.116451       -0.5  -0.500000   ...   0.290894  -0.116451   0.143433      
20240521  -0.500000       -0.5  -0.349625   ...   0.397940  -0.500000   0.324906      
20240522  -0.500000       -0.5  -0.194626   ...  -0.500000  -0.500000   0.415933      
20240523  -0.385976       -0.5  -0.112807   ...  -0.500000  -0.500000  -0.500000
```

使用data_construct.py处理后文件将以日期为文件名,以因子名为列名,以股票代码为行名。示例如下: 
```
            Alpha_001  Alpha_002  Alpha_003  ...  Alpha_099  Alpha_100  Alpha_101
000001.SZ  -0.276020  -0.005130  -0.272802   ...        0.0   0.002137  -0.621469
000002.SZ        NaN        NaN        NaN   ...        NaN        NaN        NaN
000004.SZ   0.053705   0.860182  -0.503675   ...       -1.0  -0.000466  -0.496454
000005.SZ   0.154871   0.350105   0.655695   ...        0.0   0.000811  -0.360360
000006.SZ  -0.276020  -0.442935   0.128912   ...        0.0   0.001855  -0.447761
...         ...        ...        ...        ...        ...        ...        ...
688798.SH        NaN        NaN        NaN   ...        NaN        NaN        NaN
688799.SH        NaN        NaN        NaN   ...        NaN        NaN        NaN
688800.SH        NaN        NaN        NaN   ...        NaN        NaN        NaN
688819.SH        NaN        NaN        NaN   ...        NaN        NaN        NaN
688981.SH        NaN        NaN        NaN   ...        NaN        NaN        NaN
```

## 如何使用

### 1. data_construct.py

该脚本用于从指定文件夹中预处理包含量价因子、基本面因子和预测目标文件的数据,预处理操作包括股票代码与日期的对齐、数据的拆分组织和数据的标准化(待实现)。处理后的数据将保存在指定的保存文件夹下的对应子目录。




```
data_construct.py [--log_folder LOG_FOLDER] [--log_name LOG_NAME]   
                  --quantity_price_factor_folder QUANTITY_PRICE_FACTOR_FOLDER 
                  --fundamental_factor_folder FUNDAMENTAL_FACTOR_FOLDER 
                  --label_folder LABEL_FOLDER 
                  --save_folder SAVE_FOLDER
                  [--read_format {csv,pkl,parquet,feather}] 
                  [--save_format {csv,pkl,parquet,feather}]
```

#### 可选参数
--log_folder:
日志文件的保存文件夹。(若当前目录下不存在该文件夹则创建之)
默认值: "log"   
--log_name:
日志文件的名称。
默认值: "data_construct.log"  
--read_format:
输入文件的格式。支持的格式有 csv、pkl、parquet 和 feather。
默认值: pkl  
--save_format:
输出文件的保存格式。支持的格式有 csv、pkl、parquet 和 feather。
默认值: pkl  

#### 必需参数
--quantity_price_factor_folder:
包含量价因子文件的文件夹路径。  
--fundamental_factor_folder:
包含基本面因子 pickle 文件的文件夹路径。  
--label_folder:
包含标签 pickle 文件的文件夹路径。  
--save_folder:
处理后的结果保存的文件夹路径,子目录 quantity_price_feature, fundamental_feature 和 label 将在该文件夹中创建。


### 2. dataset.py 
该工具用于从指定文件夹中获取数量-价格特征数据、基本面特征数据和标签数据,并生成训练、验证和测试数据集。数据集将根据用户指定的比例进行划分,并保存为指定格式的文件。

```
dataset.py [--log_folder LOG_FOLDER] [--log_name LOG_NAME] 
           [--quantity_price_feature_dir QUANTITY_PRICE_FEATURE_DIR]
           [--fundamental_feature_dir FUNDAMENTAL_FEATURE_DIR] 
           [--label_dir LABEL_DIR] 
           [--data_dir DATA_DIR] 
           [--file_format {csv,pkl,parquet,feather}]        
            --label_name LABEL_NAME 
           [--split_ratio SPLIT_RATIO SPLIT_RATIO SPLIT_RATIO] 
           [--mask_len MASK_LEN]
           [--mode {convert,drop,loose_drop}] 
           [--cat CAT] 
           [--dtype {FP32,FP64,FP16,BF16}] 
            --train_seq_len TRAIN_SEQ_LEN 
           [--val_seq_len VAL_SEQ_LEN]
           [--test_seq_len TEST_SEQ_LEN] 
            --save_path SAVE_PATH
```


#### 可选参数
--log_folder:
日志文件保存的文件夹路径。
默认值: "log"  
--log_name:
日志文件的名称。
默认值: "dataset.log"
--quantity_price_feature_dir:
包含量价特征数据文件的文件夹路径。
默认值: None  
--fundamental_feature_dir:
包含基本面特征数据文件的文件夹路径。
默认值: None  
--label_dir:
包含标签数据文件的文件夹路径。
默认值: None  
--data_dir:
数据文件夹路径。如果指定该路径DATA_DIR,相当于指定量价数据目录为 DATA_DIR/quantity_price_feature,基本面数据目录为 DATA_DIR/fundamental_feature,标签目录为 DATA_DIR/label。(上述分别指定将失效)
默认值: None  
--file_format:
输入文件的格式。支持的格式包括 csv、pkl、parquet 和 feather。
默认值: pkl  
--split_ratio:
训练集、验证集和测试集的划分比例。
默认值: [0.7, 0.2, 0.1] (分别对应70%训练集,20%验证集,10%测试集)  
--mask_len:
在序列预测任务中用于防止模型作弊的遮罩长度。例如在预测20日收益时,可以设置遮罩长度为20天。
默认值: 0  
--mode:
处理缺失值和无穷大值的模式。(默认值: loose_drop)  
支持三种模式: 
- convert: 将 NaN 转换为 0,将 Inf 转换为输入数据类型中可表示的最大有限值。
- drop: 删除序列中包含 NaN 或 Inf 值的股票代码。
- loose_drop: 仅在序列中所有横截面均出现 NaN 或 Inf 的股票代码上执行删除操作。  

--cat:
是否将数量-价格特征与基本面特征进行拼接。
默认值: True  
--dtype:
数据张量的类型。支持 FP32、FP64、FP16 或 BF16。
默认值: FP32  
--val_seq_len:
验证集的序列长度(天数)。如果未指定,默认为与训练集相同的长度。
默认值: None  
--test_seq_len:
测试集的序列长度(天数)。如果未指定,默认为与训练集相同的长度。
默认值: None  

#### 必需参数
--label_name:
目标标签的列名(标签文件中的列名)。
--train_seq_len:
训练集的序列长度(天数)。
--save_path:
划分后的数据集train-val-test字典的pt文件保存路径。


### 3.train_AttnFactorVAE.py

#### 从外部文件导入配置
```
train_AttnFactorVAE.py [--log_folder LOG_FOLDER] [--log_name LOG_NAME]  
                        --load_configs LOAD_CONFIGS
                       [--save_configs SAVE_CONFIGS]
```
**参数**  
--log_folder: 日志文件保存路径(默认: "log")  
--log_name: 日志文件名称(默认: train_AttnFactorVAE.log)  
--load_configs: 读取配置文件的路径  
--save_configs: 保存配置文件的路径(默认保存为 save_folder 中的 config.json)  


配置文件示例: 
```
{
    "Model": {
        "type": "AttnFactorVAE",
        "fundamental_feature_size": 31,
        "quantity_price_feature_size": 101,
        "num_gru_layers": 4,
        "gru_hidden_size": 32,
        "hidden_size": 100,
        "latent_size": 48,
        "gru_dropout": 0.1,
        "std_activation": "softplus",
        "checkpoint_path": null
    },
    "VAE_Optimizer": {
        "optimizer_type": "Lion",
        "optimizer_kwargs": {},
        "learning_rate": 0.0001,
        "lr_scheduler_type": "linear",
        "lr_scheduler_warmup_steps": 0,
        "lr_scheduler_num_cycles": 0.5,
        "lr_scheduler_power": 1.0
    },
    "Predictor_Optimizer": {
        "optimizer_type": "Lion",
        "optimizer_kwargs": {},
        "learning_rate": 0.0001,
        "lr_scheduler_type": "linear",
        "lr_scheduler_warmup_steps": 0,
        "lr_scheduler_num_cycles": 0.5,
        "lr_scheduler_power": 1.0
    },
    "Objective_Loss": {
        "gamma": 1,
        "scale": 100
    },
    "Dataset": {
        "dataset_path": "data\\dataset.pt",
        "num_workers": 4,
        "shuffle": true,
        "num_batches_per_epoch": 20,
        "mode": "loose_drop",
        "seq_len": 20
    },
    "Train": {
        "max_epoches": 40,
        "grad_clip_norm": -1,
        "grad_clip_value": -1,
        "detect_anomaly": true,
        "device": "cuda",
        "dtype": "FP32",
        "log_folder": "log",
        "sample_per_batch": 300,
        "report_per_epoch": 1,
        "save_per_epoch": 1,
        "save_folder": "model\\AttnFactorVAE\\test1",
        "save_name": "AttnFactorVAE",
        "save_format": ".pt"
    }
}
```

#### 使用命令行参数设定配置
```
train_AttnFactorVAE.py [--log_folder LOG_FOLDER] [--log_name LOG_NAME] [--save_configs SAVE_CONFIGS]
                        --dataset_path DATASET_PATH --num_workers NUM_WORKERS --shuffle SHUFFLE --num_batches_per_epoch NUM_BATCHES_PER_EPOCH 
                        --checkpoint_path CHECKPOINT_PATH --quantity_price_feature_size QUANTITY_PRICE_FEATURE_SIZE --fundamental_feature_size FUNDAMENTAL_FEATURE_SIZE --num_gru_layers NUM_GRU_LAYERS --gru_hidden_size GRU_HIDDEN_SIZE --hidden_size HIDDEN_SIZE --latent_size LATENT_SIZE --gru_dropout GRU_DROPOUT --std_activation {exp,softplus}   
                        --optimizer_type {Adam,AdamW,Lion,SGDNesterov,DAdaptation,Adafactor} --optimizer_kwargs OPTIMIZER_KWARGS [OPTIMIZER_KWARGS ...] --learning_rate LEARNING_RATE --lr_scheduler_type {constant,linear,cosine,cosine_with_restarts,polynomial,adafactor} --lr_scheduler_warmup_steps LR_SCHEDULER_WARMUP_STEPS --lr_scheduler_num_cycles LR_SCHEDULER_NUM_CYCLES --lr_scheduler_power LR_SCHEDULER_POWER 
                       [--vae_optimizer_type {Adam,AdamW,Lion,SGDNesterov,DAdaptation,Adafactor} --vae_optimizer_kwargs VAE_OPTIMIZER_KWARGS [VAE_OPTIMIZER_KWARGS ...] --vae_learning_rate VAE_LEARNING_RATE --vae_lr_scheduler_type {constant,linear,cosine,cosine_with_restarts,polynomial,adafactor} --vae_lr_scheduler_warmup_steps VAE_LR_SCHEDULER_WARMUP_STEPS --vae_lr_scheduler_num_cycles VAE_LR_SCHEDULER_NUM_CYCLES --vae_lr_scheduler_power VAE_LR_SCHEDULER_POWER]
                       [--predictor_optimizer_type {Adam,AdamW,Lion,SGDNesterov,DAdaptation,Adafactor} --predictor_optimizer_kwargs PREDICTOR_OPTIMIZER_KWARGS [PREDICTOR_OPTIMIZER_KWARGS ...] --predictor_learning_rate PREDICTOR_LEARNING_RATE --predictor_lr_scheduler_type {constant,linear,cosine,cosine_with_restarts,polynomial,adafactor} --predictor_lr_scheduler_warmup_steps PREDICTOR_LR_SCHEDULER_WARMUP_STEPS --predictor_lr_scheduler_num_cycles PREDICTOR_LR_SCHEDULER_NUM_CYCLES --predictor_lr_scheduler_power PREDICTOR_LR_SCHEDULER_POWER] 
                        --gamma GAMMA --scale SCALE 
                        --max_epoches MAX_EPOCHES --grad_clip_norm GRAD_CLIP_NORM --grad_clip_value GRAD_CLIP_VALUE --detect_anomaly DETECT_ANOMALY --dtype {FP32,FP64,FP16,BF16} --device {auto,cuda,cpu} --sample_per_batch SAMPLE_PER_BATCH --report_per_epoch REPORT_PER_EPOCH --save_per_epoch SAVE_PER_EPOCH --save_folder SAVE_FOLDER --save_name SAVE_NAME --save_format SAVE_FORMAT

```


该脚本接受多种命令行参数来配置训练过程,以下是关键参数的简要说明: 

**通用参数**  
--log_folder: 日志文件保存路径(默认: "log")  
--log_name: 日志文件名称(默认: train_AttnFactorVAE.log)  
--save_configs: 保存配置文件的路径(默认保存为 save_folder 中的 config.json)  


**数据加载参数**  
--dataset_path: 数据集 .pt 文件的路径  
--num_workers: 数据加载时使用的子进程数量(默认: 4)  
--shuffle: 是否在加载数据时打乱顺序(默认: True)  
--num_batches_per_epoch: 每个 epoch 中从所有批次中采样的批次数量(默认: -1,表示使用所有批次)  

**模型参数**  
--checkpoint_path: 加载 checkpoint 的路径(可选)  
--quantity_price_feature_size: 量价特征的输入维度  
--fundamental_feature_size: 基本面特征的输入维度  
--num_gru_layers: GRU层数  
--gru_hidden_size: GRU每层的隐藏层维度  
--hidden_size: VAE编码器、预测器和解码器的隐藏层维度  
--latent_size: VAE编码器、预测器和解码器的潜在空间维度(因子数量)   
--gru_dropout: GRU层的 dropout 概率(默认: 0.1)  
--std_activation: 标准差计算的激活函数(默认: exp, 可选: exp, softplus)  

**优化器参数**  
--optimizer_type: 优化器类型(默认: Lion,可选: Adam、AdamW、Lion、SGDNesterov、DAdaptation、Adafactor)   
--optimizer_kwargs: 优化器的其他参数。可以以关键字参数的方式传入。(可选)  
--learning_rate: 优化器的学习率(默认: 0.001)  
--lr_scheduler_type: 学习率调度器类型(默认: constant)  
--lr_scheduler_warmup_steps: 学习率调度器预热步数(默认: 0)  
--lr_scheduler_num_cycles: cosine学习率调度中的波数。(默认为 0.5)  
--lr_scheduler_power polynomial学习率调度中多项式的幂。(默认为 1)  

你也可以分别指定VAE模块(包括feature_extractor, encoder和decoder)和predictor模块的优化器参数。这两个模块会使用不同的优化器进行参数的更新。在指定上述参数时,相当于同时将相同参数赋给VAE模块和predictor模块的优化器。  

--vae_optimizer_type: VAE优化器类型(默认: Lion,可选: Adam、AdamW、Lion、SGDNesterov、DAdaptation、Adafactor)   
--vae_optimizer_kwargs: VAE优化器的其他参数。可以以关键字参数的方式传入。(可选)  
--vae_learning_rate: VAE优化器的学习率(默认: 0.001)  
--vae_lr_scheduler_type: VAE学习率调度器类型(默认: constant)  
--vae_lr_scheduler_warmup_steps: VAE学习率调度器预热步数(默认: 0)  
--vae_lr_scheduler_num_cycles: 使用cosine类型时,VAE优化器学习率调度中的波数。(默认为 0.5)  
--vae_lr_scheduler_power 使用polynomial类型时,VAE优化器学习率调度中多项式的幂。(默认为 1)  

--predictor_optimizer_type: predictor优化器类型(默认: Lion, 可选: Adam、AdamW、Lion、SGDNesterov、DAdaptation、Adafactor)   
--predictor_optimizer_kwargs: predictor优化器的其他参数。可以以关键字参数的方式传入。(可选)  
--predictor_learning_rate: predictor优化器的学习率(默认: 0.001)  
--predictor_lr_scheduler_type: predictor学习率调度器类型(默认: constant)  
--predictor_lr_scheduler_warmup_steps: predictor学习率调度器预热步数(默认: 0)  
--predictor_lr_scheduler_num_cycles: 使用cosine类型时,predictor优化器学习率调度中的波数。(默认为 0.5)  
--predictor_lr_scheduler_power 使用polynomial类型时,predictor优化器学习率调度中多项式的幂。(默认为 1)  

**训练参数**    
--gamma: KL 散度在损失函数中的系数(默认: 1)   
--scale: MSE 损失的比例系数(默认: 100)  
--max_epoches: 最大训练 epoch 数(默认: 20)  
--grad_clip_value: 梯度裁剪的绝对值限制(-1表示禁用,默认: -1)  
--grad_clip_norm: 梯度裁剪的范数限制(-1表示禁用,默认: -1)  
--detect_anomaly: 是否启用异常检测。为 autograd 引擎启用异常检测,在启用检测的情况下运行正向传递将允许反向传递打印创建失败反向函数的正向操作的回溯,则任何生成nan值的向后计算都会引发错误。(默认: False)
--dtype: 张量数据类型(默认: FP32, 可选: FP32、FP64、FP16、BF16)  
--device: 计算设备(默认: cuda,可选: cuda 或 cpu)  
--sample_per_batch: 每 n 个批次检查一次样本数据(0表示禁用,默认: 0)  
--report_per_epoch: 每 n 个 epoch 报告一次训练损失和验证损失(默认: 1)  
--save_per_epoch: 每 n 个 epoch 保存一次模型权重(默认: 1)  
--save_folder: 保存模型的文件夹路径  
--save_name: 保存模型的名称(默认: Model)
--save_format: 保存模型的文件格式(默认: .pt, 可选: .pt、.safetensor)

### 4. train_FactorVAE.py

#### 从外部文件导入配置
```
train_AttnFactorVAE.py [--log_folder LOG_FOLDER] [--log_name LOG_NAME]  
                        --load_configs LOAD_CONFIGS
                       [--save_configs SAVE_CONFIGS]
```
**参数**  
--log_folder: 日志文件保存路径(默认: "log")  
--log_name: 日志文件名称(默认: train_AttnFactorVAE.log)  
--load_configs: 读取配置文件的路径  
--save_configs: 保存配置文件的路径(默认保存为 save_folder 中的 config.json)  


配置文件示例: 
```
{
    "Model": {
        "type": "FactorVAE",
        "feature_size": 132,
        "num_gru_layers": 4,
        "gru_hidden_size": 32,
        "hidden_size": 100,
        "latent_size": 48,
        "gru_dropout": 0.1,
        "std_activation": "softplus",
        "checkpoint_path": "model\\AttnFactorVAE\\test\\AttnFactorVAE_epoch11.pt"
    },
    "VAE_Optimizer": {
        "optimizer_type": "Lion",
        "optimizer_kwargs": {},
        "learning_rate": 0.0001,
        "lr_scheduler_type": "linear",
        "lr_scheduler_warmup_steps": 0,
        "lr_scheduler_num_cycles": 0.5,
        "lr_scheduler_power": 1.0
    },
    "Predictor_Optimizer": {
        "optimizer_type": "Lion",
        "optimizer_kwargs": {},
        "learning_rate": 0.0001,
        "lr_scheduler_type": "linear",
        "lr_scheduler_warmup_steps": 0,
        "lr_scheduler_num_cycles": 0.5,
        "lr_scheduler_power": 1.0
    },
    "Objective_Loss": {
        "gamma": 1,
        "scale": 100
    },
    "Dataset": {
        "dataset_path": "data\\dataset_cat.pt",
        "num_workers": 4,
        "shuffle": true,
        "num_batches_per_epoch": 20,
        "mode": "loose_drop",
        "seq_len": 20
    },
    "Train": {
        "max_epoches": 40,
        "grad_clip_norm": 5,
        "grad_clip_value": -1,
        "detect_anomaly": true,
        "device": "cuda",
        "dtype": "FP32",
        "log_folder": "log",
        "sample_per_batch": 1,
        "report_per_epoch": 1,
        "save_per_epoch": 1,
        "save_folder": "model\\FactorVAE\\test2",
        "save_name": "FactorVAE",
        "save_format": ".pt"
    }
}
```

#### 使用命令行参数设定配置
```
train_FactorVAE.py [--log_folder LOG_FOLDER] [--log_name LOG_NAME] [--save_configs SAVE_CONFIGS]
                    --dataset_path DATASET_PATH --num_workers NUM_WORKERS --shuffle SHUFFLE --num_batches_per_epoch NUM_BATCHES_PER_EPOCH 
                    --checkpoint_path CHECKPOINT_PATH --feature_size FEATURE_SIZE --num_gru_layers NUM_GRU_LAYERS --gru_hidden_size GRU_HIDDEN_SIZE --hidden_size HIDDEN_SIZE --latent_size LATENT_SIZE --gru_dropout GRU_DROPOUT --std_activation {exp,softplus}   
                    --optimizer_type {Adam,AdamW,Lion,SGDNesterov,DAdaptation,Adafactor} --optimizer_kwargs OPTIMIZER_KWARGS [OPTIMIZER_KWARGS ...] --learning_rate LEARNING_RATE --lr_scheduler_type {constant,linear,cosine,cosine_with_restarts,polynomial,adafactor} --lr_scheduler_warmup_steps LR_SCHEDULER_WARMUP_STEPS --lr_scheduler_num_cycles LR_SCHEDULER_NUM_CYCLES --lr_scheduler_power LR_SCHEDULER_POWER 
                   [--vae_optimizer_type {Adam,AdamW,Lion,SGDNesterov,DAdaptation,Adafactor} --vae_optimizer_kwargs VAE_OPTIMIZER_KWARGS [VAE_OPTIMIZER_KWARGS ...] --vae_learning_rate VAE_LEARNING_RATE --vae_lr_scheduler_type {constant,linear,cosine,cosine_with_restarts,polynomial,adafactor} --vae_lr_scheduler_warmup_steps VAE_LR_SCHEDULER_WARMUP_STEPS --vae_lr_scheduler_num_cycles VAE_LR_SCHEDULER_NUM_CYCLES --vae_lr_scheduler_power VAE_LR_SCHEDULER_POWER]
                   [--predictor_optimizer_type {Adam,AdamW,Lion,SGDNesterov,DAdaptation,Adafactor} --predictor_optimizer_kwargs PREDICTOR_OPTIMIZER_KWARGS [PREDICTOR_OPTIMIZER_KWARGS ...] --predictor_learning_rate PREDICTOR_LEARNING_RATE --predictor_lr_scheduler_type {constant,linear,cosine,cosine_with_restarts,polynomial,adafactor} --predictor_lr_scheduler_warmup_steps PREDICTOR_LR_SCHEDULER_WARMUP_STEPS --predictor_lr_scheduler_num_cycles PREDICTOR_LR_SCHEDULER_NUM_CYCLES --predictor_lr_scheduler_power PREDICTOR_LR_SCHEDULER_POWER] 
                    --gamma GAMMA --scale SCALE 
                    --max_epoches MAX_EPOCHES --grad_clip_norm GRAD_CLIP_NORM --grad_clip_value GRAD_CLIP_VALUE --detect_anomaly DETECT_ANOMALY --dtype {FP32,FP64,FP16,BF16} --device {auto,cuda,cpu} --sample_per_batch SAMPLE_PER_BATCH --report_per_epoch REPORT_PER_EPOCH --save_per_epoch SAVE_PER_EPOCH --save_folder SAVE_FOLDER --save_name SAVE_NAME --save_format SAVE_FORMAT

```


该脚本接受多种命令行参数来配置训练过程,以下是关键参数的简要说明: 

**通用参数**  
--log_folder: 日志文件保存路径(默认: "log")  
--log_name: 日志文件名称(默认: train_FactorVAE.log)  
--save_configs: 保存配置文件的路径(默认保存为 save_folder 中的 config.json)  

**数据加载参数**  
--dataset_path: 数据集 .pt 文件的路径  
--num_workers: 数据加载时使用的子进程数量(默认: 4)  
--shuffle: 是否在加载数据时打乱顺序(默认: True)  
--num_batches_per_epoch: 每个 epoch 中从所有批次中采样的批次数量(默认: -1,表示使用所有批次)  

**模型参数**  
--checkpoint_path: 加载 checkpoint 的路径(可选)  
--quantity_price_feature_size: 量价特征的输入维度  
--fundamental_feature_size: 基本面特征的输入维度  
--num_gru_layers: GRU层数  
--gru_hidden_size: GRU每层的隐藏层维度  
--hidden_size: VAE编码器、预测器和解码器的隐藏层维度  
--latent_size: VAE编码器、预测器和解码器的潜在空间维度(因子数量)   
--gru_dropout: GRU层的 dropout 概率(默认: 0.1)  
--std_activation: 标准差计算的激活函数(默认: exp, 可选: exp, softplus)  

**优化器参数**  
--optimizer_type: 优化器类型(默认: Lion,可选: Adam、AdamW、Lion、SGDNesterov、DAdaptation、Adafactor)   
--optimizer_kwargs: 优化器的其他参数。可以以关键字参数的方式传入。(可选)  
--learning_rate: 优化器的学习率(默认: 0.001)  
--lr_scheduler_type: 学习率调度器类型(默认: constant)  
--lr_scheduler_warmup_steps: 学习率调度器预热步数(默认: 0)  
--lr_scheduler_num_cycles: cosine学习率调度中的波数。(默认为 0.5)  
--lr_scheduler_power polynomial学习率调度中多项式的幂。(默认为 1)  

你也可以分别指定VAE模块(包括feature_extractor, encoder和decoder)和predictor模块的优化器参数。这两个模块会使用不同的优化器进行参数的更新。在指定上述参数时,相当于同时将相同参数赋给VAE模块和predictor模块的优化器。  

--vae_optimizer_type: VAE优化器类型(默认: Lion,可选: Adam、AdamW、Lion、SGDNesterov、DAdaptation、Adafactor)   
--vae_optimizer_kwargs: VAE优化器的其他参数。可以以关键字参数的方式传入。(可选)  
--vae_learning_rate: VAE优化器的学习率(默认: 0.001)  
--vae_lr_scheduler_type: VAE学习率调度器类型(默认: constant)  
--vae_lr_scheduler_warmup_steps: VAE学习率调度器预热步数(默认: 0)  
--vae_lr_scheduler_num_cycles: 使用cosine类型时,VAE优化器学习率调度中的波数。(默认为 0.5)  
--vae_lr_scheduler_power 使用polynomial类型时,VAE优化器学习率调度中多项式的幂。(默认为 1)  

--predictor_optimizer_type: predictor优化器类型(默认: Lion, 可选: Adam、AdamW、Lion、SGDNesterov、DAdaptation、Adafactor)   
--predictor_optimizer_kwargs: predictor优化器的其他参数。可以以关键字参数的方式传入。(可选)  
--predictor_learning_rate: predictor优化器的学习率(默认: 0.001)  
--predictor_lr_scheduler_type: predictor学习率调度器类型(默认: constant)  
--predictor_lr_scheduler_warmup_steps: predictor学习率调度器预热步数(默认: 0)  
--predictor_lr_scheduler_num_cycles: 使用cosine类型时,predictor优化器学习率调度中的波数。(默认为 0.5)  
--predictor_lr_scheduler_power 使用polynomial类型时,predictor优化器学习率调度中多项式的幂。(默认为 1)  

**训练参数**    
--gamma: KL 散度在损失函数中的系数(默认: 1)   
--scale: MSE 损失的比例系数(默认: 100)  
--max_epoches: 最大训练 epoch 数(默认: 20)  
--grad_clip_value: 梯度裁剪的绝对值限制(-1表示禁用,默认: -1)  
--grad_clip_norm: 梯度裁剪的范数限制(-1表示禁用,默认: -1)  
--detect_anomaly: 是否启用异常检测。为 autograd 引擎启用异常检测,在启用检测的情况下运行正向传递将允许反向传递打印创建失败反向函数的正向操作的回溯,则任何生成nan值的向后计算都会引发错误。(默认: False)
--dtype: 张量数据类型(默认: FP32, 可选: FP32、FP64、FP16、BF16)  
--device: 计算设备(默认: cuda,可选: cuda 或 cpu)  
--sample_per_batch: 每 n 个批次检查一次样本数据(0表示禁用,默认: 0)  
--report_per_epoch: 每 n 个 epoch 报告一次训练损失和验证损失(默认: 1)  
--save_per_epoch: 每 n 个 epoch 保存一次模型权重(默认: 1)  
--save_folder: 保存模型的文件夹路径  
--save_name: 保存模型的名称(默认: Model)
--save_format: 保存模型的文件格式(默认: .pt, 可选: .pt、.safetensor)

#### 其他
- 权重迁移: checkpoint_path不仅可以指向FactorVAE模型权重文件，也可以指向AttnFactorVAE模型文件。若AttnFactorVAE模型文件目录下存在配置文件config.json，则检查之，若配置支持迁移，则会加载AttnFactorVAE模型中的encoder、decoder、predictor模块的权重，只随机初始化feature_extractor模块。而若配置不支持或尝试加载模型权重失败，则不会进行权重加载。
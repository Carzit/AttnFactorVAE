
# AttnFactorVAE

[![Open Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Carzit/AttnFactorVAE/blob/dev/demo/colab_demo.ipynb)

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

### 1. 数据构建与预处理
使用`data_construct.py`进行数据构建与预处理。   
该脚本用于从指定文件夹中预处理包含量价因子、基本面因子和预测目标文件的数据,预处理操作包括股票代码与日期的对齐以及数据的拆分组织。处理后的数据将保存在指定的保存文件夹下的对应子目录。

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

### 2. 数据标准化
使用`standardization.py`进行标准化（可选）。有5种可用的标准化形式，包括：
* 截面Z-Score 标准化（CSZScore） 对所有数据按日期聚合后进行 Z-Score 处理，主要目的在于保证每日横截面数据的可比性。Z-score = (x - μ)/σ
* 截面排序标准化（CSRank） 对所有数据按日期聚合后进行排序处理，将排序结果作为模型输入。此方法主要目的在于排除异常值的影响，但缺点也很明显，丧失了数据间相对大小关系的刻画。
* 数据集整体Z-Score标准化（ZScore） 截面标准化会使数据损失时序变化信息，而整个数据集做标准化可以将不同日期的相对大小关系也喂入模型进行学习。当然此处需要注意数据泄露问题，我们使用训练集算出均值和标准差后，将其用于整个数据集进行标准化。
* 数据集整体 Minmax 标准化（MinMax） 相较于 ZScore 标准化而言，MinMax 能使数据严格限制在规定的上下限范围内，且保留了数据间的大小关系。
* 数据集整体 Robust Z-Score 标准化（RobustZScore） 由于标准差的计算需要对数据均值偏差进行平方运算，会使数据对极值更敏感。而Mad = Median(|x-Median(x)|)能有效解决这一问题，使得到的均值标准差指标更加稳健。Robust-Z-score = (x - Median(x))/Mad(x)

```
standardization.py [--log_path LOG_PATH]
                    --preprocess_data_folder PREPROCESS_DATA_FOLDER
                    --save_folder SAVE_FOLDER
                   [--standardization_method {CSZScore,CSRank,ZScore,MinMax,RobustZScore}]
                   [--read_format {csv,pkl,parquet,feather}]
                   [--save_format {csv,pkl,parquet,feather}]
```

#### 可选参数
--log_path:
日志文件保存的文件夹路径及文件名。
默认值："log/standardization.log"

--standardization_method:
标准化方法。可选CSZScore,CSRank,ZScore,MinMax,RobustZScore。

--read_format:
输入文件的格式。支持的格式包括 csv、pkl、parquet 和 feather。
默认值: pkl

--save_format:
输出文件的格式。支持的格式包括 csv、pkl、parquet 和 feather。
默认值: pkl

#### 必需参数
--preprocess_data_folder:
包含业已预处理数据的文件夹路径。请保证这一文件夹下有fundamental features, quantity prices & labels。

--save_folder:
保存文件夹路径。将会在该文件夹下自动生成对应的子文件夹。

### 3. 数据集构建
使用`dataset.py`进行数据集构建。该工具用于从指定文件夹中获取量价特征数据、基本面特征数据和标签数据,并生成训练、验证和测试数据集。数据集将根据用户指定的比例进行划分,并保存为指定格式的文件。

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
是否将量价特征与基本面特征进行拼接。注意对于FactorVAE模型使用的数据集，该参数应为True；对于AttnFactorVAE和AttnRet模型使用的数据集，该参数应为False。 
默认值: False  
--dtype:
数据张量的精度类型。支持 FP32、FP64、FP16 或 BF16。
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


### 4. 训练

使用`train_AttnFactorVAE.py`进行AttnFactorVAE模型的训练。训练参数有两种载入途径：从外部文件导入配置和使用命令行参数设定配置。  

#### 4.1 从外部文件导入配置
```
train_AttnFactorVAE.py [--log_path LOG_PATH]
                        --load_configs LOAD_CONFIGS
                       [--save_configs SAVE_CONFIGS]
```

--log_folder: 日志文件保存路径(默认: "log/train_AttnFactorVAE.log")   
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

#### 4.2 使用命令行参数设定配置
```
train_AttnFactorVAE.py [--log_path LOG_PATH] [--save_configs SAVE_CONFIGS]
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
--log_path: 日志文件保存路径(默认: "log/train_AttnFactorVAE.log")   
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

你也可以分别指定VAE模块(包括feature_extractor, encoder和decoder)和predictor模块的优化器参数。这两个模块会使用不同的优化器进行参数的更新。在指定上述参数时,相当于同时将相同参数赋给VAE模块和predictor模块的优化器（例如：`--optimizer_type Adam`等价于`--vae_optimizer_type Adam --predictor_optimizer_type Adam`）。  

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
--dtype: 张量数据精度类型(默认: FP32, 可选: FP32、FP64、FP16、BF16)  
--device: 计算设备(默认: cuda,可选: cuda 或 cpu)  
--sample_per_batch: 每 n 个批次检查一次样本数据(0表示禁用,默认: 0)  
--report_per_epoch: 每 n 个 epoch 报告一次训练损失和验证损失(默认: 1)  
--save_per_epoch: 每 n 个 epoch 保存一次模型权重(默认: 1)  
--save_folder: 保存模型的文件夹路径  
--save_name: 保存模型的名称(默认: Model)
--save_format: 保存模型的文件格式(默认: .pt, 可选: .pt、.safetensor)

#### 4.3 AttnRet和FactorVAE
为了方便进行模型性能的比较，我们同时提供了AttnRet(RiskAttention)和FactorVAE模型的训练代码：`train_AttnRet.py`和`train_FactorVAE.py`。使用方式与`train_AttnFactorVAE.py`相同。  

但注意，由于模型不尽相同，模型相关的超参数存在差异。具体地表现为：
- AttnRet模型由于将VAE模块替换为MLP,因此没有AttnFactorVAE模型中的hidden_size, latent_size和std_activation参数，而是新增了num_fc_layers参数；
- FactorVAE模型则因为使用无注意力机制的特征提取器，因此不对feature种类进行区分，因此没有AttnFactorVAE模型中的quantity_price_feature_size和fundamental_feature_size参数，而是使用feature_size参数（这个参数的值即quantity_price_feature_size和fundamental_feature_size参数之和）。
- AttnRet只有单一的全局优化器，而不是像AttnFactorVAE和FactorVAE一样设置两个优化器分别优化VAE模块和predictor模块。因此AttnRet训练时没有形如`--vae_optimizer_**`、`--predictor_optimizer_**`的命令行参数，配置文件夹中也没有`"VAE_Optimizer"`、`"Predictor_VAE"`键，取而代之的是`"Optimizer"`。

您可以查看本repository的configs文件夹下的`config_train_AttnRet.json`和`config_train_FactorVAE.json`配置文件示例；同时，您也可以使用`python train_AttnRet.py`和`python train_FactorVAE.py -h`命令以查看命令行参数的详细信息。在此不做赘述。  


#### 4.4 其他
**权重迁移：** 由于AttnFactor, AttnRet和FactorVAE之间的模型架构存在相同的部分，出于加速训练和增加模型可比性的目的，上述模型的训练代码支持不同模型间相同模块部分的权重迁移，包括AttnFactor和AttnRet之间的AttnFeatureExtractor模块、AttnFactorVAE和FactorVAE之间的encoder、decoder和predictor模型。例如，在FactorVAE训练时，checkpoint_path不仅可以指向FactorVAE模型权重文件，也可以指向AttnFactorVAE模型文件。若AttnFactorVAE模型文件目录下存在配置文件config.json，则检查之，若配置支持迁移（形状匹配），则会加载AttnFactorVAE模型中的encoder、decoder、predictor模块的权重，只随机初始化feature_extractor模块。而若配置不支持或尝试加载模型权重失败，则不会进行权重加载。  

**多优化器：** 我们对于AttnFactorVAE和FactorVAE的训练设置了两个优化器，分别对VAE模块(包括attn_feature_extractor/feature_extractor, encoder和decoder模块)和predictor模块进行更新。设置多个优化器对模型的不同模块进行优化并不会改变张量的计算图和梯度的值，但对于一些内部保存全局动量的优化器如Adam，这部分的值可能会有一点差异，会与单一的全局优化器有区别。简单地，将VAE模块和Predictor模块的优化器参数设置为相同的参数（我们提供了快速这样实现的arguments选项）可以近似地视为设置了一个单一全局优化器。同时，将其中一个优化器的lr设为0可以视为冻结该模块的权重。  

### 5. 评估

使用`eval_AttnFactorVAE.py`进行AttnFactorVAE模型的训练。训练参数有两种载入途径：从外部文件导入配置和使用命令行参数设定配置。  

#### 5.1 从外部文件导入配置
```
eval_AttnFactorVAE.py [--log_path LOG_PATH]
                       --load_configs LOAD_CONFIGS
                      [--save_configs SAVE_CONFIGS]
```

--log_folder: 日志文件保存路径(默认: "log/eval_AttnFactorVAE.log")   
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
        "std_activation": "softplus"
    },
    "Dataset": {
        "dataset_path": "data\\dataset.pt",
        "subset": "test",
        "num_workers": 4,
        "mode": "loose_drop",
        "seq_len": 20
    },
    "Eval": {
        "device": "cuda",
        "dtype": "FP32",
        "metric": "IC",
        "checkpoints": ["model\\AttnFactorVAE\\test_softmax\\AttnFactorVAE_epoch10.pt"],
        "checkpoint_folder": "model\\AttnFactorVAE\\test_softmax",
        "save_folder": "eval\\AttnFactorVAE\\test_softmax",
        "plot_index": [0]
    }
}
```

#### 5.2 使用命令行参数设定配置
```
eval_AttnFactorVAE.py [--log_path LOG_PATH] --save_configs SAVE_CONFIGS
                      --dataset_path DATASET_PATH --subset SUBSET --num_workers NUM_WORKERS [--checkpoints [CHECKPOINTS ...]] [--checkpoint_folder CHECKPOINT_FOLDER]   
                      --quantity_price_feature_size QUANTITY_PRICE_FEATURE_SIZE --fundamental_feature_size FUNDAMENTAL_FEATURE_SIZE
                      --num_gru_layers NUM_GRU_LAYERS --gru_hidden_size GRU_HIDDEN_SIZE --hidden_size HIDDEN_SIZE --latent_size LATENT_SIZE --std_activation STD_ACTIVATION 
                      --dtype {FP32,FP64,FP16,BF16} --device {auto,cuda,cpu}  --metric {MSE,IC,RankIC,ICIR,RankICIR} [--plot_index PLOT_INDEX [PLOT_INDEX ...]] --save_folder SAVE_FOLDER
```

该脚本接受多种命令行参数来配置训练过程,以下是关键参数的简要说明: 

**通用参数**  
--log_path: 日志文件保存路径(默认: "log/eval_AttnFactorVAE.log")   
--save_configs: 保存配置文件的路径(默认保存为 save_folder 中的 config.json)  

**数据加载参数**  
--dataset_path: 数据集 .pt 文件的路径   
--subset: 拆分后的数据子集(默认: test,可选: train、val、test)   
--num_workers: 数据加载时使用的子进程数量(默认: 4)

**模型参数**  
--quantity_price_feature_size: 量价特征的输入维度  
--fundamental_feature_size: 基本面特征的输入维度  
--num_gru_layers: GRU层数  
--gru_hidden_size: GRU每层的隐藏层维度  
--hidden_size: VAE编码器、预测器和解码器的隐藏层维度  
--latent_size: VAE编码器、预测器和解码器的潜在空间维度(因子数量)   
--std_activation: 标准差计算的激活函数(默认: exp, 可选: exp, softplus)  

**评估参数**    
--checkpoints: 加载 checkpoint 的路径(可选, 可以传入一个或多个地址)
--checkpoint_folder: checkpoint文件夹路径(可选，若指定则相当于checkpoints即为checkpoint_folder目录下所有pt文件和safetensors文件)
--dtype: 张量数据类型(默认: FP32, 可选: FP32、FP64、FP16、BF16)  
--device: 计算设备(默认: cuda, 可选: cuda 或 cpu)  
--metric: 评估度量(默认: IC, 可选: MSE, IC, RankIC, ICIR, RankIC)  
--plot_index：绘图索引(默认: 0, 可以传入一个或多个序号)  
--save_folder: 保存评估结果表格与可视化图像的文件夹路径  

#### 5.3 AttnRet和FactorVAE
相似地，为了方便进行模型性能的比较，我们同时提供了AttnRet(RiskAttention)和FactorVAE模型的评估代码：`eval_AttnRet.py`和`eval_FactorVAE.py`。使用方式与`eval_AttnFactorVAE.py`相同。  

但注意，由于模型不尽相同，模型相关的超参数存在差异。具体地表现为：
- AttnRet模型由于将VAE模块替换为MLP,因此没有AttnFactorVAE模型中的hidden_size, latent_size和std_activation参数，而是新增了num_fc_layers参数；
- FactorVAE模型则因为使用无注意力机制的特征提取器，因此不对feature种类进行区分，因此没有AttnFactorVAE模型中的quantity_price_feature_size和fundamental_feature_size参数，而是使用feature_size参数（这个参数的值即quantity_price_feature_size和fundamental_feature_size参数之和）。

另外，模型不再需要dropout相关的参数：在评估时所有dropout会冻结为0.

#### 5.4 其他
- 评估度量： 
- - `MSE`即预测结果与真实值的平方差之均值
- - `IC`为每个batch上(每个交易日)预测结果与真实值的Pearson相关系数之均值。

$$
\text{IC}_s = \frac{(r_{\hat{y}_s} - \mathbb{E}[r_{\hat{y}_s}])^T (r_{y_s} - \mathbb{E}[r_{y_s}])}{\text{std}(r_{\hat{y}_s}) \cdot \text{std}(r_{y_s})}
$$

$$
\text{IC} = \mathbb{E}[\text{IC}_s] = \frac{1}{T_{\text{test}}} \sum_{s=1}^{T_{\text{test}}} \text{IC}_s
$$

- - `RankIC`为每个batch上(每个交易日)预测结果与真实值的Spearman相关系数之均值。

$$
\text{RankIC}_s = \frac{(\text{rank}(r_{\hat{y}_s}) - \mathbb{E}[\text{rank}(r_{\hat{y}_s})])^T (\text{rank}(r_{y_s}) - \mathbb{E}[\text{rank}(r_{y_s})])}
{\text{std}(\text{rank}(r_{\hat{y}_s})) \cdot \text{std}(\text{rank}(r_{y_s}))}
$$

$$
\text{RankIC} = \mathbb{E}[\text{RankIC}_s] = \frac{1}{T_{\text{test}}} \sum_{s=1}^{T_{\text{test}}} \text{RankIC}_s
$$

- - `ICIR`为每个batch上(每个交易日)预测结果与真实值的Pearson相关系数之均值除以其标准差。

$$
\text{ICIR} = \frac{\mathbb{E}[\text{IC}_s]}{\text{std}(\text{IC}_s)}
$$

- - `RankICIR`为每个batch上(每个交易日)预测结果与真实值的Spearman相关系数之均值除以其标准差。

$$
\text{RankICIR} = \frac{\mathbb{E}[\text{RankIC}_s]}{\text{std}(\text{RankIC}_s)}
$$

### 6. 推理

使用`eval_AttnFactorVAE.py`进行AttnFactorVAE模型的训练。训练参数有两种载入途径：从外部文件导入配置和使用命令行参数设定配置。  

#### 6.1 从外部文件导入配置
```
infer_AttnFactorVAE.py [--log_path LOG_PATH]
                       --load_configs LOAD_CONFIGS
                       [--save_configs SAVE_CONFIGS]
```

--log_folder: 日志文件保存路径(默认: "log/infer_AttnFactorVAE.log")   
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
        "checkpoint_path": "model\\AttnFactorVAE\\test_softmax\\AttnFactorVAE_epoch11.pt"
    },
    "Dataset": {
        "dataset_path": "data\\dataset_loose_drop.pt",
        "subset": "test",
        "num_workers": 4,
        "mode": "loose_drop",
        "seq_len": 20
    },
    "Infer": {
        "device": "cuda",
        "dtype": "FP32",
        "save_format": "csv",
        "save_folder": "infer\\AttnFactorVAE\\test"
    }
}
```

#### 6.2 使用命令行参数设定配置
```
infer_AttnFactorVAE.py [--log_path LOG_PATH] [--save_configs SAVE_CONFIGS] 
                       --dataset_path DATASET_PATH [--subset SUBSET] [--num_workers NUM_WORKERS] 
                       --quantity_price_feature_size QUANTITY_PRICE_FEATURE_SIZE --fundamental_feature_size FUNDAMENTAL_FEATURE_SIZE --num_gru_layers NUM_GRU_LAYERS --gru_hidden_size GRU_HIDDEN_SIZE --hidden_size HIDDEN_SIZE --latent_size LATENT_SIZE [--std_activation {exp,softplus}] --checkpoint_path CHECKPOINT_PATH 
                       [--dtype {FP32,FP64,FP16,BF16}] -[-device {auto,cuda,cpu}] --save_folder SAVE_FOLDER [--save_format {csv,pkl,parquet,feather}]
```

该脚本接受多种命令行参数来配置训练过程,以下是关键参数的简要说明: 

**通用参数**  
--log_path: 日志文件保存路径(默认: "log/eval_AttnFactorVAE.log")   
--save_configs: 保存配置文件的路径(默认保存为 save_folder 中的 config.json)  

**数据加载参数**  
--dataset_path: 数据集 .pt 文件的路径   
--subset: 拆分后的数据子集(默认: test,可选: train、val、test)   
--num_workers: 数据加载时使用的子进程数量(默认: 4)

**模型参数**  
--quantity_price_feature_size: 量价特征的输入维度  
--fundamental_feature_size: 基本面特征的输入维度  
--num_gru_layers: GRU层数  
--gru_hidden_size: GRU每层的隐藏层维度  
--hidden_size: VAE编码器、预测器和解码器的隐藏层维度  
--latent_size: VAE编码器、预测器和解码器的潜在空间维度(因子数量)     
--std_activation: 标准差计算的激活函数(默认: exp, 可选: exp, softplus)  

**评估参数**    
--checkpoints: 加载 checkpoint 的路径(可选, 可以传入一个或多个地址)
--checkpoint_folder: checkpoint文件夹路径(可选，若指定则相当于checkpoints即为checkpoint_folder目录下所有pt文件和safetensors文件)
--dtype: 张量数据精度类型(默认: FP32, 可选: FP32、FP64、FP16、BF16)  
--device: 计算设备(默认: cuda, 可选: cuda 或 cpu)  
--save_folder: 保存推理结果的文件夹路径  
--save_folder: 保存推理结果的文件格式(默认: pkl, 可选: csv, pkl, parquet, feather)   

#### 6.3 AttnRet和FactorVAE
相似地，为了方便进行模型性能的比较，我们同时提供了AttnRet(RiskAttention)和FactorVAE模型的推理代码：`infer_AttnRet.py`和`infer_FactorVAE.py`。使用方式与`infer_AttnFactorVAE.py`相同。  

但注意，由于模型不尽相同，模型相关的超参数存在差异。具体地表现为：
- AttnRet模型由于将VAE模块替换为MLP,因此没有AttnFactorVAE模型中的hidden_size, latent_size和std_activation参数，而是新增了num_fc_layers参数；
- FactorVAE模型则因为使用无注意力机制的特征提取器，因此不对feature种类进行区分，因此没有AttnFactorVAE模型中的quantity_price_feature_size和fundamental_feature_size参数，而是使用feature_size参数（这个参数的值即quantity_price_feature_size和fundamental_feature_size参数之和）。

另外，模型不再需要dropout相关的参数：在评估时所有dropout会冻结为0.


## 相关工作
-  [FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational
Autoencoder for Predicting Cross-Sectional Stock Returns.](https://ojs.aaai.org/index.php/AAAI/article/view/20369)
Yitong Duan, Lei Wang, Qizhong Zhang, Jian Li
- 基于风险注意力模型的图神经网络因子挖掘
东方证券金融工程与FOF团队 杨怡玲、薛耕

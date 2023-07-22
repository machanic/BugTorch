一、策略网络进行攻击
       以下参数在策略网络进行攻击的时候使用，其中“--targeted”可以选择“untargeted”和“targeted”，分别表示非目标攻击和目标攻击；“--dataset”表示攻击的数据集的种类，可以选择“CIFAR-10”数据集和“ImageNet”数据集；“--norm-type”表示范数的种类，可以选择“l2”和“linf”两个类型；“--policy-arch”表示策略网络使用的网络类型括：“empty”,“vgg11_inv”,
“vgg13_inv”,“vgg16_inv”,“vgg19_inv”,“carlinet_inv”,“wrn_28_10_drop_inv”,“unet','resnet20_inv”,“resnet32_inv”；
“--victim-arch”表示被攻击的网络类型，包括：“WRN-28-10-drop”，“WRN-40-10-drop”，“pyramidnet”，“densenet-bc-L190-k40”；“--policy-weight-fname”表示策略网络预训练的存储位置；“--gpu”表示攻击时需要选取的gpu编号，可根据自身设备进行选取。

以下为攻击时使用的参数列表示例：
--targeted
untargeted
--ce-lmbd
1.0
3e-3
1.5e-3
1.5e-3
--clip-grad
5
--cosine-epsilon-round
0
--current-mean-mult
1
--dataset
CIFAR-10
--decay
0
--delta-mult
1
--empty-coeff
0.5
--empty-normal-mean-norm
0.003
--epsilon-schema
fixed
--exclude-std
--fix-num-eval
--fixed-epsilon-mult
0.4
--gamma
50.0
--grad-method
policy_distance
--grad-size
0
--init-boost
--init-boost-stop
250
--init-boost-th
0.1
--init-empty-normal-mean
--init-num-eval
25
--jump-count
1
--lr
0.0001
--lr-step-freq
0
--lr-step-mult
1.0
--max-baseline
1.0
--max-num-eval
10000
--max-query
10000
--max-sharp
0.5
--mean-reward
--min-baseline
0.05
--min-epsilon-mult
0.01
--min-sharp
0.02
--minus-ca-sim
0.0
--momentum
0.0
--norm-type
l2
--num-image
10000
--num-pre-tune-step
0
--optimizer
Adam
--part-id
0
--phase
test
--policy-arch
vgg13_inv
--policy-base-width
32
--policy-bilinear
--policy-calibrate
--policy-init-std
0.003
--policy-normalization-type
none
--policy-weight-fname
train_pytorch_model/policy_driven_attack/pretrained_models/CIFAR-10/vgg13_inv_WRN-28-10-drop/model_test1.pth
--pre-tune-lr
0.0001
--pre-tune-th
1.0
--rescale-factor
0.5
--save-grad-pct
0.0
--seed
1234
--std-lr-mult
1
--std-reward
--sub-base
--tan-jump
--try-split
0.0
0.25
--victim-arch
WRN-28-10-drop
--victim-batch-size
50
--gpu
5

二、生成预训练数据
       生成预训练数据的代码和攻击的代码相同，唯一的区别就是把策略网络变成简化版的策略网络，体现在参数上便是在“--policy-arch”
传入“empty”，并且为了保存生成数据，需要将“--save-grad”传入参数列表。

如下为生成训练数据的参数示例：
--targeted
untargeted
--ce-lmbd
1.0
3e-3
1.5e-3
1.5e-3
--clip-grad
5
--cosine-epsilon-round
0
--current-mean-mult
1
--dataset
CIFAR-10
--decay
0
--delta-mult
1
--empty-coeff
0.5
--empty-normal-mean-norm
0.003
--epsilon-schema
fixed
--exclude-std
--fix-num-eval
--fixed-epsilon-mult
0.4
--gamma
50.0
--grad-method
policy_distance
--grad-size
0
--init-boost
--init-boost-stop
250
--init-boost-th
0.1
--init-empty-normal-mean
--init-num-eval
25
--jump-count
1
--lr
0.0001
--lr-step-freq
0
--lr-step-mult
1.0
--max-baseline
1.0
--max-num-eval
10000
--max-query
10000
--max-sharp
0.5
--mean-reward
--min-baseline
0.05
--min-epsilon-mult
0.01
--min-sharp
0.02
--minus-ca-sim
0.0
--momentum
0.0
--norm-type
l2
--num-image
10000
--num-pre-tune-step
0
--optimizer
Adam
--part-id
0
--phase
test
--policy-arch
empty
--policy-base-width
32
--policy-bilinear
--policy-calibrate
--policy-init-std
0.003
--policy-normalization-type
none
--policy-weight-fname
train_pytorch_model/policy_driven_attack/pretrained_models/CIFAR-10/vgg13_inv_WRN-28-10-drop/model_test1.pth
--pre-tune-lr
0.0001
--pre-tune-th
1.0
--rescale-factor
0.5
--save-grad-pct
0.0
--seed
1234
--std-lr-mult
1
--std-reward
--sub-base
--tan-jump
--try-split
0.0
0.25
--victim-arch
WRN-28-10-drop
--victim-batch-size
50
--gpu
5
--save-grad

三、根据生成需训练数据集进行训练
       以下的参数是根据生成预训练数据集进行训练的代码所需要的。其中“--num-train”表示需要训练的数据的数量；“--dataset”表示数据集的类型，可以选择“CIFAR-10”数据集和“ImageNet”数据集；“--num-test”表示训练完成后及进行测试的图片数量；“--train-data-victim-arch”表示预训练数据集生成过程中使用的受害者网络，选择攻击过程中没有用到的网络“resnet-29-16x64d”；“--victim-arch”表示测试时使用的被害者网络，包括：WRN-28-10-drop”，“WRN-40-10-drop”，“pyramidnet”，“densenet-bc-L190-k40”；“--policy-arch”表示需要训练的策略网络类型，包括：“vgg11_inv”,
“vgg13_inv”,“vgg16_inv”,“vgg19_inv”,“carlinet_inv”,“wrn_28_10_drop_inv”,“unet','resnet20_inv”,“resnet32_inv”。

以下是预训练传入参数示例：
--model-dir
train_pytorch_model/policy_driven_attack/pretrained_models
--num-train
10000
--max-query
10000
--dataset
CIFAR-10
--phase
test
--num-test
1000
--train-data-victim-arch
resnet-29-16x64d
--victim-arch
resnet-29-16x64d
--policy-arch
vgg13_inv
--policy-init-std
3e-3
--policy-bilinear
--policy-normalization-type
none
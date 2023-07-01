# What is FL-PLAS
As an emerging distributed machine learning framework, federal learning with a large number of clients cannot guarantee whether each participating user is a legitimate normal user, and every step in between, from data acquisition to the final global model building, may be attacked by malicious users. Moreover, since the server of federal learning cannot supervise the training process of malicious user data and user models, malicious users can complete the backdoor injection of global models by backdoor injection of local data and models, and federal learning also enhances the difficulty of defending against malicious user attacks on the basis of protecting user privacy. Backdoor attack is one of the very covert malicious attack methods. 
 
A model attacked and injected by a backdoor does not show abnormal performance on normal samples, and when a backdoor in the model is activated by a trigger, the backdoor model classifies the backdoor samples into the category specified by the attacker. However, current backdoor defense strategies have various problems, such as difficulty in tolerating a high proportion of malicious users in the environment and possible need for third-party data. In response, this thesis proposes a backdoor defense strategy based on layer aggregation policy. This strategy can defend against a large percentage of malicious users while eliminating the need to use auxiliary datasets on the server side. In this thesis, the user model is divided into two parts according to the function of the model: feature extractor and classifier, and only the feature extractor is uploaded while the classifier is kept local when federal learning is performed. This scheme can isolate the classifiers of benign users from those of malicious users. When faced with a backdoor sample, the backdoor in the global backdoor feature extractor is activated by the trigger in the sample and the backdoor features in the sample are extracted. The classifier for benign users can only classify the other features extracted by the feature extractor because it does not have the ability to classify the backdoor features and therefore shows good classification ability. The algorithm in this thesis can defend against a larger proportion of malicious users in the environment without using additional datasets, and can also perform well in the face of two new types of backdoor attacks.
# Using the code 
Experimental environment:
```
Python                  3.6.13
torch                   1.8.0+cu111
torchvision             0.9.0+cu111
```
Operation description:
If one wants to see the MA and BA of the model with the feature extractor and classifier with poison respectively, one can use:
```
python parameterBoard.py --dataname mnist --model lenet --backdoor_type trigger --test True 
```
If one wants to see how the model behaves for MA and BA in various scenarios, including different datasets, malicious user ratios, backdoor types, and defense methods, there are three types of command examples
```
python parameterBoard.py --lr 0.0067 --num_nets 100 --part_nets_per_round 30 --fl_round 200 --malicious_ratio 0 --dataname mnist --model lenet -- device cuda:0 --num_class 10 --backdoor_type trigger --defense_method none --cut 3 --test False

python parameterBoard.py --lr 0.0025 --num_nets 100 --part_nets_per_round 30 --fl_round 200 --malicious_ratio 0 --dataname cifar10 --model mobilenet --device cuda:0 --num_class 10 --backdoor_type trigger --defense_method none --cut 80 --test False

python parameterBoard.py --lr 0.000015 --num_nets 100 --part_nets_per_round 30 --fl_round 200 --malicious_ratio 0 --dataname cifar100 --model resnet18 --device cuda:0 --num_class 100 --backdoor_type trigger --defense_method none --cut 69 --test False
```
The malicious_ratio and defense_method of each of these classes can be modified. In particular, when the dataset is cifar10, the backdoor type in the command can be modified to semantic or edge-case.

In addition, if images need to be generated, the run.py program should be run after deleting the ma.txt and ba.txt in the folder, which will eventually generate 10 images, including 5 images of the main task accuracy and 5 images of the backdoor task accuracy. These images show how the model's main task accuracy or backdoor task accuracy changes with the percentage of malicious users when the three models are subjected to three backdoor attacks.

MIT license

Programmer:  Qichao Jin

Email: jin_qc@qq.com zjy@besti.edu.cn

北京电子科技学院CSP实验室


Readme:

Codes with CIFAR-10 as True data and CIFAR-100(select classes) as proxy data are uploaded. 
Codes for the other cases are very similar to these, with minor changes
The commands to run each code are mentioned in the respective files. Path where the Datasets can be downloaded can be modified in the command. 

Folder organization:

> CIFAR-10:
	Note: Before running any of the codes, all files from the folder 'network' should be copied to the respective folder.
	For example, to run codes from 'train_generator' folder, all files from 'network' folder should be copied to the root of 'train_generator' folder.
	> network: Contains the network architecture definition for the Teacher, Student and GAN models used for CIFAR-10. Also contains saved weights of trained Teacher network.
		> alexnet.py: Teacher architecture with CIFAR-10 as True Data
		> alexnet_half.py: Student architecture with CIFAR-10 as True Data
		> dcgan_model.py: Architecture of generator and disriminator using DCGAN and DeGAN
		> best_model.pth: Saved Teacher weights used for all CIFAR-10 experiments
	> train_teacher:
		> train_teacher.py: Code for training Teacher network
	> train_generator: 
		> dfgan.py: Code for training DCGAN/ DeGAN. Setting 'c_l' and 'd_l' to 0 is for training DCGAN.
		The classes in inc_classes can be modified to select the required classes from CIFAR-100 as part of Proxy Dataset.
	> train_student: 
		> Using_Data
			> KD_related_data.py: Code for Knowledge Distillation using related/ unrelated data
		> Using_GAN
		The code 'dfgan.py' in the folder train_generator needs to be run before this
			> KD_dfgan.py: Code for Knowledge Distillation using DCGAN/ DeGAN
		
> Other_networks_used:
	> lenet.py : Teacher architecture with Fashion-MNIST as True Data
	> lenet_half.py : Student architecture with Fashion-MNIST as True Data
	> inceptionv3_teacher.py : Teacher architecture with CIFAR-100 as True Data
	> resnet_18_student.py : Student architecture with CIFAR-100 as True Data
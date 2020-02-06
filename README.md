# Train-Predict-Landmarks-by-pix2pix

This network architecture is presented in which detects the locations of landmarks for vocal tract area.<br>
The method is based on heatmap generation and location of the argmax.<br>
The network architecuture is same as the autoencoder with skip connections in the generator of the pix2pix network.<br>


# Paper:
This code is used for the following research. If you found it usefull, please cite the following document:

https://www.nature.com/articles/s41598-020-58103-6

@article{eslami2020automatic,
  title={Automatic vocal tract landmark localization from midsagittal MRI data},
  author={Eslami, Mohammad and Neuschaefer-Rube, Christiane and Serrurier, Antoine},
  journal={Scientific Reports},
  volume={10},
  number={1},
  pages={1--13},
  year={2020},
  publisher={Nature Publishing Group}
}

Following repositories are also used for the mentioned paper:

https://github.com/mohaEs/Train-Predict-Landmarks-by-SFD

https://github.com/mohaEs/Train-Predict-Landmarks-by-DAN

https://github.com/mohaEs/Train-Predict-Landmarks-by-MCAM

https://github.com/mohaEs/Train-Predict-Landmarks-by-dlib

https://github.com/mohaEs/Train-Predict-Landmarks-by-flat-net


## set up
code is based on the tensorflow 1.14 which the embedded keras is also used.

## data
it is assumed that, the png files are stored in a folder and corresponding cvs files with same filenames are located in another folder and contains the x,y locations of the landmarks. <br>
For example one image and corresponding landmark file is placed in data folder.<br>
The landmark file contanis 20 landmarks.<br>

## training
The mentioned network were able to handle just 3 heatmaps for 256x256 images and batch size of 10.<br>
Therefore for having a full machine predicting 20 landmarks we need to train 7 different netwroks.<br>
The desired landmark for each network can be set by arguments.<br>

the input arguments are as follow: <br>
"--mode", choices=["train", "test"])<br>
"--input_dir", the path of the directory contains png files <br>
"--target_dir",  the path of directory contains csv files of corressponding ladnmarks <br>
"--checkpoint",  the path of directorry for the trained model <br> 
"--output_dir",  the path of output, in training mode the trained model would be saved here and in testing mode the results.<br>
"--landmarks",  the string contains the number of desired landmarks<br>

for example for a machine: <br>

> target_dir='../Images_Data/LoSo_1/temp_train_lm/' <br>
> input_dir='../Images_Data/LoSo_1/temp_train_png/' <br>
> checkpoint='../Images_Data/LoSo_1/Models/Models_pix2pix/Models_lm_6/' <br>
> output_dir='../Images_Data/LoSo_1/Models/Models_pix2pix/Models_lm_6/' <br>
> landmarks='18,2,14'<br>
> python3 Pr_LandMarkDetection_pix2pixArc+HeatMap.py --mode 'train'   --input_dir   $input_dir     --target_dir  $target_dir    --checkpoint  $checkpoint     --output_dir  $output_dir     --landmarks  $landmarks 

of the checkpoint is empty, the new training session would be done otherwise the training would be continued from previous session.<br>

Do not forget to set the suitable number of epochs in the python scripts.

## predicting

> target_dir='../Images_Data/LoSo_1/temp_test_lm/' <br>
> input_dir='../Images_Data/LoSo_1/temp_test_png/' <br> 
> checkpoint='../Images_Data/LoSo_1/Models/Models_pix2pix/Models_lm_6/' <br>
> output_dir='../Images_Data/LoSo_1/Results/Models_pix2pix/Models_lm_6/' <br>
> landmarks='18,2,14' <br>
> python3 Pr_LandMarkDetection_pix2pixArc+HeatMap.py --mode 'test'   --input_dir   $input_dir     --target_dir  $target_dir    --checkpoint  $checkpoint     --output_dir  $output_dir     --landmarks  $landmarks 

The predicted landmarks plus the truth locations would be saved in csv files. Also a visualization image of the prediction will be saved:

![Alt text](output-sample.png?raw=true "Title")


Notice that, in the save csv of output folder, the x and y columns may changed from the original ones in targer_dir:
x->y, 
y->x



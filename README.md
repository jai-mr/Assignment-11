* Repository github url : https://github.com/jai-mr/
* Assignment Repository : https://github.com/jai-mr/Assignment-10
* Submitted by : Jaideep Rangnekar
* Registered email id : jaideepmr@gmail.com

## Assignment

1. Write a code that draws this curve (without the arrows). In submission, you'll upload your drawn curve and code for that
[Assignment 1.1](https://github.com/jai-mr/Assignment-11/blob/main/content/images/Assignment_1.png)

2. Write a code which
    1. uses this new ResNet Architecture for Cifar10:
        1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        2. Layer1 -
            1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
            2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
            3. Add(X, R1)
        3. Layer 2 -
            1. Conv 3x3 [256k]
            2. MaxPooling2D
            3. BN
            4. ReLU
        4. Layer 3 -
            1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
            2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
            3. Add(X, R2)
        5. MaxPooling with Kernel Size 4
        6. FC Layer 
        7. SoftMax
    2. Uses One Cycle Policy such that:
        1. Total Epochs = 24
        2. Max at Epoch = 5
        3. LRMIN = FIND
        4. LRMAX = FIND
        5. NO Annihilation
    3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
    4. Batch size = 512
    5. Target Accuracy: 90%. 


## Details for Assignment Executed

## * Solution 1.1
### [Jupyter Notebook File reference executed in Colab](https://github.com/jai-mr/Assignment-11/blob/main/11.1_CodeFinal.ipynb)

### [Cyclic Triangle](https://github.com/jai-mr/Assignment-11/blob/main/content/images/0_CyclicTriangle.png)

## Solution 1.2
### * [Jupyter Notebook File reference executed in Colab 2](https://github.com/jai-mr/Assignment-11/blob/main/11_2_CodeFinal.ipynb)



### * [Data View](https://github.com/jai-mr/Assignment-11/blob/main/content/images/1_DataView.png)

### * [Mis-Classified Images](https://github.com/jai-mr/Assignment-11/blob/main/content/images/3_MisClassified.png)

### * [GradCam-Misclassified images wrt Predicted(wrong) class](https://github.com/jai-mr/Assignment-11/blob/main/content/images/4_GradCam-Misclassified%20images%20wrt%20Predicted(wrong)%20class.png)

### * [GradCam-Misclassified images wrt Actual(correct) class](https://github.com/jai-mr/Assignment-11/blob/main/content/images/5_GradCam-Misclassified%20images%20wrt%20Actual(correct)%20class.png)

### * [LR Finder Plot](https://github.com/jai-mr/Assignment-11/blob/main/content/images/2_LRFinderPlot.png)

### * [Changed LR](https://github.com/jai-mr/Assignment-11/blob/main/content/images/8_ChangeLR.png)

### * [Training/Test - Loss & Accuracy Curve](https://github.com/jai-mr/Assignment-11/blob/main/content/images/6_Loss-Accuracy%20Plot%20for%20Train%20and%20Test.png)

### * [Test vs Train Accuracy](https://github.com/jai-mr/Assignment-11/blob/main/content/images/7_TestvsTrain.png)

### * Test Accuracy : 91.32%

### * Class wise accuracies
* Accuracy of plane : 97 %
* Accuracy of   car : 91 %
* Accuracy of  bird : 88 %
* Accuracy of   cat : 77 %
* Accuracy of  deer : 92 %
* Accuracy of   dog : 89 %
* Accuracy of  frog : 96 %
* Accuracy of horse : 93 %
* Accuracy of  ship : 93 %
* Accuracy of truck : 94 %


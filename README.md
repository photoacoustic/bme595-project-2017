# BME595 Dee[?]earning
### 90 drinking coffee(blue) or no drinking coffee(orange) days
-blue is for drinking coffee days, orange is for not drinking coffee days. The different length of blue/orange color bars were represented to different consecutive drinking coffee/not drinking coffee days.
![Coffee or NoCoffee](https://github.com/photoacoustic/bme595-project-2017/blob/master/project/Screen%20Shot%202017-10-12%20at%203.58.06%20PM.png)
# Title  
- Using "Neural Network" Prediction of bodily metabolism with MyConnectome dataset.
## Team members  
-  NameB (GitHubUserB), Ho-Ching Yang (photoacoustic),
## Goals  
- We are going to train neural network with 4D functional MRI data to predict the bodily metabolism.
## Challenges
- Myconnectome dataset had recorded brain function and metabolism fluctuate in a single individual, Dr. Poldrack over the course of an entire year.  
- There is 90 scans in this dataset. Each scans has 294 timepoint which means we'll have 90*294 3D fMRI image as input of Neural Network.
- The input will be 4D fMRI images (don't know if we need to count time into another dimension), in terms of the output will be bodily metabolism(ex. this is the functional MRI image scanned after this subject drinking coffee)
- "What" should we let neural network to "See" will be a big chanllenges. We should use processed dataset, which is 
## Restrictions --optional--  
- I will be very happy if I knew a simplest way to train 4D fMRI dataset without GPU...

![2D](https://github.com/photoacoustic/bme595-project-2017/blob/master/project/Screen%20Shot%202017-10-11%20at%209.34.54%20PM.png)
![3D](https://github.com/photoacoustic/bme595-project-2017/blob/master/project/Screen%20Shot%202017-10-11%20at%209.35.09%20PM.png)

1. Myconnectome
http://myconnectome.org/wp/
2. Nathawani, D., Sharma, T. and Yang, Y., 2016. Neuroscience meets deep learning.
http://download.cnet.com/ShiftIt/3000-2072_4-76063682.html


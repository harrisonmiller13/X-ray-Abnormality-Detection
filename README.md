# X-ray-Abnormality-Detection
This project was built to classify an upper limb x-ray as normal or abnormal.  The image classification was implemented using a Convolutional Neural Network built in Pytorch and GRADcam for highlighting of the abnormality.  

## Data

The image data comes from the MURA 1.1 dataset from Stanford University School of Medicine. MURA is a dataset of musculoskeletal radiographs consisting of 14,863 studies from 12,173 patients, with a total of 40,561 multi-view radiographic images. Each belongs to one of seven standard upper extremity radiographic study types: elbow, finger, forearm, hand, humerus, shoulder, and wrist. Each study was manually labeled as normal or abnormal by board-certified radiologists from the Stanford Hospital at the time of clinical radiographic interpretation in the diagnostic radiology environment between 2001 and 2012.


## CNN

The initial CNN that I created was a 10 layer network with dropout and regularization.  The Adam optimizer was used.  The x-rays were sent through the network in batches of 8 with an initial learning rate of 0.0001. ReLU was used as the activation function.  Binary Cross Entropy was the loss function.  For the final layer, a sigmoid activation was used for the classification.

A second CNN was created using transfer learning. Using the DENSENET201 pretrained network, all layers were frozen except for the last 10 in order to specify for the dataset.

## Results
Using the 10 layer network the model achieved an accuracy of 71.4% and a kappa statistic of 0.423 which was an improvement over the baseline accuracy of 61.5%.  
Using the transfer learning model accuracy increased to 82.2% and a kappa statistic of 0.642.




### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [PyTorch](http://pytorch.org)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
<!-- * [Heroku](https://www.heroku.com) -->


## Authors

* **Harrison Miller** - *Initial work* - [GitHub](https://github.com/harrisonmiller13)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
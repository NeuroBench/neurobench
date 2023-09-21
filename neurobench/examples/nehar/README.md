Neuromorphic Human Activity Recognition
=======================================================================
Human Activity Recognition (HAR) is a time-dependent task that has applications in various aspects of human life, from healthcare to sports, safety, and smart environments. In this task, we present a comparative analysis of different SNN-based models designed for classifying raw signals (Accelerometer and Gyroscope) collected in the Wireless Sensor Data Mining (WISDM) dataset.
The WISDM dataset consists of data from 51 subjects performing 18 activities. This dataset collects signals from both the accelerometer and the gyroscope of a smartphone and a smartwatch. Each activity is recorded for 3 minutes with an acquisition rate of 20 Hz. The dataset's classes are balanced, with each activity represented in the dataset contributing approximately 5.3% to 5.8% of the total approximately 15.63 million samples.
From the whole smartwatch dataset, we selected a subset of general hand-oriented activities for our analysis. These activities include: (1) dribbling in basketball, (2) playing catch with a tennis ball, (3) typing, (4) writing, (5) clapping, (6) brushing teeth, and (7) folding clothes. We divided the signals into non-overlapping temporal windows with a length of 2 seconds. These temporal windows serve as the input layer for the benchmarked models.

## Reference

Vittorio Fra, Evelina Forno, Riccardo Pignari, Terrence Stewart, Enrico Macii, and Gianvito Urgese:  
Human activity recognition: suitability of a neuromorphic approach for on-edge AIoT applications.  
Neuromorphic Computing and Engineering 2022.

See article on [IOPscience](http://iopscience.iop.org/article/10.1088/2634-4386/ac4c38)

See project on [GitHub](https://github.com/neuromorphic-polito/NeHAR)


## Contact
* Gianvito Urgese `gianvito.urgese@polito.it`
* Benedetto Leto `benedetto.leto@studenti.polito.it`
* Vittorio Fra `vittorio.fra@polito.it`
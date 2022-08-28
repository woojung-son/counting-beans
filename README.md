## 2021 Kaggle Competition by PNU <br/>(prize-winning project, 2th place)
- Subject: Counting the number of beans in the plate as accurate as possible given images.

## counting-beans
<strong>_counting-beans_</strong> is the script for counting the number of beans in the plate given images.

## Competition Description
- Five images with different angles are provided for each instance.(beans in the plate) But this repository only provides you the one with bird-eye angle.
- In the test tiem, 30 hidden instances are given and each example has the same amount of point. 
- For training, 30 open instances are given.(Images in the folder _Open_ are from the training set)
- Training data and test data follow the same distribution.
- Only algorithmic method is allowed.(learning-based method is not permitted)
- The proposed method must finish its work within 5 min.

## Process
1. Segment the plate from the background. Use _GrabCut_ algorithm. Since it is an interactive algorithm requiring user's continuous input, we use manual center-crop and circular mask augmentation for automatic process. The result is shown below:<br/>

  
2. Detect and count beans in the plate. We use thresholds of Hue/ValueChroma to detect beans from the plate and other beans. Threhsolds are found by manual tuning. We upload the file _find_upper_lower.py_ which helps find the best threshold. Though the aformentioned process can find the boundary of each bean with comparable performance, it might struggle in the case with many beans. So we make a decision whether the input image contains many beans or not before counting. In the former case, we use regression based on the area of detected beans, fitted with the training set. In the latter case, we mask the plate to get beans and count with some post-processings. The result is shown below:<br/>
  
## References
Rother et al. _"GrabCut": interactive foreground extraction using iterated graph cuts_. ToG. 2004.
  
  
 

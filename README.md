# DATA606 - Capstone Project

## ChequeChecker - HandWritten Text Recognition

### Introduction

The world has been digitalized to a great extent and yet there are few sensitive areas where we are forced to use traditional methods. Bank cheques are one of them. Most of the financial transactions are done by a paper cheque all over the world. When these handwritten bank cheques are in play there is also a scope for manual errors. These manual errors could be costly and could save a lot of money if this problem is addressed.

### Project Objective
In my project, I would try to build an application which can scan the cheques, analyze the handwritten text and identify the amount that has to be credited or debited. This way we can avoid the scope for manual errors to a significant extent. 

### About the Data
For the data, I had to rely on the handwritten text data and found the IAM Handwriting Database. 
- Briefly the dataset characteristics are as follows:
- 657 writers contributed samples of their handwriting
- 115'320 isolated and labeled words
A preliminary analysis of the dataset gave the following details: 
- The words have been extracted from pages of scanned text and are saved as png files
- Each file has a word label given which would give us the coordinates of the box that it is chopped, tagset data (Parts of Speech tagset)
- The word itself

#### Data Source: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/iam-handwriting-database#icdar02

### Implemetation Methodology
- Clean and process the words data
- Link the word pictures with the labeling
- Try Neural Networks to find the map between words
- CNN, RNN, LSTM Networks
- Identify the words from the cheque that has to be predict using OpenCV/Google Tesseract
- Pick the best network and predict the identified words
- Visualize the results on a webapp

### References
- Zhang, X., Wang, M., Wang, L., Huo, Q., & Li, H. (2015). Building Handwriting Recognizers by Leveraging Skeletons of Both Offline and Online Samples. 2015 13th International Conference on Document Analysis and Recognition (ICDAR). doi:10.1109/icdar.2015.7333793
- Krishnan, P., Dutta, K., & Jawahar, C. (2016). Deep Feature Embedding for Accurate Recognition and Retrieval of Handwritten Text. 2016 15th International Conference on Frontiers in Handwriting Recognition (ICFHR). doi:10.1109/icfhr.2016.0062
- Jayadevan, R., Kolhe, S. R., Patil, P. M., & Pal, U. (2011). Automatic processing of handwritten bank cheque images: A survey. International Journal on Document Analysis and Recognition (IJDAR), 15(4), 267-296. doi:10.1007/s10032-011-0170-8

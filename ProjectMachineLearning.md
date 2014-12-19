***Practical Machine Learning - Course project***

**Riccardo Sorrentino**

The aim of the project is to create a prediction model about the quality
performances of six amateur athletes. Their activity, barbell lifts, is
assessed with a rating betweeen A and E. The prediction model will use
data from accelerometers on the belt, forearm, arm and -dumbell.

**Executive Summary**

A good outcome, with an accuracy rate around 99%, can be obtained using
a random forest model on 52 variables. The results on the test data are
B A B A A E D B A A B C B A E E A B B B, and appear to be correct.

**Downloading and setting data tables**

The first step has been to install the caret package and download both
the training and the testing data

    library(caret)

    setInternet2(use=TRUE)
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile="./projectml.csv")
    projectml <- read.csv("projectml.csv")
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="./testingml.csv")
    testingml <- read.csv("testingml.csv")

To perform cross validation, the training data set has been divided in
two parts: the model training set, and a cross validation set.

    set.seed(2702)
    inTrain <- createDataPartition(projectml$classe, p=0.7, list=FALSE)
    training <- projectml[inTrain,]
    CValidation <- projectml[-inTrain,]

A first look on the training data shows that the first seven columns
describe each single exercise (name of the athlete, date, time, etc.)
and are useless for prediction. Those columns have been deleted from the
training set.

    head(training[,1:7], 4)

    ##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
    ## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
    ## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
    ## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
    ## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
    ##   new_window num_window
    ## 1         no         11
    ## 2         no         11
    ## 3         no         11
    ## 5         no         12

    r <- dim(training)[2]
    training <- training[,(8:r)]

**Deleting useless predictors**

A more accurate look of the data shows that in many columns appear NAs
values, so they are useless for prediction. Those columns have been
deleted.

    training <- training[ , colSums(is.na(training)) == 0]

At the point the training set has 86 columns. It could be useful to
perform a near zero variance analysis to further reduce the number of
predictors.

    nsv <- nearZeroVar(training, saveMetrics=FALSE)
    nsv

    ##  [1]  5  6  7  8  9 10 11 12 13 36 37 38 39 40 41 45 46 47 48 49 50 51 52
    ## [24] 53 67 68 69 70 71 72 73 74 75

    training <- training[,-nsv]

The columns with limited variance have been deleted. Now the training
set has 53 columns: 52 predictors and one outcome.

**Creating the model: 1. a tree model**

The outcome is a rating of the activity quality, so the first hypothesis
is to use a tree model.

    set.seed(1001)
    modFitHyp <- train(classe ~ ., method="rpart", data=training)

    ## Loading required package: rpart

    modFitHyp

    ## CART 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## 
    ## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
    ## 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
    ##   0.03580511  0.5020642  0.34982354  0.05001894   0.07790586
    ##   0.06065846  0.4102147  0.20017587  0.06295904   0.10458913
    ##   0.11311159  0.3186464  0.05330413  0.04053058   0.06142134
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was cp = 0.03580511.

The resulting accuracy rate of 0.50 is quite low.

**Creating the model: 2. a random forest model**

To obtain a better accuracy rate it is useful to try a random forest
model.

    set.seed(1001)
    modFit <- train(classe ~ ., method="rf", data=training, allowParallel = TRUE)

    ## Loading required package: randomForest
    ## randomForest 4.6-10
    ## Type rfNews() to see new features/changes/bug fixes.

    modFit

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## 
    ## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
    ## 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
    ##    2    0.9884722  0.9854158  0.002261365  0.002852582
    ##   27    0.9885172  0.9854738  0.002420387  0.003055488
    ##   52    0.9780594  0.9722407  0.006079155  0.007696161
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 27.

    modFit$finalModel

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry, allowParallel = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.71%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 3901    4    1    0    0 0.001280082
    ## B   16 2633    9    0    0 0.009405568
    ## C    0   11 2375   10    0 0.008764608
    ## D    0    0   34 2216    2 0.015985790
    ## E    0    0    3    7 2515 0.003960396

The accuracy rate of 0.9885 is quite good. Preprocessing doesn't improve
the model performance.

**Cross Validation analysis**

The random forest model has been tested on the cross validation set. The
out of sample error is predicted to be more than 0.0115 (1 - the
accuracy rate) even if a confusion matrix on the training set shows an
in-sample-error of zero.

    pred <- predict(modFit, CValidation)
    print(confusionMatrix(pred, CValidation$classe))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    9    0    0    0
    ##          B    1 1125    5    0    1
    ##          C    0    5 1017    8    2
    ##          D    0    0    4  956    2
    ##          E    1    0    0    0 1077
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9935          
    ##                  95% CI : (0.9911, 0.9954)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9918          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9988   0.9877   0.9912   0.9917   0.9954
    ## Specificity            0.9979   0.9985   0.9969   0.9988   0.9998
    ## Pos Pred Value         0.9946   0.9938   0.9855   0.9938   0.9991
    ## Neg Pred Value         0.9995   0.9971   0.9981   0.9984   0.9990
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2841   0.1912   0.1728   0.1624   0.1830
    ## Detection Prevalence   0.2856   0.1924   0.1754   0.1635   0.1832
    ## Balanced Accuracy      0.9983   0.9931   0.9941   0.9952   0.9976

The effective out-of-sample error is 0.0061, very low.

**The final outcome**

At this point it is possible to apply the model to the testing set and
obtain the final outcome.

    answers <- predict(modFit, testingml)
    answers

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

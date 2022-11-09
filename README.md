# Exploring-Errors-in-Regression
This project studies regression of data whose underlying function varies from linear to very nonlinear using methods with different flexibility. From the class, we learned that the mean-squared errors $MSE$ is given as:



$E(y_0-\hat{f}(x_0))^2 = Var(\hat{f}(x_0)) + Bias[(\hat{f}(x_0))]^2 + Var(\epsilon)$


where $Var(\hat{f})$ is the *Variance*, which refers to the amount by which $\hat{f}$ would change if we estimated it using a different training set; $Bias(\hat{f}) = E(\hat{f}) - f$ is the $Bias$, which refers to the error that is introduced by approximating a complicated real-life problem by a much simper model; and $Var(\epsilon)$ is the irreducible error. We will study the bias-variance tradeoff for different methods on different data.


**Data Generation:** I will be generating data using the following models. Let $y = f(x) + \epsilon$, where $\epsilon$ is the random noise (e.g. Gaussian) and $f(x)$ has three forms as:
1. A linear function $f(x) = ax + b$;
2. A quadratic function $f(x) = ax^2 + bx + c$;
3. A nonlinear function: e.g. $f(x) = x\sin(x)$, $f(x) = \frac{1}{1 + 25x^2}$ ![image](https://user-images.githubusercontent.com/63213726/200918310-d94ac411-f87c-4c47-adcd-1575a85df0bc.png)

</br></br>
<ins>NOTE</ins>: I will be picking a domain and deciding how many data points to produce. I will be splitting my data into the training set and the test set randomly and so validation can be performed. For the random noise, I will be controlling the mean and variance, choosing them arbritrarily as I see fit.


**Regression Models:** I will use three regression methods to fit the data.
1. $Y = \beta_0 + \beta_1X$  $(df=2)$;
2. A smoothing cubic spline with degree of freedom 5;
3. A smoothing cubic spline with effective degree of freedom approximately 25. (<ins>NOTE</ins>: for smoothing spline I will adjust the smoothing parameters to control the level of flexibility or degree of freedom.)

**Research Objective:** I will use my data and methods to investigate the following topics.
1. The performance of methods with different flexibility on data from linear to nonlinear underlying functions.
2. How does the $MSE$, $Bias$, and $Var$ vary with method of different flexibility?
3. How does the variance of noise affect the performance of different method (e.g. similar to (1) but with a different noise level characterized by the variance). I will use the nonlinear data with different noise levels fitted by three methods, plot the model and calculate training $MSE$ and test $MSE$.



## **Data Generation**
To generate data, I will first be establishing the true population models that I will be using to arbitrarily make predictions using various regression techniques. Here are my three functions that I'm choosing for this project, all of which have extreme varying complexity:

**Linear Model:** $f(x) = \frac{3}{10}x + 1$   


**Quadratic Model:** $f(x) = \frac{1}{100}x^2 -\frac{1}{50}x + 2$   


**Non-linear Model:** $f(x) = \frac{1}{5}x - \frac{1}{20}(\frac{x^2\cos(x)}{\ln(x+2)})$   


The reason why I am using multiple models of varying complexities is because I want to observe exactly how our Regression methods react to different patterns of data. Depending on the flexibility and bias, Spline regression with a high degree of freedom for example might have a much more severe reaction to a more complex non-linear $f$ than a simple linear regression model would.


We fix our noise with $\epsilon$ := np.random.normal(0, $var$, $n$)



![alt text](https://github.com/ririye01/Exploring-Errors-in-Regression/blob/main/plots/Plotted_Data_and_True_Models.png)  



## **Regression Models**
Now that we have our data, let's evaluate how various regression methods perform when fitting to the data. As a reminder, I will be using the following methods to evaluate which method fits the data best and which method fits the model best:
1. $Y = \beta_0 + \beta_1X$  $(df=2)$;
2. A smoothing cubic spline with degree of freedom 5;
3. A smoothing cubic spline with effective degree of freedom approximately 25. (<ins>NOTE</ins>: for smoothing spline I will adjust the smoothing parameters to control the level of flexibility or degree of freedom.)



### **Splitting Our Data into Training and Testing Sets**

We have our data, but we need to split it now into a training dataset and a testing dataset. The reasoning why I am following this approach is because this enables me to both create a model $\hat{f}(x)$ appropriate for the shape of our data and test it on data that my program did not see during the training process. To split our data accordingly, I will use the cross-validation. In the cross-validation approach, the model is fit on the training set then applied to predict the response variable $y$ in the testing set. Scikit-Learn's train_test_split() performs this operation, and I will ensure that the function randomly splits 75% of our data into the training set and 25% of it on the testing data. 


### **Calculate Corresponding Smoothness for Degrees of Freedom**

Now, I will calculate the necessary smoothness values that correspond with degrees of freedom 2, 5, and 25 respectively. Degrees of freedom are associated with the amount of piecewise fragments used to calculate the cubic spline. For example, if my degrees of freedom equals 5, then there exists 4 separate piecewise fragments in the overall cubic spline piecewise function. Note that as degrees of freedom increase, flexibility of our model decreases.


### **Obtain Cubic Spline Predictions through $\hat{f}(x)$**

Now that we have the necessary smoothness values associated with our target degrees of freedom, we now can input our data into the csaps() function to obtain predicted y values based on each corresponding model. 

### **Plot Our Predictions**

Now, I will use matplotlib to plot the target $y$-values and predicted $\hat{y}$-values. By doing this, I will be able to see how our predictions differ from reality, and I'll also be able to visualize how cubic spline predictions with different flexibilites compare with one another as the degrees of freedom increases. 


![alt text](https://github.com/ririye01/Exploring-Errors-in-Regression/blob/main/plots/Regression_Estimates.png)


In the linear model, overfitting to the training data becomes extremely apparent as our degrees of freedom increases. The true model is linear and the predicted model for df=2, which is a linear prediction, follows the true model almost exactly. However, overfitting occurs in the red dotted line, and we can see that the model bends and curve to slight variations in the training data caused by Gaussian noise, which distances it from the testing data and makes it not resemble reality.



In the quadratic model, we see the same phenomenon for the most part, except in the predicted linear model for this true quadratic model when df=2, we can see that the model underfits pretty extremely.



In the complex non-linear model, increases in degrees of freedom actually resemble reality much more than predicted cubic spline models with smaller degrees of freedom. The df=2 and df=5 models extremely underfit reality here, but the red dotted line resembles the true $f(x)$ almost exactly. 



Observing $MSE$, $Bias$, and $Var$ will allow us to explore these deviations from reality even further.




## **Research Objective**

In order to calculate the $MSE$, $Bias[(\hat{f}(x))]^2$, and $Var(\hat{f}(x))$ for each of our regression models, we need to obtain $m$ training data sets by resampling. For the given data $(x,y)$ in the test set, let $\bar{\hat{y}}$ be the average of predicted $y$ at $x$ using models obtained from different training datasets.


### **Resampling via Bootstrap Method**

We will need to perform Bootstrap Sampling as our form of resampling. Bootstrap Sampling is the repetitive random selection of sample data to estimate a population parameter. To put that in Layman's terms, Bootstrap Sampling involves selecting a subset of our sample multiple times with replacement—meaning that the data selected in each subset can reappear in the next subset and so forth—to predict $f(x)$, which is the true model. Our predicted model will be denoted as $\hat{f}(x)$. We will perform Bootstrap Sampling with X_train values which were already calculated using scikit-learn's train_test_split() function. Iterating through the draw_bootstrap_sample() function repetitively will allow for us to focus on multiple segments of our training data X. This method is untraditional, because bootstrap sampling typically relies on all data X as the training set, but we are using it on X_train then taking the average of all of our results. Our results would not be that much different than if we were just using X_train, but I wanted to make sure to showcase this bootstrap method in practice. By repetitively going through this process, we can estimate the variability of our estimate $\hat{f}(x)$ without generating additional samples.

### **Calculate MSE, Bias, and Var**

The Mean-Squared Error $MSE$ Formula is as follows:
$MSE(x) = \frac{1}{n}\sum\limits_{i=1}^n(y_i-\hat{y}_i)^2$


where $y_i$ denotes the observed data, $\hat{y}_i$ denotes the predicted data from $\hat{f}(x)$, and $n$ denotes the number of data points.


The $Bias$ Formula is as follows:
$Bias[(\hat{f}(x))]^2 = \frac{1}{n}\sum\limits_{i=1}^n({\hat{f}(x_i)}-f(x_i))^2$


where $f(x_i)$ denotes the true model prediction of $y$ based on $x_i$,  $\hat{f}(x_i)$ denotes the predicted $\hat{y}_i$ based on our predicted model $\hat{f}(x)$, and $n$ denotes the number of data in our dataset.


The $Variance$ Formula is as follows:
$Var(\hat{f}(x)) = \frac{1}{n}\sum\limits_{i=1}^n(\bar{\hat{y}}-\hat{y}_i)^2$


where $\bar{\hat{y}}$ denotes the average of all the predictions by $\hat{f}$ in the model for a specific datapoint, and $\hat{y}_i$ denotes the predicted piece of data based on $\hat{f}$, and $n$ denotes the number of data in our dataset.



For the total test dataset, average of values at individual $x$ will be used. 




Note that throughout this section, whenever I refer to Bias, I am actually talking about $Bias^2$.



### **Generate Degree of Freedom Range**

I want to plot the flexibility on our x-axis with degrees of freedom instead of smoothness values, so I will generate the plotting space for these smoothness values then translate them to degrees of freedom values.

### **Plot Training MSE, Bias, and Variance**

To observe how MSE, Bias, and Variance change in each model as the degrees of freedom increases, I will plot the values against one another below.



![alt text](https://github.com/ririye01/Exploring-Errors-in-Regression/blob/main/plots/Training_MSE_Bias_Variance.png)



For every single one of these plots, the MSE and Bias decrease as the degrees of freedom increases. This phenomenon makes sense, because the lower the flexibility of our model becomes, the lower the Bias and MSE become in our training data. The increase in degrees of freedom is intended to cause our model to fit closer and closer to the training data, so in every single one of these cases, the plot's pattern more closely fits to those data values. 





I do not expect this behavior to hold true when it comes to the MSE, Bias, and Variance of our testing data, because while this fitting phenomenon may occur with the increase in fit in our training data, it may lead to severe overfitting in our testing data. 



### **Plot Testing MSE, Bias, and Variance**

Below, I will plot degrees of freedom with the testing MSE, Bias, and Variance to check how behavior changes with this decrease in flexibility. 



![alt text](https://github.com/ririye01/Exploring-Errors-in-Regression/blob/main/plots/Testing_MSE_Bias_Variance.png)




Within the true linear model, the $MSE$, $Bias$, and $Variance$ all increase as flexibility decreases, which makes sense here conceptually with the bias-variance trade-off. Additionally, this increase becomes more dramatic as we continue to add more degrees of freedom to the model. The increase in $Bias$ is because of the overfitting of our model, and the $Variance$ increases because of the larger deviations between mean predictions and each individual prediction in our model. $MSE$ is equivalent to the sum of the $Bias$ and $Variance$, and we can see that clearly in the plot above. 


Within the true quadratic model, we see a similar trend here but with one notable difference. The initial $MSE$ and $Bias$ are higher than any of the other values throughout the second plot, and the reasoning is because of the problem of underfitting. In our original plot, the linear model poorly matches reality, and while the increasing $MSE$ and $Bias$ indicate that overfitting is a problem still as we add more degrees of freedom to the cubic spline calculation, the problem is not as bad as overfitting. Predicted cubic spline models with degrees of freedom around 10 which more accurately mimic a quadratic curve lead to the most reduced $MSE$ and $Bias$, because they meet in the middle and do not under- or over-fit to the training data. The $Variance$ trend is the exact same in this model as the true linear model.



Within the true complex nonlinear model, a similar trend unveils itself as well, but the linear $\hat{f}(x)$ really shines as the model with the smallest $MSE$ and $Bias$. I believe this may be the case, because while the linear model seems to be underfit in the initial plot, the true model is a sinusoidal function that approximately revolves around a line. Because of that, the $MSE$ only increases as our model becomes less flexible, even when we use our cubic spline with df=25 that resembles the true model almost identically. The same trend applied to $Bias$, and $Variance$ is the same as the true linear and true quadratic model.



### **Conceptualizing the Difference between Training and Testing Mean-Squared Error**

Now, I want to observe the differences between Training and Testing $MSE$ in each of these true models by graphing them with each other in all of our population examples. 



![alt text](https://github.com/ririye01/Exploring-Errors-in-Regression/blob/main/plots/Training_v_Testing_MSE.png)



Within the true linear model, testing $MSE$ increases and training $MSE$ decreases as the number of degrees of freedom increases in our model. This may be because of overfitting to the training data. The training data may be fit to perfectly, causing extremely minimal deviations between the data and predicted values which minimizes the training $MSE$, but the testing $MSE$ suffers as a result because that fitting only causes further deviations between predictions and reality. 



Within the true quadratic model, the same trend occurs for the most part, except it is clear that the lowest degrees of freedom values lead to overfitting because of the absurdly large testing $MSE$ and pretty large training $MSE$ at the beginning. This is because of underfitting, but then they reach a minimum testing $MSE$ at approximately df=10. Then, as df increases from there on out, the training $MSE$ continues to decrease and the testing $MSE$ increases severely once again.



Within the true non-linear model, both the training and testing $MSE$ actually continue to decrease, which is actually very shocking. The reasoning for this phenomenon is because the reality of the data following a really complex pattern actually is fit to more closely and more closely as the flexibility of our predictive model decreases and the degrees of freedom increase. Because both of these values are decreasing, overfitting isn't an issue unlike the other two models, and the nuanced fitting to the training data creates a model that actually resembles reality.



### **Understanding How Noise Affects Training and Testing MSE**

Specifically on the non-linear model, I am now changing up the Gaussian noise parameter to understand necessarily how an increase of nuance in our data adjusts  $MSE$ behavior.


![alt text](https://github.com/ririye01/Exploring-Errors-in-Regression/blob/main/plots/Noise_Comparison.png)



With minimal Gaussian noise in the data, we see that Training and Testing $MSE$ follow the exact same trend for the most part. They both decrease primarily as flexibility decreases, so a less flexible model accurately predicts it better. I already described this plot previously, because it's the same noise that I used in my other plots in the other parts of this project.



As the level of Gaussian noise increases, however, it seems that the testing $MSE$ starts to deviate from that track of following the training $MSE$, and a predictive model with less and less degrees of freedom and more and more complexity begin to actually predict false results more and more times with larger deviations from reality. In the second plot where noise=3 instead of 1, that curve away from the decreasing trend happens earlier around 40 degrees of freedom. As we go even further and set the noise equal to 5, testing $MSE$ starts to exponentially increase more dramatically, and the testing $MSE$ at all points really does not seem to come close to the training $MSE$. In other words, overfitting may be more of a problem as Gaussian noise increases and variability in our data increases, because it becomes harder and harder for a predictive model to resemble reality when such a big standard deviation exists in our actual $y$-values.




## **References**


[1]  https://www.analyticsvidhya.com/blog/2020/02/what-is-bootstrap-sampling-in-statistics-and-machine-learning/



[2] https://stackabuse.com/matplotlib-scatter-plot-with-distribution-plots-histograms-jointplot/



[3] https://github.com/sayanam/MachineLearning/blob/master/ExperimentationWithBiasAndVariance/BiasAndVariance_V2.ipynb



[4] https://medium.com/analytics-vidhya/calculation-of-bias-variance-in-python-8f96463c8942

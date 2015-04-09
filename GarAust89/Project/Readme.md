## Gareth Austen Project for General Assembly Data Science Spring 2015

### Project Statment and Hypothesis

For my project I choese the Kaggle competition: *Titanic: Machine Learning from Disaster.* The purpose 
of this project is to try and predict what passengers survived and what passengers perished on the titanic. 
The Titanic is one of the most infamous shipwrecks in history and it occured in 1912. On the Titanic there was
a policy of saving the women and children and therefore we can expect the female and child survival rate to be 
signficantly higher than that for men over a certain age.


### Data Description

Kaggle provide two datasets, a training and a test set, that contain identical variables. The training set 
has approximately twice as many observations as the test. A complete list of the variables provided by Kaggle 
is listed below:

```
VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)
```

I began by conducted some preliminary examination into the training dataset. As mentioned earlier, I suspected 
that Male survival ratge would be signficantly lower than Female survival rate. In the mosaic plot below we can
see that this is indeed the case.

![SurvivalRatebyGender](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Project/Plots/MalevsFemaleSurvivalRate.png)

The male survival rate on the Titanic was only **18%** while the Female survival rate was **75%**. Obviously, in 
this case, you were considerably more likely to survive the sinking of the Titanic if you were Female. 

The next variable I took an indepth look at was class. The Titanic was a luxury ship and tickets for First Class 
cost approximately the equivalent of $100,000 at today's rates. In the early 1900s, it took time to acquire this 
kind of wealth (there were no Mark Zuckerberg's in 1912). The below image is a cross section of the Titanic's layout:

![TitanicCrossSection](https://github.com/GarAust89/DAT5_BOS_students/blob/master/GarAust89/Project/Plots/TitanicCrossSection.jpg)

In this image, we can see that the upper levels of the ship are for 1st Class Passengers, followed by 2nd Class Passengers in the middle and 3rd Class Passengers are located deeper in the ship. Therefore, while Passengers in 1st Class are probably older than Passengers in other classes, they are also located closer to the lifeboats and therefore are more likely to reach them first and escape the sinking ship. 

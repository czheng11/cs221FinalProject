# Modeling Indicators of Mental Health Openness in Tech Workplaces
## CS 221 - Final Project
_by Abdallah AbuHashem, Grace Hu, and Crystal Zheng_

Mental health is an issue that impacts many people worldwide, every one in four people. Mental health disorders escape undiagnosed even though they have immense impact on people and their families. In fact, someone dies from suicide every 40 seconds. In this project, we aimed to determine the key factors that contribute to the progression and magnification of mental disorders. Our goal is to better understand the relationship between employees and companies and also help companies predict the progression of mental health disorders and how to deal with them.

## Data
We took the data from 4 datasets obtained from Kaggle from the Open Sourcing Mental Illness (OSMI) nonprofit’s Mental Health in Tech Survey, 2014, 2016, 2017, and 2018.

Source: Open Sourcing Mental Illness (OSMI). 2014-2018 OSMI Mental Health in Tech Survey. Retrieved from https://osmihelp.org/research

Enclosed in this folder are the following:
1) Cleaned datasets after pre-processing for 2014, 2016, 2017, and 2018
2) Code in python / python notebook files

Some of the main models ran include:
- Neural networks
- Logistic Regression (SGD + Adam)
- PCA
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

## Takeaways
Overall, we were able to successfully test and implement multiple classifiers to 87-91% on the task of predicting whether an employee would be willing to discuss their mental health with their supervisor and whether the employee would be willing to discuss mental health with coworkers. 
* Out of all our models, our neural network with SGD optimizer performed best on the supervisor task while our SVM with sigmoid kernel performed best on the coworker(s) task. 
* Furthermore, preliminary findings in establishing interpretability to our models were promising as our extracted weights from our logistic regression models gave us some interesting insights as to what factors would strongly contribute, positive or negatively, to discussions of mental health with supervisor and coworkers. 
* Some particularly notable ones included the influence of being in a small-mid size company to willingness to discuss in both cases.

There’s still a lot of improvements to be made to address this sensitive issue in the workspace. A more consistent and larger dataset could potentially further improve results, which is encouraging since over the next few years, thousands of new data entries would be inputted and more applicable features could be measured. Ultimately, we hope that using AI for social good will bring the tech industry towards a new direction for proactively talking about mental health and wellbeing for all employees

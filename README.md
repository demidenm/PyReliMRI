# MRI Reliability

## Intro of Problem

MRI litearture is asking an increasing number of reliability questions for [task fMRI](https://doi.org/10.1177/0956797620916786) and [resting state fMRI](www.doi.org/10.1016/j.neuroimage.2019.116157). As attention to these questions increase, it is important to acknowledge the different methods to calculating reliability in fMRI, [as has been documents](www.doi.org/10.1111/j.1749-6632.2010.05446.x).

## Purpose of Script

The purpose of this tool is to provide an open-source package that will provide multiple reliability metrics, at the group and individual level, that researchers can report with their manuscripts. At the individual level metrics, we plan to provide an `ICC` function that calculates ICC(1), ICC(2,1) or ICC(3,1), for description see discussion in (Liljequist et al., 2019)[www.doi.org/10.1371/journal.pone.0219854]. In addition to the ICC, at the individual level we plan to provide a function to calculate the moment product correlation between sessions. At the group level, we plan to provide similarity calculations using Jaccard and Dice similarity coefficients. Since the Dice & Jaccard coefficients are biased by the threshold level between two images, we will provide an iterable approach to calculate the coefficients across a range of thresholds. Finally, for the Jaccard and Dice coefficients we will also include a permutation function that will allow inclusion of 2+ group level maps and calculate simiarlity coefficients for all coefficients. 

## What is included in script

The script folder current contains to scripts:

  1. `Validating_ICCCalculations.R`: This is an R script validating the manual ANOVA based calculates to the `Psych` packaged which calculates ICC1 - ICC3 using an *aov* and *lmer* approach.
  2. `Calc_Similarities.py`: This is the python script that calculates similarities at the individual and group level.

    a. the `similarity` function takes in the path to two .nii images with the specification for a) threshold and b) type of similarity coefficient. Threshold can be looped and so can be an interger =>0. The similarities coefficeints are either 'Dice' or 'Jaccards', with the default = Dice. Dice Coefficient (DC) is the interaction of two images (i.e., active voxels, T/F) divided by the union of the two images. The Jaccards Coefficient is similar to dice, except JC is *DC / (2 - DC)*. See description on [Wikipedia](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient). Hence, both coefficients will be contrained to 0 - 1 and JC will always be lower than the DC. 
    b. the `permute_similarity` functions can take a list of .nii images, pre-specified threshold (interger:>=0) and similarity coefficient (str: Dice/Jaccards). It will create create different permutations for the list and calculate similarity coefficients for all combinations. It will return a list of coefficients for the list of combinations (order is returned, too, in separate variable).
    c. the `Calc_icc` function is similar to the validation in the R script, just a python implementation using a Panda's dataframe. It takes in a wide dataframe, with specified id variables (subject/inter-subj vars) and value variables (session variables/intra-subj vars). For example, one can specific the subject varable as "Subject" and Sessions as ["Session1","Session2","Session3"] and specified the type of icc (i.e., str 'icc_1','icc_2','icc_3'). Default ICC is ICC(2,1). Consistent with recommended started ICC calculation in [Noble et al., 2021](https://www.sciencedirect.com/science/article/pii/S235215462030200X)
    d. the `MRI_ICCs` function takes in a list of session paths to nii files and computes a specified ICC. If specified sess1 and sess1 (e.g, n_sessions=2) the script will concatenate the images for all subjects using nilearns image.concat_imgs and then access pull data into numpy array. A the length of the session is used to create length N of subjects to used as the subject variable for ICC. A 4D to 2D function is used to convert the data into a 2D format, whereby positon 1 = voxels (length = 3D shape) and positioon 2 = subjects. A for loop iterates over the subjects, cuts along the rows for each session and creates a multiple column pandas dataframe, (Subject, Sess1 Voxel, Sess2 Voxel) and feeds it into the `Calc_icc'. 
    
      i. note, currently doesn't mask session 4D volumes, may be useful to ensure that only voxels in both sessions across all subjects are used.

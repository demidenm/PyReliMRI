library(psych)
library(tidyverse)

# This code is to validate the python ICC code to the ICC calculates from the psych packages
# Note, the psych package's `ICC` function uses lmer and aov calculations to calculate ICCs

# synth df
set.seed(1234)
x = rnorm(n = 100, m = 11, sd = 1)
x1 = x * .5 + rnorm(100, m = 5, sd = 1)
x2 = x * .4 + x1 * .3 + rnorm(100, m = 16, sd = 1)
df1 = data.frame(x,x1,x2)

df1$subj = 1:nrow(df1)

df1_lng = df1 %>% 
  gather(key = 'rate', value='score',x:x2) 



n = nrow(df1)
c = ncol(df1[,-4])
DF_n = n - 1
DF_c = c - 1
DF_r = (n-1)*(c-1)
  
# Sum Square Values
SS_T = sum((as.matrix(df1_lng$score)-mean(as.matrix(df1_lng$score)))^2)

# r = subjects, between
SS_R = df1_lng %>% 
  mutate(grand_mean = mean(score)) %>% 
  group_by(subj) %>% 
  mutate(subj_var = (mean(score) - grand_mean)^2) %>% 
  pull(subj_var) %>% 
  sum()

# c = columns, measurement occaisons, within measurement error
SS_C = df1_lng %>% 
  mutate(grand_mean = mean(score)) %>% 
  group_by(rate) %>% 
  mutate(rate_var = (mean(score) - grand_mean)^2) %>% 
  pull(rate_var) %>% 
  sum()

# Sum Square Errors
SSE = SS_T - SS_R - SS_C

# Sum square withins subj err
SSW = SS_C + SSE

# Mean Squared Values
MSR = SS_R/(DF_n)
MSC = SS_C/(DF_c)
MSE = SSE/(DF_r)
MSW = SSW/(n*(DF_c))


# ICC(1), Model 1
ICC_1 = (MSR-MSW) / (MSR+(DF_c*MSW))
# ICC(2,1)
ICC_2.1 = (MSR - MSE) / (MSR + (DF_c)*MSE + (c)*(MSC-MSE)/n)

# ICC(3,1)
ICC_3.1 <- (MSR - MSE) /(MSR + (DF_c) * MSE)

ICC(df1[,-4],)



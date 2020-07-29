# libmr_gpu
Stand-alone libmr without python wrapping. Converting parts of code to cuda in an effort to speed up the weibull fitting for the EVMs.

# current_status
Have converted some of the functions from weibull.c to kernels, however, we might not see improvement due to this conversion. Unless MR class is parallelized i.e. to run the weibull fitting for different instances in parallel. 

# moving_forward
Have decided to continue with PyTorch based weibull fitting from https://github.com/mlosch/python-weibullfit which leverages CUDA.
Able to get the same exact shape/scale parameters for weibulls when the data(distances) are sorted and pre-processed using the code from libmr. 
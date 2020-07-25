/*  \index
 * weibull_gpu.c provides the gpu implementation of the weibull distribution -- follows code in weibull.c and leverages gpu 
 * computations where possible.
 * @Author Touqeer Ahmad  touqeer at vast dot uccs dot edu
 * Vision and Security Tenchonolgy Lab
 * University of Colorado, Colorado Springs 
 */


#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <float.h>
#include <time.h>

#include "weibull_gpu.h"
#include "GpuSolver.h"

#ifdef __cplusplus
extern "C" {
#endif

  
static double weibull_scale_likelihood(double sigma, double* x, double* w, double xbar, int size)
{
  double v;
  double* wLocal;
  int i;
  double sumxw;
  double sumw;
  
  printf("Printing w[i] in weibull_scale_likelihood ...");
  for (int k = 0; k < size; k++)
    printf("%f\n",w[k]);
  
    
  wLocal=(double*)malloc(sizeof(double)*size);
    
  for (i=0; i<size; i++)
  {
    wLocal[i]=w[i]*exp(x[i]/sigma);
  }
  
  sumxw=0;
  sumw=0;
    
  for (i=0; i<size; i++)
  {
    sumxw+=(wLocal[i]*x[i]);
    sumw+=wLocal[i];
  }
    
  v = (sigma + xbar - sumxw / sumw);

  free(wLocal);
  return v;
}  

int weibull_fit_gpu(double* weibullparms, double* wparm_confidenceintervals, double* inputData, double alpha, int size)
{
  printf("\n\nIn weibull_fit_gpu function ...\n");
  clock_t t_start; 
  t_start = clock();  
  
  double PI =  3.141592653589793238462;
  double FULL_PRECISION_MIN = 2.225073858507201e-308; /* smalled full precision positive number anything smaller is unnormalized, for testing for underflow */
  double FULL_PRECISION_MAX = 1.797693134862315e+308; /* largest full precision positive number, for testing for overflow */
  double  tol = 1.000000000000000e-006;/* this impacts the non-linear estimation..  if your problem is highly unstable (small scale) this might be made larger but we never recommend anything greater than 10e-5.  Also if larger it will converge faster, so if yo can live with lower accuracy, you can change it */
  double n;
  double nuncensored=0;
  double ncensored=0;
  int i;
  int code;

  double *censoring= (double *)malloc(sizeof(double)*size);
  double *frequency  = (double *)malloc(sizeof(double)*size);
  double * var = (double *)malloc(sizeof(double)*size);
  double* x0 =    (double *)malloc(sizeof(double)*size);


  /*set frequency to all 1.0's */
  /*and censoring to 0.0's */
  for (i=0; i< size; i++)
  {
    frequency[i]=1.0;
    censoring[i]=0.0;
  }

  printf("Data before and after log:\n");

  /*  ********************************************** */
  for (i=0; i<size; i++)
  {
    printf("Before log = %f, ", inputData[i]);
    inputData[i]=log(inputData[i]);
    printf("After log = %f \n", inputData[i]);
  }
  /*  ********************************************** */
  {
    double mySum;
  
    mySum=0;
    for (i=0; i<size; i++)
    {
      mySum+=frequency[i];
    }
  
    printf("mySum = %f\n", mySum);
  
    n=mySum;
    
     
    /*  ********************************************** */
    {
      mySum=0;
    
      for (i=0; i<size; i++)
      {
        mySum+=(frequency[i]*censoring[i]);
      }
    
      ncensored=mySum;
      nuncensored = n - ncensored;

    }
  }

  /* declar local for max/range computation  ********************************************** */
  {
    double maxVal, minVal;
    double range, maxx;
    double tempVal;
  
    maxVal=-1000000000;
    minVal=1000000000;
  
    printf("Computing the minVal and maxVal \n");
    printf("inputData[0] = %f", inputData[0]);
    printf("inputData[size-1] = %f", inputData[size-1]);
  
  
    for (i=0; i<size; i++)
    {
      tempVal=inputData[i];
      
      if (tempVal < minVal)
        minVal=tempVal;
    
      if (tempVal > maxVal)
        maxVal=tempVal;
    }
  
    printf("Computed the minVal and maxVal \n");
    printf("minVal = %f\n", minVal);
    printf("maxVal = %f\n", maxVal);
  
    range = maxVal - minVal;
    maxx = maxVal;
    
    
    
    /*Shift x to max(x) == 0, min(x) = -1 to make likelihood eqn more stable. */
    /*  ********************************************** */
    {
      double mean, myStd;
      double sigmahat;
      double meanUncensored;
      double upper, lower;
      double search_band[2];
      
      mean=0;
      myStd=0;
      
      
      /*
      printf("Now printing x0 through CPU:\n");
       
      for (i=0; i<size; i++)
      {
        x0[i]=(inputData[i]-maxx)/range;
        printf("%f\n", x0[i]);
      }
      
       
      for (i=0; i<size; i++)
      {
        mean+=x0[i];
      }
      
      mean /= n;
      printf("Now printing mean = %f \n", mean);
      
      for (i=0; i<size; i++)
      {
        var[i] = x0[i] - mean;
      }
      
      for (i=0; i<size; i++)
      {
        myStd+=var[i]*var[i];
      }
      
      myStd/=(n-1);
      myStd=sqrt(myStd);
      printf("Now printing myStd = %f \n", myStd);
      */
      
      
      // GPU replacement of the above code
      ///*
      runKernels_ComputeMeanAndStd(inputData, x0, &mean, &myStd, maxx, range, size);
      printf("Now printing x0 through GPU:\n");
      for (i=0; i<size; i++)
        printf("%f\n", x0[i]);
      printf("Now printing mean = %f \n", mean);
      printf("Now printing myStd = %f \n", myStd);
      //*/
      
      
      sigmahat = (sqrt((double)(6.0))*myStd)/PI;
      printf("sigmahat = %f\n", sigmahat);
      
      
      
      meanUncensored=0;
      
      for (i=0; i<size; i++)
      {
        meanUncensored+=(frequency[i]*x0[i])/n;
      }
      printf("meanUncensored = %f\n", meanUncensored);
      
      if ((tempVal=weibull_scale_likelihood(sigmahat,x0,frequency,meanUncensored,size)) > 0)
      {
        printf("In the if condition ...\n");
        
        upper=sigmahat;
        lower=0.5*upper;
        
        while((tempVal=weibull_scale_likelihood(lower,x0,frequency,meanUncensored,size)) > 0)
        {
          upper = lower;
          lower = 0.5 * upper;
          
          if (lower < FULL_PRECISION_MIN)
          {
            printf("MLE in wbfit Failed to converge leading for underflow in root finding\n");
          }
        }
      }
      else
      {
        printf("In the else part ...\n");
        
        lower = sigmahat;
        upper = 2.0 * lower;
        
        while ((tempVal=weibull_scale_likelihood(upper,x0,frequency,meanUncensored,size)) < 0)
        {
          lower=upper;
          upper = 2 * lower;
        }
      }
      /* ****************************************** */
      search_band[0]=lower;
      search_band[1]=upper;
      
      printf("lower = %f\n", lower);
      printf("upper = %f\n", upper);
      
    }
    
  }
  
  //runKernels();
  
  
}

#ifdef __cplusplus
}
#endif


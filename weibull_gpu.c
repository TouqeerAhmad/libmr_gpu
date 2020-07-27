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


static int  weibull_neg_log_likelihood(double* nlogL, double* acov, double* weibulparms, double* data,
                                         double* censoring, double* frequency, int size)
{
  runKernels_NegLogLikelihood(nlogL, acov, weibulparms, data, censoring, frequency, size);
  return 0;
}

    
static double weibull_scale_likelihood(double sigma, double* x, double* w, double xbar, int size)
{ 
  return runKernels_ScaleLikelihood(sigma, x, w, xbar, size);
}

 
/* based on dfzero from fortan, it finxs the zero in the given search bands, and stops if it is within tolerance. */
static int wdfzero(double* sigmahat, double* likelihood_value, double* err, double* search_bands, double tol,
                     double* x0, double* frequency, double meanUncensored, int size)
{
  double exitflag;
  double a,b,c=0.0,d=0.0,e=0.0,m,p,q,r,s;
  double fa,fb,fc;
  double fval;
  double tolerance;
    
  exitflag=1;
  *err = exitflag;
    
  a = search_bands[0];
  b = search_bands[1];
    
  fa = weibull_scale_likelihood(a,x0,frequency,meanUncensored,size);
  fb = weibull_scale_likelihood(b,x0,frequency,meanUncensored,size);
    
  if (fa == 0)
  {
    b=a;
    *sigmahat=b;
    fval = fa;
    *likelihood_value = fval;
    return 1;
  }
  else if (fb == 0)
  {
    fval=fb;
    *likelihood_value = fval;
    *sigmahat=b;
    return 1;
  }
  else if ((fa > 0) == (fb > 0))
  {
    //WEIBULL_ERROR_HANDLER(-4,"ERROR: wdfzero says function values at the interval endpoints must differ in sign\n");
  }
    
  fc = fb;
    
  /*Main loop, exit from middle of the loop */
  while (fb != 0)
  {
    /* Insure that b is the best result so far, a is the previous */
    /* value of b, and that c is  on the opposite size of the zero from b. */
    if ((fb > 0) == (fc > 0))
    {
      c = a;
      fc = fa;
      d = b - a;
      e = d;
    }
      
    {
      double absFC;
      double absFB;
        
      absFC=fabs(fc);
      absFB=fabs(fb);
        
      if (absFC < absFB)
      {
        a = b;
        b = c;
        c = a;
        fa = fb;
        fb = fc;
        fc = fa;
      }
    }
      
    /*set up for test of Convergence, is the interval small enough? */
    m = 0.5*(c - b);
      
    {
      double absB,  absM,  absFA,absFB, absE;
      absB=fabs(b);
      absM=fabs(m);
      absFA=fabs(fa);
      absFB=fabs(fb);
      absE=fabs(e);
        
      {
        tolerance = 2.0*tol *((absB > 1.0) ? absB : 1.0);
        
        if ((absM <= tolerance) | (fb == 0.0))
          break;
          
        /*Choose bisection or interpolation */
        if ((absE < tolerance) | (absFA <= absFB))
        {
          /*Bisection */
          d = m;
          e = m;
        }
        else
        {
          /*Interpolation */
          s = fb/fa;
            
          if (a == c)
          {
            /*Linear interpolation */
            p = 2.0*m*s;
            q = 1.0 - s;
          }
          else
          {
            /*Inverse quadratic interpolation */
            q = fa/fc;
            r = fb/fc;
            p = s*(2.0*m*q*(q - r) - (b - a)*(r - 1.0));
            q = (q - 1.0)*(r - 1.0)*(s - 1.0);
          }
            
          if (p > 0)
            q = -1.0*q;
          else
            p = -1.0*p;
        }
      }
      
      
      {
        double tempTolerance = tolerance*q;
        double absToleranceQ;
        double absEQ;
        double tempEQ = (0.5 * e * q);
        absToleranceQ=fabs(tempTolerance);
        absEQ=fabs(tempEQ);
          
        /*Is interpolated point acceptable */
        if ((2.0*p < 3.0*m*q - absToleranceQ) & (p < absEQ))
        {
          e = d;
          d = p/q;
        }
        else
        {
          d = m;
          e = m;
        }
      }
        
    } /*Interpolation */
          
    /*Next point */
    a = b;
    fa = fb;
      
    if (fabs(d) > tolerance)
      b = b + d;
    else if (b > c)
      b = b - tolerance;
    else
      b = b + tolerance;
      
    fb = weibull_scale_likelihood(b,x0,frequency,meanUncensored,size);
      
  }/*Main loop (While) */
          
  fval=weibull_scale_likelihood(b,x0,frequency,meanUncensored,size);
  *likelihood_value = fval;
  *sigmahat=b;
    
  return 1;
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
      
      
      // ... Next we  go find the root (zero) of the likelihood eqn which  wil be the MLE for sigma. 
      // then  the MLE for mu has an explicit formula from that.  */
      
      {
        double err;
        double likelihood_value;
        code = wdfzero(&sigmahat,&likelihood_value,&err,search_band,tol,x0,frequency,meanUncensored,size);
        printf("code = %d\n", code);
        
      }
      
      
      /* ****************************************** */
      {
        double muHat;
        double sumfrequency;
        
        muHat=0;
        sumfrequency=0;
        
        for (i=0; i<size; i++)
        {
          tempVal=exp(x0[i]/sigmahat);
          sumfrequency +=(frequency[i]*tempVal);
        }
        
        sumfrequency = sumfrequency / nuncensored;
        muHat = sigmahat * log(sumfrequency);
        
        /* ****************************************** */
        
        /*Those were parameter estimates for the shifted, scaled data, now */
        /*transform the parameters back to the original location and scale. */
        weibullparms[0]=(range*muHat)+maxx;
        weibullparms[1]=(range*sigmahat);
      }
      
    }
  }
  
  
  {
    int rval;
    double nlogL=0, tempVal;
    double transfhat[2], se[2], probs[2],acov[4];
    
    probs[0]=alpha/2;
    probs[1]=1-alpha/2;
    /* ****************************************** */
    
    
    rval=weibull_neg_log_likelihood(&nlogL,acov,weibullparms,inputData,censoring,frequency,size);
    
    printf("nlogL  = %f\n", nlogL);
    printf("acov[0]  = %f\n", acov[0]);
    printf("acov[1]  = %f\n", acov[1]);
    printf("acov[2]  = %f\n", acov[2]);
    printf("acov[3]  = %f\n", acov[3]);
    
    //if(rval<0) WEIBULL_ERROR_HANDLER(-5,"Failed to fine final parameters settings MLE failed. Memory leaked");
    
    /* ****************************************** */
    /*Compute the Confidence Interval (CI)  for mu using a normal approximation for muhat.  Compute */
    /*the CI for sigma using a normal approximation for log(sigmahat), and */
    /*transform back to the original scale. */
    
    transfhat[0]=weibullparms[0];
    transfhat[1]=log(weibullparms[1]);
    
    se[0]=sqrt(acov[0]);
    se[1]=sqrt(acov[3]);
    se[1]=se[1]/weibullparms[1];
    
    //rval=wnorminv(wparm_confidenceintervals,probs,transfhat,se,4);
    //if(rval<0) WEIBULL_ERROR_HANDLER(-7,"Cannot compute confidence interval since wnorminv fails. Memory leaked");
    
    //wparm_confidenceintervals[2]=exp(wparm_confidenceintervals[2]);
    //wparm_confidenceintervals[3]=exp(wparm_confidenceintervals[3]);
    
    //tempVal=wparm_confidenceintervals[2];
    //wparm_confidenceintervals[2]=1/wparm_confidenceintervals[3];
    //wparm_confidenceintervals[3]=1/tempVal;
    
    //wparm_confidenceintervals[0]=exp(wparm_confidenceintervals[0]);
    //wparm_confidenceintervals[1]=exp(wparm_confidenceintervals[1]);
    
    //weibullparms[0]=exp(weibullparms[0]);
    //weibullparms[1]=1/weibullparms[1];
  }
  
  /*free all memory */
  free(x0);
  free(var);
  free(censoring);
  free(frequency);
  
  
  t_start = clock() - t_start; 
  double time_taken = ((double)t_start)/CLOCKS_PER_SEC; // in seconds 
  printf("fun() took %f seconds to execute \n", time_taken); 
  
  
}

#ifdef __cplusplus
}
#endif


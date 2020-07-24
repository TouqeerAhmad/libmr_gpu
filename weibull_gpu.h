/*
 * weibull_gpu.h:
 * @Author Touqeer Ahmad  touqeer at vast dot uccs dot edu
 * Vision and Security Tenchonolgy Lab
 * University of Colorado, Colorado Springs 

 */

#pragma once
#ifndef WEIBULL_GPU_H
#define WEIBULL_GPU_H

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif
  _declspec(dllexport) int weibull_fit_gpu(double* weibull_parms, double* wparm_confidenceintervals, double* inputData, double alpha, int size);
#ifdef __cplusplus
}
#endif
#else
#ifdef __cplusplus
extern "C" {
#endif
  int weibull_fit_gpu(double* weibullparms, double* wparm_confidenceintervals, double* inputData, double alpha, int size);
#ifdef __cplusplus
}
#endif

#endif
#endif

  
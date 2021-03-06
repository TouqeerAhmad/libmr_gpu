
#pragma once
#ifndef GPUSOLVER_H
#define GPUSOLVER_H

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif
  _declspec(dllexport) void runKernels_ComputeMeanAndStd(double * inputData, double * x0, double *mean, double *myStd, double maxx, double range, int size);
  _declspec(dllexport) double runKernels_ScaleLikelihood(double sigma, double *x, double *w, double xbar, int size);
  _declspec(dllexport) void runKernels_NegLogLikelihood(double* nlogL, double* acov, double* weibulparms, double* data, double* censoring, double* frequency, int size);
#ifdef __cplusplus
}
#endif
#else
#ifdef __cplusplus
extern "C" {
#endif
  void runKernels_ComputeMeanAndStd(double * inputData, double * x0, double *mean, double *myStd, double maxx, double range, int size);
  double runKernels_ScaleLikelihood(double sigma, double *x, double *w, double xbar, int size);
  void runKernels_NegLogLikelihood(double* nlogL, double* acov, double* weibulparms, double* data, double* censoring, double* frequency, int size);
#ifdef __cplusplus
}
#endif

#endif
#endif


#pragma once
#ifndef GPUSOLVER_H
#define GPUSOLVER_H

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif
  _declspec(dllexport) void runKernels();
  _declspec(dllexport) void runKernels_ComputeMeanAndStd(double * inputData, double * x0, double *mean, double *myStd, double maxx, double range, int size);
#ifdef __cplusplus
}
#endif
#else
#ifdef __cplusplus
extern "C" {
#endif
  void runKernels();
  void runKernels_ComputeMeanAndStd(double * inputData, double * x0, double *mean, double *myStd, double maxx, double range, int size);
#ifdef __cplusplus
}
#endif

#endif
#endif









void runKernels();
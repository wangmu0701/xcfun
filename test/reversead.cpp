#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "xcfun.h"

#include <iostream>
#include <chrono>

#include <memory>
#include "reversead/reversead.hpp"
using namespace ReverseAD;

#define TOL 1.0e-8


void test_on(int func_len, char (*func_name)[20], double* func_weight) {
  xc_functional fun = xc_new_functional();

  double d_elements[20] = {1, 2.1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                         1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8};
  
  int nout,i;

  double *output;
  double *output_ad;

  for (int i = 0; i < func_len; i++) {
    xc_set(fun, func_name[i], func_weight[i]);
  }

  size_t f_order = 5;

  xc_eval_setup(fun,
        XC_A_B_2ND_TAYLOR,
        XC_PARTIAL_DERIVATIVES,
        f_order);

  nout = xc_output_length(fun);

  output_ad = (double*)malloc(sizeof(double)*nout);
  output = (double*)malloc(sizeof(double)*nout);


  auto t1 = std::chrono::high_resolution_clock::now();
  xc_eval(fun, d_elements, output);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "Taylor method time = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
            << " milliseconds" << std::endl;
/*            
  t1 = std::chrono::high_resolution_clock::now();
  xc_eval_reversead(fun,d_elements,output_ad);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "ReverseAD time = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
            << " milliseconds" << std::endl;
*/
  t1 = std::chrono::high_resolution_clock::now();
  std::shared_ptr<DerivativeTensor<size_t, double>> tensor = xc_eval_reversead_tensor(fun, d_elements, f_order);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "ReverseAD::Tensor["<<f_order<<"] time = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
            << " milliseconds" << std::endl;
  size_t size;
  size_t** tind;
  double* values;
  tensor->get_internal_coordinate_list(0, f_order, &size, &tind, &values);  
  std::cout << "NNZ in tensor = " << size << std::endl;
/*
  bool isWrong = false;
  for (i=0;i<nout;i++) {
    //printf("%.8f == %.8f\n", output_ad[i], output[i]);
    if (fabs(output_ad[i] - output[i]) > TOL) {
      isWrong = true;
    }
  }
  if (!isWrong) {
    std::cout << "Results corrects!" << std::endl;
  } else {
    std::cout << "Results wrong!" << std::endl;
  }
*/
  free(output);
  free(output_ad);
  xc_free_functional(fun);  

}

int main(void) {

  char func_name[10][20] = {"lda", "blyp", "pbe", "bp86", "kt1", "kt2", "kt3",
                           "pbe0", "b3lyp", "b97"};
  double func_weight[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

  test_on(10, func_name, func_weight);

  return EXIT_SUCCESS;
}

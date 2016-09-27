#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <iostream>
#include "xcfun.h"
#include "xcint.hpp"

#define TOL_EPS 1.0e-8

extern struct alias_data* xcint_aliases;

void test_on(char const * const func_name, double func_weight) {
  xc_functional fun = xc_new_functional();

  double d_elements[20] = {1, 2.1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                         1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8};

  int nout,i;

  double *output;
  double *output_ad;

  xc_set(fun, func_name, func_weight);

  xc_eval_setup(fun,
        XC_A_B_2ND_TAYLOR,
        XC_PARTIAL_DERIVATIVES,
        3);
  nout = xc_output_length(fun);

  output_ad = (double*)malloc(sizeof(double)*nout);
  output = (double*)malloc(sizeof(double)*nout);
  for (int i = 0; i < nout; i++) {
    output_ad[i] = 0;
    output[i] = 0;
  }

  xc_eval(fun, d_elements, output);
  xc_eval_reversead(fun, d_elements, output_ad);

  bool isWrong = false;
  for (i=0;i<nout;i++) {
    //printf("%.8f == %.8f\n", output_ad[i], output[i]);
    if (fabs(output_ad[i] - output[i]) > TOL_EPS) {
      isWrong = true;
    }
  }

  if (!isWrong) {
    std::cout << "Functional " << func_name << " passed" << std::endl;
  } else {
    std::cout << "Functional " << func_name << " failed" << std::endl;
    for (i=0;i<nout;i++) {
      //printf("%.8f == %.8f\n", output_ad[i], output[i]);
    }
    exit(-1);
  }

  free(output);
  free(output_ad);
  xc_free_functional(fun);  

}

int main(void)
{

  test_on("blyp", 0.1);
  test_on("pbec", 0.9);
  std::set<int> black_list = {11, 12, 13};
  for (int i = 1; i < 40; i++) {
    if (black_list.find(i) == black_list.end()) {
      std::cout << "Testing on ["<<i<<"]  " << xcint_aliases[i].name << "  ";
      test_on(xcint_aliases[i].name, 1.0);
    }
  }
  return EXIT_SUCCESS;
}

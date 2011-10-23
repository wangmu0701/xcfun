#include "xcint.hpp"

static struct alias_data aliases_array[XC_MAX_ALIASES] =
{
  {"lda","Slater exchange and VWN5 correlation",{{"slaterx",1.0},{"vwn5c",1.0}} },
  {"blyp","Becke exchange and LYP correlation",{{"beckex",1.0},{"lypc",1.0}} },
};

struct alias_data *xcint_aliases = aliases_array;

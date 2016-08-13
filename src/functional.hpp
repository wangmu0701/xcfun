#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include "ctaylor.hpp"
#include "config.hpp"
#include "specmath.hpp"
#include "xcint.hpp"

#define FUNCTIONAL(F) template<> const char *fundat_db<F>::symbol = #F; template<> functional_data fundat_db<F>::d
#define EN(N,FUN) FUN<ctaylor<ireal_t,N> >,
#define ENERGY_FUNCTION(FUN) FOR_EACH(XC_MAX_ORDER,EN,FUN)\
                             FUN<adouble>,
#define PARAMETER(P) template<> const char *pardat_db<P>::symbol = #P; template<> parameter_data pardat_db<P>::d

#endif

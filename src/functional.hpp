#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include "ctaylor.hpp"
#include "config.hpp"
#include "specmath.hpp"
#include "xcint.hpp"

#define FUNCTIONAL(F) template<> const char *fundat_db<F>::symbol = #F; template<> functional_data fundat_db<F>::d
#define EN(N,FUN) FUN<ctaylor<ireal_t,N> >,

#ifdef XCFUN_REVERSEAD
#define ENERGY_FUNCTION(FUN) FOR_EACH(XC_MAX_ORDER,EN,FUN)\
                             FUN<adouble>,
#else
#ifdef XCFUN_RAPSODIA
#define ENERGY_FUNCTION(FUN) FOR_EACH(XC_MAX_ORDER,EN,FUN)\
                             FUN<RAfloatD>,
#else
#ifdef XCFUN_ADOLC
#define ENERGY_FUNCTION(FUN) FOR_EACH(XC_MAX_ORDER,EN,FUN)\
                             FUN<adouble>,
#else
#define ENERGY_FUNCTION(FUN) FOR_EACH(XC_MAX_ORDER,EN,FUN)
#endif // XCFUN_ADOLC
#endif // XCFUN_RAPSODIA
#endif // XCFUN_REVERSEAD





#define PARAMETER(P) template<> const char *pardat_db<P>::symbol = #P; template<> parameter_data pardat_db<P>::d

#endif

#ifndef B97X_H
#define B97X_H

#include "b97xc.hpp"
#define PREFACTOR 0.9305257363491002 // 1.5*pow(3/(4*PI),1.0/3.0)

namespace b97x
{
    
    // LSDA factor, for alpha and beta spin

    template<class num>
    static num e_x_LSDA_ab(const num &a_43)
    {
        return -PREFACTOR*a_43;
    }
    
    
    // parameters for enhancement factor; c0,c1,c2
    
    const parameter c_b97[3] = { 0.8094, 0.5073, 0.7481};
    const parameter c_b97_1[3] = { 0.789518, 0.573805, 0.660975};
    const parameter c_b97_2[3] = { 0.827642, 0.047840, 1.76125};
    const parameter Gamma = 0.004;
    
    
    template<class num>
    static num energy_b97x_ab(const parameter &Gamma,
                              const parameter c_params[], const num &a_43, const num &gaa)
    {
        num s2_ab = b97xc::spin_dens_gradient_ab2(gaa, a_43);
        return e_x_LSDA_ab(a_43)*b97xc::enhancement(Gamma, c_params, s2_ab);
    }
    
}

    
#endif


# fluid_sim_hist.pyx
import numpy as np
import cython
import time
import h5py
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t

from libc.math cimport sqrt, sqrtf, exp, log, cbrt, fabs, ceil, M_PI, floor
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.string cimport memcpy
from libc.time cimport time, clock_t, clock, CLOCKS_PER_SEC
from cython.parallel import prange, parallel
from openmp cimport omp_get_max_threads, omp_get_num_threads, omp_get_thread_num, omp_set_num_threads
omp_set_num_threads(8)

################### utils for fluid sim #######################################

cdef extern from * nogil:
    """
    void parallel_simd_update1(int size, float half_dt, float six_dt,
        float* restrict temp_E_ptr, float* restrict temp_Zx_ptr, float* restrict temp_Zy_ptr,
        float* restrict temp_Zz_ptr, float* restrict temp_Phi_ptr, float* restrict temp_VarPhi_ptr,
        float* restrict E_ptr, float* restrict Zx_ptr, float* restrict Zy_ptr,
        float* restrict Zz_ptr, float* restrict Phi_ptr, float* restrict VarPhi_ptr,
        const float* restrict k1_E_ptr, const float* restrict k1_Zx_ptr, const float* restrict k1_Zy_ptr,
        const float* restrict k1_Zz_ptr, const float* restrict k1_P_ptr, const float* restrict k1_V_ptr)
    {
        #pragma omp parallel for simd schedule(static) \
            aligned(temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, \
                    temp_Phi_ptr, temp_VarPhi_ptr, E_ptr, Zx_ptr, Zy_ptr, \
                    Zz_ptr, Phi_ptr, VarPhi_ptr, k1_E_ptr, k1_Zx_ptr, \
                    k1_Zy_ptr, k1_Zz_ptr, k1_P_ptr, k1_V_ptr:32)
        for (int i = 0; i < size; i++) {
            float kE = k1_E_ptr[i];
            float kZx = k1_Zx_ptr[i];
            float kZy = k1_Zy_ptr[i];
            float kZz = k1_Zz_ptr[i];
            float kP = k1_P_ptr[i];
            float kV = k1_V_ptr[i];
            temp_E_ptr[i] += half_dt * kE;
            temp_Zx_ptr[i] += half_dt * kZx;
            temp_Zy_ptr[i] += half_dt * kZy;
            temp_Zz_ptr[i] += half_dt * kZz;
            temp_Phi_ptr[i] += half_dt * kP;
            temp_VarPhi_ptr[i] += half_dt * kV;
            E_ptr[i] += six_dt * kE;
            Zx_ptr[i] += six_dt * kZx;
            Zy_ptr[i] += six_dt * kZy;
            Zz_ptr[i] += six_dt * kZz;
            Phi_ptr[i] += six_dt * kP;
            VarPhi_ptr[i] += six_dt * kV;
        }
    }
    void parallel_simd_update2(int size, float mix_dt, float half_dt, float three_dt,
        float* restrict temp_E_ptr, float* restrict temp_Zx_ptr, float* restrict temp_Zy_ptr,
        float* restrict temp_Zz_ptr, float* restrict temp_Phi_ptr, float* restrict temp_VarPhi_ptr,
        float* restrict E_ptr, float* restrict Zx_ptr, float* restrict Zy_ptr,
        float* restrict Zz_ptr, float* restrict Phi_ptr, float* restrict VarPhi_ptr,
        const float* restrict k1_E_ptr, const float* restrict k1_Zx_ptr, const float* restrict k1_Zy_ptr,
        const float* restrict k1_Zz_ptr, const float* restrict k1_P_ptr, const float* restrict k1_V_ptr,
        const float* restrict k2_E_ptr, const float* restrict k2_Zx_ptr, const float* restrict k2_Zy_ptr,
        const float* restrict k2_Zz_ptr, const float* restrict k2_P_ptr, const float* restrict k2_V_ptr)
    {
        #pragma omp parallel for simd schedule(static) \
            aligned(temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, \
                    temp_Phi_ptr, temp_VarPhi_ptr, E_ptr, Zx_ptr, Zy_ptr, \
                    Zz_ptr, Phi_ptr, VarPhi_ptr, k1_E_ptr, k1_Zx_ptr, \
                    k1_Zy_ptr, k1_Zz_ptr, k1_P_ptr, k1_V_ptr, k2_E_ptr, k2_Zx_ptr, \
                    k2_Zy_ptr, k2_Zz_ptr, k2_P_ptr, k2_V_ptr:32)
        for (int i = 0; i < size; i++) {
            float kE = k1_E_ptr[i];
            float kZx = k1_Zx_ptr[i];
            float kZy = k1_Zy_ptr[i];
            float kZz = k1_Zz_ptr[i];
            float kP = k1_P_ptr[i];
            float kV = k1_V_ptr[i];
            temp_E_ptr[i] += mix_dt * kE - half_dt * k2_E_ptr[i];
            temp_Zx_ptr[i] += mix_dt * kZx - half_dt * k2_Zx_ptr[i];
            temp_Zy_ptr[i] += mix_dt * kZy - half_dt * k2_Zy_ptr[i];
            temp_Zz_ptr[i] += mix_dt * kZz - half_dt * k2_Zz_ptr[i];
            temp_Phi_ptr[i] += mix_dt * kP - half_dt * k2_P_ptr[i];
            temp_VarPhi_ptr[i] += mix_dt * kV - half_dt * k2_V_ptr[i];
            E_ptr[i] += three_dt * kE;
            Zx_ptr[i] += three_dt * kZx;
            Zy_ptr[i] += three_dt * kZy;
            Zz_ptr[i] += three_dt * kZz;
            Phi_ptr[i] += three_dt * kP;
            VarPhi_ptr[i] += three_dt * kV;
        }
    }
    void parallel_simd_update3(int size, float six_dt,
        float* restrict E_ptr, float* restrict Zx_ptr, float* restrict Zy_ptr,
        float* restrict Zz_ptr, float* restrict Phi_ptr, float* restrict VarPhi_ptr,
        const float* restrict k1_E_ptr, const float* restrict k1_Zx_ptr, const float* restrict k1_Zy_ptr,
        const float* restrict k1_Zz_ptr, const float* restrict k1_P_ptr, const float* restrict k1_V_ptr)
    {
        #pragma omp parallel for simd schedule(static) \
            aligned(E_ptr, Zx_ptr, Zy_ptr, \
                    Zz_ptr, Phi_ptr, VarPhi_ptr, k1_E_ptr, k1_Zx_ptr, \
                    k1_Zy_ptr, k1_Zz_ptr, k1_P_ptr, k1_V_ptr:32)
        for (int i = 0; i < size; i++) {
            E_ptr[i] += six_dt * k1_E_ptr[i];
            Zx_ptr[i] += six_dt * k1_Zx_ptr[i];
            Zy_ptr[i] += six_dt * k1_Zy_ptr[i];
            Zz_ptr[i] += six_dt * k1_Zz_ptr[i];
            Phi_ptr[i] += six_dt * k1_P_ptr[i];
            VarPhi_ptr[i] += six_dt * k1_V_ptr[i];
        }
    }
    void NewtonRaphson(float* restrict T_ptr, float* restrict dv_ptr, float* restrict v2_ptr,
        float* restrict V_ptr, float* restrict p_ptr, const float* restrict E_ptr,
        const float* restrict Zx_ptr, const float* restrict Zy_ptr, const float* restrict Zz_ptr, const float* restrict Phi_ptr,
        float gam, float gam2, float delt, float del3, float lam4, float T_02, float a, int tot_size)
    {
        #pragma omp parallel for simd schedule(static) \
            aligned(T_ptr, dv_ptr, v2_ptr, V_ptr, p_ptr, \
                    E_ptr, Zx_ptr, Zy_ptr, Zz_ptr, Phi_ptr:32)
        for (int i = 0; i < tot_size; i++) {
            float T = T_ptr[i];
            float phi = Phi_ptr[i];
            float e = E_ptr[i];
            float zx = Zx_ptr[i];
            float zy = Zy_ptr[i];
            float zz = Zz_ptr[i];
            float phi2 = phi * phi;
            float D_term = phi * del3;
            float z2 = zx * zx + zy * zy + zz * zz;
            float l_term = lam4 * phi2;
            float T2 = T * T;
            float Tdiff = T2 - T_02;
            float V_eff = gam2 * Tdiff + l_term - D_term * T;
            float g_term = gam * T;
            float T1 = g_term - D_term;
            float dV_dT = T1 * phi2;
            float T3_term = T2 * T * a;
            float T4_term = T2 * T2 * a;
            float p = T4_term - V_eff * phi2;
            float Ep = p + e;
            float w = 4.0 * T4_term - T * dV_dT;
            float dp = 4.0 * T3_term - dV_dT;
            float dw = 16.0 * T3_term - g_term * phi2 - dV_dT;
            float mid = Ep - w;
            float f = mid * Ep - z2;
            float temp = 2.0 * dp - dw;
            float df = temp * Ep - w * dp;
            float frac = f / df;
            T1 = T - frac;
            T2 = T1 * T1;
            T4_term = T2 * T2 * a;
            Tdiff = T2 - T_02;
            V_eff = gam2 * Tdiff + l_term - D_term * T1;
            Ep = T4_term - V_eff * phi2 + e;
            temp = gam * Tdiff - delt * phi * T1 + 4.0 * l_term;
            float inv_Ep = 1.0 / Ep;
            dv_ptr[i] = inv_Ep;
            V_ptr[i] = temp * phi;
            v2_ptr[i] = inv_Ep * inv_Ep * z2;
            p_ptr[i] = p;
            T_ptr[i] = T1;
        }
    }
    static inline void minmod_derv_vec(const float* restrict u_ptr, float* restrict d_ptr,
        float theta_term, float dr_term, int sij, int sii, int sixm, int stride)
    {
        #pragma omp simd aligned(u_ptr, d_ptr:32)
        for (int k = 1; k < sixm; k++) {
            int idx = sij + k;
            int idout = sii + k;
            float uc = u_ptr[idx];
            float um = u_ptr[idx-stride];
            float up = u_ptr[idx+stride];
            float d1 = uc - um;
            float d2 = (up - um) * theta_term;
            float d3 = up - uc;
            float s1 = d1 * d2;
            float s2 = d1 * d3;
            if (s1 <= 0.0f || s2 <= 0.0f) {
                d_ptr[idout] = 0.0f;
            } else if (d1 > 0.0f) {
                float temp = (d1 < d2) ? d1 : d2;
                d_ptr[idout] = ((temp < d3) ? temp : d3) * dr_term;
            } else { \
                float temp = (d1 > d2) ? d1 : d2;
                d_ptr[idout] = ((temp > d3) ? temp : d3) * dr_term;
            }
        }
    }
    static inline void source_vec(const float* restrict dP_dx_ptr, const float* restrict dP_dy_ptr, const float* restrict dP_dz_ptr,
        const float* restrict Zx_ptr, const float* restrict Zy_ptr, const float* restrict Zz_ptr, const float* restrict VarPhi_ptr, 
        float* restrict E_n_ptr, float* restrict Zx_n_ptr, float* restrict Zy_n_ptr, float* restrict Zz_n_ptr, float* restrict V_n_ptr,
        float eta, int sij, int sii, int sixm)
    {
        #pragma omp simd aligned(dP_dx_ptr, dP_dy_ptr, dP_dz_ptr, Zx_ptr, Zy_ptr, Zz_ptr, VarPhi_ptr, \
            E_n_ptr, Zx_n_ptr, Zy_n_ptr, Zz_n_ptr, V_n_ptr:32)
        for (int k = 1; k < sixm; k++) {
            int idx = sij + k;
            int idd = sii + k;
            float denv = Zx_n_ptr[idx];
            float varphi_i = VarPhi_ptr[idx];
            float dpdx = dP_dx_ptr[idd];
            float dpdy = dP_dy_ptr[idd];
            float dpdz = dP_dz_ptr[idd];
            float zdp = dpdx * Zx_ptr[idx] + dpdy * Zy_ptr[idx] + dpdz * Zz_ptr[idx];
            float lorentz = 1.0f / sqrtf(1.0f - Zy_n_ptr[idx]);
            float t1 = varphi_i + denv * zdp;
            float eta_term = eta * t1 * lorentz + E_n_ptr[idx];
            V_n_ptr[idx] = -eta_term;
            E_n_ptr[idx] = eta_term * varphi_i;
            Zx_n_ptr[idx] = -eta_term * dpdx;
            Zy_n_ptr[idx] = -eta_term * dpdy;
            Zz_n_ptr[idx] = -eta_term * dpdz;
        }
    }
    static inline void dir_flux(const float* restrict E_ptr, const float* restrict Z0_ptr,
        const float* restrict Z1_ptr, const float* restrict Z2_ptr, const float* restrict P_n_ptr,
        const float* restrict dE_ptr, const float* restrict dZ0_ptr, const float* restrict dZ1_ptr,
        const float* restrict dZ2_ptr, const float* restrict dP_ptr,
        float* restrict E_n_ptr, float* restrict Z0_n_ptr, float* restrict Z1_n_ptr, float* restrict Z2_n_ptr,
        float half_dr, float inv_dr2, float c_s, int sii, int sij, int sxm, int spp, int smm)
    {
        #pragma omp simd aligned(E_ptr, Z0_ptr, Z1_ptr, Z2_ptr, P_n_ptr, \
            dE_ptr, dZ0_ptr, dZ1_ptr, dZ2_ptr, dP_ptr, E_n_ptr, Z0_n_ptr, Z1_n_ptr, Z2_n_ptr:32)
        for (int k = 1; k < sxm; k++) {
            int idd = sii + k;
            int idx = sij + k;
            int idxp = spp + k;
            int idxm = smm + k;
            float dE = dE_ptr[idd] * half_dr;
            float dZ0 = dZ0_ptr[idd] * half_dr;
            float dZ1 = dZ1_ptr[idd] * half_dr;
            float dZ2 = dZ2_ptr[idd] * half_dr;
            float dp = dP_ptr[idd] * half_dr;
            float Ep = E_ptr[idx] + dE;
            float Em = E_ptr[idx] - dE;
            float Z0p = Z0_ptr[idx] + dZ0;
            float Z0m = Z0_ptr[idx] - dZ0;
            float Z1p = Z1_ptr[idx] + dZ1;
            float Z1m = Z1_ptr[idx] - dZ1;
            float Z2p = Z2_ptr[idx] + dZ2;
            float Z2m = Z2_ptr[idx] - dZ2;
            float Pp = P_n_ptr[idx] + dp;
            float Pm = P_n_ptr[idx] - dp;
            float denom_p = (Z0p / (Ep + Pp)) + c_s;
            float denom_m = (Z0m / (Em + Pm)) - c_s;
            float f_Z0p = denom_p * Z0p + Pp;
            float f_Z0m = denom_m * Z0m + Pm;
            float f_Z1p = denom_p * Z1p;
            float f_Z1m = denom_m * Z1m;
            float f_Z2p = denom_p * Z2p;
            float f_Z2m = denom_m * Z2m;
            float f_Ep = Z0p + c_s * Ep;
            float f_Em = Z0m - c_s * Em;
            E_n_ptr[idx] -= inv_dr2 * (f_Ep - f_Em);
            E_n_ptr[idxp] += inv_dr2 * f_Ep;
            E_n_ptr[idxm] -= inv_dr2 * f_Em;
            Z0_n_ptr[idx] -= inv_dr2 * (f_Z0p - f_Z0m);
            Z0_n_ptr[idxp] += inv_dr2 * f_Z0p;
            Z0_n_ptr[idxm] -= inv_dr2 * f_Z0m;
            Z1_n_ptr[idx] -= inv_dr2 * (f_Z1p - f_Z1m);
            Z1_n_ptr[idxp] += inv_dr2 * f_Z1p;
            Z1_n_ptr[idxm] -= inv_dr2 * f_Z1m;
            Z2_n_ptr[idx] -= inv_dr2 * (f_Z2p - f_Z2m);
            Z2_n_ptr[idxp] += inv_dr2 * f_Z2p;
            Z2_n_ptr[idxm] -= inv_dr2 * f_Z2m;
        }
    }
    static inline void bound_flux(const float* restrict E_ptr, const float* restrict Z0_ptr,
        const float* restrict Z1_ptr, const float* restrict Z2_ptr, const float* restrict P_n_ptr,
        float* restrict E_n_ptr, float* restrict Z0_n_ptr, float* restrict Z1_n_ptr, float* restrict Z2_n_ptr,
        float inv_dr2, float c_s, int s_edge, int s_border, int size, int add, int size2)
    {
        for (int k = 1; k < size; k++) {
            int idx = s_border + k * add + k * size2;
            int idxp = s_edge + k * add + k * size2;
            float E = E_ptr[idx];
            float Z0 = Z0_ptr[idx];
            float Z1 = Z1_ptr[idx];
            float Z2 = Z2_ptr[idx];
            float P = P_n_ptr[idx];
            float denom = (Z0 / (E + P)) + c_s;
            float f_Z0 = denom * Z0 + P;
            float f_Z1 = denom * Z1;
            float f_Z2 = denom * Z2;
            float f_E = Z0 + c_s * E;
            E_n_ptr[idxp] += inv_dr2 * f_E;
            Z0_n_ptr[idxp] += inv_dr2 * f_Z0;
            Z1_n_ptr[idxp] += inv_dr2 * f_Z1;
            Z2_n_ptr[idxp] += inv_dr2 * f_Z2;
        }
    }
    static inline void bound_copy(float* restrict E_n_ptr, float* restrict Zx_n_ptr,
        float* restrict Zy_n_ptr, float* restrict Zz_n_ptr, float* restrict V_n_ptr,
        int s_edge, int s_border, int size, int add, int size2)
    {
        for (int k = 1; k < size; k++) {
            int idx = s_border + k * add + k * size2;
            int idxp = s_edge + k * add + k * size2;
            E_n_ptr[idx] = E_n_ptr[idxp];
            Zx_n_ptr[idx] = Zx_n_ptr[idxp];
            Zy_n_ptr[idx] = Zy_n_ptr[idxp];
            Zz_n_ptr[idx] = Zz_n_ptr[idxp];
            V_n_ptr[idx] = V_n_ptr[idxp];
        }
    }
    static inline void V_flux(const float* restrict P_ptr, float* restrict V_n_ptr, float inv_dr_sq, 
        int sixm, int sij, int sxp, int sxm, int syp, int sym, int szp, int szm)
    {
        #pragma omp simd aligned(P_ptr, V_n_ptr:32)
        for (int k = 1; k < sixm; k++) {
            int idx = sij + k;
            int xp = sxp + k;
            int xm = sxm + k;
            int yp = syp + k;
            int ym = sym + k;
            int zp = szp + k;
            int zm = szm + k;
            V_n_ptr[idx] += inv_dr_sq * (P_ptr[xp] + P_ptr[xm] + P_ptr[yp] + P_ptr[ym] + P_ptr[zp] + P_ptr[zm] - 6.0f * P_ptr[idx]);
        }
    }
    void rhsC(const float* restrict E_ptr, const float* restrict Zx_ptr, const float* restrict Zy_ptr,
            const float* restrict Zz_ptr, const float* restrict P_ptr, const float* restrict V_ptr,
            float* restrict E_n_ptr, float* restrict Zx_n_ptr, float* restrict Zy_n_ptr, float* restrict Zz_n_ptr, const float* restrict P_n_ptr, float* restrict V_n_ptr,
            float* restrict dE_ptr, float* restrict dZx_ptr, float* restrict dZy_ptr, float* restrict dZz_ptr, float* restrict dP_ptr,
            float eta, float half_dr, float inv_dr2, float inv_dr_sq, float theta_term, float dr_term, float c_s, int sxy, int sz, int sy, int sx)
    {
        int sizm = sz - 1;
        int siym = sy - 1;
        int sixm = sx - 1;
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < sizm; i++) {
            int si = sxy * i;
            int thread_id = omp_get_thread_num();
            int sii = thread_id * sx;
            int sm = sxy * (i - 1);
            int sp = sxy * (i + 1);
            for (int j = 1; j < siym; j++) {
                int sj = sx * j;
                int sxp = si + sj + 1;
                int sxm = si + sj - 1;
                int sym = si + sx * (j - 1);
                int syp = si + sx * (j + 1);
                int szm = sm + sj;
                int szp = sp + sj;
                int sij = si + sj;
                minmod_derv_vec(P_ptr, dZx_ptr, theta_term, dr_term, sij, sii, sixm, 1);
                minmod_derv_vec(P_ptr, dZy_ptr, theta_term, dr_term, sij, sii, sixm, sx);
                minmod_derv_vec(P_ptr, dZz_ptr, theta_term, dr_term, sij, sii, sixm, sxy);
                source_vec(dZx_ptr, dZy_ptr, dZz_ptr, Zx_ptr, Zy_ptr, Zz_ptr, V_ptr, E_n_ptr, Zx_n_ptr, Zy_n_ptr, Zz_n_ptr, V_n_ptr, eta, sij, sii, sixm);
                V_flux(P_ptr, V_n_ptr, inv_dr_sq, sixm, sij, sxp, sxm, syp, sym, szp, szm);
            }
        }
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < sizm; i++) {
            int si = sxy * i;
            int thread_id = omp_get_thread_num();
            int sii = thread_id * sx;
            int sm = sxy * (i - 1);
            int sp = sxy * (i + 1);
            for (int j = 1; j < siym; j++) {
                int sj = sx * j;
                int sxp = si + sj + 1;
                int sxm = si + sj - 1;
                int sym = si + sx * (j - 1);
                int syp = si + sx * (j + 1);
                int szm = sm + sj;
                int szp = sp + sj;
                int sij = si + sj;
                minmod_derv_vec(P_n_ptr, dP_ptr, theta_term, dr_term, sij, sii, sixm, 1);
                minmod_derv_vec(Zx_ptr, dZx_ptr, theta_term, dr_term, sij, sii, sixm, 1);
                minmod_derv_vec(Zy_ptr, dZy_ptr, theta_term, dr_term, sij, sii, sixm, 1);
                minmod_derv_vec(Zz_ptr, dZz_ptr, theta_term, dr_term, sij, sii, sixm, 1);
                minmod_derv_vec(E_ptr, dE_ptr, theta_term, dr_term, sij, sii, sixm, 1);
                dir_flux(E_ptr, Zx_ptr, Zy_ptr, Zz_ptr, P_n_ptr, dE_ptr, dZx_ptr, dZy_ptr, dZz_ptr, dP_ptr, E_n_ptr, Zx_n_ptr, Zy_n_ptr, Zz_n_ptr, half_dr, inv_dr2, c_s, sii, sij, sixm, sxp, sxm);
                minmod_derv_vec(P_n_ptr, dP_ptr, theta_term, dr_term, sij, sii, sixm, sx);
                minmod_derv_vec(Zx_ptr, dZx_ptr, theta_term, dr_term, sij, sii, sixm, sx);
                minmod_derv_vec(Zy_ptr, dZy_ptr, theta_term, dr_term, sij, sii, sixm, sx);
                minmod_derv_vec(Zz_ptr, dZz_ptr, theta_term, dr_term, sij, sii, sixm, sx);
                minmod_derv_vec(E_ptr, dE_ptr, theta_term, dr_term, sij, sii, sixm, sx);
                dir_flux(E_ptr, Zy_ptr, Zz_ptr, Zx_ptr, P_n_ptr, dE_ptr, dZy_ptr, dZz_ptr, dZx_ptr, dP_ptr, E_n_ptr, Zy_n_ptr, Zz_n_ptr, Zx_n_ptr, half_dr, inv_dr2, c_s, sii, sij, sixm, syp, sym);
                minmod_derv_vec(P_n_ptr, dP_ptr, theta_term, dr_term, sij, sii, sixm, sxy);
                minmod_derv_vec(Zx_ptr, dZx_ptr, theta_term, dr_term, sij, sii, sixm, sxy);
                minmod_derv_vec(Zy_ptr, dZy_ptr, theta_term, dr_term, sij, sii, sixm, sxy);
                minmod_derv_vec(Zz_ptr, dZz_ptr, theta_term, dr_term, sij, sii, sixm, sxy);
                minmod_derv_vec(E_ptr, dE_ptr, theta_term, dr_term, sij, sii, sixm, sxy);
                dir_flux(E_ptr, Zz_ptr, Zx_ptr, Zy_ptr, P_n_ptr, dE_ptr, dZz_ptr, dZx_ptr, dZy_ptr, dP_ptr, E_n_ptr, Zz_n_ptr, Zx_n_ptr, Zy_n_ptr, half_dr, inv_dr2, c_s, sii, sij, sixm, szp, szm);
            }
            int s_edge = si + sx;
            int s_border = si;
            bound_flux(E_ptr, Zy_ptr, Zz_ptr, Zx_ptr, P_n_ptr, E_n_ptr, Zy_n_ptr, Zz_n_ptr, Zx_n_ptr, inv_dr2, c_s, s_edge, s_border, sixm, 1, 0);
            s_edge = si + sxy - 2*sx;
            s_border = si + sxy - sx;
            bound_flux(E_ptr, Zy_ptr, Zz_ptr, Zx_ptr, P_n_ptr, E_n_ptr, Zy_n_ptr, Zz_n_ptr, Zx_n_ptr, (-inv_dr2), (-c_s), s_edge, s_border, sixm, 1, 0);
            s_edge = si + 1;
            s_border = si;
            bound_flux(E_ptr, Zx_ptr, Zy_ptr, Zz_ptr, P_n_ptr, E_n_ptr, Zx_n_ptr, Zy_n_ptr, Zz_n_ptr, inv_dr2, c_s, s_edge, s_border, siym, 0, sx);
            s_edge = si + sixm - 1;
            s_border = si + sixm;
            bound_flux(E_ptr, Zx_ptr, Zy_ptr, Zz_ptr, P_n_ptr, E_n_ptr, Zx_n_ptr, Zy_n_ptr, Zz_n_ptr, (-inv_dr2), (-c_s), s_edge, s_border, siym, 0, sx);
        }
        #pragma omp parallel for schedule(static)
        for (int j = 1; j < siym; j++) {
            int sj = sx * j;
            int s_edge = sxy + sj;
            int s_border = sj;
            bound_flux(E_ptr, Zz_ptr, Zx_ptr, Zy_ptr, P_n_ptr, E_n_ptr, Zz_n_ptr, Zx_n_ptr, Zy_n_ptr, inv_dr2, c_s, s_edge, s_border, sixm, 1, 0);
            bound_copy(E_n_ptr, Zx_n_ptr, Zy_n_ptr, Zz_n_ptr, V_n_ptr, s_edge, s_border, sixm, 1, 0);
            s_edge = (sizm-1) * sxy + sj;
            s_border = sizm * sxy + sj;
            bound_flux(E_ptr, Zz_ptr, Zx_ptr, Zy_ptr, P_n_ptr, E_n_ptr, Zz_n_ptr, Zx_n_ptr, Zy_n_ptr, (-inv_dr2), (-c_s), s_edge, s_border, sixm, 1, 0);
            bound_copy(E_n_ptr, Zx_n_ptr, Zy_n_ptr, Zz_n_ptr, V_n_ptr, s_edge, s_border, sixm, 1, 0);
        }
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < sizm; i++) {
            int si = sxy * i;
            int s_edge = si + sx;
            int s_border = si;
            bound_copy(E_n_ptr, Zx_n_ptr, Zy_n_ptr, Zz_n_ptr, V_n_ptr, s_edge, s_border, sixm, 1, 0);
            s_edge = si + sxy - 2*sx;
            s_border = si + sxy - sx;
            bound_copy(E_n_ptr, Zx_n_ptr, Zy_n_ptr, Zz_n_ptr, V_n_ptr, s_edge, s_border, sixm, 1, 0);
            s_edge = si + 1;
            s_border = si;
            bound_copy(E_n_ptr, Zx_n_ptr, Zy_n_ptr, Zz_n_ptr, V_n_ptr, s_edge, s_border, siym, 0, sx);
            s_edge = si + sixm - 1;
            s_border = si + sixm;
            bound_copy(E_n_ptr, Zx_n_ptr, Zy_n_ptr, Zz_n_ptr, V_n_ptr, s_edge, s_border, siym, 0, sx);
        }
    }
    """

cdef extern void parallel_simd_update1(int size, float half_dt, float six_dt, 
    float* temp_E_ptr, float* temp_Zx_ptr, float* temp_Zy_ptr, float* temp_Zz_ptr,
    float* temp_Phi_ptr, float* temp_VarPhi_ptr, float* E_ptr, float* Zx_ptr, float* Zy_ptr,
    float* Zz_ptr, float* Phi_ptr, float* VarPhi_ptr, const float* k1_E_ptr, const float* k1_Zx_ptr,
    const float* k1_Zy_ptr, const float* k1_Zz_ptr, const float* k1_P_ptr, const float* k1_V_ptr) noexcept nogil

cdef extern void parallel_simd_update2(int size, float mix_dt, float half_dt, float three_dt, 
    float* temp_E_ptr, float* temp_Zx_ptr, float* temp_Zy_ptr, float* temp_Zz_ptr,
    float* temp_Phi_ptr, float* temp_VarPhi_ptr, float* E_ptr, float* Zx_ptr, float* Zy_ptr,
    float* Zz_ptr, float* Phi_ptr, float* VarPhi_ptr, const float* k1_E_ptr, const float* k1_Zx_ptr,
    const float* k1_Zy_ptr, const float* k1_Zz_ptr, const float* k1_P_ptr, const float* k1_V_ptr, const float* k2_E_ptr, const float* k2_Zx_ptr,
    const float* k2_Zy_ptr, const float* k2_Zz_ptr, const float* k2_P_ptr, const float* k2_V_ptr) noexcept nogil

cdef extern void parallel_simd_update3(int size, float six_dt,
    float* E_ptr, float* Zx_ptr, float* Zy_ptr, float* Zz_ptr, float* Phi_ptr, float* VarPhi_ptr,
    const float* k1_E_ptr, const float* k1_Zx_ptr, const float* k1_Zy_ptr, const float* k1_Zz_ptr, const float* k1_P_ptr, const float* k1_V_ptr) noexcept nogil

cdef extern void NewtonRaphson(float* T_ptr, float* dv_ptr, float* v2_ptr, float* V_ptr, float* p_ptr, const float* E_ptr,
        const float* Zx_ptr, const float* Zy_ptr, const float* Zz_ptr, const float* Phi_ptr,
        float gam, float gam2, float delt, float del3, float lam4, float T_02, float a, int tot_size) noexcept nogil

cdef extern void rhsC(const float* E_ptr, const float* Zx_ptr, const float* Zy_ptr,
            const float* Zz_ptr, const float* P_ptr, const float* V_ptr,
            float* E_n_ptr, float* Zx_n_ptr, float* Zy_n_ptr, float* Zz_n_ptr, const float* P_n_ptr, float* V_n_ptr,
            float* dE_ptr, float* dZx_ptr, float* dZy_ptr, float* dZz_ptr, float* dP_ptr,
            float eta, float half_dr, float inv_dr2, float inv_dr_sq, float theta_term, float dr_term, float c_s, int sxy, int sz, int sy, int sx) noexcept nogil

cdef inline void init_val(double* R, double* C_F, double* Phi_b, double* K, double* T_Nn, double* t_N, double* t_C,
        list px, list py, list pz, list vals, double dr, double inv_dr, 
        double gam, double gam2, double T_02, double delt, double del3, double lamb, double lam4,
        double gstar, double eta, double T_C, double Vol, bint l_type) noexcept nogil:
    cdef double d3 = delt * delt * delt
    cdef double tc3 = T_C * T_C * T_C
    cdef double inv_l = 1.0 / (lamb * lamb * sqrt(lamb))
    cdef double sig = (2.0 * sqrt(2.0) / 81.0) * d3 * inv_l * tc3
    cdef double T_hat = T_02 / (T_C * T_C)
    cdef double M_pl = 1.22089e19
    cdef double logM = log(M_pl * M_pl * M_pl * M_pl / (tc3 * T_C))
    cdef double t_c = sqrt(90.0 / (32.0 * M_PI * M_PI * M_PI)) * M_pl / (sqrt(gstar) * T_C * T_C)
    cdef double A, L, sqrtA, cube, cube2, t_grow, T_N, tntc, k
    if l_type:
        A = delt * lamb * inv_l * 4.57
        cube = cbrt(logM / A)
        cube2 = cube * cube
        T_N = sqrt(T_hat / (1.0 - cube2 * (1.0 - T_hat))) * T_C
        tntc = (1.0 - cube2 * (1.0 - T_hat))  / T_hat
        k = 1.5 * A * (T_hat / (t_c * (1.0 - T_hat))) * sqrt((1.0 - T_hat*tntc) / (1.0 - T_hat))
    else:
        L = (4.0 / 9.0) * delt * delt * gam * T_02 * T_C * T_C / (lamb * lamb)
        A = 64.0 * M_PI * sig * sig * sig / (3.0 * L * L * T_C)
        sqrtA = sqrt(A)
        tntc = 1 + sqrtA / sqrt(logM)
        T_N = (1.0 - 0.5 * sqrt(A) / sqrt(logM)) * T_C
        k = 2.0 * sqrt(logM) * logM / (sqrtA * t_c)
    cdef double T2 = T_N * T_N
    cdef double T4 = T2 * T2
    cdef double Tdiff = T2 - T_02
    cdef double g_term = gam * Tdiff
    cdef double g_term2 = gam2 * Tdiff
    cdef double d = delt * T_N
    cdef double d1 = del3 * T_N
    cdef double d2 = d * d
    cdef double l = 2.0 * lamb
    cdef double ins = d2 - 2.0 * l * g_term
    cdef double root = sqrt(ins)
    cdef double phi_b = (d + root) / l
    cdef double pb2 = phi_b * phi_b
    cdef double mid = g_term2 - d1 * phi_b + lam4 * pb2
    cdef double V = mid * pb2
    cdef double v_w = -V / (eta * sig)
    cdef double coeff = 0.25 * V / sig
    cdef double r = -2.0 * sig / V
    cdef int r_int = <int>ceil(sqrt(2.0 * r) * inv_dr)
    cdef int max_r = 4 * r_int
    cdef int max_r2 = 12 * r_int * r_int
    cdef int z2, zy2, r2, x, y, z
    cdef float val
    cdef double fac = coeff * dr * dr
    for z in range(-max_r, max_r+1):
        z2 = z*z
        for y in range(-max_r, max_r+1):
            zy2 = z2 + y*y
            for x in range(-max_r, max_r+1):
                r2 = zy2 + x*x
                if r2 <= max_r2:
                    val = <float>(phi_b * exp(fac * r2))
                    with gil:
                        px.append(x)
                        py.append(y)
                        pz.append(z)
                        vals.append(val)
    R[0] = r
    C_F[0] = 8.0 * M_PI * v_w * v_w * v_w / (k * k * k * Vol)
    Phi_b[0] = phi_b
    K[0] = k
    T_Nn[0] = T_N
    t_N[0] = tntc * t_c
    t_C[0] = t_c

cdef inline void bubble_times(double dt, double t_init, double t_end, double k, double c_f, int* bubbles_per_step) noexcept nogil:
    cdef double inv_dt = 1.0 / dt
    cdef double t_test = 0.0
    cdef double t1 = t_init
    cdef double r_denom = 1.0 / (RAND_MAX + 1.0)
    cdef double t_finish = t_end - dt
    cdef double t2, ran
    cdef int index
    srand(time(NULL))
    while t_test < t_finish:
        ran = rand() * r_denom
        t2 = log(exp(k * t1) - c_f * log(ran)) / k
        t_test += (t2 - t1)
        if t_test >= t_finish:
            break
        index = <int>floor(t_test * inv_dt)
        bubbles_per_step[index] += 1
        t1 = t2

cdef class Fluid1D:
    cdef:
        # Basic scalars
        float t_end, dt, half_dt, six_dt, three_dt, dr, inv_dr, inv_dr2, inv_dr_sq, half_dr, cfl, theta_term, dr_term, min_phi
        float gam, gam2, T_0, T_02, delt, del3, lamb, lam4, a, eta, c_s
        double T_N, T_C, R, C_F, K, t_n, t_c, Phi_b
        double memcp_t, rhs_t, stepTv_t, t_init
        Py_ssize_t sx, sy, sz, sxy, s_tot, val_n, tot_bubbles
        bint test
        list bubble_coords
        tuple chunk_shape
        int[::1] Px, Py, Pz, BPT, c_x, c_y, c_z
        float[::1] val
        # Main arrays (state and geometry)
        float[:, :, ::1] E, Zx, Zy, Zz, Phi, VarPhi, T
        float[:, :, ::1] temp_E, temp_Zx, temp_Zy, temp_Zz, temp_Phi, temp_VarPhi
        float[:, :, ::1] k1_E, k1_Zx, k1_Zy, k1_Zz, k1_P, k1_V, k2_E, k2_Zx, k2_Zy, k2_Zz, k2_P, k2_V
        float[:, ::1] dE, dZx, dZy, dZz, dP
    
    def __cinit__(self, int n, double x_diam, double cfl, double t_end,
                 double gamma, double T_0, double delta, double lambd, double gstar, double eta, double t_init,
                 double theta=2.0, double yfac=1.0, double zfac=1.0, bint test = False, bint l_type = False):
        self.dr = <float>(x_diam / n)
        self.inv_dr = <float>(1.0 / self.dr)
        self.inv_dr2 = <float>(0.5 * self.inv_dr)
        self.inv_dr_sq = <float>(self.inv_dr * self.inv_dr)
        self.half_dr = <float>(0.5 * self.dr)
        self.dt = <float>(cfl * self.dr)
        self.cfl = <float>(cfl)
        self.t_end = <float>(t_end)
        self.half_dt = <float>(0.5 * self.dt)
        self.six_dt = <float>(self.dt / 6.0)
        self.three_dt = <float>(self.dt / 3.0)
        self.theta_term = <float>(0.5 / theta)
        self.dr_term = <float>(theta * self.inv_dr)
        self.t_init = t_init
        self.sx = n
        self.sy = int(n * yfac)
        self.sz = int(n * zfac)
        self.s_tot = self.sx * self.sy * self.sz
        self.sxy = self.sx * self.sy
        self.c_s = <float>(np.sqrt(1.0/3.0))
        self.gam = <float>(gamma)
        self.gam2 = <float>(0.5 * gamma)
        self.T_0 = <float>(T_0)
        self.T_02 = <float>(T_0 * T_0)
        self.delt = <float>(delta)
        self.del3 = <float>(delta / 3.0)
        self.lamb = <float>(lambd)
        self.lam4 = <float>(0.25 * lambd)
        cdef double a = gstar * np.pi * np.pi / 30.0
        self.a = <float>(a / 3.0)
        self.T_C = np.sqrt(self.T_02 / (1.0 - 2.0 * delta * delta / (9.0 * lambd * gamma)))
        self.eta = <float>(eta * self.T_C)
        self.test = test
        self.memcp_t = 0.0
        self.rhs_t = 0.0
        self.stepTv_t = 0.0
        self.chunk_shape = (min(50, self.sz), min(50, self.sy), min(50, self.sx))
        cdef tuple shape = (self.sz, self.sy, self.sx)
        cdef tuple s_x = (8, self.sx)
        self.Phi = np.array(np.zeros(shape, dtype=np.float32), order='C')
        cdef double R, C_F, Phi_b, K, T_Nn, t_n, t_c
        cdef list px = [], py = [], pz = [], vals = []
        init_val(&R, &C_F, &Phi_b, &K, &T_Nn, &t_n, &t_c, px, py, pz, vals, 
        self.dr, self.inv_dr, gamma, (0.5*gamma), (T_0*T_0), delta, (delta/3.0), lambd, (0.25*lambd),
        gstar, (eta * self.T_C), self.T_C, x_diam*x_diam*x_diam*yfac*zfac, l_type)
        print(f"Radius = {R:.5f}, T_Co = {self.T_C:.5f}, T_Nuc = {T_Nn:.5f}, co_time = {t_c:.6g}, nuc_time = {t_n:.6g}")
        self.R = R
        self.C_F = C_F
        self.K = K
        self.T_N = T_Nn
        self.t_c = t_c
        self.t_n = t_n
        self.Phi_b = Phi_b
        self.min_phi = <float>(0.9 * self.Phi_b)
        self.BPT = np.ascontiguousarray(np.zeros(int(np.ceil(t_end/self.dt) + 2.0)).astype(np.int32))
        cdef int[::1] B_P_T = self.BPT
        cdef int* BPT_ptr = &B_P_T[0]
        bubble_times(<double>(self.dt), t_init, t_end, self.K, self.C_F, BPT_ptr)
        cdef int tot_bubbles = <int>np.sum(np.asarray(self.BPT))
        self.tot_bubbles = tot_bubbles
        print(f"Total Timesteps: {int(np.round(t_end / self.dt))}")
        print(f"Possible Bubbles: {tot_bubbles}")
        print(f"Mean: {np.mean(np.asarray(self.BPT)):.3f}, Median: {int(np.median(np.asarray(self.BPT)))}, Max: {int(np.max(np.asarray(self.BPT)))}")
        self.Px = np.ascontiguousarray(np.array(px).astype(np.int32))
        self.Py = np.ascontiguousarray(np.array(py).astype(np.int32))
        self.Pz = np.ascontiguousarray(np.array(pz).astype(np.int32))
        self.val = np.ascontiguousarray(np.array(vals).astype(np.float32))
        self.val_n = len(vals)
        del px, py, pz, vals
        cdef int B = np.ceil(3.5 * sqrt(2.0 * R) / self.dr + 2).astype(int)
        cdef object rng = np.random.default_rng()
        self.c_z = np.ascontiguousarray(np.zeros(2).astype(np.int32))
        self.c_y = np.ascontiguousarray(np.zeros(2).astype(np.int32))
        self.c_x = np.ascontiguousarray(np.zeros(2).astype(np.int32))
        if test:
            self.c_z[0] = np.ceil(self.sz/2).astype(int)
            self.c_y[0] = np.ceil(self.sy/2).astype(int)
            self.c_x[0] = np.ceil(self.sx/2).astype(int)
        else:
            self.c_z[0] = rng.integers(B, (self.sz-B))
            self.c_y[0] = rng.integers(B, (self.sy-B))
            self.c_x[0] = rng.integers(B, (self.sx-B))
        print(f"Initial Bubble Coords: (x, y, z) = ({self.c_x[0]}, {self.c_y[0]}, {self.c_z[0]})")
        self.bubble_coords = []
        self.Phi = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.bubble_insert(0, self.bubble_coords)
        if test == False:
            self.c_z = np.ascontiguousarray(rng.integers(B, high=(self.sz-B), size=(tot_bubbles+1)).astype(np.int32))
            self.c_y = np.ascontiguousarray(rng.integers(B, high=(self.sy-B), size=(tot_bubbles+1)).astype(np.int32))
            self.c_x = np.ascontiguousarray(rng.integers(B, high=(self.sx-B), size=(tot_bubbles+1)).astype(np.int32))
        self.E = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.Zx = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.Zy = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.Zz = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.VarPhi = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.T = np.array(np.full(shape, fill_value = <float>(self.T_N), dtype=np.float32), order='C')
        self.k1_E = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.k2_E = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.k1_Zx = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.k2_Zx = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.k1_Zy = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.k2_Zy = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.k1_Zz = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.k2_Zz = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.k1_P = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.k2_P = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.k1_V = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.k2_V = np.array(np.zeros(shape, dtype=np.float32), order='C')
        cdef float T2 = <float>(self.T_N * self.T_N)
        cdef float T02 = self.T_02
        cdef float Tdiff = T2 - T02
        cdef float g_t = self.gam2 * Tdiff
        cdef float g1 = self.gam * Tdiff
        cdef float gd = self.gam * T2
        cdef float T4 = T2 * T2
        cdef float d_term = <float>(self.del3 * self.T_N)
        cdef float d1 = <float>(self.delt * self.T_N)
        cdef float delt3 = self.del3
        cdef float l4 = self.lam4
        cdef float l1 = self.lamb
        cdef float a3 = self.a
        cdef float[:, :, ::1] Phi = self.Phi
        cdef float[:, :, ::1] E = self.E
        cdef float[:, :, ::1] k1_E = self.k1_E
        cdef float[:, :, ::1] k1_P = self.k1_P
        cdef float[:, :, ::1] k1_Zx = self.k1_Zx
        cdef float* Phi_ptr = &Phi[0,0,0]
        cdef float* E_ptr = &E[0,0,0]
        cdef float* k1_E_ptr = &k1_E[0,0,0]
        cdef float* k1_P_ptr = &k1_P[0,0,0]
        cdef float* k1_Zx_ptr = &k1_Zx[0,0,0]
        cdef float phi, phi2, Veff, dv_dt, e, p, vdp, dvdp, denom
        for i in prange(self.s_tot, nogil=True, schedule = "static"):
            phi = Phi_ptr[i]
            phi2 = phi * phi
            Veff = g_t - d_term * phi + l4 * phi2
            vdp = g1 - d1 * phi + l1 * phi2
            dv_dt = gd - d_term * phi
            e = a * T4 + Veff * phi2 - dv_dt * phi2
            p = a3 * T4 - Veff * phi2
            E_ptr[i] = e
            k1_P_ptr[i] = p
            k1_E_ptr[i] = (vdp * phi)
            k1_Zx_ptr[i] = <float>(1.0 / (e+p))
        self.temp_E = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.temp_Zx = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.temp_Zy = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.temp_Zz = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.temp_Phi = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.temp_VarPhi = np.array(np.zeros(shape, dtype=np.float32), order='C')
        self.dE = np.ascontiguousarray(np.zeros(s_x, dtype=np.float32))
        self.dZx = np.ascontiguousarray(np.zeros(s_x, dtype=np.float32))
        self.dZy = np.ascontiguousarray(np.zeros(s_x, dtype=np.float32))
        self.dZz = np.ascontiguousarray(np.zeros(s_x, dtype=np.float32))
        self.dP = np.ascontiguousarray(np.zeros(s_x, dtype=np.float32))

    cpdef init_Scalar(self):
        return np.asarray(self.Phi)

    cdef inline void bubble_insert(self, Py_ssize_t BPT_val, list bubble_coords) noexcept nogil:
        cdef float[:, :, ::1] Phi = self.Phi
        cdef int[::1] cx = self.c_x
        cdef int[::1] cy = self.c_y
        cdef int[::1] cz = self.c_z
        cdef float* Phi_ptr = &Phi[0,0,0]
        cdef int* x_ptr = &cx[0]
        cdef int* y_ptr = &cy[0]
        cdef int* z_ptr = &cz[0]
        cdef Py_ssize_t sx = self.sx
        cdef Py_ssize_t sxy = self.sxy
        cdef Py_ssize_t ox = x_ptr[BPT_val]
        cdef Py_ssize_t oy = y_ptr[BPT_val]
        cdef Py_ssize_t oz = z_ptr[BPT_val]
        cdef Py_ssize_t idx = ox + oy * sx + oz * sxy
        if Phi_ptr[idx] >= self.min_phi:
            return
        with gil:
            bubble_coords.append([ox, oy, oz])
        cdef int p_n = self.val_n
        cdef float value
        cdef int i, x, y, z
        cdef int[::1] Px = self.Px
        cdef int[::1] Py = self.Py
        cdef int[::1] Pz = self.Pz
        cdef int* px_ptr = &Px[0]
        cdef int* py_ptr = &Py[0]
        cdef int* pz_ptr = &Pz[0]
        cdef float[::1] val = self.val
        cdef float* val_ptr = &val[0]
        for i in range(p_n):
            x = ox + px_ptr[i]
            y = oy + py_ptr[i]
            z = oz + pz_ptr[i]
            value = val_ptr[i]
            idx = sxy * z + sx * y + x
            if Phi_ptr[idx] < value:
                Phi_ptr[idx] = value

    cdef inline void advance_rk4(self) noexcept nogil:
        cdef:
            int i
            Py_ssize_t size = self.s_tot
            float half_dt = self.half_dt
            float dt = self.dt
            float six_dt = self.six_dt
            float three_dt = self.three_dt
            float[:, :, ::1] E = self.E
            float[:, :, ::1] Zx = self.Zx
            float[:, :, ::1] Zy = self.Zy
            float[:, :, ::1] Zz = self.Zz
            float[:, :, ::1] Phi = self.Phi
            float[:, :, ::1] VarPhi = self.VarPhi
            float[:, :, ::1] temp_E = self.temp_E
            float[:, :, ::1] temp_Zx = self.temp_Zx
            float[:, :, ::1] temp_Zy = self.temp_Zy
            float[:, :, ::1] temp_Zz = self.temp_Zz
            float[:, :, ::1] temp_Phi = self.temp_Phi
            float[:, :, ::1] temp_VarPhi = self.temp_VarPhi
            float[:, :, ::1] T = self.T
            float[:, :, ::1] k1_E = self.k1_E
            float[:, :, ::1] k2_E = self.k2_E
            float[:, :, ::1] k1_Zx = self.k1_Zx
            float[:, :, ::1] k2_Zx = self.k2_Zx
            float[:, :, ::1] k1_Zy = self.k1_Zy
            float[:, :, ::1] k2_Zy = self.k2_Zy
            float[:, :, ::1] k1_Zz = self.k1_Zz
            float[:, :, ::1] k2_Zz = self.k2_Zz
            float[:, :, ::1] k1_P = self.k1_P
            float[:, :, ::1] k2_P = self.k2_P
            float[:, :, ::1] k1_V = self.k1_V
            float[:, :, ::1] k2_V = self.k2_V
            float[:, ::1] dE = self.dE
            float[:, ::1] dZx = self.dZx
            float[:, ::1] dZy = self.dZy
            float[:, ::1] dZz = self.dZz
            float[:, ::1] dP = self.dP
            float* E_ptr = &E[0,0,0]
            float* Zx_ptr = &Zx[0,0,0]
            float* Zy_ptr = &Zy[0,0,0]
            float* Zz_ptr = &Zz[0,0,0]
            float* Phi_ptr = &Phi[0,0,0]
            float* VarPhi_ptr = &VarPhi[0,0,0]
            float* temp_E_ptr = &temp_E[0,0,0]
            float* temp_Zx_ptr = &temp_Zx[0,0,0]
            float* temp_Zy_ptr = &temp_Zy[0,0,0]
            float* temp_Zz_ptr = &temp_Zz[0,0,0]
            float* temp_Phi_ptr = &temp_Phi[0,0,0]
            float* temp_VarPhi_ptr = &temp_VarPhi[0,0,0]
            float* k1_E_ptr = &k1_E[0,0,0]
            float* k1_Zx_ptr = &k1_Zx[0,0,0]
            float* k1_Zy_ptr = &k1_Zy[0,0,0]
            float* k1_Zz_ptr = &k1_Zz[0,0,0]
            float* k1_P_ptr = &k1_P[0,0,0]
            float* k1_V_ptr = &k1_V[0,0,0]
            float* k2_E_ptr = &k2_E[0,0,0]
            float* k2_Zx_ptr = &k2_Zx[0,0,0]
            float* k2_Zy_ptr = &k2_Zy[0,0,0]
            float* k2_Zz_ptr = &k2_Zz[0,0,0]
            float* k2_P_ptr = &k2_P[0,0,0]
            float* k2_V_ptr = &k2_V[0,0,0]
            float* T_ptr = &T[0,0,0]
            float* dE_ptr = &dE[0,0]
            float* dZx_ptr = &dZx[0,0]
            float* dZy_ptr = &dZy[0,0]
            float* dZz_ptr = &dZz[0,0]
            float* dP_ptr = &dP[0,0]
            double start, end

        # Stage 1
        memcpy(temp_E_ptr, E_ptr, size * sizeof(float))
        memcpy(temp_Zx_ptr, Zx_ptr, size * sizeof(float))
        memcpy(temp_Zy_ptr, Zy_ptr, size * sizeof(float))
        memcpy(temp_Zz_ptr, Zz_ptr, size * sizeof(float))
        memcpy(temp_Phi_ptr, Phi_ptr, size * sizeof(float))
        memcpy(temp_VarPhi_ptr, VarPhi_ptr, size * sizeof(float))

        start = <double>clock() / CLOCKS_PER_SEC
        rhsC(temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, temp_Phi_ptr, temp_VarPhi_ptr,
            k1_E_ptr, k1_Zx_ptr, k1_Zy_ptr, k1_Zz_ptr, k1_P_ptr, k1_V_ptr,
            dE_ptr, dZx_ptr, dZy_ptr, dZz_ptr, dP_ptr,
            self.eta, self.half_dr, self.inv_dr2, self.inv_dr_sq, self.theta_term, self.dr_term, self.c_s, self.sxy, self.sz, self.sy, self.sx)
        memcpy(k1_P_ptr, temp_VarPhi_ptr, size * sizeof(float))
        end = <double>clock() / CLOCKS_PER_SEC
        self.rhs_t += end - start

        # Stage 2
        start = <double>clock() / CLOCKS_PER_SEC
        parallel_simd_update1(size, half_dt, six_dt, 
        temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, temp_Phi_ptr, temp_VarPhi_ptr,
        E_ptr, Zx_ptr, Zy_ptr, Zz_ptr, Phi_ptr, VarPhi_ptr,
        k1_E_ptr, k1_Zx_ptr, k1_Zy_ptr, k1_Zz_ptr, k1_P_ptr, k1_V_ptr)
        end = <double>clock() / CLOCKS_PER_SEC
        self.memcp_t += end-start
        start = <double>clock() / CLOCKS_PER_SEC
        NewtonRaphson(T_ptr, k2_Zx_ptr, k2_Zy_ptr, k2_E_ptr, k2_P_ptr,
        temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, temp_Phi_ptr,
        self.gam, self.gam2, self.delt, self.del3, self.lam4, self.T_02, self.a, size)
        end = <double>clock() / CLOCKS_PER_SEC
        self.stepTv_t += end - start
        start = <double>clock() / CLOCKS_PER_SEC
        rhsC(temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, temp_Phi_ptr, temp_VarPhi_ptr,
            k2_E_ptr, k2_Zx_ptr, k2_Zy_ptr, k2_Zz_ptr, k2_P_ptr, k2_V_ptr,
            dE_ptr, dZx_ptr, dZy_ptr, dZz_ptr, dP_ptr,
            self.eta, self.half_dr, self.inv_dr2, self.inv_dr_sq, self.theta_term, self.dr_term, self.c_s, self.sxy, self.sz, self.sy, self.sx)
        memcpy(k2_P_ptr, temp_VarPhi_ptr, size * sizeof(float))
        end = <double>clock() / CLOCKS_PER_SEC
        self.rhs_t += end - start

        # Stage 3
        start = <double>clock() / CLOCKS_PER_SEC
        parallel_simd_update2(size, half_dt, half_dt, three_dt, 
        temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, temp_Phi_ptr, temp_VarPhi_ptr,
        E_ptr, Zx_ptr, Zy_ptr, Zz_ptr, Phi_ptr, VarPhi_ptr,
        k2_E_ptr, k2_Zx_ptr, k2_Zy_ptr, k2_Zz_ptr, k2_P_ptr, k2_V_ptr,
        k1_E_ptr, k1_Zx_ptr, k1_Zy_ptr, k1_Zz_ptr, k1_P_ptr, k1_V_ptr)
        end = <double>clock() / CLOCKS_PER_SEC
        self.memcp_t += end-start
        start = <double>clock() / CLOCKS_PER_SEC
        NewtonRaphson(T_ptr, k1_Zx_ptr, k1_Zy_ptr, k1_E_ptr, k1_P_ptr,
        temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, temp_Phi_ptr,
        self.gam, self.gam2, self.delt, self.del3, self.lam4, self.T_02, self.a, size)
        end = <double>clock() / CLOCKS_PER_SEC
        self.stepTv_t += end - start

        start = <double>clock() / CLOCKS_PER_SEC
        rhsC(temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, temp_Phi_ptr, temp_VarPhi_ptr,
            k1_E_ptr, k1_Zx_ptr, k1_Zy_ptr, k1_Zz_ptr, k1_P_ptr, k1_V_ptr,
            dE_ptr, dZx_ptr, dZy_ptr, dZz_ptr, dP_ptr,
            self.eta, self.half_dr, self.inv_dr2, self.inv_dr_sq, self.theta_term, self.dr_term, self.c_s, self.sxy, self.sz, self.sy, self.sx)
        memcpy(k1_P_ptr, temp_VarPhi_ptr, size * sizeof(float))
        end = <double>clock() / CLOCKS_PER_SEC
        self.rhs_t += end - start

        # Stage 4
        start = <double>clock() / CLOCKS_PER_SEC
        parallel_simd_update2(size, dt, half_dt, three_dt, 
        temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, temp_Phi_ptr, temp_VarPhi_ptr,
        E_ptr, Zx_ptr, Zy_ptr, Zz_ptr, Phi_ptr, VarPhi_ptr,
        k1_E_ptr, k1_Zx_ptr, k1_Zy_ptr, k1_Zz_ptr, k1_P_ptr, k1_V_ptr,
        k2_E_ptr, k2_Zx_ptr, k2_Zy_ptr, k2_Zz_ptr, k2_P_ptr, k2_V_ptr)
        end = <double>clock() / CLOCKS_PER_SEC
        self.memcp_t += end - start
        start = <double>clock() / CLOCKS_PER_SEC
        NewtonRaphson(T_ptr, k2_Zx_ptr, k2_Zy_ptr, k2_E_ptr, k2_P_ptr,
        temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, temp_Phi_ptr,
        self.gam, self.gam2, self.delt, self.del3, self.lam4, self.T_02, self.a, size)
        end = <double>clock() / CLOCKS_PER_SEC
        self.stepTv_t += end - start

        start = <double>clock() / CLOCKS_PER_SEC
        rhsC(temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, temp_Phi_ptr, temp_VarPhi_ptr,
            k2_E_ptr, k2_Zx_ptr, k2_Zy_ptr, k2_Zz_ptr, k2_P_ptr, k2_V_ptr,
            dE_ptr, dZx_ptr, dZy_ptr, dZz_ptr, dP_ptr,
            self.eta, self.half_dr, self.inv_dr2, self.inv_dr_sq, self.theta_term, self.dr_term, self.c_s, self.sxy, self.sz, self.sy, self.sx)
        memcpy(k2_P_ptr, temp_VarPhi_ptr, size * sizeof(float))
        end = <double>clock() / CLOCKS_PER_SEC
        self.rhs_t += end - start

        start = <double>clock() / CLOCKS_PER_SEC
        parallel_simd_update3(size, six_dt, 
        E_ptr, Zx_ptr, Zy_ptr, Zz_ptr, Phi_ptr, VarPhi_ptr,
        k2_E_ptr, k2_Zx_ptr, k2_Zy_ptr, k2_Zz_ptr, k2_P_ptr, k2_V_ptr)
        end = <double>clock() / CLOCKS_PER_SEC
        self.memcp_t += end - start
        start = <double>clock() / CLOCKS_PER_SEC
        NewtonRaphson(T_ptr, k1_Zx_ptr, k1_Zy_ptr, k1_E_ptr, k1_P_ptr,
        temp_E_ptr, temp_Zx_ptr, temp_Zy_ptr, temp_Zz_ptr, temp_Phi_ptr,
        self.gam, self.gam2, self.delt, self.del3, self.lam4, self.T_02, self.a, size)
        end = <double>clock() / CLOCKS_PER_SEC
        self.stepTv_t += end - start

    cdef void snap_hdf5_cdef(self, float t, object group, int* index) noexcept:
        cdef object snapshot_group = group.create_group(f'snapshot_{index[0]:04d}')
        snapshot_group.attrs["t"] = t
        cdef list arrays = [self.k1_Zy, self.T, self.Phi, self.E, self.k1_P
        ]
        cdef list names = ['v2', 'T', 'Phi','E', 'press'
        ]
        cdef Py_ssize_t lx = self.sy
        cdef Py_ssize_t ly = self.sx
        cdef Py_ssize_t lz = self.sz
        
        cdef int i
        for i in range(len(arrays)):
            snapshot_group.create_dataset(
                names[i],
                data=np.asarray(arrays[i]),
                dtype=np.float32,
                compression='lzf',
                chunks=self.chunk_shape,
                shuffle=True
            )
        index[0] += 1

    cpdef run_sim(self, int snaps, str name="scalar_sim_3d.h5"):
        cdef:
            float dt = self.dt
            float t = 0.0000
            float t_end = self.t_end
            float next_time = 0.0
            int per_snap, bpt_single, n, nuc_act
            int bpt_val = 0
            int b_tot = self.tot_bubbles
            str key
            object f, initial_group, snapshots_group
            list snap_times = (np.linspace(0.0, t_end, num=(snaps+1))).tolist()
            list bubble_coords = self.bubble_coords
            bint nucleating = True
            int[::1] bpt = self.BPT
            int* bpt_ptr = &bpt[0]
        cdef dict initial = {
            "gamma": self.gam, "T_0": self.T_0, "delta": self.delt, "lambda": self.lamb, "a": self.a, "eta": self.eta,
            "cfl": self.cfl, "dim_x": self.sx, "dim_y": self.sy, "dim_z": self.sz, "theta": self.dr_term / self.dr, 'dr': self.dr,
            "T_C": self.T_C, "T_N": self.T_N, "t_c": self.t_c, "t_n": self.t_n, "t_init": self.t_init, "t_end": self.t_end,
            "dt": self.dt, "Radius": self.R, "Phi_B": self.Phi_b, "testing": self.test, "nuc_possible": self.tot_bubbles
        }
        if self.test:
            nucleating = False
        cdef int steps = int(t_end / dt + 1)
        cdef int count = 0
        with h5py.File(name, 'w') as f:
            initial_group = f.create_group('init')
            for key, value in initial.items():
                initial_group.attrs[key] = value
            snapshots_group = f.create_group('snapshots')
            self.snap_hdf5_cdef(t, snapshots_group, &count)
            snap_times.pop(0)
            next_time = <float>snap_times[0]
            for i in range(steps):
                with nogil:
                    self.advance_rk4()
                    t += dt
                    bpt_single = bpt_ptr[i]
                if nucleating and (bpt_single > 0):
                    for n in range(bpt_single):
                        self.bubble_insert((bpt_val + n), bubble_coords)
                    bpt_val += bpt_single
                if t >= next_time:
                    self.snap_hdf5_cdef(next_time, snapshots_group, &count)
                    print(f"Progress: t = {next_time:.3f}, End: t = {t_end:.3f}", end='\r', flush=True)
                    snap_times.pop(0)
                    if len(snap_times) == 0:
                        break
                    next_time = <float>snap_times[0]
            initial_group.create_dataset(
                "bubble_coords",
                data=np.asarray(bubble_coords),
                dtype=np.int32,
                compression='lzf',
                shuffle=True)
            nuc_act = len(bubble_coords) - 1
            initial_group.attrs["nuc_actual"] = nuc_act
        print(f"Possible: {b_tot}, Actual: {nuc_act}, Ratio: {(nuc_act/b_tot):.3f}")
        print(f"Adding: {self.memcp_t}, RHS: {self.rhs_t}, NewtonRaphson: {self.stepTv_t}")
        return

    cpdef rerun_sim(self, int snaps, int[:, :] bubble_arr, str name="scalar_sim_3d.h5"):
        cdef:
            float dt = self.dt
            float t = 0.0000
            float t_end = self.t_end
            float next_time = 0.0
            int per_snap, n
            int num_bub = (np.shape(bubble_arr)[0])
            str key
            object f, initial_group, snapshots_group
            list snap_times = (np.linspace(0.0, t_end, num=(snaps+1))).tolist()
        self.c_x = np.ascontiguousarray(bubble_arr[:,2])
        self.c_y = np.ascontiguousarray(bubble_arr[:,1])
        self.c_z = np.ascontiguousarray(bubble_arr[:,0])
        cdef list temp_list = []
        cdef dict initial = {
            "gamma": self.gam, "T_0": self.T_0, "delta": self.delt, "lambda": self.lamb, "a": self.a, "eta": self.eta,
            "cfl": self.cfl, "dim_x": self.sx, "dim_y": self.sy, "dim_z": self.sz, "theta": self.dr_term / self.dr, 'dr': self.dr,
            "T_C": self.T_C, "T_N": self.T_N, "t_c": self.t_c, "t_n": self.t_n, "t_init": self.t_init, "t_end": self.t_end,
            "dt": self.dt, "Radius": self.R, "Phi_B": self.Phi_b,
        }
        cdef int steps = int(t_end / dt + 1)
        cdef int count = 0
        for n in range(num_bub):
            self.bubble_insert(n, temp_list)
        del temp_list
        with h5py.File(name, 'w') as f:
            initial_group = f.create_group('init')
            for key, value in initial.items():
                initial_group.attrs[key] = value
            snapshots_group = f.create_group('snapshots')
            self.snap_hdf5_cdef(t, snapshots_group, &count)
            snap_times.pop(0)
            next_time = <float>snap_times[0]
            for i in range(steps):
                with nogil:
                    self.advance_rk4()
                    t += dt
                if t >= next_time:
                    self.snap_hdf5_cdef(next_time, snapshots_group, &count)
                    print(f"Progress: t = {next_time:.3f}, End: t = {t_end:.3f}", end='\r', flush=True)
                    snap_times.pop(0)
                    if len(snap_times) == 0:
                        break
                    next_time = <float>snap_times[0]
        print(f"Adding: {self.memcp_t}, RHS: {self.rhs_t}, NewtonRaphson: {self.stepTv_t}")
        return
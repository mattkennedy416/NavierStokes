/*
 ============================================================================
 Name        : NavierStokesSolver.cu
 Author      : Matt Kennedy
 Version     : 1.0
 Description : Solve the Navier-Stokes over a flat plate
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace std;



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



// ================================================================
// ================================================================
// define any configuration parameters needed by both CPU and GPU
// ================================================================
// ================================================================

// note that we'll probably get segfaults if the cpu and gpu variables are set to different values!
// (they didn't seem to be happy trying to set one from the other, so need to manually change both for now)
// (and host code isn't happy about reading from a device global variable)
int jmax = 70;
int kmax = 70;
__constant__ int jmax_d = 70;
__constant__ int kmax_d = 70;

double plateLength = 0.00001;
__constant__ double plateLength_d = 0.00001;

double CFL = 0.2; // can fudge this to help stability
__constant__ double CFL_d = 0.2; // can fudge this to help stability

double M0 = 4.0;

double dx = plateLength / (jmax - 1); // uniform for now
//double dy = plateLength / (kmax - 1);
double dy = 1.1869 * pow(10.0,-7); // calculated from boundary layer


__constant__ double u0_d = 1361.12;



// ================================================================
// ================================================================
// define any global variables needed by the GPU
// ================================================================
// ================================================================

__constant__ double gam_d = 1.4; // seems "gamma" is a protected name from the numeric library
__constant__ double Pr_d = 0.71; // Prandtl number
__constant__ double R_d = 287.0; // specific gas constant
__constant__ double Cv_d = 0.7171*1000; // specific heat capacity of air
__constant__ double Cp_d = 1.006 * 1000; // specific heat capacity of air
__constant__ double mu0_d = 1.7894E-5; // dynamic viscosity of air
__constant__ double T0_d = 288.16;
__constant__ double p0_d = 101325.0;



// ================================================================
// ================================================================
// define any global variables needed by the CPU
// ================================================================
// ================================================================


double a0 = 340.28;
double u0 = M0*a0;

double p0 = 101325.0;
double T0 = 288.16;


double v0 = 0;

double gam = 1.4; // seems "gamma" is a protected name from the numeric library
double Pr = 0.71; // Prandtl number
double R = 287.0; // specific gas constant
double Cv = 0.7171*1000; // specific heat capacity of air
double Cp = 1.006 * 1000; // specific heat capacity of air

double rho0 = p0 / (R * T0);
double e0 = T0 * Cv;

double mu0 = 1.7894E-5; // dynamic viscosity of air
double Re = rho0 * u0 * plateLength / mu0;







__device__ void calc_Q(double* Q_d, double* u_d, double* v_d, double* p_d, double* T_d, int j, int k) {

	int ind2 = j*kmax_d + k; // flattened index for our 2d arrays
	int ind3_0 = (j + 0*jmax_d)*kmax_d + k; // flattened index for the first dim of our 3d arrays
	int ind3_1 = (j + 1*jmax_d)*kmax_d + k; // stack them like extra rows
	int ind3_2 = (j + 2*jmax_d)*kmax_d + k;
	int ind3_3 = (j + 3*jmax_d)*kmax_d + k;

	double rho_val = p_d[ind2] / (R_d * T_d[ind2]);
	double e_val = Cv_d * T_d[ind2]; // energy of air based on temp
	double Et_val = rho_val * (e_val + 0.5*(u_d[ind2]*u_d[ind2] + v_d[ind2]*v_d[ind2]));

	Q_d[ind3_0] = rho_val;
	Q_d[ind3_1] = rho_val * u_d[ind2];
	Q_d[ind3_2] = rho_val * v_d[ind2];
	Q_d[ind3_3] = Et_val;

}


__device__ void heatFluxParameters(double* T_d, double mu_val, bool isPredictor, int j, int k, double dx, double dy, double* q) {

	double dTdx;
	double dTdy;

	if (isPredictor) { // scheme is forward, make this backward

		if (j > 0)
			dTdx = (T_d[j*kmax_d + k] - T_d[(j-1)*kmax_d + k])/dx;
		else if (j == 0)
			dTdx = (T_d[(j+1)*kmax_d + k] - T_d[j*kmax_d + k])/dx;

		if (k > 0)
			dTdy = (T_d[j*kmax_d + k] - T_d[j*kmax_d + (k-1)])/dy;
		else if (k == 0)
			dTdy = (T_d[j*kmax_d + k + 1] - T_d[j*kmax_d + k])/dy;

	}
	else { // scheme is backward, make this forward

		if (j < jmax_d-1)
			dTdx = (T_d[(j+1)*kmax_d + k] - T_d[j*kmax_d + k])/dx;
		else if (j == jmax_d - 1)
			dTdx = (T_d[j*kmax_d + k] - T_d[(j-1)*kmax_d + k]) / dx;

		if (k < kmax_d-1)
			dTdy = (T_d[j*kmax_d+k+1] - T_d[j*kmax_d + k]) / dy;
		else if (k == kmax_d - 1)
			dTdy = (T_d[j*kmax_d + k] - T_d[j*kmax_d + k-1]) / dy;

	}

	double k_cond = mu_val * Cp_d / Pr_d;

	q[0] = -k_cond * dTdx;
	q[1] = -k_cond * dTdy;
}

__device__ void shearParameters(double* u_d, double* v_d, double mu, bool isPredictor, int j, int k, double dx, double dy, double* shears) {

	// calculate shear for a single location (j,k)
	// inputs are assumed to be entire matrices

	double dvdx_FB;
	double dudx_FB;
	double dvdy_FB;
	double dudy_FB;
	double dvdx_C;
	double dudx_C;
	double dvdy_C;
	double dudy_C;

	// calculate the forward or backward differenced versions
	if (isPredictor) {
	    // want opposite direction from scheme step differencing
	    // scheme is forward, make this backward

	    if (j > 0) {
	        dvdx_FB = (v_d[j*kmax_d + k] - v_d[(j-1)*kmax_d + k])/dx;
	        dudx_FB = (u_d[j*kmax_d + k] - u_d[(j-1)*kmax_d + k])/dx;
	    }
	    else {
	        dvdx_FB = (v_d[(j+1)*kmax_d + k] - v_d[j*kmax_d + k])/dx; // except first point forward
	        dudx_FB = (u_d[(j+1)*kmax_d + k] - u_d[j*kmax_d + k])/dx; // except first point forward
	    }



	    if (k > 0) {
	        dudy_FB = (u_d[j*kmax_d+k] - u_d[j*kmax_d+k-1])/dy;
	        dvdy_FB = (v_d[j*kmax_d+k] - v_d[j*kmax_d+k-1])/dy;
	    }
	    else {
	        dudy_FB = (u_d[j*kmax_d+k+1] - u_d[j*kmax_d+k])/dy; // except first point forward
	        dvdy_FB = (v_d[j*kmax_d+k+1] - v_d[j*kmax_d+k])/dy; // except first point forward
	    }

	}
	else {

	    // scheme is backward, make this forward

	    if (j < jmax_d - 1) {
	        dvdx_FB = (v_d[(j+1)*kmax_d + k] - v_d[j*kmax_d + k])/dx;
	        dudx_FB = (u_d[(j+1)*kmax_d + k] - u_d[j*kmax_d + k])/dx;
	    }
	    else {
	        dvdx_FB = (v_d[j*kmax_d+k] - v_d[(j-1)*kmax_d + k])/dx; // except jmax backward
	        dudx_FB = (u_d[j*kmax_d+k] - u_d[(j-1)*kmax_d + k])/dx; // except jmax backward
	    }

	    if (k < kmax_d-1) {
	        dudy_FB = (u_d[j*kmax_d + k+1] - u_d[j*kmax_d + k])/dy;
	        dvdy_FB = (v_d[j*kmax_d + k+1] - v_d[j*kmax_d + k])/dy;
	    }
	    else {
	    	dudy_FB = (u_d[j*kmax_d + k] - u_d[j*kmax_d + k-1])/dy; // except kmax backward
			dvdy_FB = (v_d[j*kmax_d + k] - v_d[j*kmax_d + k-1])/dy; // except kmax backward
	    }


	}

	// and then we want centeral differenced versions

	if (j == 0) {
	    dvdx_C = (v_d[(j+1)*kmax_d + k] - v_d[j*kmax_d + k])/dx;
	    dudx_C = (u_d[(j+1)*kmax_d + k] - u_d[j*kmax_d + k])/dx;
	}
	else if (j == jmax_d - 1)
	{
	    dvdx_C = (v_d[j*kmax_d + k] - v_d[(j-1)*kmax_d + k])/dx;
	    dudx_C = (u_d[j*kmax_d + k] - u_d[(j-1)*kmax_d + k])/dx;
	}
	else {
	    dvdx_C = (v_d[(j+1)*kmax_d + k] - v_d[(j-1)*kmax_d + k])/(2*dx);
	    dudx_C = (u_d[(j+1)*kmax_d + k] - u_d[(j-1)*kmax_d + k])/(2*dx);
	}



	if (k == 0) {
	    dudy_C = (u_d[j*kmax_d + k+1] - u_d[j*kmax_d + k])/dy;
	    dvdy_C = (v_d[j*kmax_d + k+1] - v_d[j*kmax_d + k])/dy;
	}
	else if (k == kmax_d-1) {
	    dudy_C = (u_d[j*kmax_d + k] - u_d[j*kmax_d + k-1])/dy;
	    dvdy_C = (v_d[j*kmax_d + k] - v_d[j*kmax_d + k-1])/dy;
	}
	else {
	    dudy_C = (u_d[j*kmax_d + k+1] - u_d[j*kmax_d + k-1])/(2*dy);
	    dvdy_C = (v_d[j*kmax_d + k+1] - v_d[j*kmax_d + k-1])/(2*dy);
	}


	// these come from page 65 and 66 in Anderson

	double lambda = -(2.0/3.0) * mu; // second viscosity coefficient estimated by Stokes

	// use the forward/backward du/dx and central dv/dy for both F and G
	double txx = lambda * ( dudx_FB + dvdy_C ) + 2 * mu * dudx_FB;

	// use the forward/backward dv/dy and central du/dx for both F and G
	double tyy = lambda * ( dudx_C + dvdy_FB ) + 2 * mu * dvdy_FB;

	double txy_F = mu * ( dvdx_FB + dudy_C );
	double txy_G = mu * ( dvdx_C + dudy_FB );

	shears[0] = txx;
	shears[1] = tyy;
	shears[2] = txy_F;
	shears[3] = txy_G;

}


__device__ void calc_FG(double* F_d, double* G_d, double* u_d, double* v_d, double* p_d, double* T_d, bool isPredictor, int j, int k, double dx, double dy) {

	int ind2 = j*kmax_d + k; // flattened index for our 2d arrays
	int ind3_0 = (j + 0*jmax_d)*kmax_d + k; // flattened index for the first dim of our 3d arrays
	int ind3_1 = (j + 1*jmax_d)*kmax_d + k; // stack them like extra rows
	int ind3_2 = (j + 2*jmax_d)*kmax_d + k;
	int ind3_3 = (j + 3*jmax_d)*kmax_d + k;

	double rho_val = p_d[ind2] / (R_d * T_d[ind2]);
	double e_val = Cv_d * T_d[ind2]; // energy of air based on temp
	double Et_val = rho_val * (e_val + 0.5*(u_d[ind2]*u_d[ind2] + v_d[ind2]*v_d[ind2]));

	double mu_val = mu0_d * pow(T_d[ind2] / T0_d, 1.5) * (T0_d + 110)/(T_d[ind2] + 110); // sutherlands law

	double q[2];
	double shears[4];

	heatFluxParameters(T_d, mu_val, isPredictor, j, k, dx, dy, q);
	shearParameters(u_d, v_d, mu_val, isPredictor, j, k, dx, dy, shears);

	// and unpack these for easier use
	double qx = q[0];
	double qy = q[1];
	double txx = shears[0];
	double tyy = shears[1];
	double txy_F = shears[2];
	double txy_G = shears[3];


	F_d[ind3_0] = rho_val * u_d[ind2];
	F_d[ind3_1] = rho_val * pow(u_d[ind2],2) + p_d[ind2] - txx;
	F_d[ind3_2] = rho_val * u_d[ind2]*v_d[ind2] - txy_F;
	F_d[ind3_3] = (Et_val + p_d[ind2]) * u_d[ind2] - u_d[ind2] * txx - v_d[ind2] * txy_F + qx;

	G_d[ind3_0] = rho_val * v_d[ind2];
	G_d[ind3_1] = rho_val * u_d[ind2] * v_d[ind2] - txy_G;
	G_d[ind3_2] = rho_val * pow(v_d[ind2],2) + p_d[ind2] - tyy;
	G_d[ind3_3] = (Et_val + p_d[ind2]) * v_d[ind2] - u_d[ind2] * txy_G - v_d[ind2] * tyy + qy;

}


__device__ void MacCormackPredictorUniform(double* Q_pred_d, double* Q_d, double* F_d, double* G_d, double dt, int j, int k, double dx, double dy) {

// DO MACCORMACKS FOR INTERIOR POINTS ONLY
	if (j == 0 || k == 0 || j == jmax_d-1 || k == kmax_d-1)
		return;

	// have each thread calculate all 4 dimensions at a single loc
	double flux;
	for (int dim=0; dim<4; dim++) {

		int ind_this = (j + dim*jmax_d)*kmax_d + k;
		int ind_nextJ = (j+1 + dim*jmax_d)*kmax_d + k;
		int ind_nextK = (j + dim*jmax_d)*kmax_d + k+1;

		flux = (F_d[ind_nextJ] - F_d[ind_this])/dx + (G_d[ind_nextK] - G_d[ind_this])/dy;
		Q_pred_d[ind_this] = Q_d[ind_this] - dt * flux;
	}
}


__device__ void MacCormackCorrectorUniform(double* Q_pred_d, double* Q_d, double* F_d, double* G_d, double dt, int j, int k, double dx, double dy) {

	// DO MACCORMACKS FOR INTERIOR POINTS ONLY
	if (j == 0 || k == 0 || j == jmax_d-1 || k == kmax_d-1)
		return;

	// have each thread calculate all 4 dimensions at a single (j,k) location
	double flux;
	for (int dim=0; dim<4; dim++) {

		int ind_this = (j + dim*jmax_d)*kmax_d + k;
		int ind_prevJ = (j-1 + dim*jmax_d)*kmax_d + k;
		int ind_prevK = (j + dim*jmax_d)*kmax_d + k-1;

		flux = (F_d[ind_this] - F_d[ind_prevJ])/dx + (G_d[ind_this] - G_d[ind_prevK])/dy;
		Q_d[ind_this] = 0.5*( Q_d[ind_this] + Q_pred_d[ind_this] - dt*flux );
	}
}


__device__ void primativesFromQ(double* Q_d, double* rho_d, double* u_d, double* v_d, double* p_d, double* T_d, double* e_d, int j, int k) {

	int ind2 = j*kmax_d + k; // flattened index for our 2d arrays
	int ind3_0 = (j + 0*jmax_d)*kmax_d + k; // flattened index for the first dim of our 3d arrays
	int ind3_1 = (j + 1*jmax_d)*kmax_d + k; // stack them like extra rows
	int ind3_2 = (j + 2*jmax_d)*kmax_d + k;
	int ind3_3 = (j + 3*jmax_d)*kmax_d + k;

	rho_d[ind2] = Q_d[ind3_0];
	u_d[ind2] = Q_d[ind3_1] / Q_d[ind3_0];
	v_d[ind2] = Q_d[ind3_2] / Q_d[ind3_0];
	e_d[ind2] = Q_d[ind3_3] / Q_d[ind3_0] - 0.5*( pow(u_d[ind2], 2) + pow(v_d[ind2], 2) );

	T_d[ind2] = e_d[ind2] / Cv_d;
	p_d[ind2] = Q_d[ind3_0] * R_d * T_d[ind2];

}


__device__ void enforceBC_nonSurface(double* u_d, double* v_d, double* p_d, double* T_d, int j, int k) {

	// need to first establish all the boundary conditions at the non-surface
	// values, and then go back and do the surface boundary conditions

	// this is really only needed if the surface goes all the way to the outflow
	// so that the last surface point can be interpolated with updated values

	int ind = j*kmax_d + k;

	if ( j == 0 && k == 0) {  // leading edge

		u_d[ind] = 0;
		v_d[ind] = 0;
		p_d[ind] = p0_d;
		T_d[ind] = T0_d;
	}
	else if (j == 0 || k == kmax_d-1) { // inflow from upstream OR upper boundary

		u_d[ind] = u0_d;
		v_d[ind] = 0;
		p_d[ind] = p0_d;
		T_d[ind] = T0_d;
	}
	else if (j == jmax_d-1) { // outflow -- extrapolate from interior values
		int ind1 = (j-1)*kmax_d + k;
		int ind2 = (j-2)*kmax_d + k;

		u_d[ind] = 2*u_d[ind1] - u_d[ind2];
		v_d[ind] = 2*v_d[ind1] - v_d[ind2];
		p_d[ind] = 2*p_d[ind1] - p_d[ind2];
		T_d[ind] = 2*T_d[ind1] - T_d[ind2];

	}
}


__device__ void enforceBC_surface(double* u_d, double* v_d, double* p_d, double* T_d, int j, int k) {

	// need to first establish all the boundary conditions at the non-surface
	// values, and then go back and do the surface boundary conditions

	// this is really only needed if the surface goes all the way to the outflow
	// so that the last surface point can be interpolated with updated values

	int ind = j*kmax_d + k;

	if (k == 0 && j > 0){
		u_d[ind] = 0;
		v_d[ind] = 0;
		p_d[ind] = 2*p_d[j*kmax_d + 1] - p_d[j*kmax_d + 2];
		T_d[ind] = T_d[j*kmax_d + 1];
	}

}




__global__ void iterateScheme_part1(double* x_d, double* y_d, double* u_d, double* v_d, double* p_d, double* T_d, double* rho_d, double* e_d, double* Q_d, double* Q_pred_d, double* F_d, double* G_d, double dx, double dy, double dt) {

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < jmax_d && k < kmax_d)
	{

		calc_Q(Q_d, u_d, v_d, p_d, T_d, j, k);

		bool isPredictor = true;
		calc_FG(F_d, G_d, u_d, v_d, p_d, T_d, isPredictor, j, k, dx, dy);


		// think we need to actually do different kernel launches here ...
		// seems to be no easy way to sync all blocks, and inherently not all blocks may be executed at once if the grid gets too large

//		MacCormackPredictorUniform(Q_pred_d, Q_d, F_d, G_d, dt, j, k, dx, dy);

	}
}


__global__ void iterateScheme_part2(double* x_d, double* y_d, double* u_d, double* v_d, double* p_d, double* T_d, double* rho_d, double* e_d, double* Q_d, double* Q_pred_d, double* F_d, double* G_d, double dx, double dy, double dt) {

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < jmax_d && k < kmax_d)
	{

		// think we need to actually do different kernel launches here ...
		// seems to be no easy way to sync all blocks, and inherently not all blocks may be executed at once if the grid gets too large

		MacCormackPredictorUniform(Q_pred_d, Q_d, F_d, G_d, dt, j, k, dx, dy);

		primativesFromQ(Q_pred_d, rho_d, u_d, v_d, p_d, T_d, e_d, j, k);

	}
}


__global__ void iterateScheme_part3(double* x_d, double* y_d, double* u_d, double* v_d, double* p_d, double* T_d, double* rho_d, double* e_d, double* Q_d, double* Q_pred_d, double* F_d, double* G_d, double dx, double dy, double dt) {

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < jmax_d && k < kmax_d)
	{

		enforceBC_nonSurface(u_d, v_d, p_d, T_d, j, k);

	}
}


__global__ void iterateScheme_part4(double* x_d, double* y_d, double* u_d, double* v_d, double* p_d, double* T_d, double* rho_d, double* e_d, double* Q_d, double* Q_pred_d, double* F_d, double* G_d, double dx, double dy, double dt) {

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < jmax_d && k < kmax_d)
	{

		enforceBC_surface(u_d, v_d, p_d, T_d, j, k);

	}
}


__global__ void iterateScheme_part5(double* x_d, double* y_d, double* u_d, double* v_d, double* p_d, double* T_d, double* rho_d, double* e_d, double* Q_d, double* Q_pred_d, double* F_d, double* G_d, double dx, double dy, double dt) {

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < jmax_d && k < kmax_d)
	{

		bool isPredictor = false;
		calc_FG(F_d, G_d, u_d, v_d, p_d, T_d, isPredictor, j, k, dx, dy);

	}
}


__global__ void iterateScheme_part6(double* x_d, double* y_d, double* u_d, double* v_d, double* p_d, double* T_d, double* rho_d, double* e_d, double* Q_d, double* Q_pred_d, double* F_d, double* G_d, double dx, double dy, double dt) {

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < jmax_d && k < kmax_d)
	{

		MacCormackCorrectorUniform(Q_pred_d, Q_d, F_d, G_d, dt, j, k, dx, dy);

		primativesFromQ(Q_d, rho_d, u_d, v_d, p_d, T_d, e_d, j, k);

	}
}


__global__ void iterateScheme_part7(double* x_d, double* y_d, double* u_d, double* v_d, double* p_d, double* T_d, double* rho_d, double* e_d, double* Q_d, double* Q_pred_d, double* F_d, double* G_d, double dx, double dy, double dt) {

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < jmax_d && k < kmax_d)
	{

		enforceBC_nonSurface(u_d, v_d, p_d, T_d, j, k);

	}
}


__global__ void iterateScheme_part8(double* x_d, double* y_d, double* u_d, double* v_d, double* p_d, double* T_d, double* rho_d, double* e_d, double* Q_d, double* Q_pred_d, double* F_d, double* G_d, double dx, double dy, double dt) {

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < jmax_d && k < kmax_d)
	{

		enforceBC_surface(u_d, v_d, p_d, T_d, j, k);

	}
}






double BoundaryLayerThickness() {
	return 5 * plateLength / sqrt(Re);
}

void setupGrid(double* x, double* y) {
	// just do a uniform grid for now

	for (int j=0; j<jmax; j++)
		x[j] = j*dx;
	for (int k=0; k<kmax; k++)
		y[k] = k*dy;
}

void initializePrimatives(double* u, double* v, double* p, double* T, double* rho, double* e) {

	for (int j=0; j<jmax; j++) {
		for (int k=0; k<kmax; k++) {
			u[j*kmax+k] = u0;
			v[j*kmax+k] = v0;
			p[j*kmax+k] = p0;
			T[j*kmax+k] = T0;
			rho[j*kmax+k] = rho0;
			e[j*kmax+k] = e0;
		}
	}
}


void applyBC_UpperPlate(double* u, double* v, double* p, double* T) {


	// leading edge
	u[0*kmax+0] = 0;
	v[0*kmax+0] = 0;
	p[0*kmax+0] = p0;
	T[0*kmax+0] = T0;

	// inflow (j=0, k=all)
	for (int k=0; k<kmax; k++) {
		u[0*kmax+k] = u0;
		v[0*kmax+k] = 0;
		p[0*kmax+k] = p0;
		T[0*kmax+k] = T0;
	}

	// upper boundary (j=all, k=kmax-1)
	for (int j=0; j<jmax; j++) {
		u[j*kmax+kmax-1] = u0;
		v[j*kmax+kmax-1] = 0;
		p[j*kmax+kmax-1] = p0;
		T[j*kmax+kmax-1] = T0;
	}

	// outflow (j=jmax-1, k=all)
	// extrapolate from interior values
	for (int k=0; k<kmax; k++) {
		u[(jmax-1)*kmax + k] = 2*u[(jmax-2)*kmax + k] - u[(jmax-3)*kmax + k];
		v[(jmax-1)*kmax + k] = 2*v[(jmax-2)*kmax + k] - v[(jmax-3)*kmax + k];
		p[(jmax-1)*kmax + k] = 2*p[(jmax-2)*kmax + k] - p[(jmax-3)*kmax + k];
		T[(jmax-1)*kmax + k] = 2*T[(jmax-2)*kmax + k] - T[(jmax-3)*kmax + k];
	}

	// and plate surface (j=all, k=0)
	for (int j=0; j<jmax; j++) {
		u[j*kmax + 0] = 0;
		v[j*kmax + 0] = 0;
		p[j*kmax + 0] = 2*p[j*kmax + 1] - p[j*kmax + 2];
		T[j*kmax + 0] = T[j*kmax + 1];
	}


}


double calc_dt(double* u, double* v, double* p, double* T, double dx, double dy) {

	// not sure the best way to do this on the GPU
	// seems to be some parallel reduce functions which can be called as their own kernels
	// which would at least prevent us from having to copy back the primative variables to the host every iteration
	// but lets not worry about that for now

	double rho_val;
	double mu_val;
	double temp_val;

	double vprime = -INFINITY;
	for (int j=0; j<jmax; j++) {
		for (int k=0; k<kmax; k++) {

			int ind = j*kmax + k;

			rho_val = p[ind] / (R * T[ind]);
			mu_val = mu0 * pow(T[ind]/T0, 1.5) * (T0 + 110)/(T[ind] + 110);

			temp_val = (4/3) * mu_val * (gam * mu_val / Pr) / rho_val; // find the max of this
			if (temp_val > vprime)
				vprime = temp_val;
		}
	}

	double spaceUnit = pow( 1/(dx*dx) + 1/(dy*dy), 0.5 );
	double term1;
	double term2;
	double term3;
	double term4;
	double dt_cfl;

	double dt = INFINITY;
	for (int j=0; j<jmax; j++) {
		for (int k=0; k<kmax; k++) {

			int ind = j*kmax + k;

			rho_val = p[ind] / (R * T[ind]);

			term1 = abs( u[ind] ) / dx;
			term2 = abs( v[ind] ) / dy;
			term3 = pow( gam*p[ind]/rho_val, 0.5 ) * spaceUnit;
			term4 = 2 * vprime * pow(spaceUnit, 2);

			dt_cfl = 1/(term1 + term2 + term3 + term4);

			if (CFL*dt_cfl < dt)
				dt = CFL*dt_cfl;
		}
	}

	return dt;
}



void arrayToCSV(double* values, char* filename, int numDims) {
 FILE *fp;
 fp = fopen(filename, "w+");

 for (int dim=0; dim<numDims; dim++) {
	 for (int j=0; j<jmax; j++) {
		 for (int k=0; k<kmax; k++) {

			 fprintf(fp, ", %f", values[(j + dim*jmax)*kmax + k]);
		 }
		 fprintf(fp, "\n");
	 }

	 if (dim < numDims-1)
		 fprintf(fp, "Dimension Starting: %i\n", dim+1);
 }


}


int main(void)
{

	double* x = (double*)malloc(jmax*sizeof(double));
	double* y = (double*)malloc(kmax*sizeof(double));

	double* u = (double*)malloc( jmax*kmax*sizeof(double) );
	double* v = (double*)malloc( jmax*kmax*sizeof(double) );
	double* p = (double*)malloc( jmax*kmax*sizeof(double) );
	double* T = (double*)malloc( jmax*kmax*sizeof(double) );
	double* rho = (double*)malloc( jmax*kmax*sizeof(double) );
	double* e = (double*)malloc( jmax*kmax*sizeof(double) );



	initializePrimatives(u, v, p, T, rho, e);
	applyBC_UpperPlate(u, v, p, T);




	// technically only needed in GPU memory, but I assume we may want to copy back intermediate results for debugging
	// calculating these will be a main component of what's being done in parallel, so don't need to initialize anything
	double* Q = (double*)malloc( 4*jmax*kmax*sizeof(double));
	double* Q_pred = (double*)malloc( 4*jmax*kmax*sizeof(double));
	double* F = (double*)malloc( 4*jmax*kmax*sizeof(double));
	double* G = (double*)malloc( 4*jmax*kmax*sizeof(double));


	// iniitialize and allocate device variables
	double* x_d;
	double* y_d;
	double* u_d;
	double* v_d;
	double* p_d;
	double* T_d;
	double* rho_d;
	double* e_d;
	double* Q_d;
	double* Q_pred_d;
	double* F_d;
	double* G_d;

	cudaError_t err;

	err = cudaMalloc((void**)&x_d, jmax*kmax*sizeof(double) );
	err = cudaMalloc((void**)&y_d, jmax*kmax*sizeof(double) );
	err = cudaMalloc((void**)&u_d, jmax*kmax*sizeof(double) );
	err = cudaMalloc((void**)&v_d, jmax*kmax*sizeof(double) );
	err = cudaMalloc((void**)&p_d, jmax*kmax*sizeof(double) );
	err = cudaMalloc((void**)&T_d, jmax*kmax*sizeof(double) );
	err = cudaMalloc((void**)&rho_d, jmax*kmax*sizeof(double) );
	err = cudaMalloc((void**)&e_d, jmax*kmax*sizeof(double) );

	// these are all 3d arrays
	err = cudaMalloc((void**)&Q_d, 4*jmax*kmax*sizeof(double) );
	err = cudaMalloc((void**)&Q_pred_d, 4*jmax*kmax*sizeof(double) );
	err = cudaMalloc((void**)&F_d, 4*jmax*kmax*sizeof(double) );
	err = cudaMalloc((void**)&G_d, 4*jmax*kmax*sizeof(double) );


	err = cudaMemcpy(x_d, x, jmax*kmax*sizeof(double), cudaMemcpyHostToDevice);
	err = cudaMemcpy(y_d, y, jmax*kmax*sizeof(double), cudaMemcpyHostToDevice);
	err = cudaMemcpy(u_d, u, jmax*kmax*sizeof(double), cudaMemcpyHostToDevice);
	err = cudaMemcpy(v_d, v, jmax*kmax*sizeof(double), cudaMemcpyHostToDevice);
	err = cudaMemcpy(p_d, p, jmax*kmax*sizeof(double), cudaMemcpyHostToDevice);
	err = cudaMemcpy(T_d, T, jmax*kmax*sizeof(double), cudaMemcpyHostToDevice);
	err = cudaMemcpy(rho_d, rho, jmax*kmax*sizeof(double), cudaMemcpyHostToDevice);
	gpuErrchk( cudaMemcpy(e_d, e, jmax*kmax*sizeof(double), cudaMemcpyHostToDevice) );






	dim3 threadsPerBlock(16,16);
	dim3 numBlocks(jmax/threadsPerBlock.x + 1, kmax/threadsPerBlock.y + 1);

	// so to force the threads to sync I think it's safer to just do the different stages in different kernel calls, at least initially
	int maxIter = 1000;
	for (int iter=0; iter<maxIter; iter++) {

		printf("Calculating iteration %i / %i\n", iter+1, maxIter);

		double dt = calc_dt(u, v, p, T, dx, dy);

		// calculate F, G, and Q
		iterateScheme_part1<<<numBlocks, threadsPerBlock>>> (x_d, y_d, u_d, v_d, p_d, T_d, rho_d, e_d, Q_d, Q_pred_d, F_d, G_d, dx, dy, dt);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );


		// calculate MacCormack's Predictor and get back primatives out of Q
		iterateScheme_part2<<<numBlocks, threadsPerBlock>>> (x_d, y_d, u_d, v_d, p_d, T_d, rho_d, e_d, Q_d, Q_pred_d, F_d, G_d, dx, dy, dt);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// enforce boundary conditions at non-surface points
		iterateScheme_part3<<<numBlocks, threadsPerBlock>>> (x_d, y_d, u_d, v_d, p_d, T_d, rho_d, e_d, Q_d, Q_pred_d, F_d, G_d, dx, dy, dt);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// enforce boundary conditions at surface points
		iterateScheme_part4<<<numBlocks, threadsPerBlock>>> (x_d, y_d, u_d, v_d, p_d, T_d, rho_d, e_d, Q_d, Q_pred_d, F_d, G_d, dx, dy, dt);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// update F and G for corrected primatives
		iterateScheme_part5<<<numBlocks, threadsPerBlock>>> (x_d, y_d, u_d, v_d, p_d, T_d, rho_d, e_d, Q_d, Q_pred_d, F_d, G_d, dx, dy, dt);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// calculate MacCormack's Corrector and get back primatives out of Q
		iterateScheme_part6<<<numBlocks, threadsPerBlock>>> (x_d, y_d, u_d, v_d, p_d, T_d, rho_d, e_d, Q_d, Q_pred_d, F_d, G_d, dx, dy, dt);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// enforce boundary conditions at non-surface points
		iterateScheme_part7<<<numBlocks, threadsPerBlock>>> (x_d, y_d, u_d, v_d, p_d, T_d, rho_d, e_d, Q_d, Q_pred_d, F_d, G_d, dx, dy, dt);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// enforce boundary conditions at surface points
		iterateScheme_part8<<<numBlocks, threadsPerBlock>>> (x_d, y_d, u_d, v_d, p_d, T_d, rho_d, e_d, Q_d, Q_pred_d, F_d, G_d, dx, dy, dt);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		cudaMemcpy(u, u_d, jmax*kmax*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(v, v_d, jmax*kmax*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(p, p_d, jmax*kmax*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(T, T_d, jmax*kmax*sizeof(double), cudaMemcpyDeviceToHost);

	}


	cudaMemcpy(F, F_d, 4*jmax*kmax*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(G, G_d, 4*jmax*kmax*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Q, Q_d, 4*jmax*kmax*sizeof(double), cudaMemcpyDeviceToHost);

	arrayToCSV(u, "u.csv", 1);
	arrayToCSV(v, "v.csv", 1);
	arrayToCSV(p, "p.csv", 1);
	arrayToCSV(T, "T.csv", 1);

	arrayToCSV(F, "F.csv", 4);
	arrayToCSV(G, "G.csv", 4);
	arrayToCSV(Q, "Q.csv", 4);






	free(x);
	free(y);
	free(u);
	free(v);
	free(p);
	free(T);
	free(rho);
	free(e);
	free(Q);
	free(Q_pred);
	free(F);
	free(G);

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(u_d);
	cudaFree(v_d);
	cudaFree(p_d);
	cudaFree(T_d);
	cudaFree(rho_d);
	cudaFree(e_d);

	cudaFree(Q_d);
	cudaFree(Q_pred_d);
	cudaFree(F_d);
	cudaFree(G_d);

	printf("Finishing!\n");

	return 0;
}


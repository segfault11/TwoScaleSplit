//-----------------------------------------------------------------------------
//  WCSPHSolver.cu
//  FastTurbulentFluids
//
//  Created by Arno in Wolde L�bke on 13.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#include <thrust\sort.h>
#include <thrust\device_ptr.h>
#include <thrust\for_each.h>
#include <thrust\iterator\zip_iterator.h>
#include "WCSPHSolver.h"
#include <iostream>

//-----------------------------------------------------------------------------
//  Macros
//-----------------------------------------------------------------------------
#define EMPTY_CELL 0xFFFFFFFF
#define PI 3.14159265358979323846
//-----------------------------------------------------------------------------
//  Constants on device
//-----------------------------------------------------------------------------
__constant__ float gdDomainOrigin[2];
__constant__ int   gdDomainDimensions[2];
__constant__ int   gdDomainDimensionsHigh[2];
__constant__ float gdEffectiveRadius;
__constant__ float gdEffectiveRadiusHigh;
__constant__ float gdRestDensity;
__constant__ float gdTaitCoefficient;
__constant__ float gdSpeedSound;
__constant__ float gdAlpha;
__constant__ float gdM4KernelCoeff; 
__constant__ float gdM4KernelGradCoeff; 
__constant__ float gdM4KernelCoeffHigh; 
__constant__ float gdM4KernelGradCoeffHigh; 
__constant__ float gdFluidParticleMass;
__constant__ float gdFluidParticleMassHigh;
__constant__ float gdBoundaryParticleMass;
__constant__ float gdTensionCoefficient;

texture<float, cudaTextureType1D, cudaReadModeElementType> 
    gdBoundaryPositionsTex;
texture<int, cudaTextureType1D, cudaReadModeElementType> 
    gdBoundaryParticleIDsTex;
texture<int, cudaTextureType1D, cudaReadModeElementType> 
    gdBoundaryCellStartIndicesTex;
texture<int, cudaTextureType1D, cudaReadModeElementType> 
    gdBoundaryCellEndIndicesTex;
//-----------------------------------------------------------------------------

//=============================================================================
//  DEVICE KERNELS
//=============================================================================
 
//-----------------------------------------------------------------------------
__device__ inline float mexicanHat2D (float x, float y)
{
    #define MEXICAN_HAT_C 0.8673250705840776

	float x2 = x*x;
	float y2 = y*y;

	return MEXICAN_HAT_C*(1.0f - (x2 + y2))*exp(-(x2 + y2)/2.0f);
}
//-----------------------------------------------------------------------------
__device__ inline float evaluateBoundaryForceWeight (float xNorm)  
{
    float q = xNorm*2.0f/gdEffectiveRadius;
   
    float c = 0.02f*gdSpeedSound*gdSpeedSound/(xNorm*xNorm);

    if (q < 2.0f/3.0f)
    {
        return c*2.0f/3.0f;
    }
    else if (q < 1.0f)
    {
        return c*(2.0f*q -3.0f/2.0f*q*q);
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;
        return c*0.5f*a*a;
    }
    else
    {
        return 0.0f;
    }
}
//-----------------------------------------------------------------------------
__device__ inline float evaluateBoundaryForceWeightHigh (float xNorm)  
{
    float q = xNorm*2.0f/gdEffectiveRadiusHigh;
   
    float c = 0.02f*gdSpeedSound*gdSpeedSound/(xNorm*xNorm);

    if (q < 2.0f/3.0f)
    {
        return c*2.0f/3.0f;
    }
    else if (q < 1.0f)
    {
        return c*(2.0f*q -3.0f/2.0f*q*q);
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;
        return c*0.5f*a*a;
    }
    else
    {
        return 0.0f;
    }
}
//-----------------------------------------------------------------------------
__device__ inline float evaluateM4Kernel (float xNorm)  
{
    float q = xNorm*2.0f/gdEffectiveRadius;
    
    if (q < 1.0f)
    {
        float a = 2.0f - q;
        float b = 1.0f - q;

        return gdM4KernelCoeff*(a*a*a - 4.0f*b*b*b);
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;

        return gdM4KernelCoeff*a*a*a;
    }
    else
    {
        return 0.0f;
    }
}
//-----------------------------------------------------------------------------
__device__ inline float evaluateM4KernelHigh (float xNorm)  
{
    float q = xNorm*2.0f/gdEffectiveRadiusHigh;
    
    if (q < 1.0f)
    {
        float a = 2.0f - q;
        float b = 1.0f - q;

        return gdM4KernelCoeffHigh*(a*a*a - 4.0f*b*b*b);
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;

        return gdM4KernelCoeffHigh*a*a*a;
    }
    else
    {
        return 0.0f;
    }
}
//-----------------------------------------------------------------------------
__device__ inline void evaluateGradientM4Kernel (float& gradX, float& gradY,
    const float2& x, float xNorm)  
{
    // NOTE xNorm == 0 lead to devision by zero

    float q = xNorm*2.0f/gdEffectiveRadius;

    if (q < 1.0f)
    {
        float a = 2.0f - q;
        float b = 1.0f - q;
        float c = gdM4KernelGradCoeff*(a*a - 4.0f*b*b)/xNorm;

        gradX = c*x.x;
        gradY = c*x.y;

        return;
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;
        float c = gdM4KernelGradCoeff*a*a/xNorm;

        gradX = c*x.x;
        gradY = c*x.y;

        return;
    }
    else
    {
        gradX = 0.0f;
        gradY = 0.0f;

        return;
    }
}
//-----------------------------------------------------------------------------
__device__ inline void evaluateGradientM4KernelHigh 
(
    float& gradX, 
    float& gradY,
    const float2& x, 
    float xNorm
)  
{
    // NOTE xNorm == 0 lead to devision by zero

    float q = xNorm*2.0f/gdEffectiveRadiusHigh;

    if (q < 1.0f)
    {
        float a = 2.0f - q;
        float b = 1.0f - q;
        float c = gdM4KernelGradCoeffHigh*(a*a - 4.0f*b*b)/xNorm;

        gradX = c*x.x;
        gradY = c*x.y;

        return;
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;
        float c = gdM4KernelGradCoeffHigh*a*a/xNorm;

        gradX = c*x.x;
        gradY = c*x.y;

        return;
    }
    else
    {
        gradX = 0.0f;
        gradY = 0.0f;

        return;
    }
}
//-----------------------------------------------------------------------------
__device__ inline int2 computeGridCoordinate (const float2& pos)
{
    int2 coord;
    coord.x = (pos.x - gdDomainOrigin[0])/gdEffectiveRadius;
    coord.y = (pos.y - gdDomainOrigin[1])/gdEffectiveRadius;
    
    coord.x = min(max(0, coord.x), gdDomainDimensions[0] - 1);
    coord.y = min(max(0, coord.y), gdDomainDimensions[1] - 1);

    return coord;
}
//-----------------------------------------------------------------------------
__device__ inline int2 computeGridCoordinateHigh (const float2& pos)
{
    int2 coord;
    coord.x = (pos.x - gdDomainOrigin[0])/gdEffectiveRadiusHigh;
    coord.y = (pos.y - gdDomainOrigin[1])/gdEffectiveRadiusHigh;
    
    coord.x = min(max(0, coord.x), gdDomainDimensionsHigh[0] - 1);
    coord.y = min(max(0, coord.y), gdDomainDimensionsHigh[1] - 1);

    return coord;
}
//-----------------------------------------------------------------------------
__device__ inline int2 computeGridCoordinate (const float2& pos, 
    float offset)
{
    int2 coord;
    coord.x = (pos.x + offset - gdDomainOrigin[0])/gdEffectiveRadius;
    coord.y = (pos.y + offset - gdDomainOrigin[1])/gdEffectiveRadius;
    
    coord.x = min(max(0, coord.x), gdDomainDimensions[0] - 1);
    coord.y = min(max(0, coord.y), gdDomainDimensions[1] - 1);
    
    return coord;
}
//-----------------------------------------------------------------------------
__device__ inline int2 computeGridCoordinateHigh (const float2& pos, 
    float offset)
{
    int2 coord;
    coord.x = (pos.x + offset - gdDomainOrigin[0])/gdEffectiveRadiusHigh;
    coord.y = (pos.y + offset - gdDomainOrigin[1])/gdEffectiveRadiusHigh;
    
    coord.x = min(max(0, coord.x), gdDomainDimensionsHigh[0] - 1);
    coord.y = min(max(0, coord.y), gdDomainDimensionsHigh[1] - 1);
    
    return coord;
}
//-----------------------------------------------------------------------------
__device__ inline int computeHash (const float2& pos)
{
    int2 coord = computeGridCoordinate(pos);
   
    return coord.y*gdDomainDimensions[0] + coord.x; 
}
//-----------------------------------------------------------------------------
__device__ inline int computeHashHigh (const float2& pos)
{
    int2 coord = computeGridCoordinateHigh(pos);
   
    return coord.y*gdDomainDimensionsHigh[0] + coord.x; 
}
//-----------------------------------------------------------------------------
__device__ inline int computeHash (const int2& coord)
{ 
    return coord.y*gdDomainDimensions[0] + coord.x; 
}
//-----------------------------------------------------------------------------
__device__ inline int computeHashHigh (const int2& coord)
{ 
    return coord.y*gdDomainDimensionsHigh[0] + coord.x; 
}
//-----------------------------------------------------------------------------
__device__ inline int computeHash (unsigned int i, unsigned int j)
{ 
    return j*gdDomainDimensions[0] + i; 
}
//-----------------------------------------------------------------------------
__device__ inline int computeHashHigh (unsigned int i, unsigned int j)
{ 
    return j*gdDomainDimensionsHigh[0] + i; 
}
//-----------------------------------------------------------------------------
__device__ inline float norm (const float2& v)
{
    return sqrt(v.x*v.x + v.y*v.y);
}
//-----------------------------------------------------------------------------
__device__ inline float dot (const float2& a, const float2& b)
{
    return a.x*b.x + a.y*b.y;
}
//-----------------------------------------------------------------------------
__device__ inline void updateDensityCell
(
    float& density, 
    const float2& pos,
    const float* const dParticlePositons, 
    const int* const dParticleIDs,
    int start, 
    int end
)
{
    for (int i = start; i < end; i++)
    {
        int j = dParticleIDs[i] & 0x7FFFFFFF;
        float2 posj;
        posj.x = dParticlePositons[2*j + 0];
        posj.y = dParticlePositons[2*j + 1];

        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        
        float dist = norm(posij);
        
        if (dist <= gdEffectiveRadius)
        {
            density += (evaluateM4Kernel(dist) + evaluateM4KernelHigh(dist))/2.0f;
        }
    }
}
//-----------------------------------------------------------------------------
__device__ inline void updateDensityCellComplement
(
    float& density, 
    const float2& pos,
    const float* const dParticlePositons, 
    const unsigned char* const dStates,
    const int* const dParticleIDs,
    int start, 
    int end,
    unsigned char statei
)
{     
    float lambda[4] = {1.0f, 0.0f, 1.0f, 0.0f}; // smells

    for (int i = start; i < end; i++)
    {
        unsigned int j = dParticleIDs[i];    
        unsigned char statej = dStates[j];        

        float2 posj;
        posj.x = dParticlePositons[2*j + 0];
        posj.y = dParticlePositons[2*j + 1];

        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        
        float dist = norm(posij);
        
        if (dist <= gdEffectiveRadius)
        {
            density += lambda[((statei & 0x0001) << 1) | (statej & 0x0001)]*
                (evaluateM4Kernel(dist) + evaluateM4KernelHigh(dist))/2.0f;
        }
    }
}
//-----------------------------------------------------------------------------
__device__ inline void updateDensityCellHigh
(
    float& density, 
    const float2& pos,
    const float* const dParticlePositions, 
    const unsigned char* const dStates,
    const int* const dParticleIDs,
    int start, 
    int end,
    unsigned char statei
)
{
    float lambda[4] = {1.0f, 0.0f, 1.0f, 1.0f};

    for (int i = start; i < end; i++)
    {
        int j = dParticleIDs[i];
        unsigned char statej = dStates[j];

        float2 posj;
        posj.x = dParticlePositions[2*j + 0];
        posj.y = dParticlePositions[2*j + 1];

        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        
        float dist = norm(posij);
        
        if (dist <= gdEffectiveRadiusHigh)
        {
            density += lambda[((statei & 0x0001) << 1) | (statej & 0x0001)]*
                (evaluateM4Kernel(dist) + evaluateM4KernelHigh(dist))/2.0f;
        }
    }
}
//-----------------------------------------------------------------------------
__device__ inline void updateDensityCellComplementHigh
(
    float& density, 
    const float2& pos,
    const float* const dParticlePositions, 
    const unsigned char* const dStates,
    const int* const dParticleIDs,
    int start, 
    int end,
    unsigned char statei
)
{
    float lambda[4] = {1.0f, 1.0f, 1.0f, 0.0f};

    for (int i = start; i < end; i++)
    {
        int j = dParticleIDs[i];
        unsigned char statej = dStates[j];

        float2 posj;
        posj.x = dParticlePositions[2*j + 0];
        posj.y = dParticlePositions[2*j + 1];

        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        
        float dist = norm(posij);
        
        if (dist <= gdEffectiveRadiusHigh)
        {
            density += lambda[((statei & 0x0001) << 1) | (statej & 0x0001)]*
                (evaluateM4Kernel(dist) + evaluateM4KernelHigh(dist))/2.0f;
        }
    }
}
//-----------------------------------------------------------------------------
__device__ inline void updateAccCell
(   
    float2& acc, 
    float2& accT, 
    float2& accB, 
    float2& ene,
    float& psiSum,
    const float2& pos, 
    const float2& vel, 
    float dens, 
    float pre, 
    const float* const dParticlePositions, 
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const float* const dParticleVelocities, 
    const unsigned char* const dStates,
    const int* const dParticleIDs, 
    int start, 
    int end, 
    int startB, 
    int endB
)
{
    float dens2 = dens*dens;
    float deleted[3] = {1.0f, 1.0f, 0.0f};

    for (int i = start; i < end; i++)
    {
        int j = dParticleIDs[i];
        unsigned char statej = dStates[j];

        float2 posj;
        posj.x = dParticlePositions[2*j + 0];
        posj.y = dParticlePositions[2*j + 1];
        float2 velj;
        velj.x = dParticleVelocities[2*j + 0];
        velj.y = dParticleVelocities[2*j + 1];
        float densj = dParticleDensities[j];
        float prej = dParticlePressures[j];
        float densj2 = densj*densj;
        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        float dist = norm(posij);

        if (dist != 0.0f && dist < gdEffectiveRadius)
        {
            float l = deleted[statej];

            // compute pressure contribution
            float coeff = pre/dens2 + prej/densj2;

            // compute artificial velocity
            float2 velij;
            velij.x = vel.x - velj.x;
            velij.y = vel.y - velj.y;
            float dvp = dot(velij, posij);

            if (dvp < 0.0f)
            {
                coeff -= dvp/(dist*dist + 0.01f*gdEffectiveRadius*
                    gdEffectiveRadius)*2.0f*gdEffectiveRadius*gdAlpha*
                    gdSpeedSound/(dens + densj);
            }
            
            float2 gradI;
            float2 gradJ;
            evaluateGradientM4Kernel(gradI.x, gradI.y, posij, dist);
            evaluateGradientM4KernelHigh(gradJ.x, gradJ.y, posij, dist);

            acc.x += l*coeff*(gradI.x + gradJ.x)/2.0f;
            acc.y += l*coeff*(gradI.y + gradJ.y)/2.0f;

            float w = evaluateM4Kernel(dist);
            float w2 = evaluateM4KernelHigh(dist);
            accT.x += l*posij.x*(w + w2)/2.0f;
            accT.y += l*posij.y*(w + w2)/2.0f; 

            float mh = mexicanHat2D(posij.x/gdEffectiveRadius, 
                posij.y/gdEffectiveRadius);
            ene.x += velj.x*mh;
            ene.y += velj.y*mh;
            psiSum += mh;
        }
    }

    float c = gdBoundaryParticleMass/(gdFluidParticleMass + 
        gdBoundaryParticleMass);

    for (int i = startB; i < endB; i++)
    {
        int k = tex1Dfetch(gdBoundaryParticleIDsTex, i);
        float2 posk;
        posk.x = tex1Dfetch(gdBoundaryPositionsTex, 2*k + 0);
        posk.y = tex1Dfetch(gdBoundaryPositionsTex, 2*k + 1);
        float2 posik;
        posik.x = pos.x - posk.x;
        posik.y = pos.y - posk.y;
        float dist = norm(posik);
        float gamma = evaluateBoundaryForceWeight(dist);

        accB.x += c*gamma*posik.x;
        accB.y += c*gamma*posik.y; 
    }
}
//-----------------------------------------------------------------------------
__device__ inline void updateAccCellComplement
(   
    float2& acc, 
    float2& accT, 
    float2& accB, 
    const float2& pos, 
    const float2& vel, 
    float dens, 
    float pre, 
    const float* const dParticlePositions, 
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const float* const dParticleVelocities, 
    const unsigned char* const dStates,
    const int* const dParticleIDs, 
    int start, 
    int end,
    unsigned char statei
)
{
    float lambda[4] = {1.0f, 0.0f, 1.0f, 0.0f}; 

    float dens2 = dens*dens;

    for (int i = start; i < end; i++)
    {
        unsigned int j = dParticleIDs[i];    
        unsigned char statej = dStates[j];
        
        float2 posj;
        posj.x = dParticlePositions[2*j + 0];
        posj.y = dParticlePositions[2*j + 1];
        float2 velj;
        velj.x = dParticleVelocities[2*j + 0];
        velj.y = dParticleVelocities[2*j + 1];
        float densj = dParticleDensities[j];
        float prej = dParticlePressures[j];
        float densj2 = densj*densj;
        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        float dist = norm(posij);

        if (dist != 0.0f && dist < gdEffectiveRadius)
        {
            float l = lambda[((statei & 0x0001) << 1) | (statej & 0x0001)];

            // compute pressure contribution
            float coeff = pre/dens2 + prej/densj2;

            // compute artifcial velocity
            float2 velij;
            velij.x = vel.x - velj.x;
            velij.y = vel.y - velj.y;
            float dvp = dot(velij, posij);
            float h = (gdEffectiveRadius + gdEffectiveRadiusHigh)/2.0f;

            if (dvp < 0.0f)
            {
                coeff -= dvp/(dist*dist + 0.01f*h*h)*2.0f*h*gdAlpha*
                    gdSpeedSound/(dens + densj);
            }
            
            float2 gradI;
            float2 gradJ;
            evaluateGradientM4Kernel(gradI.x, gradI.y, posij, dist);
            evaluateGradientM4KernelHigh(gradJ.x, gradJ.y, posij, dist);

            acc.x += l*coeff*(gradI.x + gradJ.x)/2.0f;
            acc.y += l*coeff*(gradI.y + gradJ.y)/2.0f;


            float w = evaluateM4Kernel(dist);
            float w2 = evaluateM4KernelHigh(dist);
            accT.x += l*posij.x*(w + w2)/2.0f;
            accT.y += l*posij.y*(w + w2)/2.0f; 
        }
    }
}
//-----------------------------------------------------------------------------
__device__ inline void updateAccCellHigh
(   
    float2& acc, 
    float2& accT, 
    const float2& pos, 
    const float2& vel, 
    float dens, 
    float pre, 
    const float* const dParticlePositions, 
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const float* const dParticleVelocities, 
    const unsigned char* const dStates,
    const int* const dParticleIDs, 
    int start, 
    int end,
    unsigned char statei
)
{
    float lambda[4] = {1.0f, 0.0f, 1.0f, 1.0f};
    float dens2 = dens*dens;

    for (int i = start; i < end; i++)
    {
        int j = dParticleIDs[i];
        unsigned char statej = dStates[j];

        float2 posj;
        posj.x = dParticlePositions[2*j + 0];
        posj.y = dParticlePositions[2*j + 1];
        float2 velj;
        velj.x = dParticleVelocities[2*j + 0];
        velj.y = dParticleVelocities[2*j + 1];
        float densj = dParticleDensities[j];
        float prej = dParticlePressures[j];
        float densj2 = densj*densj;
        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        float dist = norm(posij);

        if (dist != 0.0f && dist < gdEffectiveRadiusHigh)
        {
            float l = lambda[((statei & 0x0001) << 1) | (statej & 0x0001)];

            // compute pressure contribution
            float coeff = pre/dens2 + prej/densj2;

            // compute artifcial velocity
            float2 velij;
            velij.x = vel.x - velj.x;
            velij.y = vel.y - velj.y;
            float dvp = dot(velij, posij);

            if (dvp < 0.0f)
            {
                coeff -= dvp/(dist*dist + 0.01f*gdEffectiveRadiusHigh*
                    gdEffectiveRadiusHigh)*2.0f*gdEffectiveRadiusHigh*gdAlpha*
                    gdSpeedSound/(dens + densj);
            }
            
            float2 gradI;
            float2 gradJ;
            evaluateGradientM4Kernel(gradI.x, gradI.y, posij, dist);
            evaluateGradientM4KernelHigh(gradJ.x, gradJ.y, posij, dist);

            acc.x += l*coeff*(gradI.x + gradJ.x)/2.0f;
            acc.y += l*coeff*(gradI.y + gradJ.y)/2.0f;

            float w = evaluateM4Kernel(dist);
            float w2 = evaluateM4KernelHigh(dist);
            accT.x += l*posij.x*(w + w2)/2.0f;
            accT.y += l*posij.y*(w + w2)/2.0f; 
        }
    }
}
//-----------------------------------------------------------------------------
__device__ inline void updateAccCellHighRelax
(   
    float2& acc, 
    float2& accT, 
    const float2& pos, 
    const float2& vel, 
    float dens, 
    float pre, 
    const float* const dParticlePositions, 
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const float* const dParticleVelocities, 
    const int* const dParticleIDs, 
    int start, 
    int end
)
{
    float dens2 = dens*dens;

    for (int i = start; i < end; i++)
    {
        int j = dParticleIDs[i];
        float2 posj;
        posj.x = dParticlePositions[2*j + 0];
        posj.y = dParticlePositions[2*j + 1];
        float densj = dParticleDensities[j];
        float prej = dParticlePressures[j];
        float densj2 = densj*densj;
        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        float dist = norm(posij);

        if (dist != 0.0f && dist < gdEffectiveRadiusHigh)
        {
            // compute pressure contribution
            float coeff = pre/dens2 + prej/densj2;
            
            float2 grad;
            evaluateGradientM4KernelHigh(grad.x, grad.y, posij, dist);

            acc.x += coeff*grad.x;
            acc.y += coeff*grad.y;
        }
    }
}
//-----------------------------------------------------------------------------
__device__ inline void updateAccCellComplementHigh
(   
    float2& acc, 
    float2& accT, 
    float2& accB, 
    const float2& pos, 
    const float2& vel, 
    float dens, 
    float pre, 
    const float* const dParticlePositions, 
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const float* const dParticleVelocities, 
    const unsigned char* const dStates,
    const int* const dParticleIDs, 
    int start, 
    int end, 
    int startB, 
    int endB,
    unsigned char statei
)
{
    float lambda[4] = {1.0f, 1.0f, 1.0f, 0.0f};
    float deleted[3] = {1.0f, 1.0f, 0.0f};
    
    float dens2 = dens*dens;

    for (int i = start; i < end; i++)
    {
        int j = dParticleIDs[i];
        unsigned char statej = dStates[j];

        float2 posj;
        posj.x = dParticlePositions[2*j + 0];
        posj.y = dParticlePositions[2*j + 1];
        float2 velj;
        velj.x = dParticleVelocities[2*j + 0];
        velj.y = dParticleVelocities[2*j + 1];
        float densj = dParticleDensities[j];
        float prej = dParticlePressures[j];
        float densj2 = densj*densj;
        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        float dist = norm(posij);

        if (dist != 0.0f && dist < gdEffectiveRadius)
        {
            // compute pressure contribution
            float coeff = pre/dens2 + prej/densj2;
            float l = lambda[((statei & 0x0001) << 1) | (statej & 0x0001)]*
                deleted[statej];

            // compute artifcial velocity
            float2 velij;
            velij.x = vel.x - velj.x;
            velij.y = vel.y - velj.y;
            float dvp = dot(velij, posij);
            float h = (gdEffectiveRadius + gdEffectiveRadiusHigh)/2.0f;
            
            if (dvp < 0.0f)
            {
                coeff -= dvp/(dist*dist + 0.01f*h*h)*2.0f*h*gdAlpha*
                    gdSpeedSound/(dens + densj);
            }
            
            float2 gradI;
            float2 gradJ;
            evaluateGradientM4Kernel(gradI.x, gradI.y, posij, dist);
            evaluateGradientM4KernelHigh(gradJ.x, gradJ.y, posij, dist);

            acc.x += l*coeff*(gradI.x + gradJ.x)/2.0f;
            acc.y += l*coeff*(gradI.y + gradJ.y)/2.0f;

            float w = evaluateM4Kernel(dist);
            float w2 = evaluateM4KernelHigh(dist);
            accT.x += l*posij.x*(w + w2)/2.0f;
            accT.y += l*posij.y*(w + w2)/2.0f; 
        }
    }

    float c = gdBoundaryParticleMass/(gdFluidParticleMassHigh + 
        gdBoundaryParticleMass);

    for (int i = startB; i < endB; i++)
    {
        int k = tex1Dfetch(gdBoundaryParticleIDsTex, i);
        float2 posk;
        posk.x = tex1Dfetch(gdBoundaryPositionsTex, 2*k + 0);
        posk.y = tex1Dfetch(gdBoundaryPositionsTex, 2*k + 1);
        float2 posik;
        posik.x = pos.x - posk.x;
        posik.y = pos.y - posk.y;
        float dist = norm(posik);
        float gamma = evaluateBoundaryForceWeightHigh(dist);

        accB.x += c*gamma*posik.x;
        accB.y += c*gamma*posik.y; 
    }
}
//-----------------------------------------------------------------------------
__device__ inline void updateAccCellComplementHighRelax
(   
    float2& acc, 
    float2& accT, 
    float2& accB, 
    const float2& pos, 
    const float2& vel, 
    float dens, 
    float pre, 
    const float* const dParticlePositions, 
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const float* const dParticleVelocities, 
    const int* const dParticleIDs, 
    int start, 
    int end, 
    int startB, 
    int endB
)
{
    float dens2 = dens*dens;

    for (int i = start; i < end; i++)
    {
        int j = dParticleIDs[i];
        float2 posj;
        posj.x = dParticlePositions[2*j + 0];
        posj.y = dParticlePositions[2*j + 1];
        float densj = dParticleDensities[j];
        float prej = dParticlePressures[j];
        float densj2 = densj*densj;
        float2 posij;
        posij.x = pos.x - posj.x;
        posij.y = pos.y - posj.y;
        float dist = norm(posij);

        if (dist != 0.0f && dist < gdEffectiveRadiusHigh)
        {
            // compute pressure contribution
            float coeff = pre/dens2 + prej/densj2;
            
            float2 grad;
            evaluateGradientM4KernelHigh(grad.x, grad.y, posij, dist);
            
            acc.x += coeff*grad.x;
            acc.y += coeff*grad.y;
        }
    }

    float c = gdBoundaryParticleMass/(gdFluidParticleMassHigh + 
        gdBoundaryParticleMass);

    for (int i = startB; i < endB; i++)
    {
        int k = tex1Dfetch(gdBoundaryParticleIDsTex, i);
        float2 posk;
        posk.x = tex1Dfetch(gdBoundaryPositionsTex, 2*k + 0);
        posk.y = tex1Dfetch(gdBoundaryPositionsTex, 2*k + 1);
        float2 posik;
        posik.x = pos.x - posk.x;
        posik.y = pos.y - posk.y;
        float dist = norm(posik);
        float gamma = evaluateBoundaryForceWeightHigh(dist);

        accB.x += c*gamma*posik.x;
        accB.y += c*gamma*posik.y; 
    }
}
//-----------------------------------------------------------------------------
__device__ void initSubParticlesAndAddToList
(
    float* const dParticlePositionsHigh,
    float* const dParticleVelocitiesHigh,
    unsigned char* const dStatesHigh,
    int* const dParticleIDsHigh,
    int* const dGLIndicesHigh,
    int numParticlesHigh,                   // current # of high particles             
    unsigned int id,                        // id of the low particle
    const float2& pos,
    const float2& vel,
    float radius
)
{
    // add high res particles to the list & and mark as transient
    unsigned int idH = 4*id + 0;
    //dGLIndicesHigh[numParticlesHigh + 0] = idH;
    dParticleIDsHigh[numParticlesHigh + 0] = idH;
    dStatesHigh[idH] = 0x0001;
    idH = 4*id + 1;
    //dGLIndicesHigh[numParticlesHigh + 1] = idH;
    dParticleIDsHigh[numParticlesHigh + 1] = idH ;
    dStatesHigh[idH] = 0x0001;
    idH = 4*id + 2;
    //dGLIndicesHigh[numParticlesHigh + 2] = idH;
    dParticleIDsHigh[numParticlesHigh + 2] = idH;
    dStatesHigh[idH] = 0x0001;
    idH = 4*id + 3;
    //dGLIndicesHigh[numParticlesHigh + 3] = idH;
    dParticleIDsHigh[numParticlesHigh + 3] = idH;
    dStatesHigh[idH] = 0x0001;

    dParticlePositionsHigh[8*id + 0] = pos.x + radius;
    dParticlePositionsHigh[8*id + 1] = pos.y + radius;
    dParticlePositionsHigh[8*id + 2] = pos.x - radius;
    dParticlePositionsHigh[8*id + 3] = pos.y + radius;
    dParticlePositionsHigh[8*id + 4] = pos.x + radius;
    dParticlePositionsHigh[8*id + 5] = pos.y - radius;
    dParticlePositionsHigh[8*id + 6] = pos.x - radius;
    dParticlePositionsHigh[8*id + 7] = pos.y - radius;

    dParticleVelocitiesHigh[8*id + 0] = vel.x;
    dParticleVelocitiesHigh[8*id + 1] = vel.y;
    dParticleVelocitiesHigh[8*id + 2] = vel.x;
    dParticleVelocitiesHigh[8*id + 3] = vel.y;
    dParticleVelocitiesHigh[8*id + 4] = vel.x;
    dParticleVelocitiesHigh[8*id + 5] = vel.y;
    dParticleVelocitiesHigh[8*id + 6] = vel.x;
    dParticleVelocitiesHigh[8*id + 7] = vel.y;
}
//-----------------------------------------------------------------------------
__device__ inline void adjustPosition 
(
    float2& posH, 
    const float2& posL,
    float density
)
{
    float2 posji;
    posji.x = posH.x - posL.x;
    posji.y = posH.y - posL.y;
    float dist = norm(posji);
    float r = min(dist, 0.75f*sqrt(gdFluidParticleMass/(density*PI)));

    posH.x = posL.x + r/dist*posji.x;
    posH.y = posL.y + r/dist*posji.y;
}
//-----------------------------------------------------------------------------

//=============================================================================
//  GLOBAL KERNELS
//=============================================================================

//-----------------------------------------------------------------------------
__global__ void setQuantities
(
    float* const dQuantities,
    const float* const dPositions,
    const float* const dDensities,
    const int* const dParticleIDs, 
    unsigned int numParticles
)
{
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    unsigned int id = dParticleIDs[idx];

    float density = dDensities[id];

    dQuantities[id] = min(abs(density - gdRestDensity), 400.0f)/400.0f;
}
//-----------------------------------------------------------------------------
__global__ void injectTransientHighD
(
    unsigned char* const dStates,
    unsigned char* const dStatesHigh,
    const float* const dDensities,
    const float* const dDensitiesHigh,
    const int* const dTransientIDsLow,
    unsigned int numTransientLow
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numTransientLow)
    {
        return;
    }

    // get id of the particle
    unsigned int id = dTransientIDsLow[idx];

    // compute local density fluctuation
    float density = dDensities[id];
    float fluc = 0.0f;

    for (unsigned int i = 0; i < 4; i++)
    {
        fluc += abs(gdRestDensity - dDensitiesHigh[4*id + i]);
    }

    fluc = fluc/(4.0f*gdRestDensity);

    if (false)
    {
        // set high res particles to default
        for (unsigned int i = 0; i < 4; i++)
        {
            dStatesHigh[4*id + i] = 0x0000;
        }    

        // set state to deleted
        dStates[id] = 0x0002;
    }
}
//-----------------------------------------------------------------------------
__global__ void adjustOrInjectTransientHighD
(
    float* const dPositionsHigh,
    float* const dVelocitiesHigh,
    unsigned char* const dStatesHigh,
    int* const dParticleIDsLowNew,
    int* const dGLParticleIDsLow,
    int* const dTransientIDsLowNew,
    int* const dParticleCountLow,
    int* const dTransientCountLow,
    const float* const dPositionsLow,
    const float* const dVelocitiesLow,
    const float* const dDensities,
    const int* const dTransientIDsLow,
    unsigned int numTransientLow
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numTransientLow)
    {
        return;
    }

    // get id of the particle
    unsigned int id = dTransientIDsLow[idx];    
    float density = dDensities[id];

    if (false)
    {
        dStatesHigh[4*id + 0] = 0x0000;
        dStatesHigh[4*id + 1] = 0x0000;
        dStatesHigh[4*id + 2] = 0x0000;
        dStatesHigh[4*id + 3] = 0x0000;
        return;
    }


    int index = atomicAdd(dParticleCountLow, 1);
    dParticleIDsLowNew[index] = id;
    dGLParticleIDsLow[index] = id;
    
    int index2 = atomicAdd(dTransientCountLow, 1);
    dTransientIDsLowNew[index2] = id;



    // get pos and vel
    float2 pos;
    pos.x = dPositionsLow[2*id + 0];
    pos.y = dPositionsLow[2*id + 1];
    float2 vel;
    vel.x = dVelocitiesLow[2*id + 0]; 
    vel.y = dVelocitiesLow[2*id + 1]; 

    // adjust the positions of the child particles
    for (unsigned int i = 0; i < 4; i++)
    {
        float2 posH;
        posH.x = dPositionsHigh[8*id + 2*i + 0];
        posH.y = dPositionsHigh[8*id + 2*i + 1];
        adjustPosition(posH, pos, density);
        dPositionsHigh[8*id + 2*i + 0] = posH.x;
        dPositionsHigh[8*id + 2*i + 1] = posH.y;
    }

    // adjust the velocities of the child particles
    //dVelocitiesHigh[8*id + 0] = vel.x;
    //dVelocitiesHigh[8*id + 1] = vel.y;
    //dVelocitiesHigh[8*id + 2] = vel.x;
    //dVelocitiesHigh[8*id + 3] = vel.y;
    //dVelocitiesHigh[8*id + 4] = vel.x;
    //dVelocitiesHigh[8*id + 5] = vel.y;
    //dVelocitiesHigh[8*id + 6] = vel.x;
    //dVelocitiesHigh[8*id + 7] = vel.y;
}
//-----------------------------------------------------------------------------
__global__ void adjustTransientHighD
(
    float* const dPositionsHigh,
    float* const dVelocitiesHigh,
    const float* const dPositionsLow,
    const float* const dVelocitiesLow,
    const float* const dDensities,
    const int* const dTransientIDsLow,
    unsigned int numTransientLow
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numTransientLow)
    {
        return;
    }

    // get id of the particle
    unsigned int id = dTransientIDsLow[idx];    


    float density = dDensities[id];

    // get pos and vel
    float2 pos;
    pos.x = dPositionsLow[2*id + 0];
    pos.y = dPositionsLow[2*id + 1];
    float2 vel;
    vel.x = dVelocitiesLow[2*id + 0]; 
    vel.y = dVelocitiesLow[2*id + 1]; 

    // adjust the positions of the child particles
    for (unsigned int i = 0; i < 4; i++)
    {
        float2 posH;
        posH.x = dPositionsHigh[8*id + 2*i + 0];
        posH.y = dPositionsHigh[8*id + 2*i + 1];
        adjustPosition(posH, pos, density);
        dPositionsHigh[8*id + 2*i + 0] = posH.x;
        dPositionsHigh[8*id + 2*i + 1] = posH.y;
    }

    // adjust the velocities of the child particles
    dVelocitiesHigh[8*id + 0] = vel.x;
    dVelocitiesHigh[8*id + 1] = vel.y;
    dVelocitiesHigh[8*id + 2] = vel.x;
    dVelocitiesHigh[8*id + 3] = vel.y;
    dVelocitiesHigh[8*id + 4] = vel.x;
    dVelocitiesHigh[8*id + 5] = vel.y;
    dVelocitiesHigh[8*id + 6] = vel.x;
    dVelocitiesHigh[8*id + 7] = vel.y;
}
//-----------------------------------------------------------------------------
__global__ void initParticleIDs 
(   
    int* const dParticleIDs, 
    unsigned int maxParticles
)
{
    // initially the particle id list stores particle ids in ascending order

    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= maxParticles)
    {
        return;
    }

    dParticleIDs[idx] = idx;
}
//-----------------------------------------------------------------------------
__global__ void computeParticleHash 
(   
    int* const dParticleHashs, 
    int* const dParticleIDs, 
    const float* const dParticlePositions, 
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    // get particle id
    int id = dParticleIDs[idx] & 0x7FFFFFFF;

    // get particle position
    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];
    
    // set particle hash
    dParticleHashs[idx] = computeHash(pos);
}
//-----------------------------------------------------------------------------
__global__ void computeParticleHashHigh
(   
    int* const dParticleHashs, 
    int* const dParticleIDs, 
    const float* const dParticlePositions, 
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    // get particle id
    int id = dParticleIDs[idx] & 0x7FFFFFFF;

    // get particle position
    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];
    
    // set particle hash
    dParticleHashs[idx] = computeHashHigh(pos);
}
//-----------------------------------------------------------------------------
__global__ void computeCellStartEndIndices 
(
    int* const dCellStartIndices,
    int* const dCellEndIndices, 
    const int* const dParticleHashs, 
    unsigned int numParticles
)
{
    extern __shared__ int sharedHash[];
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles) 
    {
        return;
    }

    int hash = dParticleHashs[idx];
    sharedHash[threadIdx.x + 1] = hash;
        
    if (idx > 0 && threadIdx.x == 0) 
    {
        sharedHash[0] = dParticleHashs[idx - 1];
    }

    __syncthreads();

    if (idx == 0 || hash != sharedHash[threadIdx.x])
    {
        dCellStartIndices[hash] = idx;
        
        if (idx > 0) 
        {
            dCellEndIndices[sharedHash[threadIdx.x]] = idx;
        }
    }

    if (idx == numParticles - 1)
    {
        dCellEndIndices[hash] = idx + 1;
    }
}
//-----------------------------------------------------------------------------
__global__ void computeParticleDensityPressure 
(
    float* const dParticleDensities, 
    float* const dParticlePressures,
    const int* const dParticleIDs, 
    const int* const dCellStartIndices,
    const int* const dCellEndIndices, 
    const float* const dParticlePositions,
    const unsigned char* const dStates,
    const int* const dParticleIDsHigh, 
    const int* const dCellStartIndicesHigh,
    const int* const dCellEndIndicesHigh, 
    const float* const dParticlePositionsHigh,
    const unsigned char* const dStatesHigh,
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    // get id and state of the particle
    unsigned int id = dParticleIDs[idx];    
    unsigned char state = dStates[id];

    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];

    int2 cs = computeGridCoordinate(pos, -gdEffectiveRadius);
    int2 ce = computeGridCoordinate(pos, gdEffectiveRadius);
    float density = 0.0f;

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHash(i, j);
            int start = dCellStartIndices[hash];
            int end = dCellEndIndices[hash];
            updateDensityCell(density, pos, dParticlePositions, 
                 dParticleIDs, start, end);
        }
    }


    // add contribution of the high res field
    cs = computeGridCoordinateHigh(pos, -gdEffectiveRadius);
    ce = computeGridCoordinateHigh(pos, gdEffectiveRadius);
    float densityH = 0.0f;

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHashHigh(i, j);
            int start = dCellStartIndicesHigh[hash];
            int end = dCellEndIndicesHigh[hash];
            updateDensityCellComplement
            (
                densityH, 
                pos, 
                dParticlePositionsHigh,
                dStatesHigh, 
                dParticleIDsHigh, 
                start, 
                end,
                state
            );
        }
    }

    density *= gdFluidParticleMass;
    densityH *= gdFluidParticleMassHigh;

    dParticleDensities[id] = (density + densityH);
    float a = (density + densityH)/gdRestDensity;
    float a3 = a*a*a;
    dParticlePressures[id] = gdTaitCoefficient*(a3*a3*a - 1.0f);
}
//-----------------------------------------------------------------------------
__global__ void computeParticleDensityPressureHigh
(
    float* const dParticleDensities, 
    float* const dParticlePressures,
    const int* const dParticleIDs, 
    const int* const dCellStartIndices,
    const int* const dCellEndIndices, 
    const float* const dParticlePositions,
    const unsigned char* const dStates,
    const int* const dParticleIDsLow, 
    const int* const dCellStartIndicesLow,
    const int* const dCellEndIndicesLow, 
    const float* const dParticlePositionsLow,
    const unsigned char* const dStatesLow,
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    unsigned int id = dParticleIDs[idx]; 
    unsigned char state = dStates[id];  
    
    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];

    int2 cs = computeGridCoordinateHigh(pos, -gdEffectiveRadiusHigh);
    int2 ce = computeGridCoordinateHigh(pos, gdEffectiveRadiusHigh);
    float density = 0.0f;

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHashHigh(i, j);
            int start = dCellStartIndices[hash];
            int end = dCellEndIndices[hash];
            updateDensityCellHigh
            (
                density, 
                pos, 
                dParticlePositions, 
                dStates,
                dParticleIDs, 
                start, 
                end,
                state
            );
        }
    }


    cs = computeGridCoordinate(pos, -gdEffectiveRadius);
    ce = computeGridCoordinate(pos, gdEffectiveRadius);
    float densityL = 0.0f;

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHash(i, j);
            int start = dCellStartIndicesLow[hash];
            int end = dCellEndIndicesLow[hash];
            updateDensityCellComplementHigh
            (
                density, 
                pos, 
                dParticlePositions, 
                dStates,
                dParticleIDs, 
                start, 
                end,
                state
            );
        }
    }

    density *= gdFluidParticleMassHigh;
    densityL *= gdFluidParticleMass;
    density += densityL;

    dParticleDensities[id] = density;
    float a = density/gdRestDensity;
    float a3 = a*a*a;
    dParticlePressures[id] = gdTaitCoefficient*(a3*a3*a - 1.0f);
}
//-----------------------------------------------------------------------------
__global__ void computeParticleDensityPressureHighRelax
(
    float* const dParticleDensities, 
    float* const dParticlePressures,
    const int* const dTransientIDs,
    const int* const dParticleIDs, 
    const int* const dCellStartIndices,
    const int* const dCellEndIndices, 
    const float* const dParticlePositions,
    const int* const dParticleIDsLow, 
    const int* const dCellStartIndicesLow,
    const int* const dCellEndIndicesLow, 
    const float* const dParticlePositionsLow,
    unsigned int numParticles
)
{
    //unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    //if (idx >= numParticles)
    //{
    //    return;
    //}

    //unsigned int id = dTransientIDs[idx];    
    //
    //float2 pos;
    //pos.x = dParticlePositions[2*id + 0];
    //pos.y = dParticlePositions[2*id + 1];

    //int2 cs = computeGridCoordinateHigh(pos, -gdEffectiveRadiusHigh);
    //int2 ce = computeGridCoordinateHigh(pos, gdEffectiveRadiusHigh);
    //float density = 0.0f;

    //for (int j = cs.y; j <= ce.y; j++)
    //{
    //    for (int i = cs.x; i <= ce.x; i++)
    //    {
    //        int hash = computeHashHigh(i, j);
    //        int start = dCellStartIndices[hash];
    //        int end = dCellEndIndices[hash];
    //        updateDensityCellHigh
    //        (
    //            density, 
    //            pos, 
    //            dParticlePositions, 
    //            dParticleIDs, 
    //            start, 
    //            end
    //        );
    //    }
    //}


    //cs = computeGridCoordinate(pos, -gdEffectiveRadiusHigh);
    //ce = computeGridCoordinate(pos, gdEffectiveRadiusHigh);
    //float densityL = 0.0f;

    //for (int j = cs.y; j <= ce.y; j++)
    //{
    //    for (int i = cs.x; i <= ce.x; i++)
    //    {
    //        int hash = computeHash(i, j);
    //        int start = dCellStartIndicesLow[hash];
    //        int end = dCellEndIndicesLow[hash];
    //        updateDensityCellHigh
    //        (
    //            densityL, 
    //            pos, 
    //            dParticlePositionsLow, 
    //            dParticleIDsLow, 
    //            start, 
    //            end
    //        );
    //    }
    //}

    //density *= gdFluidParticleMassHigh;
    //densityL *= gdFluidParticleMass;
    //density += densityL;

    //dParticleDensities[id] = density;
    //float a = density/gdRestDensity;
    //float a3 = a*a*a;
    //dParticlePressures[id] = gdTaitCoefficient*(a3*a3*a - 1.0f);
}
//-----------------------------------------------------------------------------
__global__ void advance 
(
    float* const dPositions, 
    float* const dVelocities, 
    float* const dAccelerations, 
    const int* const dParticleIDs,
    float dt, 
    unsigned int numParticles
)
{
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    unsigned int id = dParticleIDs[idx];  

    float2 acc;
    acc.x = dAccelerations[2*id + 0];
    acc.y = dAccelerations[2*id + 1];
    
    float2 pos;
    pos.x = dPositions[2*id + 0];
    pos.y = dPositions[2*id + 1];
    
    float2 vel;
    vel.x = dVelocities[2*id + 0]; 
    vel.y = dVelocities[2*id + 1]; 

    vel.x += dt*acc.x; 
    vel.y += dt*acc.y;

    pos.x += dt*vel.x;
    pos.y += dt*vel.y;
        
    dVelocities[2*id + 0] = vel.x;
    dVelocities[2*id + 1] = vel.y;
    
    dPositions[2*id + 0] = pos.x;
    dPositions[2*id + 1] = pos.y;
}
//-----------------------------------------------------------------------------
__global__ void computeParticleAcceleration 
(
    float* const dAccelerations,
    float* const dVisualQuantities,
    const float* const dParticlePositions, 
    const float* const dParticleVelocities,
    unsigned char* const dStates,
    unsigned char* const dStatesHigh,
    int* const dParticleIDsNew,
    int* const dParticleIDsHighNew,
    int* const dGLIndices,
    int* const dGLIndicesHigh,
    int* const dTransientIDsLow,          
    int* const dParticleCount,
    int* const dParticleCountHigh,
    int* const dTransientParticleCountLow,
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    float* const dParticlePositionsHigh, 
    float* const dParticleVelocitiesHigh,
    const float* const dParticleDensitiesHigh, 
    const float* const dParticlePressuresHigh,
    const int* const dParticleIDs, 
    const int* const dCellStartIndices,
    const int* const dCellEndIndices, 
    const int* const dParticleIDsHigh, 
    const int* const dCellStartIndicesHigh,
    const int* const dCellEndIndicesHigh, 
    float dt, 
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    // get id and state of the particle
    unsigned int id = dParticleIDs[idx];    
    unsigned char state = dStates[id];

    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];
    float2 vel;
    vel.x = dParticleVelocities[2*id + 0]; 
    vel.y = dParticleVelocities[2*id + 1]; 
    float density = dParticleDensities[id];
    float pressure = dParticlePressures[id];
    
    float2 acc;
    acc.x = 0.0f;
    acc.y = 0.0f;

    float2 accH;
    accH.x = 0.0f;
    accH.y = 0.0f;

    float2 accT;
    accT.x = 0.0f;
    accT.y = 0.0f;

    float2 accB;
    accB.x = 0.0f;
    accB.y = 0.0f;

    float2 ene;
    ene.x = 0.0f;
    ene.y = 0.0f;
    float psiSum = 0.0f;

    int2 cs = computeGridCoordinate(pos, -gdEffectiveRadius);
    int2 ce = computeGridCoordinate(pos, gdEffectiveRadius);

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHash(i, j);
            int start = dCellStartIndices[hash];
            int end = dCellEndIndices[hash];
            int startB = tex1Dfetch(gdBoundaryCellStartIndicesTex, hash);
            int endB = tex1Dfetch(gdBoundaryCellEndIndicesTex, hash);
            updateAccCell
            (
                acc, 
                accT, 
                accB, 
                ene,
                psiSum,
                pos, 
                vel, 
                density, 
                pressure, 
                dParticlePositions, 
                dParticleDensities, 
                dParticlePressures, 
                dParticleVelocities, 
                dStates,
                dParticleIDs,
                start, 
                end, 
                startB, 
                endB
            );
        }
    }

    cs = computeGridCoordinateHigh(pos, -gdEffectiveRadius);
    ce = computeGridCoordinateHigh(pos, gdEffectiveRadius);

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHashHigh(i, j);
            int start = dCellStartIndicesHigh[hash];
            int end = dCellEndIndicesHigh[hash];
            updateAccCellComplement
            (
                accH, 
                accT, 
                accB, 
                pos, 
                vel, 
                density, 
                pressure, 
                dParticlePositionsHigh, 
                dParticleDensitiesHigh, 
                dParticlePressuresHigh, 
                dParticleVelocitiesHigh, 
                dStatesHigh,
                dParticleIDsHigh, 
                start, 
                end, 
                state
            );
        }
    }

    acc.x *= -gdFluidParticleMass;
    acc.y *= -gdFluidParticleMass;

    accH.x *= -gdFluidParticleMassHigh;
    accH.y *= -gdFluidParticleMassHigh;

    acc.x += accH.x;
    acc.y += accH.y;

    acc.x -= gdTensionCoefficient*accT.x;
    acc.y -= gdTensionCoefficient*accT.y;

    acc.x += accB.x;
    acc.y += accB.y;
    
    acc.y -= 9.81f;

    dAccelerations[2*id + 0] = acc.x;
    dAccelerations[2*id + 1] = acc.y;

    vel.x += dt*acc.x; 
    vel.y += dt*acc.y;

    pos.x += dt*vel.x;
    pos.y += dt*vel.y;

    dVisualQuantities[id] = min(1.0f/(psiSum*psiSum*gdEffectiveRadius)*
        (ene.x*ene.x + ene.y*ene.y), 800.0f)/800.0f;
    
    // if particle is marked for deletion don't add it to the list
    if (state == 0x0001)
    {
        return;
    }
    
    // add particle to id list (Low)
    int index = atomicAdd(dParticleCount, 1);
    dParticleIDsNew[index] = id;
    
    //if (pos.x < 0.5f && state == 0x0000)
    {
        dGLIndices[index] = id;
    }

    // if particle is not transient and split condition is true ...
    if (pos.x > 0.5f && state == 0x0000)
    {        
        // mark as transient
        dStates[id] = 0x0001;

        // add particle id to transient list (low)
        int numTransient = atomicAdd(dTransientParticleCountLow, 1);
        dTransientIDsLow[numTransient] = id;

        // create highres particles and add to id list (high)
        int numParticlesHigh = atomicAdd(dParticleCountHigh, 4);


        float radius = 0.5f*sqrt(gdFluidParticleMass/(2.0f*density*PI));

        initSubParticlesAndAddToList
        (
            dParticlePositionsHigh,
            dParticleVelocitiesHigh,  
            dStatesHigh,           
            dParticleIDsHighNew,
            dGLIndicesHigh, 
            numParticlesHigh,  
            id, 
            pos,
            vel,
            radius
        );
    }
    
}
//-----------------------------------------------------------------------------
__global__ void computeParticleAccelerationAndAdvanceHigh
(
    float* const dParticlePositions, 
    float* const dParticleVelocities,
    float* const dAccelerations,
    int* const dParticleIDsNew,
    int* const dGLIndices,
    int* const dParticleCount,
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const unsigned char* const dStates,
    const float* const dParticlePositionsLow, 
    const float* const dParticleVelocitiesLow,
    const float* const dParticleDensitiesLow, 
    const float* const dParticlePressuresLow,
    const unsigned char* const dStatesLow,
    const int* const dParticleIDs, 
    const int* const dCellStartIndices,
    const int* const dCellEndIndices, 
    const int* const dParticleIDsLow, 
    const int* const dCellStartIndicesLow,
    const int* const dCellEndIndicesLow, 
    float dt, 
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    unsigned int id = dParticleIDs[idx];  
    unsigned char state = dStates[id];  

    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];
    float2 vel;
    vel.x = dParticleVelocities[2*id + 0]; 
    vel.y = dParticleVelocities[2*id + 1]; 
    float density = dParticleDensities[id];
    float pressure = dParticlePressures[id];

    int2 cs = computeGridCoordinateHigh(pos, -gdEffectiveRadiusHigh);
    int2 ce = computeGridCoordinateHigh(pos, gdEffectiveRadiusHigh);
    
    float2 acc;
    acc.x = 0.0f;
    acc.y = 0.0f;
    
    float2 accL;
    acc.x = 0.0f;
    acc.y = 0.0f;

    float2 accT;
    accT.x = 0.0f;
    accT.y = 0.0f;

    float2 accB;
    accB.x = 0.0f;
    accB.y = 0.0f;

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHashHigh(i, j);
            int start = dCellStartIndices[hash];
            int end = dCellEndIndices[hash];
            int startB = tex1Dfetch(gdBoundaryCellStartIndicesTex, hash);
            int endB = tex1Dfetch(gdBoundaryCellEndIndicesTex, hash);
            updateAccCellHigh
            (
                acc, 
                accT, 
                pos, 
                vel, 
                density, 
                pressure, 
                dParticlePositions, 
                dParticleDensities, 
                dParticlePressures, 
                dParticleVelocities, 
                dStates,
                dParticleIDs, 
                start, 
                end,
                state
            );
        }
    }

    cs = computeGridCoordinate(pos, -gdEffectiveRadius);
    ce = computeGridCoordinate(pos, gdEffectiveRadius);

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHash(i, j);
            int start = dCellStartIndicesLow[hash];
            int end = dCellEndIndicesLow[hash];
            int startB = tex1Dfetch(gdBoundaryCellStartIndicesTex, hash);
            int endB = tex1Dfetch(gdBoundaryCellEndIndicesTex, hash);
            updateAccCellComplementHigh
            (
                accL, 
                accT, 
                accB, 
                pos, 
                vel, 
                density, 
                pressure, 
                dParticlePositionsLow, 
                dParticleDensitiesLow, 
                dParticlePressuresLow, 
                dParticleVelocitiesLow, 
                dStatesLow,
                dParticleIDsLow, 
                start,
                end, 
                startB, 
                endB,
                state
            );
        }
    }

    acc.x *= -gdFluidParticleMassHigh;
    acc.y *= -gdFluidParticleMassHigh;

    accL.x *= -gdFluidParticleMass;
    accL.y *= -gdFluidParticleMass;

    acc.x -= gdTensionCoefficient*accT.x;
    acc.y -= gdTensionCoefficient*accT.y;

    acc.x += accL.x;
    acc.y += accL.y;

    acc.x += accB.x;
    acc.y += accB.y;
    
    acc.y -= 9.81f;

    vel.x += dt*acc.x; 
    vel.y += dt*acc.y;

    pos.x += dt*vel.x;
    pos.y += dt*vel.y;

    dAccelerations[2*id + 0] = acc.x;
    dAccelerations[2*id + 1] = acc.y;

    dParticleVelocities[2*id + 0] = vel.x;
    dParticleVelocities[2*id + 1] = vel.y;
    
    dParticlePositions[2*id + 0] = pos.x;
    dParticlePositions[2*id + 1] = pos.y;


    int index = atomicAdd(dParticleCount, 1);
    dParticleIDsNew[index] = id;
    dGLIndices[index] = id;
}
//-----------------------------------------------------------------------------
__global__ void computeParticleAccelerationAndAdvanceHighRelax
(
    float* const dParticlePositions, 
    float* const dParticleVelocities,
    const float* const dParticleDensities, 
    const float* const dParticlePressures, 
    const  float* const dParticlePositionsLow, 
    const  float* const dParticleVelocitiesLow,
    const float* const dParticleDensitiesLow, 
    const float* const dParticlePressuresLow,
    const int* const dTransientIDs, 
    const int* const dParticleIDs, 
    const int* const dCellStartIndices,
    const int* const dCellEndIndices, 
    const int* const dParticleIDsLow, 
    const int* const dCellStartIndicesLow,
    const int* const dCellEndIndicesLow, 
    float dt, 
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    unsigned int id = dTransientIDs[idx];    

    float2 pos;
    pos.x = dParticlePositions[2*id + 0];
    pos.y = dParticlePositions[2*id + 1];
    float2 vel;
    vel.x = dParticleVelocities[2*id + 0]; 
    vel.y = dParticleVelocities[2*id + 1]; 
    float density = dParticleDensities[id];
    float pressure = dParticlePressures[id];

    int2 cs = computeGridCoordinateHigh(pos, -gdEffectiveRadiusHigh);
    int2 ce = computeGridCoordinateHigh(pos, gdEffectiveRadiusHigh);
    
    float2 acc;
    acc.x = 0.0f;
    acc.y = 0.0f;
    
    float2 accL;
    acc.x = 0.0f;
    acc.y = 0.0f;

    float2 accT;
    accT.x = 0.0f;
    accT.y = 0.0f;

    float2 accB;
    accB.x = 0.0f;
    accB.y = 0.0f;

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHashHigh(i, j);
            int start = dCellStartIndices[hash];
            int end = dCellEndIndices[hash];
            int startB = tex1Dfetch(gdBoundaryCellStartIndicesTex, hash);
            int endB = tex1Dfetch(gdBoundaryCellEndIndicesTex, hash);
            updateAccCellHighRelax(acc, accT, pos, vel, density, pressure, 
                dParticlePositions, dParticleDensities, dParticlePressures, 
                dParticleVelocities, dParticleIDs, start, end);
        }
    }

    cs = computeGridCoordinate(pos, -gdEffectiveRadiusHigh);
    ce = computeGridCoordinate(pos, gdEffectiveRadiusHigh);

    for (int j = cs.y; j <= ce.y; j++)
    {
        for (int i = cs.x; i <= ce.x; i++)
        {
            int hash = computeHash(i, j);
            int start = dCellStartIndicesLow[hash];
            int end = dCellEndIndicesLow[hash];
            int startB = tex1Dfetch(gdBoundaryCellStartIndicesTex, hash);
            int endB = tex1Dfetch(gdBoundaryCellEndIndicesTex, hash);
            updateAccCellComplementHighRelax
            (
                accL, accT, accB, pos, vel, density, pressure, 
                dParticlePositionsLow, dParticleDensitiesLow, 
                dParticlePressuresLow, dParticleVelocitiesLow, 
                dParticleIDsLow, start, end, startB, endB
            );
        }
    }

    acc.x *= -gdFluidParticleMassHigh;
    acc.y *= -gdFluidParticleMassHigh;

    accL.x *= -gdFluidParticleMass;
    accL.y *= -gdFluidParticleMass;

    //acc.x -= gdTensionCoefficient*accT.x;
    //acc.y -= gdTensionCoefficient*accT.y;

    acc.x += accL.x;
    acc.y += accL.y;

    acc.x += accB.x;
    acc.y += accB.y;
    
    //acc.y -= 9.81f;

    vel.x += dt*acc.x; 
    vel.y += dt*acc.y;

    pos.x += dt*vel.x;
    pos.y += dt*vel.y;

    dParticleVelocities[2*id + 0] = vel.x;
    dParticleVelocities[2*id + 1] = vel.y;
    
    dParticlePositions[2*id + 0] = pos.x;
    dParticlePositions[2*id + 1] = pos.y;
}
//-----------------------------------------------------------------------------
//__global__ void computeParticleAccelerationAndAdvanceHighRelax
//(
//    float* const dParticlePositions, 
//    float* const dParticleVelocities,
//    const float* const dParticleDensities, 
//    const float* const dParticlePressures, 
//    const  float* const dParticlePositionsLow, 
//    const  float* const dParticleVelocitiesLow,
//    const float* const dParticleDensitiesLow, 
//    const float* const dParticlePressuresLow,
//    const int* const dTransientIDs, 
//    const int* const dParticleIDs, 
//    const int* const dCellStartIndices,
//    const int* const dCellEndIndices, 
//    const int* const dParticleIDsLow, 
//    const int* const dCellStartIndicesLow,
//    const int* const dCellEndIndicesLow, 
//    float dt, 
//    unsigned int numParticles
//)
//{
//    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
//
//    if (idx >= numParticles)
//    {
//        return;
//    }
//
//    unsigned int id = dTransientIDs[idx];    
//
//    float2 pos;
//    pos.x = dParticlePositions[2*id + 0];
//    pos.y = dParticlePositions[2*id + 1];
//    float2 vel;
//    vel.x = dParticleVelocities[2*id + 0]; 
//    vel.y = dParticleVelocities[2*id + 1]; 
//    float density = dParticleDensities[id];
//    float pressure = dParticlePressures[id];
//
//    int2 cs = computeGridCoordinateHigh(pos, -gdEffectiveRadiusHigh);
//    int2 ce = computeGridCoordinateHigh(pos, gdEffectiveRadiusHigh);
//    
//    float2 acc;
//    acc.x = 0.0f;
//    acc.y = 0.0f;
//    
//    float2 accL;
//    acc.x = 0.0f;
//    acc.y = 0.0f;
//
//    float2 accT;
//    accT.x = 0.0f;
//    accT.y = 0.0f;
//
//    float2 accB;
//    accB.x = 0.0f;
//    accB.y = 0.0f;
//
//    for (int j = cs.y; j <= ce.y; j++)
//    {
//        for (int i = cs.x; i <= ce.x; i++)
//        {
//            int hash = computeHashHigh(i, j);
//            int start = dCellStartIndices[hash];
//            int end = dCellEndIndices[hash];
//            int startB = tex1Dfetch(gdBoundaryCellStartIndicesTex, hash);
//            int endB = tex1Dfetch(gdBoundaryCellEndIndicesTex, hash);
//            updateAccCellHigh(acc, accT, pos, vel, density, pressure, 
//                dParticlePositions, dParticleDensities, dParticlePressures, 
//                dParticleVelocities, dParticleIDs, start, end);
//        }
//    }
//
//    cs = computeGridCoordinate(pos, -gdEffectiveRadiusHigh);
//    ce = computeGridCoordinate(pos, gdEffectiveRadiusHigh);
//
//    for (int j = cs.y; j <= ce.y; j++)
//    {
//        for (int i = cs.x; i <= ce.x; i++)
//        {
//            int hash = computeHash(i, j);
//            int start = dCellStartIndicesLow[hash];
//            int end = dCellEndIndicesLow[hash];
//            int startB = tex1Dfetch(gdBoundaryCellStartIndicesTex, hash);
//            int endB = tex1Dfetch(gdBoundaryCellEndIndicesTex, hash);
//            updateAccCellComplementHigh
//            (
//                accL, accT, accB, pos, vel, density, pressure, 
//                dParticlePositionsLow, dParticleDensitiesLow, 
//                dParticlePressuresLow, dParticleVelocitiesLow, dParticleIDsLow, 
//                start, end, startB, endB
//            );
//        }
//    }
//
//    acc.x *= -gdFluidParticleMassHigh;
//    acc.y *= -gdFluidParticleMassHigh;
//
//    accL.x *= -gdFluidParticleMass;
//    accL.y *= -gdFluidParticleMass;
//
//    acc.x -= gdTensionCoefficient*accT.x;
//    acc.y -= gdTensionCoefficient*accT.y;
//
//    acc.x += accL.x;
//    acc.y += accL.y;
//
//    acc.x += accB.x;
//    acc.y += accB.y;
//    
//    acc.y -= 9.81f;
//
//    vel.x += dt*acc.x; 
//    vel.y += dt*acc.y;
//
//    pos.x += dt*vel.x;
//    pos.y += dt*vel.y;
//
//    dParticleVelocities[2*id + 0] = vel.x;
//    dParticleVelocities[2*id + 1] = vel.y;
//    
//    dParticlePositions[2*id + 0] = pos.x;
//    dParticlePositions[2*id + 1] = pos.y;
//}
//-----------------------------------------------------------------------------
//  Definiton of WCSPHConfig
//-----------------------------------------------------------------------------
WCSPHConfig::WCSPHConfig 
(
    float xs, 
    float ys, 
    float xe,
    float ye, 
    float effectiveRadius, 
    float effectiveRadiusHigh,
    float restDensity, 
    float taitCoeff, 
    float speedSound, 
    float alpha, 
    float tensionCoefficient,
    float timeStep
)
: 
    EffectiveRadius(effectiveRadius), 
    EffectiveRadiusHigh(effectiveRadiusHigh),
    RestDensity(restDensity), 
    TaitCoeffitient(taitCoeff), 
    SpeedSound(speedSound),
    Alpha(alpha), 
    TensionCoefficient(tensionCoefficient),
    TimeStep(timeStep)
{
    if (xs >= xe || ys >= ye)
    {
        UTIL::ThrowException("Invalid configuration parameters", __FILE__, 
            __LINE__);
    }

    DomainOrigin[0] = xs;
    DomainOrigin[1] = ys;
    DomainEnd[0] = xe;
    DomainEnd[1] = ye;

    DomainDimensions[0] = static_cast<int>(std::ceil((DomainEnd[0] - 
        DomainOrigin[0])/EffectiveRadius));
    DomainDimensions[1] = static_cast<int>(std::ceil((DomainEnd[1] - 
        DomainOrigin[1])/EffectiveRadius));

    DomainDimensionsHigh[0] = static_cast<int>(std::ceil((DomainEnd[0] - 
        DomainOrigin[0])/EffectiveRadiusHigh));
    DomainDimensionsHigh[1] = static_cast<int>(std::ceil((DomainEnd[1] - 
        DomainOrigin[1])/EffectiveRadiusHigh));
}
//-----------------------------------------------------------------------------
WCSPHConfig::~WCSPHConfig ()
{

}
//-----------------------------------------------------------------------------
// DEFINITION: WCSPHSolver
//-----------------------------------------------------------------------------
//  - Nested Class : Neighbour Grid 
//-----------------------------------------------------------------------------
WCSPHSolver::NeighborGrid::NeighborGrid (const int gridDimensions[2], 
    int maxParticles)
{
    // malloc device mem
    int sizeIds = maxParticles*sizeof(int);
    CUDA_SAFE_CALL( cudaMalloc(&dParticleHashs,  sizeIds) );

    int sizeCellLists = gridDimensions[0]*gridDimensions[1]*sizeof(int);
    CUDA_SAFE_CALL( cudaMalloc(&dCellStart, sizeCellLists) );
    CUDA_SAFE_CALL( cudaMalloc(&dCellEnd, sizeCellLists) );

    // alloc particle id lists
    CUDA_SAFE_CALL( cudaMalloc(&dParticleIDs[0], sizeIds) );
    CUDA_SAFE_CALL( cudaMalloc(&dParticleIDs[1], sizeIds) );

    // init particle id lists
    int* data = new int[maxParticles];
    for (unsigned int i = 0; i < maxParticles; i++)
    {
        data[i] = i;
    }
    CUDA_SAFE_CALL( cudaMemcpy
    (
        dParticleIDs[0],
        data, 
        sizeIds, 
        cudaMemcpyHostToDevice
    ) );
    CUDA_SAFE_CALL( cudaMemcpy
    (
        dParticleIDs[1],
        data, 
        sizeIds, 
        cudaMemcpyHostToDevice
    ) );
    delete[] data;

}
//-----------------------------------------------------------------------------
WCSPHSolver::NeighborGrid::~NeighborGrid ()
{
    CUDA::SafeFree<int>(&dParticleHashs);
    CUDA::SafeFree<int>(&dCellStart);
    CUDA::SafeFree<int>(&dCellEnd);
    CUDA::SafeFree<int>(&dParticleIDs[0]);
    CUDA::SafeFree<int>(&dParticleIDs[1]);
}
//-----------------------------------------------------------------------------
WCSPHSolver::WCSPHSolver 
(
    const WCSPHConfig& config, 
    ParticleSystem& fluidParticles, 
    ParticleSystem& fluidParticlesHigh, 
    ParticleSystem& boundaryParticles
)
: 
    mEffectiveRadius(config.EffectiveRadius), 
    mEffectiveRadiusHigh(config.EffectiveRadiusHigh),
    mRestDensity(config.RestDensity),
    mTaitCoeffitient(config.TaitCoeffitient),
    mSpeedSound(config.SpeedSound),
    mAlpha(config.Alpha), 
    mTensionCoeffient(config.TensionCoefficient), 
    mTimeStep(config.TimeStep),
    mFluidParticles(&fluidParticles),
    mFluidParticlesHigh(&fluidParticlesHigh),
    mBoundaryParticles(&boundaryParticles), 
    mIsBoundaryInit(false),
    mFluidParticleGrid
    (
        config.DomainDimensions, 
        fluidParticles.GetMaxParticles()
    ),
    mFluidParticleGridHigh
    (
        config.DomainDimensionsHigh, 
        fluidParticlesHigh.GetMaxParticles()
    ),
    mBlockDim(256, 1 , 1),
    mActive(0),
    mTransientParticleCountLow(0),
    mTransientParticleCountHigh(0)
{
    mDomainOrigin[0] = config.DomainOrigin[0];
    mDomainOrigin[1] = config.DomainOrigin[1];
    mDomainEnd[0] = config.DomainEnd[0];
    mDomainEnd[1] = config.DomainEnd[1];

    mDomainDimensions[0] = config.DomainDimensions[0];
    mDomainDimensions[1] = config.DomainDimensions[1];
    mDomainDimensionsHigh[0] = config.DomainDimensionsHigh[0];
    mDomainDimensionsHigh[1] = config.DomainDimensionsHigh[1];

    // set cuda block and grid dimensions
    mBlockDim.x = 256; 
    mBlockDim.y = 1; 
    mBlockDim.z = 1; 
    unsigned int numParticles = mFluidParticles->GetNumParticles();
    unsigned int numBlocks = numParticles/mBlockDim.x;
    mGridDim.x = numParticles % mBlockDim.x == 0 ? numBlocks : numBlocks + 1;
    mGridDim.y = 1;
    mGridDim.z = 1;
    CUDA_SAFE_CALL( cudaMalloc(&mdParticleCount, sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemset(mdParticleCount, 0, sizeof(int)) );

    // set cuda block and grid dimensions (high)
    unsigned int numParticlesHigh = mFluidParticlesHigh->GetNumParticles();
    unsigned int numBlocksHigh = numParticlesHigh/mBlockDim.x;
    mGridDimHigh.x = numParticlesHigh % mBlockDim.x == 0 ? numBlocksHigh : 
        numBlocksHigh + 1;
    mGridDimHigh.y = 1;
    mGridDimHigh.z = 1;
    CUDA_SAFE_CALL( cudaMalloc(&mdParticleCountHigh, sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemset(mdParticleCountHigh, 0, sizeof(int)) );

    // allocate extra device memory for neighbor search (boundaries)
    unsigned int size = sizeof(float)*mBoundaryParticles->GetNumParticles();
    CUDA_SAFE_CALL( cudaMalloc(&mdBoundaryParticleHashs, size) );
    CUDA_SAFE_CALL( cudaMalloc(&mdBoundaryParticleIDs, size) );
    unsigned int domainSize = mDomainDimensions[0]*mDomainDimensions[1]*
        sizeof(int);
    CUDA_SAFE_CALL( cudaMalloc(&mdBoundaryCellStartIndices, domainSize) );
    CUDA_SAFE_CALL( cudaMalloc(&mdBoundaryCellEndIndices, domainSize) );

    // allocate memory for transient particle (High)
    CUDA_SAFE_CALL( cudaMalloc(&mdTransientParticleCountHigh, sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemset(mdTransientParticleCountHigh, 0, sizeof(int)) );
    CUDA_SAFE_CALL( cudaMalloc
    (
        &mdTransientIDsHigh, 
        sizeof(int)*mFluidParticlesHigh->mMaxParticles
    ) );



    // allocate memory for transient particle (Low)
    CUDA_SAFE_CALL( cudaMalloc(&mdTransientParticleCountLow, sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemset(mdTransientParticleCountLow, 0, sizeof(int)) );
    CUDA_SAFE_CALL( cudaMalloc
    (
        &mdTransientIDsLow[0], 
        sizeof(int)*mFluidParticles->mMaxParticles
    ) );
    CUDA_SAFE_CALL( cudaMalloc
    (
        &mdTransientIDsLow[1], 
        sizeof(int)*mFluidParticles->mMaxParticles
    ) );


    // alloc and init particle states (high)
    CUDA_SAFE_CALL( cudaMalloc
    (
        &mdStatesHigh,
        sizeof(unsigned char)*mFluidParticlesHigh->mMaxParticles
    ) );
    CUDA_SAFE_CALL( cudaMemset
    (
        mdStatesHigh,
        0,
        sizeof(unsigned char)*mFluidParticlesHigh->mMaxParticles
    ) );


    // alloc and init particle states (low)
    CUDA_SAFE_CALL( cudaMalloc
    (
        &mdStates,
        sizeof(unsigned char)*mFluidParticles->mMaxParticles
    ) );
    CUDA_SAFE_CALL( cudaMemset
    (
        mdStates,
        0,
        sizeof(unsigned char)*mFluidParticles->mMaxParticles
    ) );

}
//-----------------------------------------------------------------------------
WCSPHSolver::~WCSPHSolver ()
{
    CUDA::SafeFree<int>(&mdBoundaryParticleHashs);
    CUDA::SafeFree<int>(&mdBoundaryParticleIDs);
    CUDA::SafeFree<int>(&mdBoundaryCellStartIndices);
    CUDA::SafeFree<int>(&mdBoundaryCellEndIndices);
    CUDA::SafeFree<int>(&mdTransientIDsHigh);
    CUDA::SafeFree<int>(&mdTransientParticleCountHigh);
    CUDA::SafeFree<int>(&mdTransientIDsLow[0]);
    CUDA::SafeFree<int>(&mdTransientIDsLow[1]);
    CUDA::SafeFree<int>(&mdTransientParticleCountLow);
    CUDA::SafeFree<unsigned char>(&mdStates);
    CUDA::SafeFree<unsigned char>(&mdStatesHigh);
}
//-----------------------------------------------------------------------------
void WCSPHSolver::Bind () const
{
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdDomainOrigin, mDomainOrigin, 
        2*sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdDomainDimensions, mDomainDimensions, 
        2*sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdDomainDimensionsHigh, mDomainDimensionsHigh, 
        2*sizeof(int)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdEffectiveRadius, &mEffectiveRadius,
        sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdEffectiveRadiusHigh, &mEffectiveRadiusHigh,
        sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdRestDensity, &mRestDensity,
        sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdTaitCoefficient, &mTaitCoeffitient,
        sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdSpeedSound, &mSpeedSound,
        sizeof(float)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdAlpha, &mAlpha, sizeof(float)) );    
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(gdTensionCoefficient, &mTensionCoeffient,
        sizeof(float)) );

    float m4KernelCoeff = 20.0f/(14.0f*PI*mEffectiveRadius*mEffectiveRadius);
    float m4GradKernelCoeff = -120.0f/(14.0f*PI*mEffectiveRadius*mEffectiveRadius*
        mEffectiveRadius);
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (
        gdM4KernelCoeff, 
        &m4KernelCoeff,
        sizeof(float)) 
    );
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (   
        gdM4KernelGradCoeff, 
        &m4GradKernelCoeff,
        sizeof(float)
    ) );
    float m4KernelCoeffHigh = 
        20.0f/(14.0f*PI*mEffectiveRadiusHigh*mEffectiveRadiusHigh);
    float m4GradKernelCoeffHigh = 
        -120.0f/(14.0f*PI*mEffectiveRadiusHigh*mEffectiveRadiusHigh*
        mEffectiveRadiusHigh);
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (
        gdM4KernelCoeffHigh, 
        &m4KernelCoeffHigh,
        sizeof(float)
    ) );
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (
        gdM4KernelGradCoeffHigh, 
        &m4GradKernelCoeffHigh,
        sizeof(float)
    ) );
    float mass = mFluidParticles->GetMass();
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (
        gdFluidParticleMass, 
        &mass, 
        sizeof(float)
    ) );
    mass = mFluidParticlesHigh->GetMass();
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (
        gdFluidParticleMassHigh, 
        &mass, 
        sizeof(float)
    ) );
    mass = mBoundaryParticles->GetMass();
    CUDA_SAFE_CALL ( cudaMemcpyToSymbol
    (
        gdBoundaryParticleMass, 
        &mass, 
        sizeof(float)
    ) );

    // init boundary handling
    if (!mIsBoundaryInit)
    {
        initBoundaries();
        mIsBoundaryInit = true;
    }

    // bind boundary handling information to textures
    mBoundaryParticles->Map();
    cudaChannelFormatDesc descf = cudaCreateChannelDesc(32, 0, 0, 0,
		cudaChannelFormatKindFloat);
    cudaChannelFormatDesc desci = cudaCreateChannelDesc(32, 0, 0, 0,
		cudaChannelFormatKindSigned);
    CUDA_SAFE_CALL( cudaBindTexture(0, gdBoundaryPositionsTex, 
        mBoundaryParticles->Positions(), descf, 
        2*mBoundaryParticles->GetNumParticles()*sizeof(float)) );
    CUDA_SAFE_CALL( cudaBindTexture(0, gdBoundaryParticleIDsTex, 
        mdBoundaryParticleIDs, desci, 
        mBoundaryParticles->GetNumParticles()*sizeof(int)) );
    unsigned int domainSize = mDomainDimensions[0]*mDomainDimensions[1]*
        sizeof(int);
    CUDA_SAFE_CALL( cudaBindTexture(0, gdBoundaryCellStartIndicesTex, 
        mdBoundaryCellStartIndices, desci, domainSize) );
    CUDA_SAFE_CALL( cudaBindTexture(0, gdBoundaryCellEndIndicesTex, 
        mdBoundaryCellEndIndices, desci, domainSize) );
    mBoundaryParticles->Unmap();
}
//-----------------------------------------------------------------------------
void WCSPHSolver::Unbind () const
{
    CUDA_SAFE_CALL( cudaUnbindTexture(gdBoundaryPositionsTex) );
    CUDA_SAFE_CALL( cudaUnbindTexture(gdBoundaryParticleIDsTex) );
    CUDA_SAFE_CALL( cudaUnbindTexture(gdBoundaryCellStartIndicesTex) );
    CUDA_SAFE_CALL( cudaUnbindTexture(gdBoundaryCellEndIndicesTex) );
}
//-----------------------------------------------------------------------------
void WCSPHSolver::Advance ()
{
    //static unsigned char activeID = 0;

    //CUDA::Timer timer;

    //mFluidParticles->Map();
    //mFluidParticlesHigh->Map();
    //mBoundaryParticles->Map();

    //this->updateNeighborGrid(activeID);
    //
    //timer.Start();
    //computeParticleDensityPressure<<<mGridDim, mBlockDim>>>
    //(
    //    mFluidParticles->Densities(), mFluidParticles->Pressures(), 
    //    mFluidParticles->mdParticleIDs[activeID], 
    //    mFluidParticleGrid.dCellStart, 
    //    mFluidParticleGrid.dCellEnd, 
    //    mFluidParticles->Positions(), 
    //    mFluidParticlesHigh->mdParticleIDs[activeID], 
    //    mFluidParticleGridHigh.dCellStart, 
    //    mFluidParticleGridHigh.dCellEnd, 
    //    mFluidParticlesHigh->Positions(), 
    //    mFluidParticlesHigh->mNumParticles
    //); 
    //timer.Stop();
    //timer.DumpElapsed();   

    //this->updatePositions(activeID);

    //mBoundaryParticles->Unmap();
    //mFluidParticlesHigh->Unmap();
    //mFluidParticles->Unmap();

    //activeID = (activeID + 1) % 2; 
}
//-----------------------------------------------------------------------------
void WCSPHSolver::AdvanceHigh ()
{
    //static unsigned char activeID = 0;

    //CUDA::Timer timer;

    //mFluidParticles->Map();
    //mFluidParticlesHigh->Map();
    //mBoundaryParticles->Map();
    //
    //this->updateNeighborGridHigh(activeID);
    //
    //timer.Start();

    //computeParticleDensityPressureHigh<<<mGridDimHigh, mBlockDim>>>
    //(
    //    mFluidParticlesHigh->Densities(), 
    //    mFluidParticlesHigh->Pressures(), 
    //    mFluidParticlesHigh->mdParticleIDs[activeID], 
    //    mFluidParticleGridHigh.dCellStart, 
    //    mFluidParticleGridHigh.dCellEnd, 
    //    mFluidParticlesHigh->Positions(), 
    //    mFluidParticles->mdParticleIDs[0], 
    //    mFluidParticleGrid.dCellStart, 
    //    mFluidParticleGrid.dCellEnd, 
    //    mFluidParticles->Positions(), 
    //    mFluidParticlesHigh->mNumParticles
    //);    
    //timer.Stop();
    //
    //this->updatePositionsHigh(activeID);
   
    //mBoundaryParticles->Unmap();
    //mFluidParticlesHigh->Unmap();
    //mFluidParticles->Unmap();
    //
    //activeID = (activeID + 1) % 2; 
}
//-----------------------------------------------------------------------------
void WCSPHSolver::AdvanceTS ()
{
    static unsigned char activeID = 0;
   
    mFluidParticles->Map();
    mFluidParticlesHigh->Map();
    mBoundaryParticles->Map();

    this->updateNeighborGridHigh(activeID);
    CUDA::CheckLastError("1");
    this->updateNeighborGrid(activeID);
    CUDA::CheckLastError("2");
    //this->relaxTransient(activeID);
    this->computePressureDensityHigh(activeID);
    CUDA::CheckLastError("3");
    this->computePressureDensity(activeID);
    //this->injectTransientHigh(activeID);

    CUDA::CheckLastError("4");
    this->computeAccelerations(activeID);
    CUDA::CheckLastError("5");
    this->updatePositionsHigh(activeID);



    CUDA::CheckLastError("6");
    this->updatePositions(activeID);
    CUDA::CheckLastError("7");


    this->adjustOrInjectTransientHigh(activeID);

    // copy back the # of current particles in the list
    // update particle system information
    CUDA_SAFE_CALL( cudaMemcpy
    (
        &mFluidParticles->mNumParticles, 
        mdParticleCount, 
        sizeof(int), 
        cudaMemcpyDeviceToHost
    ) );

    CUDA_SAFE_CALL( cudaMemcpy
    (
        &mFluidParticlesHigh->mNumParticles, 
        mdParticleCountHigh, 
        sizeof(int), 
        cudaMemcpyDeviceToHost
    ) );  
    CUDA_SAFE_CALL( cudaMemcpy
    (
        &mTransientParticleCountLow, 
        mdTransientParticleCountLow, 
        sizeof(int),
        cudaMemcpyDeviceToHost
    ) );


    activeID = (activeID + 1) % 2; 
    mActive = activeID;
    
    mGridDim.x = std::ceil
    (
        static_cast<float>(mFluidParticles->mNumParticles)/mBlockDim.x
    );  

    mGridDimHigh.x = std::ceil
    (
        static_cast<float>(mFluidParticlesHigh->mNumParticles)/mBlockDim.x
    );

    mGridDimTransientLow.x = static_cast<unsigned int>(std::ceil
    (
        static_cast<float>(mTransientParticleCountLow)/mBlockDim.x
    ));

    
    this->setVisualQuantities(activeID);

    CUDA::CheckLastError("7");

    mFluidParticles->Unmap();
    mFluidParticlesHigh->Unmap();
    mBoundaryParticles->Unmap();
}
//-----------------------------------------------------------------------------
//  - private methods
//-----------------------------------------------------------------------------
void WCSPHSolver::updateNeighborGrid (unsigned char activeID)
{
    CUDA::Timer timer;
    
    // compute hash of active particles
    computeParticleHash<<<mGridDim, mBlockDim>>>
    (
        mFluidParticleGrid.dParticleHashs, 
        mFluidParticleGrid.dParticleIDs[activeID], 
        mFluidParticles->Positions(),
        mFluidParticles->mNumParticles
    );
    
    // sort active ids by hash
    thrust::sort_by_key
    (
        thrust::device_ptr<int>(mFluidParticleGrid.dParticleHashs),
        thrust::device_ptr<int>(mFluidParticleGrid.dParticleHashs +
            mFluidParticles->mNumParticles), 
        thrust::device_ptr<int>(mFluidParticleGrid.dParticleIDs[activeID])
    );

    // set all grid cells to be empty
    unsigned int size = mDomainDimensions[0]*mDomainDimensions[1]*
        sizeof(int);

    CUDA_SAFE_CALL(cudaMemset(mFluidParticleGrid.dCellStart, 
        EMPTY_CELL, size)); 
    CUDA_SAFE_CALL(cudaMemset(mFluidParticleGrid.dCellEnd, 
        EMPTY_CELL, size)); 

    // fill grid cells according to current particles
    int sharedMemSize = sizeof(int)*(mBlockDim.x + 1);
    computeCellStartEndIndices<<<mGridDim, mBlockDim, sharedMemSize>>>
    (
        mFluidParticleGrid.dCellStart, 
        mFluidParticleGrid.dCellEnd, 
        mFluidParticleGrid.dParticleHashs, 
        mFluidParticles->mNumParticles
    );
}
//-----------------------------------------------------------------------------
void WCSPHSolver::computePressureDensity (unsigned char activeID)
{
    computeParticleDensityPressure<<<mGridDim, mBlockDim>>>
    (
        mFluidParticles->Densities(), 
        mFluidParticles->Pressures(), 
        mFluidParticleGrid.dParticleIDs[activeID], 
        mFluidParticleGrid.dCellStart, 
        mFluidParticleGrid.dCellEnd, 
        mFluidParticles->Positions(), 
        mdStates,
        mFluidParticleGridHigh.dParticleIDs[activeID], 
        mFluidParticleGridHigh.dCellStart, 
        mFluidParticleGridHigh.dCellEnd, 
        mFluidParticlesHigh->Positions(),         
        mdStatesHigh,
        mFluidParticles->mNumParticles
    );
}
//-----------------------------------------------------------------------------
void WCSPHSolver::computeAccelerations (unsigned char activeID)
{
    CUDA::Timer timer;

    // reset particle count to zero
    CUDA_SAFE_CALL(cudaMemset(mdParticleCount, 0, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(mdTransientParticleCountLow, 0, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(mdParticleCountHigh, 0, sizeof(int)));
    
    //timer.Start();
    computeParticleAcceleration<<<mGridDim, mBlockDim>>>
    (
        mFluidParticles->mdAccelerations,
        mFluidParticles->mdVisualQuantities,
        mFluidParticles->Positions(), 
        mFluidParticles->Velocities(), 
        mdStates,
        mdStatesHigh,
        mFluidParticleGrid.dParticleIDs[(activeID + 1) % 2], 
        mFluidParticleGridHigh.dParticleIDs[(activeID + 1) % 2],
        mFluidParticles->mdParticleIDs,
        mFluidParticlesHigh->mdParticleIDs,
        mdTransientIDsLow[(activeID + 1) % 2], 
        mdParticleCount,
        mdParticleCountHigh,
        mdTransientParticleCountLow, 
        mFluidParticles->Densities(),
        mFluidParticles->Pressures(),
        mFluidParticlesHigh->mdPositions,
        mFluidParticlesHigh->mdVelocities,
        mFluidParticlesHigh->mdDensities,
        mFluidParticlesHigh->mdPressures,
        mFluidParticleGrid.dParticleIDs[activeID], 
        mFluidParticleGrid.dCellStart, 
        mFluidParticleGrid.dCellEnd, 
        mFluidParticleGridHigh.dParticleIDs[activeID], 
        mFluidParticleGridHigh.dCellStart, 
        mFluidParticleGridHigh.dCellEnd, 
        mTimeStep, 
        mFluidParticles->mNumParticles
    );


}
//-----------------------------------------------------------------------------
void WCSPHSolver::updatePositions (unsigned char activeID)
{
    advance<<<mGridDim, mBlockDim>>>
    (
        mFluidParticles->Positions(), 
        mFluidParticles->Velocities(),
        mFluidParticles->mdAccelerations,
        mFluidParticleGrid.dParticleIDs[activeID],
        mTimeStep,
        mFluidParticles->mNumParticles
    );
}
//-----------------------------------------------------------------------------
void WCSPHSolver::updateNeighborGridHigh (unsigned char activeID)
{
    CUDA::Timer timer;

    // set all grid cells to be empty
    unsigned int size = mDomainDimensionsHigh[0]*mDomainDimensionsHigh[1]*
        sizeof(unsigned int);

    CUDA_SAFE_CALL ( cudaMemset(mFluidParticleGridHigh.dCellStart, 
        EMPTY_CELL, size) ); 
    CUDA_SAFE_CALL ( cudaMemset(mFluidParticleGridHigh.dCellEnd, 
        EMPTY_CELL, size) ); 

    if (mFluidParticlesHigh->mNumParticles == 0)
    {
        return;
    }
        
    // compute hash of active particles
    computeParticleHashHigh<<<mGridDimHigh, mBlockDim>>>
    (
        mFluidParticleGridHigh.dParticleHashs, 
        mFluidParticleGridHigh.dParticleIDs[activeID], 
        mFluidParticlesHigh->Positions(),
        mFluidParticlesHigh->mNumParticles
    );
    
    // sort active ids by hash
    thrust::sort_by_key
    (
        thrust::device_ptr<int>(mFluidParticleGridHigh.dParticleHashs),
        thrust::device_ptr<int>(mFluidParticleGridHigh.dParticleHashs +
            mFluidParticlesHigh->mNumParticles), 
        thrust::device_ptr<int>(mFluidParticleGridHigh.dParticleIDs[activeID])
    );

    // fill grid cells according to current particles
    int sharedMemSize = sizeof(int)*(mBlockDim.x + 1);
    computeCellStartEndIndices<<<mGridDimHigh, mBlockDim, sharedMemSize>>>
    (
        mFluidParticleGridHigh.dCellStart, 
        mFluidParticleGridHigh.dCellEnd, 
        mFluidParticleGridHigh.dParticleHashs, 
        mFluidParticlesHigh->mNumParticles
    );
}
//-----------------------------------------------------------------------------
void WCSPHSolver::updatePositionsHigh (unsigned char activeID)
{
    if (mFluidParticlesHigh->mNumParticles == 0)
    {
        return;
    }

    CUDA::Timer timer;
       
    // reset particle count to zero
    //timer.Start();
    computeParticleAccelerationAndAdvanceHigh<<<mGridDimHigh, mBlockDim>>>
    (
        mFluidParticlesHigh->Positions(), 
        mFluidParticlesHigh->Velocities(), 
        mFluidParticlesHigh->mdAccelerations,
        mFluidParticleGridHigh.dParticleIDs[(activeID + 1) % 2],
        mFluidParticlesHigh->mdParticleIDs,
        mdParticleCountHigh,
        mFluidParticlesHigh->Densities(),
        mFluidParticlesHigh->Pressures(),
        mdStatesHigh,
        mFluidParticles->mdPositions,
        mFluidParticles->mdVelocities,
        mFluidParticles->mdDensities,
        mFluidParticles->mdPressures,
        mdStates,
        mFluidParticleGridHigh.dParticleIDs[activeID], 
        mFluidParticleGridHigh.dCellStart, 
        mFluidParticleGridHigh.dCellEnd, 
        mFluidParticleGrid.dParticleIDs[activeID], 
        mFluidParticleGrid.dCellStart, 
        mFluidParticleGrid.dCellEnd, 
        mTimeStep, 
        mFluidParticlesHigh->mNumParticles
    );  
}
//-----------------------------------------------------------------------------
void WCSPHSolver::computePressureDensityHigh(unsigned char activeID)
{
    if (mFluidParticlesHigh->mNumParticles == 0)
    {
        return;
    }

    computeParticleDensityPressureHigh<<<mGridDimHigh, mBlockDim>>>
    (
        mFluidParticlesHigh->Densities(), 
        mFluidParticlesHigh->Pressures(), 
        mFluidParticleGridHigh.dParticleIDs[activeID], 
        mFluidParticleGridHigh.dCellStart, 
        mFluidParticleGridHigh.dCellEnd, 
        mFluidParticlesHigh->Positions(), 
        mdStatesHigh,
        mFluidParticleGrid.dParticleIDs[activeID], 
        mFluidParticleGrid.dCellStart, 
        mFluidParticleGrid.dCellEnd, 
        mFluidParticles->Positions(), 
        mdStates,
        mFluidParticlesHigh->mNumParticles
    );
}
//-----------------------------------------------------------------------------
//void WCSPHSolver::relaxTransient (unsigned char activeID)
//{
//    CUDA_SAFE_CALL(cudaMemcpy
//    (
//        &mTransientParticleCountHigh, 
//        mdTransientParticleCountHigh, 
//        sizeof(int),
//        cudaMemcpyDeviceToHost
//    ))
//
//    if (mTransientParticleCountHigh == 0)
//    {
//        return;
//    }
//
//    mGridDimTransientHigh.x = std::ceil
//    (
//        static_cast<float>(mTransientParticleCountHigh)/mBlockDim.x
//    );
//
//    for (unsigned int i = 0; i < 4; i++)
//    {
//        computeParticleDensityPressureHighRelax
//        <<<mGridDimTransientHigh, mBlockDim>>>
//        (
//            mFluidParticlesHigh->Densities(), 
//            mFluidParticlesHigh->Pressures(), 
//            mdTransientIDsHigh,
//            mFluidParticlesHigh->mdParticleIDs[activeID], 
//            mFluidParticleGridHigh.dCellStart, 
//            mFluidParticleGridHigh.dCellEnd, 
//            mFluidParticlesHigh->Positions(), 
//            mFluidParticles->mdParticleIDs[activeID], 
//            mFluidParticleGrid.dCellStart, 
//            mFluidParticleGrid.dCellEnd, 
//            mFluidParticles->Positions(), 
//            mTransientParticleCountHigh
//        );
//
//        computeParticleAccelerationAndAdvanceHighRelax
//        <<<mGridDimTransientHigh, mBlockDim>>>
//        (
//            mFluidParticlesHigh->Positions(), 
//            mFluidParticlesHigh->Velocities(), 
//            mFluidParticlesHigh->Densities(),
//            mFluidParticlesHigh->Pressures(),
//            mFluidParticles->mdPositions,
//            mFluidParticles->mdVelocities,
//            mFluidParticles->mdDensities,
//            mFluidParticles->mdPressures,
//            mdTransientIDsHigh,
//            mFluidParticlesHigh->mdParticleIDs[activeID], 
//            mFluidParticleGridHigh.dCellStart, 
//            mFluidParticleGridHigh.dCellEnd, 
//            mFluidParticles->mdParticleIDs[activeID], 
//            mFluidParticleGrid.dCellStart, 
//            mFluidParticleGrid.dCellEnd, 
//            mTimeStep/10.0f, 
//            mTransientParticleCountHigh
//        );  
//    }
//
//    //std::cout << mTransientParticleCountHigh << std::endl;
//
//
//    CUDA_SAFE_CALL( cudaMemset(mdTransientParticleCountHigh, 0, sizeof(int)) );
//}
//-----------------------------------------------------------------------------
void WCSPHSolver::adjustOrInjectTransientHigh (unsigned char activeID)
{
    // return if there are no particles in transition (low) 
    if (mTransientParticleCountLow == 0)
    {
        return;
    }    


    adjustOrInjectTransientHighD<<<mGridDimTransientLow, mBlockDim>>>
    (
        mFluidParticlesHigh->mdPositions,
        mFluidParticlesHigh->mdVelocities,
        mdStatesHigh,
        mFluidParticleGrid.dParticleIDs[(activeID + 1) % 2], 
        mFluidParticles->mdParticleIDs,
        mdTransientIDsLow[(activeID + 1) % 2], 
        mdParticleCount,
        mdTransientParticleCountLow, 
        mFluidParticles->mdPositions,
        mFluidParticles->mdVelocities,
        mFluidParticles->mdDensities,
        mdTransientIDsLow[activeID],
        mTransientParticleCountLow
    );


}
//-----------------------------------------------------------------------------
void WCSPHSolver::adjustTransientHigh (unsigned char activeID)
{
    // return if there are no particles in transition (low) 
    if (mTransientParticleCountLow == 0)
    {
        return;
    }


    adjustTransientHighD<<<mGridDimTransientLow, mBlockDim>>>
    (
        mFluidParticlesHigh->mdPositions,
        mFluidParticlesHigh->mdVelocities,
        mFluidParticles->mdPositions,
        mFluidParticles->mdVelocities,
        mFluidParticles->mdDensities,
        mdTransientIDsLow[activeID],
        mTransientParticleCountLow
    );

}
//-----------------------------------------------------------------------------
void WCSPHSolver::injectTransientHigh(unsigned char activeID)
{
    if (mTransientParticleCountLow == 0)
    {
        return;
    }

    injectTransientHighD<<<mGridDimTransientLow, mBlockDim>>>
    (
        mdStates,
        mdStatesHigh,
        mFluidParticles->mdDensities,
        mFluidParticlesHigh->mdDensities,
        mdTransientIDsLow[activeID],
        mTransientParticleCountLow
    );   
}
//-----------------------------------------------------------------------------
void WCSPHSolver::setVisualQuantities(unsigned char activeID)
{
    setQuantities<<<mGridDim, mBlockDim>>>
    (
        mFluidParticles->mdVisualQuantities, 
        mFluidParticles->mdPositions,
        mFluidParticles->mdDensities, 
        mFluidParticleGrid.dParticleIDs[activeID],
        mFluidParticles->mNumParticles
    );

    if (mFluidParticlesHigh->mNumParticles > 0)
    {
        setQuantities<<<mGridDimHigh, mBlockDim>>>
        (
            mFluidParticlesHigh->mdVisualQuantities, 
            mFluidParticlesHigh->mdPositions, 
            mFluidParticlesHigh->mdDensities, 
            mFluidParticleGridHigh.dParticleIDs[activeID],
            mFluidParticlesHigh->mNumParticles
        );
    }
}
//-----------------------------------------------------------------------------
void WCSPHSolver::initBoundaries () const
{
    dim3 gridDim;
    dim3 blockDim;      
    blockDim.x = 256; 
    blockDim.y = 1; 
    blockDim.z = 1; 
    unsigned int numBoundaryParticles = mBoundaryParticles->GetNumParticles();
    unsigned int numBlocks = numBoundaryParticles/blockDim.x;  
    gridDim.x = numBoundaryParticles % blockDim.x == 0 ? numBlocks : numBlocks + 1;
    gridDim.y = 1;
    gridDim.z = 1;
    mBoundaryParticles->Map();
    initParticleIDs<<<gridDim, blockDim>>>(mdBoundaryParticleIDs, 
        numBoundaryParticles);
    computeParticleHash<<<gridDim, blockDim>>>
        (mdBoundaryParticleHashs, mdBoundaryParticleIDs, 
        mBoundaryParticles->Positions(), numBoundaryParticles);
    thrust::sort_by_key(thrust::device_ptr<int>(mdBoundaryParticleHashs),
        thrust::device_ptr<int>(mdBoundaryParticleHashs + numBoundaryParticles), 
        thrust::device_ptr<int>(mdBoundaryParticleIDs));

    unsigned int domainSize = mDomainDimensions[0]*mDomainDimensions[1]*
        sizeof(int); 
    CUDA_SAFE_CALL ( cudaMemset(mdBoundaryCellStartIndices, EMPTY_CELL,
        domainSize) ); 
    CUDA_SAFE_CALL ( cudaMemset(mdBoundaryCellEndIndices, EMPTY_CELL, 
        domainSize) );
    int sharedMemSize = sizeof(int)*(blockDim.x + 1);
    computeCellStartEndIndices<<<gridDim, blockDim, sharedMemSize>>>
        (mdBoundaryCellStartIndices, mdBoundaryCellEndIndices, 
        mdBoundaryParticleHashs, numBoundaryParticles);
    mBoundaryParticles->Unmap();
}
//-----------------------------------------------------------------------------

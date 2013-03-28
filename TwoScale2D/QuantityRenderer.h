//-----------------------------------------------------------------------------
//  QuantityRenderer.h
//  FastTurbulentFluids
//
//  Created by Arno in Wolde L�bke on 25.03.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#pragma once

#include "stdutil.h"
#include "OpenGL\OpenGL.h"
#include "ParticleSystem.h"
#include "Rectangle.h"

class QuantityRenderer 
{
    DECL_DEFAULTS (QuantityRenderer);

public:
    QuantityRenderer (const ParticleSystem& particleSystem, float xs, float ys,
        float xe, float ye, float pointSize);
    ~QuantityRenderer ();

    void SetDisplayRectangle (const CGTK::Rectangle& rectangle);
    void Render () const;

private:
    const ParticleSystem* mParticleSystem;
    
    GLuint mProgram;
    GLuint mVAO;
};
//-----------------------------------------------------------------------------
//  Rectangle.h
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 28.03.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#pragma once

#include <stdexcept>
#include <cmath>

namespace CGTK
{

struct Rectangle
{
    float XS, YS;   // lower left point of the rectangle
    float XE, YE;   // upper right point of the rectangle

    Rectangle ()
    : XS(0.0f), YS(0.0f), XE(1.0f), YE(1.0f)    
    {
    
    }

    Rectangle (float xs, float ys, float xe, float ye)
    : XS(xs), YS(ys), XE(xe), YE(ye)
    {
        if (xs > xe || ys > ye)
        {
            throw std::runtime_error("invalid rectangle state");
        }
    }

    ~Rectangle ()
    {

    }

    inline void Translate (float dx, float dy)
    {
        XS += dx;
        YS += dy;
        XE += dx;
        YE += dy;
    }

    inline void Scale (float s)
    {
        float xse = XS - XE;
        float yse = YS - YE;
        float mag = std::sqrt(xse*xse + yse*yse);
        xse /= mag;
        yse /= mag;

        XS -= s*xse;
        YS -= s*yse;
        XE += s*xse;
        YE += s*yse;
    }
    
    inline float GetWidth () const
    {
        return XE - XS;
    } 

    inline float GetHeight () const
    {
        return YE - YS;
    }
};

}
//-----------------------------------------------------------------------------
//  PointRendererGeometryShader.glsl
//  SPHFLUIDS2D
//
//  Created by Arno in Wolde Lübke on 04.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#version 330 core

uniform float xs;
uniform float ys;
uniform float width;
uniform float height;
uniform float pointSize;

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;


in VertexData
{
    float quantity;
}
gVertexData;

out GeometryData
{
    vec2 relCoord;
    float quantity;
}
gGeometryData;

void main() 
{
    float dx = pointSize;
    
    gl_Position = gl_in[0].gl_Position + vec4(dx, dx, 0.0f, 0.0f);
    gGeometryData.relCoord = vec2(1.0f, 1.0f);
    gGeometryData.quantity = gVertexData.quantity;
    EmitVertex();
    
    gl_Position = gl_in[0].gl_Position + vec4(-dx, dx, 0.0f, 0.0f);
    gGeometryData.relCoord = vec2(-1.0f, 1.0f);
    gGeometryData.quantity = gVertexData.quantity;
    EmitVertex();
    
    gl_Position = gl_in[0].gl_Position + vec4(dx, -dx, 0.0f, 0.0f);
    gGeometryData.relCoord = vec2(1.0f, -1.0f);
    gGeometryData.quantity = gVertexData.quantity;
    EmitVertex();
    
    gl_Position = gl_in[0].gl_Position + vec4(-dx, -dx, 0.0f, 0.0f);
    gGeometryData.relCoord = vec2(-1.0f, -1.0f);
    gGeometryData.quantity = gVertexData.quantity;
    EmitVertex();
    
    EndPrimitive();
}
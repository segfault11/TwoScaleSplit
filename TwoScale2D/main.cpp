#include <iostream>
#include <cstdlib>
#include <memory>
#include "VideoWriter\VideoWriter.h"
#include "OpenGL\OpenGL.h"
#include "CUDA\cuda.h"
#include "ParticleSystem.h"
#include "PointRenderer.h"
#include "QuantityRenderer.h"
#include "WCSPHSolver.h"
#include "Rectangle.h"

#define PI 3.14159265358979323846
#define WIDTH  1100
#define HEIGHT 1100

ParticleSystem* gFluidParticles;
ParticleSystem* gFluidParticlesHigh;
ParticleSystem* gBoundaryParticles;
QuantityRenderer* gFluidRenderer;
QuantityRenderer* gFluidRendererHigh;
PointRenderer* gBoundaryRenderer;
static CGTK::Rectangle gsDisplayRectangle;
WCSPHSolver* gSolver;

static VideoWriter gsVideoWriter("video.avi", WIDTH, HEIGHT);

void display ();
void keyboard (unsigned char key, int x, int y);

void init ();
void init2 ();
void tearDown ();
void saveScreenshot (const std::string& filename);

int main (int argc, char* argv[])
{  
    //unsigned int i = 1;

    //std::cout << (i << 1) << std::endl;
    //std::system("pause");


    cudaGLSetGLDevice(0); 
    glutInit(&argc, argv);
    glutInitContextVersion(3, 3);
    glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
    glutInitContextProfile(GLUT_CORE_PROFILE);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("");
	glewExperimental = TRUE;
	glewInit();
    init();
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, 
        GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutMainLoop();
    tearDown();

    return 0;
}

void display () 
{
    static int i = 0;
    try
    {
    gSolver->AdvanceTS();
    //gSolver->AdvanceHigh(); 
    
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   
   
    gFluidRenderer->SetDisplayRectangle(gsDisplayRectangle);
    gFluidRendererHigh->SetDisplayRectangle(gsDisplayRectangle);
    gBoundaryRenderer->SetDisplayRectangle(gsDisplayRectangle);
    gFluidRenderer->Render();
    gFluidRendererHigh->Render();    
    gBoundaryRenderer->Render();

    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();

    }
    catch (const std::runtime_error& e)
    {
        std::cout << e.what() << std::endl;
        std::system("pause");
    }

    if (i % 6 == 0)
    {
        gsVideoWriter.CaptureFrame();
    }

    i++;
    
}

void keyboard (unsigned char key, int x, int y)
{
    switch (key) 
    {
        case 'r':
            gsDisplayRectangle.Scale(0.01f);
            return;
        case 'f':
            gsDisplayRectangle.Scale(-0.01f);
            return;
        case 'a':
            gsDisplayRectangle.Translate(-0.01f, 0.0f);
            return;
        case 's':
            gsDisplayRectangle.Translate(0.0f, -0.01f);
            return;
        case 'd':
            gsDisplayRectangle.Translate(0.01f, 0.0f);
            return;
        case 'w':
            gsDisplayRectangle.Translate(0.0f, 0.01f);
            return;
	    default:
		    return; 
	}
}

void init ()
{
    //glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR);
    glBlendColor(0.5f, 0.5f, 0.5f, 0.5f);

    // Dam break simulation with about 15000 fluid particles
    try
    {
        // set configuration parameters
        float restDensity = 1000.0f;
        float fluidVol = 100.0f*0.0025f*150.0f*0.0025f;
        unsigned int numParticles = 101*151;
        float particleMass = fluidVol*restDensity/numParticles;
        float effectiveRadius = std::sqrtf(fluidVol*25.0f/(numParticles*PI));
        float effectiveRadiusHigh = 0.5f*effectiveRadius;
        float taitCoeff = 1119.0714f;
        float speedSound = 88.1472f;
        float alpha = 0.04f;
        float tensionCoefficient = 0.08f;
        float timeStep = 0.0002f;

        gFluidParticles = CreateParticleBox
        (
            0.205f, 0.105f, 0.0025f, 101, 151, 
            particleMass
        );     
        gFluidParticlesHigh = new ParticleSystem
        (
            gFluidParticles->GetMaxParticles()*4,
            gFluidParticles->GetMass()/4.0f
        );
        //gFluidParticlesHigh = CreateParticleBox
        //(
        //    0.705f, 0.105f, 0.00125f, 101, 151, 
        //    particleMass/4.0f
        //);
        gBoundaryParticles = CreateParticleBoxCanvas
        (
            0.1f, 0.1f, 0.0025f,321, 321, 3, 3, 
            1000.0f*0.25f*0.25f/(101.0f*151.0f)
        );
        gFluidRenderer = new QuantityRenderer
        (
            *gFluidParticles, 0.09f, 0.09f, 0.91f,
            0.91f, 0.005f  
        );
        gFluidRendererHigh = new QuantityRenderer
        (
            *gFluidParticlesHigh, 0.09f, 0.09f, 0.91f,
            0.91f, 0.003f       
        );
        gBoundaryRenderer = new PointRenderer
        (
            *gBoundaryParticles, 0.09f, 0.09f, 
            0.91f, 0.91f, 0.0f, 0.0f, 0.0f, 1.0f, 0.005f  
        );
        WCSPHConfig config
        (   
            0.0f, 0.0f, 1.0f, 1.0f, effectiveRadius, 
            effectiveRadiusHigh, restDensity, taitCoeff, speedSound, 
            alpha, tensionCoefficient, timeStep
        );     
        gSolver = new WCSPHSolver
        (
            config, *gFluidParticles, *gFluidParticlesHigh,*gBoundaryParticles
        );
        gSolver->Bind();      
    }
    catch (const std::runtime_error& e)
    {
        std::cout << e.what() << std::endl;
    }
    
}



//void init2 ()
//{
//    // Dam break simulation with about 60000 fluid particles
//    try
//    {
//        float restDensity = 1000.0f;
//        float fluidVol = 200.0f*0.00125f*300.0f*0.00125f;
//        unsigned int numParticles = 201*301;
//        float particleMass = fluidVol*restDensity/numParticles;
//        float effectiveRadius = std::sqrtf(fluidVol*30.0f/(numParticles*PI));
//        float taitCoeff = 1119.0714f;
//        float speedSound = 88.1472;
//        float alpha = 0.046f;
//        float tensionCoefficient = 0.08f;
//        float timeStep = 0.0002f;
//
//        gFluidParticles = CreateParticleBox(0.105f, 0.105f, 0.00125f, 201, 301, 
//            particleMass);
//        
//        gBoundaryParticles = CreateParticleBoxCanvas(0.1f, 0.1f, 0.00125f, 
//            641, 641, 4, 4, particleMass);
//
//        gFluidRenderer = new PointRenderer(*gFluidParticles, 0.0f, 0.0f, 1.0f,
//            1.0f, 0.0f, 0.6f, 0.8f, 1.0f);
//        gBoundaryRenderer = new PointRenderer(*gBoundaryParticles, 0.0f, 0.0f, 
//            1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
//
//        
//        WCSPHConfig config(0.0f, 0.0f, 1.0f, 1.0f, effectiveRadius, restDensity, 
//            taitCoeff, speedSound, alpha, tensionCoefficient, timeStep);
//
//        
//        
//        gSolver = new WCSPHSolver(config, *gFluidParticles, *gBoundaryParticles);
//        gSolver->Bind();  
//    }
//    catch (const std::runtime_error& e)
//    {
//        std::cout << e.what() << std::endl;
//    }
//}


void tearDown ()
{
    delete gFluidParticles;
    delete gBoundaryParticles;
    delete gFluidParticlesHigh;
    delete gFluidRenderer;
    delete gFluidRendererHigh;
    delete gBoundaryRenderer;
    delete gSolver;
}

void saveScreenshot (const std::string& filename)
{
    gsVideoWriter.SaveScreenshot(filename);
}
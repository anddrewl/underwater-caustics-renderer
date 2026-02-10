#include "ray-tracer.h"

Window window;

 
// draw something in each frame
static void draw() {
    int cycleframe = globalFrameCount % 400;
    float g = 0.0f + cycleframe*0.0025f;
    float b = 1.0f - cycleframe*0.0025f;


    for (int j = 0; j < globalHeight; j++) {
        float r = 0.0f;
        for (int i = 0; i < globalWidth; i++) {
            //FrameBuffer.pixel(i, j) = float3(PCG32::rand()); // noise
            FrameBuffer.pixel(i, j) = float3(r,g,b);
            r += 0.0025f;
            //FrameBuffer.pixel(i, j) = float3(0.5f * (cos((i + globalFrameCount) * 0.1f) + 1.0f)); // moving cosine
        }
        g += 0.0025f;
        b -= 0.0025f;
        if (g > 1.0f) {
            g = 0.0f;
        }
        if (b < 0.0f) {
            b = 1.0f;
        }
    }
}
/*
static void hello_world(int argc, const char* argv[]) {
    // set the function to be called in the main loop
    window.process = draw;
}*/



// setting up lighting
static PointLightSource light;
static PointLightSource light2;
static void setupLightSource() {
    //light.position = float3(0.5f, 4.0f, 1.0f); // use this for sponza.obj
    light.position = float3(0.25f, 0.30f, 0.25f);
    light.wattage = float3(30.0f, 30.0f, 30.0f);
    globalScene.addLight(&light);
    light2.position = float3(-0.25f, 0.30f, -0.25f);
    light2.wattage = float3(10.0f, 10.0f, 10.0f);
    globalScene.addLight(&light2);
    
}



// loading .obj file from the command line arguments
static TriangleMesh mesh;
static TriangleMesh* surface;
static void setupScene(int argc, const char* argv[]) {
    if (argc > 1) {
        bool objLoadSucceed = mesh.load(argv[1]);
        surface = generateWaterSurface(float3(-0.390286, -0.385896, -0.392431), 0.3 ,128, 0.8, 0.8,5000);
        if (!objLoadSucceed) {
            printf("Invalid .obj file.\n");
            printf("Making a single triangle instead.\n");
            mesh.createSingleTriangle();
        }
    } else {
        printf("Specify .obj file in the command line arguments. Example: window.exe cornellbox.obj\n");
        printf("Making a single triangle instead.\n");
        mesh.createSingleTriangle();
    }
    globalScene.addObject(&mesh);
    globalScene.addObject(surface);
}
static void ray_tracer(int argc, const char* argv[]) {
    setupScene(argc, argv);
    setupLightSource();

    globalRenderType = RENDER_RAYTRACE;
}

static void rasterizer(int argc, const char* argv[]) {
    setupScene(argc, argv);
    setupLightSource();
    globalRenderType = RENDER_RASTERIZE;
}

static void physics(int argc, const char* argv[]) {
    globalEnableParticles = true;
    setupLightSource();
    globalRenderType = RENDER_RASTERIZE;
    if (argc > 1) globalParticleSystem.sphereMeshFilePath = argv[1];
    globalParticleSystem.initialize();
}

static void Final(int argc, const char* argv[]) {
    setupScene(argc, argv);
    setupLightSource();

    globalRenderType = RENDER_PHOTON;
}


int main(int argc, const char* argv[]) {
    // hello_world(argc, argv);
    // Complex RoU[] = {Complex(1,0), Complex(-1,0)}
    // ray_tracer(argc, argv);
    // rasterizer(argc, argv);
    // physics(argc, argv);
    
    Final(argc, argv);
    window.start();
}

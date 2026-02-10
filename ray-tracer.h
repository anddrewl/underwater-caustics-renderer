#pragma once
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX


// OpenGL
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>


// image loader and writer
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


// linear algebra 
#include "linalg.h"
using namespace linalg::aliases;
using namespace linalg::ostream_overloads;

// animated GIF writer
#include "gif.h"

//final Project
#include "photon.h"


// misc
#include <iostream>
#include <vector>
#include <cfloat>
#include <random>
#include <cmath>

// main window
static GLFWwindow* globalGLFWindow;


// window size and resolution
// (do not make it too large - will be slow!)
constexpr int globalWidth = 512;
constexpr int globalHeight = 384;


// degree and radian
constexpr float PI = 3.14159265358979f;
constexpr float DegToRad = PI / 180.0f;
constexpr float RadToDeg = 180.0f / PI;


// for ray tracing
constexpr float Epsilon = 5e-5f;


// amount the camera moves with a mouse and a keyboard
constexpr float ANGFACT = 0.2f;
constexpr float SCLFACT = 0.1f;


// fixed camera parameters
constexpr float globalAspectRatio = float(globalWidth / float(globalHeight));
constexpr float globalFOV = 45.0f; // vertical field of view
constexpr float globalDepthMin = Epsilon; // for rasterization
constexpr float globalDepthMax = 100.0f; // for rasterization
constexpr float globalFilmSize = 0.032f; //for ray tracing
const float globalDistanceToFilm = globalFilmSize / (2.0f * tan(globalFOV * DegToRad * 0.5f)); // for ray tracing


// particle system related
bool globalEnableParticles = false;
constexpr float deltaT = 0.002f;
constexpr float3 globalGravity = float3(0.0f, -9.8f, 0.0f) * 10;
constexpr int globalNumParticles = 50;
bool globalSimpleCollision = true;
bool globalSphereCollision = false;
bool globalGravitationalField = false;
bool globalVolumetricParticle = false;
const float GravationalConstant = 0.00001;//6.674E-11;

// dynamic camera parameters
float3 globalEye = float3(0.0f, 0.0f, 1.5f);
float3 globalLookat = float3(0.0f, 0.0f, 0.0f);
float3 globalUp = normalize(float3(0.0f, 1.0f, 0.0f));
float3 globalViewDir; // should always be normalize(globalLookat - globalEye)
float3 globalRight; // should always be normalize(cross(globalViewDir, globalUp));
bool globalShowRaytraceProgress = false; // for ray tracing
bool globalShadow = false;
bool globalPointLightSource = true; //must turn on! don't know what will happen without it
bool globalEnvironmentImage = false;

//for final project: number of photons
const int NPHOTON = 100000;
bool globalPhotonTracing = true;
bool waterGeneration = true;
bool globalPhotonMap = true;


const char* pathToImage = "../media/uffizi_probe.hdr";

// mouse event
static bool mouseLeftPressed;
static double m_mouseX = 0.0;
static double m_mouseY = 0.0;


// rendering algorithm
enum enumRenderType {
	RENDER_RASTERIZE,
	RENDER_RAYTRACE,
	RENDER_IMAGE,
	RENDER_PHOTON, //ADDED FOR FINAL PROJECT
};
enumRenderType globalRenderType = RENDER_IMAGE;
int globalFrameCount = 0;
static bool globalRecording = false;
static GifWriter globalGIFfile;
constexpr int globalGIFdelay = 1;


// OpenGL related data (do not modify it if it is working)
static GLuint GLFrameBufferTexture;
static GLuint FSDraw;
static const std::string FSDrawSource = R"(
    #version 120

    uniform sampler2D input_tex;
    uniform vec4 BufInfo;

    void main()
    {
        gl_FragColor = texture2D(input_tex, gl_FragCoord.st * BufInfo.zw);
    }
)";
static const char* PFSDrawSource = FSDrawSource.c_str();



// fast random number generator based pcg32_fast
#include <stdint.h>
namespace PCG32 {
	static uint64_t mcg_state = 0xcafef00dd15ea5e5u;	// must be odd
	static uint64_t const multiplier = 6364136223846793005u;
	uint32_t pcg32_fast(void) {
		uint64_t x = mcg_state;
		const unsigned count = (unsigned)(x >> 61);
		mcg_state = x * multiplier;
		x ^= x >> 22;
		return (uint32_t)(x >> (22 + count));
	}
	float rand() {
		return float(double(pcg32_fast()) / 4294967296.0);
	}
}



// image with a depth buffer
// (depth buffer is not always needed, but hey, we have a few GB of memory, so it won't be an issue...)
class Image {
public:
	std::vector<float3> pixels;
	std::vector<float> depths;
	int width = 0, height = 0;

	static float toneMapping(const float r) {
		// you may want to implement better tone mapping
		return std::max(std::min(1.0f, r), 0.0f);
	}

	static float gammaCorrection(const float r, const float gamma = 1.0f) {
		// assumes r is within 0 to 1
		// gamma is typically 2.2, but the default is 1.0 to make it linear
		return pow(r, 1.0f / gamma);
	}

	void resize(const int newWdith, const int newHeight) {
		this->pixels.resize(newWdith * newHeight);
		this->depths.resize(newWdith * newHeight);
		this->width = newWdith;
		this->height = newHeight;
	}

	void clear() {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				this->pixel(i, j) = float3(0.0f);
				this->depth(i, j) = 0.0f;
			}
		}
	}

	Image(int _width = 0, int _height = 0) {
		this->resize(_width, _height);
		this->clear();
	}

	bool valid(const int i, const int j) const {
		return (i >= 0) && (i < this->width) && (j >= 0) && (j < this->height);
	}

	float& depth(const int i, const int j) {
		return this->depths[i + j * width];
	}

	float3& pixel(const int i, const int j){
		// optionally can check with "valid", but it will be slow
		return this->pixels[i + j * width];
	}

	const float3 pixelConst(const int i, const int j){
		// optionally can check with "valid", but it will be slow
		return this->pixels[i + j * width];
	}

	void load(const char* fileName) {
		int comp, w, h;
		float* buf = stbi_loadf(fileName, &w, &h, &comp, 3);
		if (!buf) {
			std::cerr << "Unable to load: " << fileName << std::endl;
			return;
		}

		this->resize(w, h);
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				this->pixels[i + j * width] = float3(buf[k], buf[k + 1], buf[k + 2]);
				k += 3;
			}
		}
		delete[] buf;
		printf("Loaded \"%s\".\n", fileName);
	}
	void save(const char* fileName) {
		unsigned char* buf = new unsigned char[width * height * 3];
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).x)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).y)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).z)));
			}
		}
		stbi_write_png(fileName, width, height, 3, buf, width * 3);
		delete[] buf;
		printf("Saved \"%s\".\n", fileName);
	}
};

// main image buffer to be displayed
Image FrameBuffer(globalWidth, globalHeight);

// you may want to use the following later for progressive ray tracing
Image AccumulationBuffer(globalWidth, globalHeight);
unsigned int sampleCount = 0;



// keyboard events (you do not need to modify it unless you want to)
void keyFunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
			case GLFW_KEY_R: {
				if (globalRenderType == RENDER_RAYTRACE) {
					printf("(Switched to rasterization)\n");
					glfwSetWindowTitle(window, "Rasterization mode");
					globalRenderType = RENDER_RASTERIZE;
				} else if (globalRenderType == RENDER_RASTERIZE) {
					printf("(Switched to ray tracing)\n");
					AccumulationBuffer.clear();
					sampleCount = 0;
					glfwSetWindowTitle(window, "Ray tracing mode");
					globalRenderType = RENDER_RAYTRACE;
				}
			break;}

			case GLFW_KEY_P: {
				if (globalPhotonMap) {
					globalPhotonMap = false;
				}
				else {
					globalPhotonMap = true;
				}
				break; }

			case GLFW_KEY_ESCAPE: {
				glfwSetWindowShouldClose(window, GL_TRUE);
			break;}

			case GLFW_KEY_I: {
				char fileName[1024];
				sprintf(fileName, "output%d.png", int(1000.0 * PCG32::rand()));
				FrameBuffer.save(fileName);
			break;}

			case GLFW_KEY_F: {
				if (!globalRecording) {
					char fileName[1024];
					sprintf(fileName, "output%d.gif", int(1000.0 * PCG32::rand()));
					printf("Saving \"%s\"...\n", fileName);
					GifBegin(&globalGIFfile, fileName, globalWidth, globalHeight, globalGIFdelay);
					globalRecording = true;
					printf("(Recording started)\n");
				} else {
					GifEnd(&globalGIFfile);
					globalRecording = false;
					printf("(Recording done)\n");
				}
			break;}

			case GLFW_KEY_U: {
				if (globalRenderType == RENDER_RAYTRACE) {
					globalRenderType = RENDER_PHOTON;
				}
				else if (globalRenderType == RENDER_PHOTON) {
					globalRenderType = RENDER_RAYTRACE;
				}
				break; }


			case GLFW_KEY_W: {
				globalEye += SCLFACT * globalViewDir;
				globalLookat += SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_S: {
				globalEye -= SCLFACT * globalViewDir;
				globalLookat -= SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_Q: {
				globalEye += SCLFACT * globalUp;
				globalLookat += SCLFACT * globalUp;
			break;}

			case GLFW_KEY_Z: {
				globalEye -= SCLFACT * globalUp;
				globalLookat -= SCLFACT * globalUp;
			break;}

			case GLFW_KEY_A: {
				globalEye -= SCLFACT * globalRight;
				globalLookat -= SCLFACT * globalRight;
			break;}

			case GLFW_KEY_D: {
				globalEye += SCLFACT * globalRight;
				globalLookat += SCLFACT * globalRight;
			break;}

			default: break;
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void mouseButtonFunc(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			mouseLeftPressed = true;
		} else if (action == GLFW_RELEASE) {
			mouseLeftPressed = false;
			if (globalRenderType == RENDER_RAYTRACE) {
				AccumulationBuffer.clear();
				sampleCount = 0;
			}
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void cursorPosFunc(GLFWwindow* window, double mouse_x, double mouse_y) {
	if (mouseLeftPressed) {
		const float xfact = -ANGFACT * float(mouse_y - m_mouseY);
		const float yfact = -ANGFACT * float(mouse_x - m_mouseX);
		float3 v = globalViewDir;

		// local function in C++...
		struct {
			float3 operator()(float theta, const float3& v, const float3& w) {
				const float c = cosf(theta);
				const float s = sinf(theta);

				const float3 v0 = dot(v, w) * w;
				const float3 v1 = v - v0;
				const float3 v2 = cross(w, v1);

				return v0 + c * v1 + s * v2;
			}
		} rotateVector;

		v = rotateVector(xfact * DegToRad, v, globalRight);
		v = rotateVector(yfact * DegToRad, v, globalUp);
		globalViewDir = v;
		globalLookat = globalEye + globalViewDir;
		globalRight = cross(globalViewDir, globalUp);

		m_mouseX = mouse_x;
		m_mouseY = mouse_y;

		if (globalRenderType == RENDER_RAYTRACE) {
			AccumulationBuffer.clear();
			sampleCount = 0;
		}
	} else {
		m_mouseX = mouse_x;
		m_mouseY = mouse_y;
	}
}




class PointLightSource {
public:
	float3 position, wattage;
};



class Ray {
public:
	float3 o, d;
	Ray() : o(), d(float3(0.0f, 0.0f, 1.0f)) {}
	Ray(const float3& o, const float3& d) : o(o), d(d) {}
};



// "type" will tell the actual type
enum enumMaterialType {
	MAT_LAMBERTIAN,
	MAT_METAL,
	MAT_GLASS
};
class Material {
public:
	std::string name;

	enumMaterialType type = MAT_LAMBERTIAN;
	float eta = 1.0f;
	float glossiness = 1.0f;

	float3 Ka = float3(0.0f);
	float3 Kd = float3(0.9f);
	float3 Ks = float3(0.0f);
	float Ns = 0.0;

	// support 8-bit texture
	bool isTextured = false;
	unsigned char* texture = nullptr;
	int textureWidth = 0;
	int textureHeight = 0;

	Material() {};
	virtual ~Material() {};

	void setReflectance(const float3& c) {
		if (type == MAT_LAMBERTIAN) {
			Kd = c;
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
	}

	float3 fetchTexture(const float2& tex) const {
		// repeating
		int x = int(tex.x * textureWidth) % textureWidth;
		int y = int(tex.y * textureHeight) % textureHeight;
		if (x < 0) x += textureWidth;
		if (y < 0) y += textureHeight;

		int pix = (x + y * textureWidth) * 3;
		const unsigned char r = texture[pix + 0];
		const unsigned char g = texture[pix + 1];
		const unsigned char b = texture[pix + 2];
		return float3(r, g, b) / 255.0f;
	}

	float3 BRDF(const float3& wi, const float3& wo, const float3& n) const {
		float3 brdfValue = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// BRDF
			brdfValue = Kd / PI;
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
		return brdfValue;
	};

	float PDF(const float3& wGiven, const float3& wSample) const {
		// probability density function for a given direction and a given sample
		// it has to be consistent with the sampler
		float pdfValue = 0.0f;
		if (type == MAT_LAMBERTIAN) {
			// empty
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
		return pdfValue;
	}

	float3 sampler(const float3& wGiven, float& pdfValue) const {
		// sample a vector and record its probability density as pdfValue
		float3 smp = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// empty
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}

		pdfValue = PDF(wGiven, smp);
		return smp;
	}
};







class HitInfo {
public:
	float t; // distance
	float3 P; // location
	float3 N; // shading normal vector
	float3 GN; //Geometric normal vector
	float2 T; // texture coordinate
	const Material* material; // const pointer to the material of the intersected object
	bool photon; // flag this hitInfo as a photon
	HitInfo() {
		photon = false;
	}
};



// axis-aligned bounding box
class AABB {
private:
	float3 minp, maxp, size;

public:
	float3 get_minp() { return minp; };
	float3 get_maxp() { return maxp; };
	float3 get_size() { return size; };


	AABB() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	void reset() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	int getLargestAxis() const {
		if ((size.x > size.y) && (size.x > size.z)) {
			return 0;
		} else if (size.y > size.z) {
			return 1;
		} else {
			return 2;
		}
	}

	void fit(const float3& v) {
		if (minp.x > v.x) minp.x = v.x;
		if (minp.y > v.y) minp.y = v.y;
		if (minp.z > v.z) minp.z = v.z;

		if (maxp.x < v.x) maxp.x = v.x;
		if (maxp.y < v.y) maxp.y = v.y;
		if (maxp.z < v.z) maxp.z = v.z;

		size = maxp - minp;
	}

	float area() const {
		return (2.0f * (size.x * size.y + size.y * size.z + size.z * size.x));
	}


	bool intersect(HitInfo& minHit, const Ray& ray) const {
		// set minHit.t as the distance to the intersection point
		// return true/false if the ray hits or not
		float tx1 = (minp.x - ray.o.x) / ray.d.x;
		float ty1 = (minp.y - ray.o.y) / ray.d.y;
		float tz1 = (minp.z - ray.o.z) / ray.d.z;

		float tx2 = (maxp.x - ray.o.x) / ray.d.x;
		float ty2 = (maxp.y - ray.o.y) / ray.d.y;
		float tz2 = (maxp.z - ray.o.z) / ray.d.z;

		if (tx1 > tx2) {
			const float temp = tx1;
			tx1 = tx2;
			tx2 = temp;
		}

		if (ty1 > ty2) {
			const float temp = ty1;
			ty1 = ty2;
			ty2 = temp;
		}

		if (tz1 > tz2) {
			const float temp = tz1;
			tz1 = tz2;
			tz2 = temp;
		}

		float t1 = tx1; if (t1 < ty1) t1 = ty1; if (t1 < tz1) t1 = tz1;
		float t2 = tx2; if (t2 > ty2) t2 = ty2; if (t2 > tz2) t2 = tz2;

		if (t1 > t2) return false;
		if ((t1 < 0.0) && (t2 < 0.0)) return false;

		minHit.t = t1;
		return true;
	}
};




// triangle
struct Triangle {
	float3 positions[3];
	float3 normals[3];
	float2 texcoords[3];
	int idMaterial = 0;
	AABB bbox;
	float3 center;
};


bool gaussianElimination3(float3x4 sys, float3* result) {
	float4 r1 = sys.row(0);
	float4 r2 = sys.row(1);
	float4 r3 = sys.row(3);




	r2 = (r2[0] != 0) ? r1 - r2 * (r1[0] / r2[0]) : r2;
	if (r2[1] == 0) return false;
	r3 = (r3[0] != 0) ? (r1 - r3 * (r1[0] / r3[0])) : r3;
	r3 = (r3[1] != 0) ? r2 - r3 * (r2[1] / r3[1]) : r3;
	if (r3[2] == 0) return false;
	r3 = r3 / r3[2];
	r2 = (r2 * (r3[2] / r2[2]) - r3);
	r2 = r2 / r2[1];
	r1 = (r1 * (r3[2] / r1[2]) - r3);
	r1 = (r1 * r2[1] / r1[1] - r2);
	r1 = r1 / r1[0];
	//std::cout << r1 << std::endl << r2 << std::endl << r3 << std::endl;

	*result = { r1[3],r2[3],r3[3] };
	return true;
}

void gaussianTest() {
	float3x4 test = { {3,2,5},{2,3,-3},{-4,3,1},{3,15,14} };
	float3 res = { 0,0,0 };
	gaussianElimination3(test, &res);
	std::cout << res[0] << res[1] << res[2] << std::endl;
}
Photon DUMMY_P = Photon();
// triangle mesh
static float3 shade(const HitInfo& hit, const float3& viewDir, const int level = 0, const float incomingEta = 1.0f, Photon& pprev = DUMMY_P, Photon& p = DUMMY_P, bool MC = false);
class TriangleMesh {
public:
	std::vector<Triangle> triangles;
	std::vector<Material> materials;
	AABB bbox;

	void transform(const float4x4& m) {
		// ====== implement it if you want =====
		// matrix transformation of an object	
		// m is a matrix that transforms an object
		// implement proper transformation for positions and normals
		// (hint: you will need to have float4 versions of p and n)
		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			for (int k = 0; k <= 2; k++) {
				const float3 &p = this->triangles[i].positions[k];
				const float3 &n = this->triangles[i].normals[k]; 
				// not doing anything right now
			}
		}
	}

	

	bool checkInTriangle(const float3 a, const float3 b, const float3 c, const float3 d, bool print = false) const {
		if (print) std::cout << "checking" << std::endl;
		float3 normA = linalg::normalize(linalg::cross(a - c, a - b));
		float3 normB = linalg::normalize(linalg::cross(b - c, b - a));
		float3 normC = linalg::normalize(linalg::cross(c - a, c - b));
		if (linalg::dot(normA, linalg::normalize(linalg::cross(a - c, a - d))) < 0) return false;
		if (linalg::dot(normB, linalg::normalize(linalg::cross(b - c, b - d))) < 0) return false;
		if (linalg::dot(normC, linalg::normalize(linalg::cross(c - a, c - d))) < 0) return false;
		if (linalg::dot(-normA, linalg::normalize(linalg::cross(a - b, a - d))) < 0) return false;
		if (linalg::dot(-normB, linalg::normalize(linalg::cross(b - a, b - d))) < 0) return false;
		if (linalg::dot(-normC, linalg::normalize(linalg::cross(c - b, c - d))) < 0) return false;
		//std::cout << linalg::dot(linalg::normalize(linalg::cross(a - c, a - b)), linalg::normalize(linalg::cross(a - c, a - d))) << std::endl;
		return true;
	}

	float3 barymetric2D(float2 a, float2 b, float2 c, float2 d) const {
		float2x2 eq = { a - c, b - c };
		float2x2 x1eq = { d - c, b - c };
		float2x2 x2eq = { a - c, d - c };
		float detEq = linalg::determinant(eq);
		float alpha = linalg::determinant(x1eq) / detEq;
		float beta = linalg::determinant(x2eq) / detEq;
		float gamma = 1 - alpha - beta;
		return { alpha, beta, gamma };
	}


	void rasterizeTriangle(const Triangle& tri, const float4x4& plm) const {
		// rasterization of a triangle
		// "plm" should be a matrix that contains perspective projection and the camera matrix
		// you do not need to implement clipping
		// you may call the "shade" function to get the pixel value
		// (you may ignore viewDir for now)
		float4 a = { tri.positions[0],1 };
		float2 tex_a = tri.texcoords[0];
		float4 b = { tri.positions[1],1 };
		float2 tex_b = tri.texcoords[1];
		float4 c = { tri.positions[2],1 };
		float2 tex_c = tri.texcoords[2];
		float4 NDCA = linalg::mul(plm, a);
		float3 pixA = { (NDCA[0] / NDCA[3] + 1) * globalWidth / 2, (NDCA[1] / NDCA[3] + 1) * globalHeight / 2 , 0 };
		float W_A = 1 / NDCA[3];
		float Z_A = NDCA[2] * W_A;
		float2 interpolatedTex_A = tex_a * W_A;
		float4 NDCB = linalg::mul(plm, b);
		float3 pixB = { (NDCB[0] / NDCB[3] + 1) * globalWidth / 2, (NDCB[1] / NDCB[3] + 1) * globalHeight / 2, 0 };
		float W_B = 1 / NDCB[3];
		float2 interpolatedTex_B = tex_b * W_B;
		float Z_B = NDCB[2] * W_B;
		float4 NDCC = linalg::mul(plm, c);
		float3 pixC = { (NDCC[0] / NDCC[3] + 1) * globalWidth / 2, (NDCC[1] / NDCC[3] + 1) * globalHeight / 2, 0 };
		float W_C = 1 / NDCC[3];
		float2 interpolatedTex_C = tex_c * W_C;
		float Z_C = NDCC[2] * W_C; 

		//std::cout << interpolatedTex_A << " " << interpolatedTex_B << " " << interpolatedTex_C << " " << std::endl;
		//std::cout << "points in space" << a << " " << b << " " << c << " " << std::endl;
		//std::cout << "rasterized points" << pixA << " " << pixB << " " << pixC << " " << std::endl;
		//std::cout << NDCA[2] << " " << NDCA[3] << std::endl;
		//std::cout << NDCB[2] << " " << NDCB[3] << std::endl;
		//std::cout << NDCC[2] << " " << NDCC[3] << std::endl;
		//std::cout << Z_A << " " << Z_B << " " << Z_C << std::endl;
		//checkInTriangle(pixA, pixB, pixC, d, true);
		for (int i = 0; i < globalWidth; i++) {
			for (int j = 0; j < globalHeight; j++) {
				float3 coord = { i + 0.5f, j + 0.5f , 0 };
				if (checkInTriangle(pixA, pixB, pixC, coord)) {
					//if(FrameBuffer.depth(i,j) < Z_)
					float3 interpolateFactors = barymetric2D({ pixA[0], pixA[1] }, { pixB[0], pixB[1] }, { pixC[0], pixC[1] }, { i + 0.5f, j + 0.5f });
					float depth = interpolateFactors[0] * Z_A + interpolateFactors[1] * Z_B + interpolateFactors[2] * Z_C;
					float interpolatedW = interpolateFactors[0] * W_A + interpolateFactors[1] * W_B + interpolateFactors[2] * W_C;
					if (FrameBuffer.depth(i, j) < depth) {
						FrameBuffer.depth(i, j) = depth;
						if (materials[tri.idMaterial].textureHeight == 0) {
							FrameBuffer.pixel(i, j) = materials[tri.idMaterial].Kd;
							continue;
						}
						HitInfo hi;
						hi.T = (interpolateFactors[0] * interpolatedTex_A + interpolateFactors[1] * interpolatedTex_B + interpolateFactors[2] * interpolatedTex_C)/interpolatedW;
						hi.material = &materials[tri.idMaterial];
						FrameBuffer.pixel(i,j) = shade(hi, { 0,0,0 },-1);

					}

				}
				//else FrameBuffer.pixel(i, j) = float3(0.0f);

			}
		}
		//FrameBuffer.pixel(256, 0) = float3(0.0f, 0.0f, 1.0f);

		
		//if (FrameBuffer.valid(int(floorf(pixA[0])), int(floorf(pixA[1])))) FrameBuffer.pixel(int(floorf(pixA[0])), int(floorf(pixA[1]))) = float3(1.0f);
		//if (FrameBuffer.valid(int(floorf(pixB[0])), int(floorf(pixB[1])))) FrameBuffer.pixel(int(floorf(pixB[0])), int(floorf(pixB[1]))) = float3(1.0f);
		//if (FrameBuffer.valid(int(floorf(pixC[0])), int(floorf(pixC[1])))) FrameBuffer.pixel(int(floorf(pixC[0])), int(floorf(pixC[1]))) = float3(1.0f);

	}


	bool raytraceTriangle(HitInfo& result, const Ray& ray, const Triangle& tri, float tMin, float tMax) const {
		float3 o = ray.o;
		float3 d = ray.d;
		float3 a = tri.positions[0];
		float3 b = tri.positions[1];
		float3 c = tri.positions[2];
		float3x3 eq = { a - c, b - c, -d};
		//if no solution to system of equations, the line is parrallel to the surface, no hits
		//if (!gaussianElimination3(eq, &solution)) return false;
		float detEq = linalg::determinant(eq);
		float alpha = linalg::dot(linalg::cross(o - c, b - c), -d)/detEq;
		float beta = linalg::dot(linalg::cross(a - c, o - c), -d)/detEq;
		float t = linalg::dot(linalg::cross(a - c, b - c), o - c)/detEq;
		float3 geonormal = linalg::normalize(linalg::cross(b - a, c - a));
		if (linalg::dot(geonormal, d) > linalg::dot(-geonormal, d)) geonormal = -geonormal;
		
		//the point must be in the triangle
		float gamma = 1-alpha-beta;
		if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1 || gamma < 0 || gamma > 1) return false;
		//if the distance is negative, it is not a solution: camera cannot capture what is behind it.
		if (t < tMin || t > tMax) return false;
		
		float3 i = t * d + o;
		//float3 shadenormal = tri.normals[0];
		float3 shadev = (alpha * tri.normals[0] + beta * tri.normals[1] + gamma * tri.normals[2]);
		float3 shadenormal = linalg::normalize(shadev);
		if (linalg::dot(shadenormal, d) > linalg::dot(-shadenormal, d)) shadenormal = -shadenormal;
		float2 texnormal = alpha * tri.texcoords[0] + beta * tri.texcoords[1] + gamma * tri.texcoords[2];
		//std::cout << a << std::endl << b << std::endl << c << std::endl << o << std::endl << d << std::endl 
			//<< i << std::endl
			//<< alpha << std::endl << beta << std::endl << gamma << std::endl
			//<< shadenormal << std::endl << texnormal << std::endl  << t << std::endl;
		//std::cout << shadev << std::endl << shadenormal << std::endl;
		result.material = & TriangleMesh::materials.at(tri.idMaterial);
		result.N = shadenormal;
		result.P = i;
		result.T = texnormal;
		result.t = t;
		result.GN = geonormal;
		return true;
	}


	// some precalculation for bounding boxes (you do not need to change it)
	void preCalc() {
		bbox.reset();
		for (int i = 0, _n = (int)triangles.size(); i < _n; i++) {
			this->triangles[i].bbox.reset();
			this->triangles[i].bbox.fit(this->triangles[i].positions[0]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[1]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[2]);

			this->triangles[i].center = (this->triangles[i].positions[0] + this->triangles[i].positions[1] + this->triangles[i].positions[2]) * (1.0f / 3.0f);

			this->bbox.fit(this->triangles[i].positions[0]);
			this->bbox.fit(this->triangles[i].positions[1]);
			this->bbox.fit(this->triangles[i].positions[2]);
		}
	}


	// load .obj file (you do not need to modify it unless you want to change something)
	bool load(const char* filename, const float4x4& ctm = linalg::identity) {
		int nVertices = 0;
		float* vertices;
		float* normals;
		float* texcoords;
		int nIndices;
		int* indices;
		int* matid = nullptr;

		printf("Loading \"%s\"...\n", filename);
		ParseOBJ(filename, nVertices, &vertices, &normals, &texcoords, nIndices, &indices, &matid);
		if (nVertices == 0) return false;
		this->triangles.resize(nIndices / 3);

		if (matid != nullptr) {
			for (unsigned int i = 0; i < materials.size(); i++) {
				// convert .mlt data into BSDF definitions
				// you may change the followings in the final project if you want
				materials[i].type = MAT_LAMBERTIAN;
				if (materials[i].Ns == 100.0f) {
					materials[i].type = MAT_METAL;
				}
				if (materials[i].name.compare(0, 5, "glass", 0, 5) == 0) {
					materials[i].type = MAT_GLASS;
					materials[i].eta = 1.5f;
				}
			}
		} else {
			// use default Lambertian
			this->materials.resize(1);
		}

		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			const int v0 = indices[i * 3 + 0];
			const int v1 = indices[i * 3 + 1];
			const int v2 = indices[i * 3 + 2];

			this->triangles[i].positions[0] = float3(vertices[v0 * 3 + 0], vertices[v0 * 3 + 1], vertices[v0 * 3 + 2]);
			this->triangles[i].positions[1] = float3(vertices[v1 * 3 + 0], vertices[v1 * 3 + 1], vertices[v1 * 3 + 2]);
			this->triangles[i].positions[2] = float3(vertices[v2 * 3 + 0], vertices[v2 * 3 + 1], vertices[v2 * 3 + 2]);

			if (normals != nullptr) {
				this->triangles[i].normals[0] = float3(normals[v0 * 3 + 0], normals[v0 * 3 + 1], normals[v0 * 3 + 2]);
				this->triangles[i].normals[1] = float3(normals[v1 * 3 + 0], normals[v1 * 3 + 1], normals[v1 * 3 + 2]);
				this->triangles[i].normals[2] = float3(normals[v2 * 3 + 0], normals[v2 * 3 + 1], normals[v2 * 3 + 2]);
			} else {
				// no normal data, calculate the normal for a polygon
				const float3 e0 = this->triangles[i].positions[1] - this->triangles[i].positions[0];
				const float3 e1 = this->triangles[i].positions[2] - this->triangles[i].positions[0];
				const float3 n = normalize(cross(e0, e1));

				this->triangles[i].normals[0] = n;
				this->triangles[i].normals[1] = n;
				this->triangles[i].normals[2] = n;
			}

			// material id
			this->triangles[i].idMaterial = 0;
			if (matid != nullptr) {
				// read texture coordinates
				if ((texcoords != nullptr) && materials[matid[i]].isTextured) {
					this->triangles[i].texcoords[0] = float2(texcoords[v0 * 2 + 0], texcoords[v0 * 2 + 1]);
					this->triangles[i].texcoords[1] = float2(texcoords[v1 * 2 + 0], texcoords[v1 * 2 + 1]);
					this->triangles[i].texcoords[2] = float2(texcoords[v2 * 2 + 0], texcoords[v2 * 2 + 1]);
				} else {
					this->triangles[i].texcoords[0] = float2(0.0f);
					this->triangles[i].texcoords[1] = float2(0.0f);
					this->triangles[i].texcoords[2] = float2(0.0f);
				}
				this->triangles[i].idMaterial = matid[i];
			} else {
				this->triangles[i].texcoords[0] = float2(0.0f);
				this->triangles[i].texcoords[1] = float2(0.0f);
				this->triangles[i].texcoords[2] = float2(0.0f);
			}
		}
		printf("Loaded \"%s\" with %d triangles.\n", filename, int(triangles.size()));

		delete[] vertices;
		delete[] normals;
		delete[] texcoords;
		delete[] indices;
		delete[] matid;

		return true;
	}

	~TriangleMesh() {
		materials.clear();
		triangles.clear();
	} 


	bool bruteforceIntersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) {
		// bruteforce ray tracing (for debugging)
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		for (int i = 0; i < triangles.size(); ++i) {
			if (raytraceTriangle(tempMinHit, ray, triangles[i], tMin, tMax)) {
				if (tempMinHit.t < result.t) {
					hit = true;
					result = tempMinHit;
				}
			}
		}

		return hit;
	}

	void createSingleTriangle() {
		triangles.resize(1);
		materials.resize(1);

		triangles[0].idMaterial = 0;

		triangles[0].positions[0] = float3(-0.5f, -0.5f, 0.0f);
		triangles[0].positions[1] = float3(0.5f, -0.5f, 0.0f);
		triangles[0].positions[2] = float3(0.0f, 0.5f, 0.0f);

		const float3 e0 = this->triangles[0].positions[1] - this->triangles[0].positions[0];
		const float3 e1 = this->triangles[0].positions[2] - this->triangles[0].positions[0];
		const float3 n = normalize(cross(e0, e1));

		triangles[0].normals[0] = n;
		triangles[0].normals[1] = n;
		triangles[0].normals[2] = n;

		triangles[0].texcoords[0] = float2(0.0f, 0.0f);
		triangles[0].texcoords[1] = float2(0.0f, 1.0f);
		triangles[0].texcoords[2] = float2(1.0f, 0.0f);
	}


private:
	// === you do not need to modify the followings in this class ===
	void loadTexture(const char* fname, const int i) {
		int comp;
		materials[i].texture = stbi_load(fname, &materials[i].textureWidth, &materials[i].textureHeight, &comp, 3);
		if (!materials[i].texture) {
			std::cerr << "Unable to load texture: " << fname << std::endl;
			return;
		}
	}

	std::string GetBaseDir(const std::string& filepath) {
		if (filepath.find_last_of("/\\") != std::string::npos) return filepath.substr(0, filepath.find_last_of("/\\"));
		return "";
	}
	std::string base_dir;

	void LoadMTL(const std::string fileName) {
		FILE* fp = fopen(fileName.c_str(), "r");

		Material mtl;
		mtl.texture = nullptr;
		char line[81];
		while (fgets(line, 80, fp) != nullptr) {
			float r, g, b, s;
			std::string lineStr;
			lineStr = line;
			int i = int(materials.size());

			if (lineStr.compare(0, 6, "newmtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				mtl.name = lineStr;
				mtl.isTextured = false;
			} else if (lineStr.compare(0, 2, "Ka", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ka = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Kd", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Kd = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ks", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ks = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ns", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f\n", &s);
				mtl.Ns = s;
				mtl.texture = nullptr;
				materials.push_back(mtl);
			} else if (lineStr.compare(0, 6, "map_Kd", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				materials[i - 1].isTextured = true;
				loadTexture((base_dir + lineStr).c_str(), i - 1);
			}
		}

		fclose(fp);
	}

	void ParseOBJ(const char* fileName, int& nVertices, float** vertices, float** normals, float** texcoords, int& nIndices, int** indices, int** materialids) {
		// local function in C++...
		struct {
			void operator()(char* word, int* vindex, int* tindex, int* nindex) {
				const char* null = " ";
				char* ptr;
				const char* tp;
				const char* np;

				// by default, the texture and normal pointers are set to the null string
				tp = null;
				np = null;

				// replace slashes with null characters and cause tp and np to point
				// to character immediately following the first or second slash
				for (ptr = word; *ptr != '\0'; ptr++) {
					if (*ptr == '/') {
						if (tp == null) {
							tp = ptr + 1;
						} else {
							np = ptr + 1;
						}

						*ptr = '\0';
					}
				}

				*vindex = atoi(word);
				*tindex = atoi(tp);
				*nindex = atoi(np);
			}
		} get_indices;

		base_dir = GetBaseDir(fileName);
		#ifdef _WIN32
			base_dir += "\\";
		#else
			base_dir += "/";
		#endif

		FILE* fp = fopen(fileName, "r");
		int nv = 0, nn = 0, nf = 0, nt = 0;
		char line[81];
		if (!fp) {
			printf("Cannot open \"%s\" for reading\n", fileName);
			return;
		}

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (lineStr.compare(0, 6, "mtllib", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				LoadMTL(base_dir + lineStr);
			}

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					nn++;
				} else if (line[1] == 't') {
					nt++;
				} else {
					nv++;
				}
			} else if (line[0] == 'f') {
				nf++;
			}
		}
		fseek(fp, 0, 0);

		float* n = new float[3 * (nn > nf ? nn : nf)];
		float* v = new float[3 * nv];
		float* t = new float[2 * nt];

		int* vInd = new int[3 * nf];
		int* nInd = new int[3 * nf];
		int* tInd = new int[3 * nf];
		int* mInd = new int[nf];

		int nvertices = 0;
		int nnormals = 0;
		int ntexcoords = 0;
		int nindices = 0;
		int ntriangles = 0;
		bool noNormals = false;
		bool noTexCoords = false;
		bool noMaterials = true;
		int cmaterial = 0;

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					float x, y, z;
					sscanf(&line[2], "%f %f %f\n", &x, &y, &z);
					float l = sqrt(x * x + y * y + z * z);
					x = x / l;
					y = y / l;
					z = z / l;
					n[nnormals] = x;
					nnormals++;
					n[nnormals] = y;
					nnormals++;
					n[nnormals] = z;
					nnormals++;
				} else if (line[1] == 't') {
					float u, v;
					sscanf(&line[2], "%f %f\n", &u, &v);
					t[ntexcoords] = u;
					ntexcoords++;
					t[ntexcoords] = v;
					ntexcoords++;
				} else {
					float x, y, z;
					sscanf(&line[1], "%f %f %f\n", &x, &y, &z);
					v[nvertices] = x;
					nvertices++;
					v[nvertices] = y;
					nvertices++;
					v[nvertices] = z;
					nvertices++;
				}
			}
			if (lineStr.compare(0, 6, "usemtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				if (materials.size() != 0) {
					for (unsigned int i = 0; i < materials.size(); i++) {
						if (lineStr.compare(materials[i].name) == 0) {
							cmaterial = i;
							noMaterials = false;
							break;
						}
					}
				}

			} else if (line[0] == 'f') {
				char s1[32], s2[32], s3[32];
				int vI, tI, nI;
				sscanf(&line[1], "%s %s %s\n", s1, s2, s3);

				mInd[ntriangles] = cmaterial;

				// indices for first vertex
				get_indices(s1, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for second vertex
				get_indices(s2, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for third vertex
				get_indices(s3, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				ntriangles++;
			}
		}

		*vertices = new float[ntriangles * 9];
		if (!noNormals) {
			*normals = new float[ntriangles * 9];
		} else {
			*normals = 0;
		}

		if (!noTexCoords) {
			*texcoords = new float[ntriangles * 6];
		} else {
			*texcoords = 0;
		}

		if (!noMaterials) {
			*materialids = new int[ntriangles];
		} else {
			*materialids = 0;
		}

		*indices = new int[ntriangles * 3];
		nVertices = ntriangles * 3;
		nIndices = ntriangles * 3;

		for (int i = 0; i < ntriangles; i++) {
			if (!noMaterials) {
				(*materialids)[i] = mInd[i];
			}

			(*indices)[3 * i] = 3 * i;
			(*indices)[3 * i + 1] = 3 * i + 1;
			(*indices)[3 * i + 2] = 3 * i + 2;

			(*vertices)[9 * i] = v[3 * vInd[3 * i]];
			(*vertices)[9 * i + 1] = v[3 * vInd[3 * i] + 1];
			(*vertices)[9 * i + 2] = v[3 * vInd[3 * i] + 2];

			(*vertices)[9 * i + 3] = v[3 * vInd[3 * i + 1]];
			(*vertices)[9 * i + 4] = v[3 * vInd[3 * i + 1] + 1];
			(*vertices)[9 * i + 5] = v[3 * vInd[3 * i + 1] + 2];

			(*vertices)[9 * i + 6] = v[3 * vInd[3 * i + 2]];
			(*vertices)[9 * i + 7] = v[3 * vInd[3 * i + 2] + 1];
			(*vertices)[9 * i + 8] = v[3 * vInd[3 * i + 2] + 2];

			if (!noNormals) {
				(*normals)[9 * i] = n[3 * nInd[3 * i]];
				(*normals)[9 * i + 1] = n[3 * nInd[3 * i] + 1];
				(*normals)[9 * i + 2] = n[3 * nInd[3 * i] + 2];

				(*normals)[9 * i + 3] = n[3 * nInd[3 * i + 1]];
				(*normals)[9 * i + 4] = n[3 * nInd[3 * i + 1] + 1];
				(*normals)[9 * i + 5] = n[3 * nInd[3 * i + 1] + 2];

				(*normals)[9 * i + 6] = n[3 * nInd[3 * i + 2]];
				(*normals)[9 * i + 7] = n[3 * nInd[3 * i + 2] + 1];
				(*normals)[9 * i + 8] = n[3 * nInd[3 * i + 2] + 2];
			}

			if (!noTexCoords) {
				(*texcoords)[6 * i] = t[2 * tInd[3 * i]];
				(*texcoords)[6 * i + 1] = t[2 * tInd[3 * i] + 1];

				(*texcoords)[6 * i + 2] = t[2 * tInd[3 * i + 1]];
				(*texcoords)[6 * i + 3] = t[2 * tInd[3 * i + 1] + 1];

				(*texcoords)[6 * i + 4] = t[2 * tInd[3 * i + 2]];
				(*texcoords)[6 * i + 5] = t[2 * tInd[3 * i + 2] + 1];
			}

		}
		fclose(fp);

		delete[] n;
		delete[] v;
		delete[] t;
		delete[] nInd;
		delete[] vInd;
		delete[] tInd;
		delete[] mInd;
	}
};


// for FFT
class Complex {
	
	public:
		float real;
		float imaginary;
		Complex(float real, float imaginary): real(real), imaginary(imaginary) {}
		Complex(){}
		Complex operator+ (Complex const& other) {
			Complex res = Complex(real + other.real, imaginary + other.imaginary);
			return res;
		}

		Complex operator- (Complex const& other) {
			Complex res = Complex(real - other.real, imaginary - other.imaginary);
			return res;
		}

		Complex operator* (Complex const& other) {
			Complex res = Complex(real * other.real - imaginary * other.imaginary, real * other.imaginary + other.real * imaginary);
			return res;
		}

		Complex operator/ (float const& other) {
			Complex res = Complex(real / other, imaginary / other);
			return res;
		}

		Complex conjugate() {
			return Complex(real, -imaginary);
		}
		Complex sqr() {
			return *this * (*this);
		}
};

Complex complexSqrt(float x) {
	if (x < 0) {
		return Complex(0, std::sqrtf(-x));
	}
	return Complex(std::sqrtf(x), 0);
}


Complex* generateRoU(int n) {
	Complex* arr = new Complex[n];
	for (int i = 0; i < n; i++) {
		arr[i] = Complex(std::cosf(std::_Pi * 2 * i / n), std::sinf(std::_Pi * 2 * i / n));
	}
	return arr;
}


void FFT(Complex* arr, Complex* RoU, int length) {
	if (length == 1) {
		//std::cout << arr[0].real << ", " << arr[0].imaginary << std::endl;
		return;
	}
	Complex* arr1 = new Complex[length / 2];
	Complex* arr2 = new Complex[length / 2];
	Complex* RoU1 = new Complex[length / 2];
	Complex* RoU2 = new Complex[length / 2];
	for (int i = 0; i < length / 2; i++) {
		arr1[i] = arr[2 * i];
		arr2[i] = arr[2 * i + 1];
		RoU1[i] = RoU[i].sqr();
		//std::cout << i << ", " << RoU1[i].real << ", " << RoU1[i].imaginary << std::endl;
		RoU2[i] = RoU[i].sqr();
	}
	FFT(arr1, RoU1, length / 2);
	FFT(arr2, RoU2, length / 2);
	//std::cout << arr1[0].real << ", " << arr1[0].imaginary << std::endl;
	//std::cout << arr2[0].real << ", " << arr2[0].imaginary << std::endl;
	for (int i = 0; i < length / 2; i++) {
		Complex p = arr1[i];
		Complex q = arr2[i] * RoU[i];
		arr[i] = p + q;
		arr[i + length / 2] = p - q;
	}
	
	delete[] arr1;
	delete[] arr2;
	delete[] RoU1;
	delete[] RoU2;

}

void revFFT(Complex* arr, Complex* RoU, int length) {
	for (int i = 0; i < length; i++) {
		arr[i] = arr[i].conjugate();
	}
	FFT(arr, RoU, length);
	for (int i = 0; i < length; i++) {
		arr[i] = arr[i].conjugate();
		arr[i] = arr[i] / length;
	}
}




class BVHNode {
public:
	bool isLeaf;
	int idLeft, idRight;
	int triListNum;
	int* triList;
	AABB bbox;
};



class BVH {
public:
	const TriangleMesh* triangleMesh = nullptr;
	BVHNode* node = nullptr;

	const float costBBox = 1.0f;
	const float costTri = 1.0f;

	int leafNum = 0;
	int nodeNum = 0;

	BVH() {}
	void build(const TriangleMesh* mesh);

	bool intersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		// bvh
		if (this->node[0].bbox.intersect(tempMinHit, ray)) {
			hit = traverse(result, ray, 0, tMin, tMax);
		}
		if (result.t != FLT_MAX) hit = true;

		return hit;
	}
	bool traverse(HitInfo& result, const Ray& ray, int node_id, float tMin, float tMax) const;

private:
	void sortAxis(int* obj_index, const char axis, const int li, const int ri) const;
	int splitBVH(int* obj_index, const int obj_num, const AABB& bbox);

};


// sort bounding boxes (in case you want to build SAH-BVH)
void BVH::sortAxis(int* obj_index, const char axis, const int li, const int ri) const {
	int i, j;
	float pivot;
	int temp;

	i = li;
	j = ri;

	pivot = triangleMesh->triangles[obj_index[(li + ri) / 2]].center[axis];

	while (true) {
		while (triangleMesh->triangles[obj_index[i]].center[axis] < pivot) {
			++i;
		}

		while (triangleMesh->triangles[obj_index[j]].center[axis] > pivot) {
			--j;
		}

		if (i >= j) break;

		temp = obj_index[i];
		obj_index[i] = obj_index[j];
		obj_index[j] = temp;

		++i;
		--j;
	}

	if (li < (i - 1)) sortAxis(obj_index, axis, li, i - 1);
	if ((j + 1) < ri) sortAxis(obj_index, axis, j + 1, ri);
}


//#define SAHBVH // use this in once you have SAH-BVH
int BVH::splitBVH(int* obj_index, const int obj_num, const AABB& bbox) {
#ifndef SAHBVH
	int bestAxis, bestIndex;
	AABB bboxL, bboxR, bestbboxL, bestbboxR;
	int* sorted_obj_index = new int[obj_num];

	// split along the largest axis
	bestAxis = bbox.getLargestAxis();

	// sorting along the axis
	this->sortAxis(obj_index, bestAxis, 0, obj_num - 1);
	for (int i = 0; i < obj_num; ++i) {
		sorted_obj_index[i] = obj_index[i];
	}

	// split in the middle
	bestIndex = obj_num / 2 - 1;

	bboxL.reset();
	for (int i = 0; i <= bestIndex; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxL.fit(tri.positions[0]);
		bboxL.fit(tri.positions[1]);
		bboxL.fit(tri.positions[2]);
	}

	bboxR.reset();
	for (int i = bestIndex + 1; i < obj_num; ++i) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];
		bboxR.fit(tri.positions[0]);
		bboxR.fit(tri.positions[1]);
		bboxR.fit(tri.positions[2]);
	}

	bestbboxL = bboxL;
	bestbboxR = bboxR;
#else
	// implelement SAH-BVH here
#endif

	if (obj_num <= 4) {
		delete[] sorted_obj_index;

		this->nodeNum++;
		this->node[this->nodeNum - 1].bbox = bbox;
		this->node[this->nodeNum - 1].isLeaf = true;
		this->node[this->nodeNum - 1].triListNum = obj_num;
		this->node[this->nodeNum - 1].triList = new int[obj_num];
		for (int i = 0; i < obj_num; i++) {
			this->node[this->nodeNum - 1].triList[i] = obj_index[i];
		}
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->leafNum++;

		return temp_id;
	}
	else {
		// split obj_index into two 
		int* obj_indexL = new int[bestIndex + 1];
		int* obj_indexR = new int[obj_num - (bestIndex + 1)];
		for (int i = 0; i <= bestIndex; ++i) {
			obj_indexL[i] = sorted_obj_index[i];
		}
		for (int i = bestIndex + 1; i < obj_num; ++i) {
			obj_indexR[i - (bestIndex + 1)] = sorted_obj_index[i];
		}
		delete[] sorted_obj_index;
		int obj_numL = bestIndex + 1;
		int obj_numR = obj_num - (bestIndex + 1);

		// recursive call to build a tree
		this->nodeNum++;
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->node[temp_id].bbox = bbox;
		this->node[temp_id].isLeaf = false;
		this->node[temp_id].idLeft = splitBVH(obj_indexL, obj_numL, bestbboxL);
		this->node[temp_id].idRight = splitBVH(obj_indexR, obj_numR, bestbboxR);

		delete[] obj_indexL;
		delete[] obj_indexR;

		return temp_id;
	}
}


// you may keep this part as-is
void BVH::build(const TriangleMesh* mesh) {
	triangleMesh = mesh;

	// construct the bounding volume hierarchy
	const int obj_num = (int)(triangleMesh->triangles.size());
	int* obj_index = new int[obj_num];
	for (int i = 0; i < obj_num; ++i) {
		obj_index[i] = i;
	}
	this->nodeNum = 0;
	this->node = new BVHNode[obj_num * 2];
	this->leafNum = 0;

	// calculate a scene bounding box
	AABB bbox;
	for (int i = 0; i < obj_num; i++) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];

		bbox.fit(tri.positions[0]);
		bbox.fit(tri.positions[1]);
		bbox.fit(tri.positions[2]);
	}

	// ---------- buliding BVH ----------
	printf("Building BVH...\n");
	splitBVH(obj_index, obj_num, bbox);
	printf("Done.\n");

	delete[] obj_index;
}


// you may keep this part as-is
bool BVH::traverse(HitInfo& minHit, const Ray& ray, int node_id, float tMin, float tMax) const {
	bool hit = false;
	HitInfo tempMinHit, tempMinHitL, tempMinHitR;
	bool hit1, hit2;

	if (this->node[node_id].isLeaf) {
		for (int i = 0; i < (this->node[node_id].triListNum); ++i) {
			if (triangleMesh->raytraceTriangle(tempMinHit, ray, triangleMesh->triangles[this->node[node_id].triList[i]], tMin, tMax)) {
				hit = true;
				if (tempMinHit.t < minHit.t) minHit = tempMinHit;
			}
		}
	}
	else {
		hit1 = this->node[this->node[node_id].idLeft].bbox.intersect(tempMinHitL, ray);
		hit2 = this->node[this->node[node_id].idRight].bbox.intersect(tempMinHitR, ray);

		hit1 = hit1 && (tempMinHitL.t < minHit.t);
		hit2 = hit2 && (tempMinHitR.t < minHit.t);

		if (hit1 && hit2) {
			if (tempMinHitL.t < tempMinHitR.t) {
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
			}
			else {
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
			}
		}
		else if (hit1) {
			hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
		}
		else if (hit2) {
			hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
		}
	}

	return hit;
}









// fill in the missing parts
class Particle {
public:
	float3 position = float3(0.0f);
	float3 velocity = float3(0.0f);
	float3 prevPosition = position;
	float3 gravityconstant = globalGravity* deltaT * deltaT;
	void reset() {
		position = float3(PCG32::rand(), PCG32::rand(), PCG32::rand()) - float(0.5f);
		velocity = 10.0f * float3((PCG32::rand() - 0.5f), 0.0f, (PCG32::rand() - 0.5f));
		if (globalSphereCollision) {
			position = 0.5f * linalg::normalize(position);
		}
		prevPosition = position;
		position += velocity * deltaT;
		if (globalSphereCollision) {
			position = 0.5f * linalg::normalize(position);
		}
		//std::cout << position << std::endl;
	}

	void step() {
		float3 temp = position;
		//gravity
		position = position + (position - prevPosition) + gravityconstant;
		

		//simple collision
		if (globalSimpleCollision) {
			if (position.x < -0.5f) {

				temp = float3(-1 - temp.x + (0.5f + position.x), temp.y, temp.z);
				position = float3(-0.5f, position.y, position.z);
			}
			else if (position.x > 0.5f) {

				temp = float3(1 - temp.x + (-0.5f + position.x), temp.y, temp.z);
				position = float3(0.5f, position.y, position.z);

			}
			if (position.y < -0.5f) {

				temp = float3(temp.x, (0.5f + position.y) - 1 - temp.y, temp.z);
				position = float3(position.x, -0.5f, position.z);


			}
			else if (position.y > 0.5f) {

				temp = float3(temp.x, (-0.5f + position.y) + 1 - temp.y, temp.z);
				position = float3(position.x, 0.5f, position.z);

			}
			if (position.z < -0.5f) {

				temp = float3(temp.x, temp.y, -1 - temp.z + (0.5f + position.z));
				position = float3(position.x, position.y, -0.5f);

			}
			else if (position.z > 0.5f) {

				temp = float3(temp.x, temp.y, 1 - temp.z + (-0.5f + position.z));
				position = float3(position.x, position.y, 0.5f);

			}
		}
		
		if (globalSphereCollision) {
			position = 0.5f * position / linalg::length(position);
		}
		velocity = (position - temp) / deltaT;
		//std::cout << position << std::endl;
		// update the particle position and velocity here
		prevPosition = temp;
	}
	void step(float3 force) {
		float3 temp = position;
		//gravity
		position = position + (position - prevPosition) + force;


		//simple collision
		if (globalSimpleCollision) {
			if (position.x < -0.5f) {

				temp = float3(-1 - temp.x + (0.5f + position.x), temp.y, temp.z);
				position = float3(-0.5f, position.y, position.z);
			}
			else if (position.x > 0.5f) {

				temp = float3(1 - temp.x + (-0.5f + position.x), temp.y, temp.z);
				position = float3(0.5f, position.y, position.z);

			}
			if (position.y < -0.5f) {

				temp = float3(temp.x, (0.5f + position.y) - 1 - temp.y, temp.z);
				position = float3(position.x, -0.5f, position.z);


			}
			else if (position.y > 0.5f) {

				temp = float3(temp.x, (-0.5f + position.y) + 1 - temp.y, temp.z);
				position = float3(position.x, 0.5f, position.z);

			}
			if (position.z < -0.5f) {

				temp = float3(temp.x, temp.y, -1 - temp.z + (0.5f + position.z));
				position = float3(position.x, position.y, -0.5f);

			}
			else if (position.z > 0.5f) {

				temp = float3(temp.x, temp.y, 1 - temp.z + (-0.5f + position.z));
				position = float3(position.x, position.y, 0.5f);

			}
		}

		if (globalSphereCollision) {
			position = 0.5f * position / linalg::length(position);
		}
		velocity = (position - temp) / deltaT;

		// update the particle position and velocity here
		prevPosition = temp;
	}
};


class ParticleSystem {
public:
	std::vector<Particle> particles;
	TriangleMesh particlesMesh;
	TriangleMesh sphere;
	const char* sphereMeshFilePath = NULL;
	
	float sphereSize = 0.0f;
	ParticleSystem() {};
	const float GravationalConstant = 0.00001;//6.674E-11;

	void updateMesh() {
		// you can optionally update the other mesh information (e.g., bounding box, BVH - which is tricky)
		if (sphereSize > 0) {
			const int n = int(sphere.triangles.size());
			for (int i = 0; i < globalNumParticles; i++) {
				for (int j = 0; j < n; j++) {
					particlesMesh.triangles[i * n + j].positions[0] = sphere.triangles[j].positions[0] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[1] = sphere.triangles[j].positions[1] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[2] = sphere.triangles[j].positions[2] + particles[i].position;
					particlesMesh.triangles[i * n + j].normals[0] = sphere.triangles[j].normals[0];
					particlesMesh.triangles[i * n + j].normals[1] = sphere.triangles[j].normals[1];
					particlesMesh.triangles[i * n + j].normals[2] = sphere.triangles[j].normals[2];
				}
			}
		} else {
			const float particleSize = 0.005f;
			for (int i = 0; i < globalNumParticles; i++) {
				// facing toward the camera
				particlesMesh.triangles[i].positions[0] = particles[i].position;
				particlesMesh.triangles[i].positions[1] = particles[i].position + particleSize * globalUp;
				particlesMesh.triangles[i].positions[2] = particles[i].position + particleSize * globalRight;
				particlesMesh.triangles[i].normals[0] = -globalViewDir;
				particlesMesh.triangles[i].normals[1] = -globalViewDir;
				particlesMesh.triangles[i].normals[2] = -globalViewDir;
			}
		}
	}

	void initialize() {
		particles.resize(globalNumParticles);
		particlesMesh.materials.resize(1);
		for (int i = 0; i < globalNumParticles; i++) {
			particles[i].reset();
		}

		if (sphereMeshFilePath) {
			if (sphere.load(sphereMeshFilePath)) {
				particlesMesh.triangles.resize(sphere.triangles.size() * globalNumParticles);
				sphere.preCalc();
				sphereSize = sphere.bbox.get_size().x * 0.5f;
			} else {
				particlesMesh.triangles.resize(globalNumParticles);
			}
		} else {
			particlesMesh.triangles.resize(globalNumParticles);
		}
		updateMesh();
	}

	void step() {
		// add some particle-particle interaction here
		// spherical particles can be implemented here
		for (int i = 0; i < globalNumParticles; i++) {
			if (globalGravitationalField) {
				float3 force = float3(0.0f);
				for (int j = 0; j < globalNumParticles; j++) {
					if (j == i) continue;
					float3 diff = particles[j].position - particles[i].position;
					force += GravationalConstant * diff / (linalg::length2(diff) * linalg::length(diff));
				}
				particles[i].step(force);
			}

			else {
				particles[i].step();
				//std::cout << "stepped" << particles[i].position << std::endl;
			}

		}


		if (globalVolumetricParticle) {
			bool noCollision = false;
			//std::cout << noCollision << std::endl;
			while (!noCollision) {
				noCollision = true;
				for (int i = 0; i < globalNumParticles; i++) {
					for (int j = 0; j < globalNumParticles; j++) {
						if (j == i) continue;
						float3 diff = particles[i].position - particles[j].position;
						if (linalg::length(diff) < 2 * sphereSize) {
							//std::cout << "collided" << i << ", " << j << std::endl;
							noCollision = false;
							float3 oldVi = linalg::normalize(particles[i].velocity);
							float3 oldVj = linalg::normalize(particles[j].velocity);
							float3 newVi = linalg::normalize(2 * (linalg::dot((-oldVi), diff) * diff) + oldVi);
							float3 newVj = linalg::normalize(2 * (linalg::dot((-oldVj), -diff) * (-diff)) + oldVj);
							float3 offset = (linalg::normalize(diff) * (2 * sphereSize - linalg::length(diff)) / 2);
							particles[i].position += offset;
							particles[j].position -= offset;
							particles[i].prevPosition = particles[i].position - deltaT * linalg::length(particles[i].velocity) * newVi;
							particles[j].prevPosition = particles[j].position - deltaT * linalg::length(particles[j].velocity) * newVj;
							particles[i].velocity = linalg::length(particles[i].velocity) * newVi;
							particles[j].velocity = linalg::length(particles[j].velocity) * newVj;
						}
					}
				}
				//std::cout << noCollision << std::endl;
			}
		}
		
		//std::cout << "updating mesh" << std::endl;
		updateMesh();
	}
};
static ParticleSystem globalParticleSystem;








// scene definition
// The function EmitPhoton() and Raytrace() will be called once each for the final project
class Scene {
public:
	std::vector<TriangleMesh*> objects;
	std::vector<PointLightSource*> pointLightSources;
	std::vector<BVH> bvhs;

	//for Final Project
	std::vector<Photon> photons;
	std::vector<Photon> causticPhotons;
	KDTree* kdt; 
	KDTree* causticKdt;
	float** height;
	int step;
	float stepLength;
	float stepWidth;
	float3 origin;


	Image image;
	void addObject(TriangleMesh* pObj) {
		objects.push_back(pObj);
		if (image.height == 0 && globalEnvironmentImage) {
			image.load(pathToImage);
		}
		std::cout << "there are currently " << objects.size() << " meshes loaded" << std::endl;
	}
	void addLight(PointLightSource* pObj) {
		pointLightSources.push_back(pObj);
	}

	void preCalc() {
		bvhs.resize(objects.size());
		for (int i = 0; i < objects.size(); i++) {
			objects[i]->preCalc();
			bvhs[i].build(objects[i]);
		}
	}

	// ray-scene intersection
	bool intersect(HitInfo& minHit, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		minHit.t = FLT_MAX;

		for (int i = 0, i_n = (int)objects.size(); i < i_n; i++) {
			//if (objects[i]->bruteforceIntersect(tempMinHit, ray, tMin, tMax)) { // for debugging
			if (bvhs[i].intersect(tempMinHit, ray, tMin, tMax)) {
				if (tempMinHit.t < minHit.t) {
					hit = true;
					minHit = tempMinHit;
				}
			}
		}
		return hit;
	}

	float4x4 perspectiveMatrix(float fovy, float aspect, float zNear, float zFar) const {
		float4x4 m;
		const float f = 1.0f / (tan(fovy * DegToRad / 2.0f));
		m[0] = { f / aspect, 0.0f, 0.0f, 0.0f };
		m[1] = { 0.0f, f, 0.0f, 0.0f };
		m[2] = { 0.0f, 0.0f, 1.0f / (zNear - zFar), -1.0f };
		m[3] = { 0.0f, 0.0f, -zFar/ (zNear - zFar), 0.0f };

		return m;
	}

	float4x4 lookatMatrix(const float3& _eye, const float3& _center, const float3& _up) const {
		// transformation to the camera coordinate
		float4x4 m;
		const float3 f = normalize(_center - _eye);
		const float3 upp = normalize(_up);
		const float3 s = normalize(cross(f, upp));
		const float3 u = cross(s, f);

		m[0] = { s.x, s.y, s.z, 0.0f };
		m[1] = { u.x, u.y, u.z, 0.0f };
		m[2] = { -f.x, -f.y, -f.z, 0.0f };
		m[3] = { 0.0f, 0.0f, 0.0f, 1.0f };
		m = transpose(m);

		// translation according to the camera location
		const float4x4 t = float4x4{ {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, { -_eye.x, -_eye.y, -_eye.z, 1.0f} };

		m = mul(m, t);
		return m;
	}

	// rasterizer
	void Rasterize() const {
		// fill in plm by a proper matrix
		const float4x4 pm = perspectiveMatrix(globalFOV, globalAspectRatio, globalDepthMin, globalDepthMax);
		const float4x4 lm = lookatMatrix(globalEye, globalLookat, globalUp);
		//globalDepthMin, globalDepthMax
		const float4x4 plm = mul(pm, lm);
		//std::cout << "plm" << plm << std::endl;

		FrameBuffer.clear();
		for (int n = 0, n_n = (int)objects.size(); n < n_n; n++) {
			for (int k = 0, k_n = (int)objects[n]->triangles.size(); k < k_n; k++) {
				objects[n]->rasterizeTriangle(objects[n]->triangles[k], plm);
			}
		}
	}

	Ray eyeRay(int x, int y) const {
		// compute the camera coordinate system 
		const float3 wDir = normalize(float3(-globalViewDir));
		const float3 uDir = normalize(cross(globalUp, wDir));
		const float3 vDir = cross(wDir, uDir);

		// compute the pixel location in the world coordinate system using the camera coordinate system
		// trace a ray through the center of each pixel
		const float imPlaneUPos = (x + 0.5f) / float(globalWidth) - 0.5f;
		const float imPlaneVPos = (y + 0.5f) / float(globalHeight) - 0.5f;

		const float3 pixelPos = globalEye + float(globalAspectRatio * globalFilmSize * imPlaneUPos) * uDir + float(globalFilmSize * imPlaneVPos) * vDir - globalDistanceToFilm * wDir;

		return Ray(globalEye, normalize(pixelPos - globalEye));
	}

	void Raytrace()  {
		FrameBuffer.clear();
		// loop over all pixels in the image
		for (int j = 0; j < globalHeight; ++j) {
			for (int i = 0; i < globalWidth; ++i) {
				const Ray ray = eyeRay(i, j);
				HitInfo hitInfo;
				if (intersect(hitInfo, ray)) {
					FrameBuffer.pixel(i, j) = shade(hitInfo, -ray.d);
				} else {
					if (globalEnvironmentImage) {
						float r = (1 / PI) * acosf(ray.d[2]) / sqrt(ray.d[0] * ray.d[0] + ray.d[1] * ray.d[1]);
						float u = ray.d[0] * r;
						float v = ray.d[1] * r;
						int pixx = static_cast<int>((u + 1) * this->image.width / 2.0f);
						int pixy = static_cast<int>((v + 1) * this->image.height / 2.0f);
						float3 renderedPixel = image.pixelConst(pixx, pixy);
						FrameBuffer.pixel(i, j) = renderedPixel;
					}
					else{
						FrameBuffer.pixel(i,j) = float3(0.0f);
					}
				}
			}

			// show intermediate process
			if (globalShowRaytraceProgress) {
				constexpr int scanlineNum = 64;
				if ((j % scanlineNum) == (scanlineNum - 1)) {
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0]);
					glRecti(1, 1, -1, -1);
					glfwSwapBuffers(globalGLFWindow);
					printf("Rendering Progress: %.3f%%\r", j / float(globalHeight - 1) * 100.0f);
					fflush(stdout);
				}
			}
		}
	}

	bool PhotonIntersect(HitInfo& minHit, const Photon& p, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		minHit.t = FLT_MAX;
		const Ray r = Ray(p.position, p.direction);
		for (int i = 0, i_n = (int)objects.size(); i < i_n; i++) {
			//if (objects[i]->bruteforceIntersect(tempMinHit, ray, tMin, tMax)) { // for debugging
			if (bvhs[i].intersect(tempMinHit, r, tMin, tMax)) {
				if (tempMinHit.t < minHit.t) {
					hit = true;
					minHit = tempMinHit;
				}
			}
		}
		return hit;
	}
	


	bool PhotonTrace(Photon p) {
		HitInfo hi;
		hi.photon = true;
		bool intersected = PhotonIntersect(hi, p);
		

		if (intersected) {
			/*
			std::cout << hi.P << std::endl;
			std::cout << hi.material->name << std::endl;
			std::cout << hi.material->type << std::endl;
			std::cout << hi.material->Kd << std::endl;*/
			p.position = hi.P;
			
			photons.push_back(p);
			Photon np = Photon();
			
			hi.photon = true;
			//std::cout << "photon hit: " << hi.photon << std::endl;
			shade(hi, p.direction, 0, 1.0F, p, np);
		}
		return intersected;
	}


	void EmitPhoton() {
		std::random_device rd;
		std::mt19937 gen(rd()); 
		std::uniform_real_distribution<> udist(0.0, 1.0);
		std::normal_distribution<> gdist(0.0, 1.0);
		//Assume there is only one lightsource for now
		/*
		float3 photonIllum = pointLightSources[0]->wattage / NPHOTON;
		float3 lsPos = pointLightSources[0]->position;
		//uniformly distributed points on a sphere:
		for (int i = 0; i < NPHOTON; i++) {
			float theta = udist(gen) * 2 * std::_Pi;
			float u = udist(gen) * 2 - 1;
			float3 dir = float3(
				std::sqrtf(1 - u * u) * std::cosf(theta),
				std::sqrtf(1 - u * u) * std::sinf(theta),
				u
			);
			//std::cout << u << std::endl;
			//dir = float3(0.0, -1, 0.0);
			Photon p = Photon();
			p.direction = dir;
			p.flux = photonIllum;
			p.position = lspos;
			if (!PhotonTrace(p)) {
				i--;
			}
		}*/
		float totalWattage = 0;
		for (auto it = pointLightSources.begin(); it != pointLightSources.end(); it++) {
			totalWattage += ((*it)->wattage.x + (*it)->wattage.y + (*it)->wattage.z);
		}
		float photonIllum = totalWattage / NPHOTON;
		int numLS = pointLightSources.size();
		int nphotonUnAllocated = NPHOTON;
		for (int j = 0; j < numLS - 1; j++) {
			int num = (int)((pointLightSources[j]->wattage[0] + pointLightSources[j]->wattage[1] + pointLightSources[j]->wattage[2]) / photonIllum);
			float3 thisPhotonIllum = pointLightSources[j]->wattage / num;
			for (int i = 0; i < num; i++) {
				float theta = udist(gen) * 2 * std::_Pi;
				float u = udist(gen) * 2 - 1;
				float3 dir = float3(
					std::sqrtf(1 - u * u) * std::cosf(theta),
					std::sqrtf(1 - u * u) * std::sinf(theta),
					u
				);
				//std::cout << u << std::endl;
				//dir = float3(0.0, -1, 0.0);
				Photon p = Photon();
				p.direction = dir;
				p.flux = thisPhotonIllum;
				p.position = pointLightSources[j]->position;
				if (!PhotonTrace(p)) {
					i--;
				}
			}
			std::cout << (pointLightSources[j]->wattage) << std::endl;
			std::cout << num << std::endl;
			std::cout << thisPhotonIllum << std::endl;
			nphotonUnAllocated -= num;
		}
		int num = nphotonUnAllocated;
		float3 thisPhotonIllum = pointLightSources[numLS - 1]->wattage / num;
		for (int i = 0; i < num; i++) {
			float theta = udist(gen) * 2 * std::_Pi;
			float u = udist(gen) * 2 - 1;
			float3 dir = float3(
				std::sqrtf(1 - u * u) * std::cosf(theta),
				std::sqrtf(1 - u * u) * std::sinf(theta),
				u
			);
			//std::cout << u << std::endl;
			//dir = float3(0.0, -1, 0.0);
			Photon p = Photon();
			p.direction = dir;
			p.flux = thisPhotonIllum;
			p.position = pointLightSources[numLS - 1]->position;
			if (!PhotonTrace(p)) {
				i--;
			}
		}
		std::cout << (pointLightSources[numLS - 1]->wattage) << std::endl;
		std::cout << num << std::endl;
		std::cout << thisPhotonIllum << std::endl;


		
		
		std::cout << "recorded " << photons.size() << "photons" << std::endl;
		kdt = new KDTree(photons);
		causticKdt = new KDTree(causticPhotons);
	}

	float3 bruteforceRadianceEstimate(const HitInfo& hi) const {
		const float3 position = hi.P;
		const float radius = 0.01f;
		float3 radiance = float3(0.0f); // if no photons are nearby, the radiance will be zero
		std::vector<Photon> closePhotons;
		for (auto it = photons.begin(); it != photons.end(); it++) {
			if (linalg::length(it->position - position) < radius) {
				closePhotons.push_back(*it);
			}
		}
		float totalConeFilterWeight = 0.0f; // we sum the cone filter weights to later normalize radiance to reduce the noise
		int i = 0;
		float k = 2;
		for (const auto& photon : closePhotons) {
			float weight = 1.0f - linalg::length(position - photon.position) / (k * radius); // cone filter weight of the photon
			radiance += photon.flux * weight * hi.material->BRDF(float3(), float3(), float3());
			totalConeFilterWeight += weight;
			i++;
		}

		//std::cout << "photons: " << i << std::endl;
		if (totalConeFilterWeight > 0.0f) radiance /= totalConeFilterWeight;
		const float coneFilterArea = std::_Pi * radius * radius;
		// divide by the area of the cone filter to get the estimated radiance around the position the photon hits
		return float3(radiance.x / coneFilterArea, radiance.y / coneFilterArea, radiance.z / coneFilterArea);
	}

	float3 radianceEstimate(const HitInfo& hi) const {
		const float3 position = hi.P;
		const float radius = 0.005f;
		float3 radiance = float3(0.0f); // if no photons are nearby, the radiance will be zero
		const std::vector<Photon> closePhotons = kdt->nearbyPhotons(position, radius);
		float totalConeFilterWeight = 0.0f; // we sum the cone filter weights to later normalize radiance to reduce the noise
		int i = 0;
		float k = 1;
		for (const auto& photon : closePhotons) {
			float weight = 1.0f - linalg::length(position - photon.position) /(k * radius); // cone filter weight of the photon
			radiance += photon.flux * weight * hi.material->BRDF(float3(), float3(), float3());
			totalConeFilterWeight += weight;
			i++;
		}

		//std::cout << "photons: " << i << std::endl;
		if (totalConeFilterWeight > 0.0f) radiance /= totalConeFilterWeight;
		const float coneFilterArea = std::_Pi * radius * radius;
		// divide by the area of the cone filter to get the estimated radiance around the position the photon hits
		return float3(radiance.x / coneFilterArea, radiance.y / coneFilterArea, radiance.z / coneFilterArea);
	}

	~Scene() {
		delete kdt;
		delete causticKdt;
		for (int i = 0; i < step; i++) {
			delete[] height[i];
		}
		delete[] height;
	}

};
static Scene globalScene;








//If hit.photon is set to true, p is used to store the new found photons. 
//We assume NO specular reflections happen on lambertian surfaces, and NO diffusing reflections happen on specular surfaces

static float3 shade(const HitInfo& hit, const float3& viewDir, const int level, const float incomingEta, Photon& pprev, Photon& p, bool mc) {
	
	int MAXRECURSE = 6;
	if (!hit.photon) {
		MAXRECURSE = 6;
	}
	float LOWERBOUND = 0.06f;
	//std::cout << "photon: " << hit.photon << std::endl;
	if (globalScene.image.height == 0) {
		globalScene.image.load(pathToImage);
		//std::cout << globalScene.image.width << ", " << globalScene.image.height << std::endl;
	}
	if (level == -1) {
		float3 pos = hit.P;
	}
	if (level > MAXRECURSE) {
		return float3();
	}


	//return hit.material->fetchTexture(hit.T);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> udist(0.0, 1.0);
	std::normal_distribution<> gdist(0.0, 1.0);
	float3 kd = hit.material->Kd;
	float3 ks = hit.material->Ks;


	
	
	Image i2;
	if (hit.material->type == MAT_LAMBERTIAN) {
		//std::cout << hit.material->name << std::endl;
		float3 L = float3(0.0f);
		float3 brdf, irradiance;
		// loop over all of the point light sources
		if (hit.photon) {
			if (level > MAXRECURSE) return float3();
			//roll dice for russian roulette
			
			float pd = std::max({kd[0], kd[1], kd[2]});
			float dice = udist(gen);
			if (dice < pd) {
				//absorbed
				return float3();
			}

			//determine the random direction the photon is going after diffusion

			float theta = udist(gen) * 2 * std::_Pi;
			float u = udist(gen) * 2 - 1;
			float3 dir = float3(
				std::sqrtf(1 - u * u) * std::cosf(theta),
				std::sqrtf(1 - u * u) * std::sinf(theta),
				u
			);
			float3 norm = hit.N;
			if (linalg::dot(norm, viewDir) < 0) {
				norm = -norm;
			}
			while (linalg::dot(norm, dir) < 0) {
				theta = udist(gen) * 2 * std::_Pi;
				u = udist(gen) * 2 - 1;
				dir = float3(
					std::sqrtf(1 - u * u) * std::cosf(theta),
					std::sqrtf(1 - u * u) * std::sinf(theta),
					u
				);
			}
			//std::cout << hit.material->name << hit.material->Kd << std::endl;
			//std::cout << pprev.flux << std::endl;
			float3 pRGB = float3(pprev.flux[0] * kd[0] / pd, pprev.flux[1] * kd[1] / pd, pprev.flux[2] * kd[2] / pd);
			//std::cout << pRGB << std::endl;
			p.caustic = pprev.caustic;
			p.direction = dir;
			p.flux = pRGB;
			p.position = hit.P;
			HitInfo nhit;
			nhit.photon = true;
			
			bool intersected = globalScene.PhotonIntersect(nhit, p);
			if (!intersected) return float3();
			nhit.photon = true;
			p.position = nhit.P;
			globalScene.photons.push_back(p);
			//std::cout << p.flux << std::endl;
			if (p.caustic) globalScene.causticPhotons.push_back(p);
			Photon np = Photon();
			return shade(nhit, -dir, level + 1, incomingEta, p, np);
		}
		float3 causticL = float3();
		float3 outgoingDir = linalg::normalize(2 * (linalg::dot(viewDir, hit.N) * hit.N) - viewDir);
		if (globalRenderType == RENDER_PHOTON) {
			float3 np = hit.P + LOWERBOUND * hit.GN;
			causticL = globalScene.radianceEstimate(hit);
			//causticL = globalScene.bruteforceRadianceEstimate(hit);
			if(globalPhotonMap) return causticL;
			//std::cout << L << std::endl;
			


		}
		for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
			float3 np = hit.P + LOWERBOUND * hit.GN;
			float3 l = globalScene.pointLightSources[i]->position - np;
			
			float d = linalg::distance(globalScene.pointLightSources[i]->position, np);
			if (globalShadow) {
				bool obscured = false;
				for (int j = 0; j < globalScene.objects.size(); j++) {
					TriangleMesh* obj = globalScene.objects[j];
					BVH bvh = globalScene.bvhs[j];
					HitInfo hi;
					Ray r = Ray(np, linalg::normalize(l));
					if (bvh.intersect(hi, r)) {
						if (hi.t < d) {
							obscured = true;
							break;
						}
					}
				}
				if (obscured) continue;
			}
			// the inverse-squared falloff
			const float falloff = length2(l);

			// normalize the light direction
			l /= sqrtf(falloff);

			// get the irradiance
			irradiance = float(std::max(0.0f, dot(hit.N, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;
			brdf = hit.material->BRDF(l, viewDir, hit.N);

			if (hit.material->isTextured) {
				brdf *= hit.material->fetchTexture(hit.T);
			}
			if (!globalPointLightSource) {
				return brdf * PI; //debug output
			}

			L += irradiance * brdf;
		}
		return (L + causticL) / 2;
	} 
	else if (hit.material->type == MAT_METAL) {
		
		float3 np = hit.P + LOWERBOUND * hit.GN;
		float3 KS = hit.material->Ks;
		float pd = std::max({ ks[0], ks[1], ks[2] });
		float dice = udist(gen);
		if (dice < pd) {
			//absorbed
			return float3();
		}
		float3 pRGB = float3(pprev.flux[0] * ks[0] / pd, pprev.flux[1] * ks[1] / pd, pprev.flux[2] * ks[2] / pd);



		float3 outgoingDir = linalg::normalize(2 * (linalg::dot(viewDir, hit.N) * hit.N) - viewDir);
		
		
		if (level == MAXRECURSE) {
			return float3(0.0f);
		}
		HitInfo minhi;
		minhi.t = -1;

		if (hit.photon) {
			HitInfo nhi;
			nhi.photon = true;
			p.direction = outgoingDir;
			p.flux = pprev.flux;
			p.caustic = true;
			bool intersected = globalScene.PhotonIntersect(nhi, p);
			if (!intersected) return float3();
			p.position = nhi.P;
			nhi.photon = true;
			Photon np = Photon();
			return shade(nhi, -outgoingDir, level + 1, incomingEta, p, np);
		}
		for (int j = 0; j < globalScene.bvhs.size(); j++) {

			TriangleMesh* obj = globalScene.objects[j];
			BVH bvh = globalScene.bvhs[j];
			HitInfo hi;
			Ray r = Ray(np, outgoingDir);
			if (bvh.intersect(hi,r)) {
				if (minhi.t == -1 || minhi.t > hi.t) {
					minhi = hi;
				}
			}
		}
		if (minhi.t == -1) {
			if (globalEnvironmentImage) {
				float r = (1 / PI) * acosf(outgoingDir[2]) / sqrt(outgoingDir[0] * outgoingDir[0] + outgoingDir[1] * outgoingDir[1]);
				float u = outgoingDir[0] * r;
				float v = outgoingDir[1] * r;
				int pixx = static_cast<int>((u + 1) * globalScene.image.width / 2.0f);
				int pixy = static_cast<int>((v + 1) * globalScene.image.height / 2.0f);
				//std::cout << u << ", " << v << std::endl;
				//std::cout << pixx << ", " << pixy << std::endl;
				return globalScene.image.pixel(pixx, pixy);
			}
			else {
				return float3(0.0f);
			}
		}
		return shade(minhi, -outgoingDir, level + 1);
		
		//return float3(0.0f); // replace this
	} 
	else if (hit.material->type == MAT_GLASS) {
		/*
		float ps = std::max({ ks[0], ks[1], ks[2] });
		float dice = udist(gen);
		if (dice < ps) {
			//absorbed
			return float3();
		}
		float3 pRGB = float3(pprev.flux[0] * ks[0] / ps, pprev.flux[1] * ks[1] / ps, pprev.flux[2] * ks[2] / ps);
		*/

		//std::cout << level << std::endl;
		float3 np = hit.P + LOWERBOUND * hit.GN;
		float3 refractnp = hit.P - LOWERBOUND * hit.GN;
		float3 KS = hit.material->Ks;
		float3 KD = hit.material->Ka;
		float eta = hit.material->eta;
		
		float nextEta = eta;
		
		if (incomingEta == eta ) {
			int posX = (int)((np.x - globalScene.origin.x) / globalScene.stepWidth);
			int posZ = (int)((np.z - globalScene.origin.z) / globalScene.stepLength);
			if (waterGeneration && posX >= 0 && posZ >= 0 && posX < globalScene.step && posX < globalScene.step) {
				float y = globalScene.height[posX][posZ];
				if (np.y < y) {
					//std::cout << "still in water" << std::endl;
					nextEta = 1.33F;
					
				}
				else {
					nextEta = 1.0F;
				}
			}
			else {
				nextEta = 1.0F;
			}
		}
		//std::cout << nextEta << std::endl;
		//check total internal reflections
		float tir = 1 - (incomingEta / nextEta) * (incomingEta / nextEta)*(1 - linalg::dot((-viewDir), hit.N) * linalg::dot((-viewDir), hit.N));
		if(tir < 0) tir = 1 - (incomingEta / nextEta) * (incomingEta / nextEta) * (1 - linalg::dot((-viewDir), hit.GN) * linalg::dot((-viewDir), hit.GN));
		float3 outgoingDir = linalg::normalize(2 * (linalg::dot(viewDir, hit.GN) * hit.GN) - viewDir);
		if (level == MAXRECURSE) {
			return float3(0.0f);
		}
		HitInfo minreflecthi;
		HitInfo minrefracthi;
		minreflecthi.t = -1;
		minrefracthi.t = -1;
		Ray r = Ray(np, outgoingDir);
		Ray refractr;
		float3 dir;
		if (tir >= 0) {
			dir = (incomingEta / nextEta) * (-viewDir - linalg::dot(-viewDir, hit.N) * hit.N) - sqrtf(tir) * hit.N;
			dir = linalg::normalize(dir);
			refractr = Ray(refractnp, dir);
		}
		float R = 1;
		// implement fresenal:
		if (tir >= 0) {
			float cosi = linalg::dot(outgoingDir, hit.N);
			float coso = linalg::dot(dir, -hit.N);
			float r1 = (cosi * incomingEta - coso * nextEta) / (cosi * incomingEta + coso * nextEta);
			float r2 = (coso * incomingEta - cosi * nextEta) / (coso * incomingEta + cosi * nextEta);
			R = (r1 * r1 + r2 * r2) / 2;
		}
		
		



		//processing for photon
		if (hit.photon) {
			float dice = udist(gen);
			//If refraction
			if (dice > R) {
				
				HitInfo nhi;
				nhi.photon = true;
				p.direction = dir;
				p.flux = pprev.flux;
				p.caustic = true;
				p.position = np;
				bool intersected = globalScene.PhotonIntersect(nhi, p);
				if (!intersected) return float3();
				
				p.position = nhi.P;
				//std::cout << p.position << std::endl;
				nhi.photon = true;
				Photon np2 = Photon();
				return shade(nhi, -dir, level + 1, nextEta, p, np2); 
			}
			//If refraction
			else {
				HitInfo nhi;
				nhi.photon = true;
				p.direction = outgoingDir;
				p.flux = pprev.flux;
				p.caustic = true;
				p.position = refractnp;
				bool intersected = globalScene.PhotonIntersect(nhi, p);
				if (!intersected) return float3();
				p.position = nhi.P;
				//std::cout << p.position << std::endl;
				nhi.photon = true;
				Photon np2 = Photon();
				return shade(nhi, -outgoingDir, level + 1, incomingEta, p, np2);
			}
		}
		


		
		for (int j = 0; j < globalScene.objects.size(); j++) {

			TriangleMesh* obj = globalScene.objects[j];
			BVH bvh = globalScene.bvhs[j];
			HitInfo hi;

			
			if (bvh.intersect(hi, r)) {
				if (minreflecthi.t == -1 || minreflecthi.t > hi.t) {
					minreflecthi = hi;
				}
			}
			if (tir >= 0 && bvh.intersect(hi, refractr)) {
				if (minrefracthi.t == -1 || minrefracthi.t > hi.t) {
					minrefracthi = hi;
				}
			}
		}
		float3 reflectL = float3(0.0f);
		float3 refractL = float3(0.0f);

		if (minreflecthi.t != -1) reflectL = R * shade(minreflecthi, -outgoingDir, level + 1,incomingEta);
		else {
			if (globalEnvironmentImage) {
				float r = (1 / PI) * acosf(outgoingDir[2]) / sqrt(outgoingDir[0] * outgoingDir[0] + outgoingDir[1] * outgoingDir[1]);
				float u = outgoingDir[0] * r;
				float v = outgoingDir[1] * r;
				int pixx = static_cast<int>((u + 1) * globalScene.image.width / 2.0f);
				int pixy = static_cast<int>((v + 1) * globalScene.image.height / 2.0f);
				//std::cout << u << ", " << v << std::endl;
				//std::cout << pixx << ", " << pixy << std::endl;
				reflectL = globalScene.image.pixel(pixx, pixy);
			}
			else reflectL = float3(0.0f);
		}
		if (tir < 0) return reflectL;
		if (minrefracthi.t != -1) refractL = (1 - R) * shade(minrefracthi, -dir, level + 1, nextEta);
		else if (tir >= 0) {
			if (globalEnvironmentImage) {
				float r = (1 / PI) * acosf(dir[2]) / sqrt(dir[0] * dir[0] + dir[1] * dir[1]);
				float u = dir[0] * r;
				float v = dir[1] * r;
				int pixx = static_cast<int>((u + 1) * globalScene.image.width / 2.0f);
				int pixy = static_cast<int>((v + 1) * globalScene.image.height / 2.0f);
				//std::cout << u << ", " << v << std::endl;
				//std::cout << pixx << ", " << pixy << std::endl;
				refractL = globalScene.image.pixel(pixx, pixy);
			}
			else refractL = float3(0.0f);
		}
		return refractL + reflectL;

		
	} else {
		// something went wrong - make it apparent that it is an error
		std::cout << "error" << std::endl;
		return float3(100.0f, 0.0f, 100.0f);
	}
	
}



TriangleMesh* generateWaterSurface(float3 origin, float baseHeight, int step = 16, float width = 0.5, float length = 0.5, float heightFactor = 10000, float maxTime = 5, float periodTime = 1, float windSpeed = 50, float2 windDir = float2(1, 1)) {

	//for randomly generating starting height
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> dist(-1.0, 1.0);
	std::normal_distribution<> dist2(0.0, 1.0);

	//std::cout << width << " " << length << std::endl;

	float stepLength = length / step;
	float stepWidth = width / step;
	float** height = new float* [step];
	float** xPos = new float* [step];
	float** zPos = new float* [step];
	float** freq = new float* [step];
	float** ampH = new float* [step];
	Complex* ampF = new Complex[step * step];
	float** normal = new float* [step];
	for (int i = 0; i < step; i++) {
		height[i] = new float[step];
		xPos[i] = new float[step];
		zPos[i] = new float[step];
		freq[i] = new float[step];
		ampH[i] = new float[step];
		normal[i] = new float[step];
	}
	TriangleMesh* tm = new TriangleMesh();
	// the only material here is water
	tm->materials.resize(1);
	tm->materials[0].eta = 1.33; //real life data
	tm->materials[0].type = MAT_GLASS; //Basically the same as glass

	//These terms are: the mesh of the top surface, the mesh of the top wave from the side, the mesh for the side that is completely underwater


	//Randomly generate height of water at each point of the grid to use as basis

	Complex* RoU = generateRoU(step * step);
	/*
	for (int i = 0; i < step * step; i++) {
		std::cout << RoU[i].real << " " << RoU[i].imaginary << std::endl;
	}*/
	float time = maxTime * (1 + dist(gen));
	float freq0 = 2 * std::_Pi / periodTime;

	for (int i = 0; i < step; i++) {
		for (int k = 0; k < step; k++) {
			//std::cout << i << " " << k << std::endl;
			float randomFactor = dist(gen);
			zPos[i][k] = k * stepLength + origin[2];
			xPos[i][k] = i * stepWidth + origin[0];

			float2 vecK = float2((i - step / 2) / width * 2 * std::_Pi, (k - step / 2) / length * 2 * std::_Pi);

			float magK = linalg::length(vecK);
			//std::cout << vecK << " " << magK << std::endl;
			freq[i][k] = std::floor(std::sqrt(magK) / freq0) * freq0;
			/*
			if (i == 0 || k == 0) {
				ampH[i][k] = 0;
			}
			else {
				//std::cout << std::exp(-1 / (magK * magK * windSpeed * windSpeed)) << std::endl;

				ampH[i][k] = std::exp(-1 / (magK * magK * windSpeed * windSpeed)) / std::pow(magK, 4) * std::abs(linalg::dot(linalg::normalize(vecK), linalg::normalize(windDir)));
			}*/
			if (magK != 0) {
				ampH[i][k] = 3 * std::exp(-1 / (magK * magK * windSpeed * windSpeed)) / std::pow(magK, 4) * std::abs(linalg::dot(linalg::normalize(vecK), linalg::normalize(windDir)));
			}
			else {
				ampH[i][k] = 0;
			}

			//std::cout << ampH[i][k] << std::endl;

		}
	}

	for (int i = 0; i < step; i++) {
		for (int k = 0; k < step; k++) {
			//std::cout << i << " " << k << std::endl;
			//std::cout << ampH[i][k] << std::endl;
			float freqK = freq[i][k];
			float sqrP = std::sqrtf(ampH[i][k]) / std::sqrtf(2);

			//std::cout << sqrP << std::endl;
			Complex h0 = Complex(dist2(gen) * sqrP, dist2(gen) * sqrP);
			Complex factor1 = Complex(std::cosf(freqK * time), std::sinf(freqK * time));
			Complex factor2 = Complex(std::cosf(-freqK * time), std::sinf(-freqK * time));
			ampF[i * step + k] = h0 * factor1 + h0.conjugate() * factor2;
			//std::cout << ampF[i * step + k].real << " " << ampF[i * step + k].imaginary << std::endl;
		}

	}

	revFFT(ampF, RoU, step * step);
	float total = 0;
	float min = ampF[0].real;
	float max = ampF[0].real;
	for (int i = 0; i < step; i++) {
		for (int k = 0; k < step; k++) {
			//std::cout << ampF[i * step + k].real << " " << ampF[i * step + k].imaginary << std::endl;
			Complex pos = ampF[i * step + k];

			height[i][k] = heightFactor * pos.real + baseHeight + origin[1];
			total += pos.real;
			if (pos.real > max) max = pos.real;
			if (pos.real < min) min = pos.real;
			//std::cout << height[i][k] << std::endl;
		}
	}
	std::cout << "mean: " << total / (step * step) << std::endl;
	std::cout << "min: " << min << std::endl;
	std::cout << "max: " << max << std::endl;


	globalScene.height = height;
	globalScene.step = step;
	globalScene.origin = origin;
	globalScene.stepLength = stepLength;
	globalScene.stepWidth = stepWidth;





	int index = 0;
	/*
	tm->triangles.resize((step - 1)* (step - 1) * 2 + 8);


	for (int i = 0; i < step - 1; i++) {
		for (int k = 0; k < step - 1; k++) {

			tm->triangles[index].positions[0] = float3(xPos[i][k], height[i][k], zPos[i][k]);
			tm->triangles[index].positions[1] = float3(xPos[i+1][k], height[i+1][k], zPos[i+1][k]);
			tm->triangles[index].positions[2] = float3(xPos[i+1][k+1], height[i+1][k+1], zPos[i+1][k+1]);


			tm->triangles[index + 1].positions[0] = float3(xPos[i][k], height[i][k], zPos[i][k]);
			tm->triangles[index + 1].positions[1] = float3(xPos[i][k + 1], height[i][k + 1], zPos[i][k + 1]);
			tm->triangles[index + 1].positions[2] = float3(xPos[i + 1][k + 1], height[i + 1][k + 1], zPos[i + 1][k + 1]);


			index += 2;
		}
	}*/

	tm->triangles.resize((step - 2) * (step - 2) / 2 + 8 + 4 * (step - 2));
	std::cout << tm->triangles.size() << std::endl;
	for (int i = 0; i < step / 2 - 1; i++) {
		for (int k = 0; k < step / 2 - 1; k++) {

			tm->triangles[index].positions[0] = float3(xPos[2 * i][2 * k], height[2 * i][2 * k], zPos[2 * i][2 * k]);
			tm->triangles[index].positions[1] = float3(xPos[2 * (i + 1)][2 * k], height[2 * (i + 1)][2 * k], zPos[2 * (i + 1)][2 * k]);
			tm->triangles[index].positions[2] = float3(xPos[2 * (i + 1)][2 * (k + 1)], height[2 * (i + 1)][2 * (k + 1)], zPos[2 * (i + 1)][2 * (k + 1)]);


			tm->triangles[index + 1].positions[0] = float3(xPos[2 * i][2 * k], height[2 * i][2 * k], zPos[2 * i][2 * k]);
			tm->triangles[index + 1].positions[1] = float3(xPos[2 * i][2 * (k + 1)], height[2 * i][2 * (k + 1)], zPos[2 * i][2 * (k + 1)]);
			tm->triangles[index + 1].positions[2] = float3(xPos[2 * (i + 1)][2 * (k + 1)], height[2 * (i + 1)][2 * (k + 1)], zPos[2 * (i + 1)][2 * (k + 1)]);


			index += 2;
		}
	}
	std::cout << index << std::endl;

	//left side
	float minHeight = height[0][0];

	for (int i = 0; i < step / 2; i++) {
		float newheight = height[0][2 * i];
		if (newheight < minHeight) minHeight = newheight;
	}
	/*
	std::cout << minHeight << std::endl;*/

	for (int i = 0; i < step / 2 - 1; i++) {
		tm->triangles[index].positions[0] = float3(xPos[0][2 * i], minHeight, zPos[0][2 * i]);
		tm->triangles[index].positions[1] = float3(xPos[0][2 * (i + 1)], height[0][2 * (i + 1)], zPos[0][2 * (i + 1)]);
		tm->triangles[index].positions[2] = float3(xPos[0][2 * i], height[0][2 * i], zPos[0][2 * i]);
		tm->triangles[index + 1].positions[0] = float3(xPos[0][2 * i], minHeight, zPos[0][2 * i]);
		tm->triangles[index + 1].positions[1] = float3(xPos[0][2 * (i + 1)], minHeight, zPos[0][2 * (i + 1)]);
		tm->triangles[index + 1].positions[2] = float3(xPos[0][2 * (i + 1)], height[0][2 * (i + 1)], zPos[0][2 * (i + 1)]);

		index += 2;
	}
	std::cout << index << std::endl;
	/*std::cout << minHeight << std::endl; */
	tm->triangles[index].positions[0] = float3(xPos[0][0], minHeight, zPos[0][0]);
	tm->triangles[index].positions[1] = origin;
	tm->triangles[index].positions[2] = float3(xPos[0][step - 2], minHeight, zPos[0][step - 2]);
	tm->triangles[index + 1].positions[0] = float3(xPos[0][step - 2], origin[1], zPos[0][step - 2]);
	tm->triangles[index + 1].positions[1] = origin;
	tm->triangles[index + 1].positions[2] = float3(xPos[0][step - 2], minHeight, zPos[0][step - 2]);
	index += 2;
	std::cout << index << std::endl;
	//right
	minHeight = height[step - 2][0];

	for (int i = 0; i < step / 2; i++) {
		float newheight = height[step - 2][2 * i];
		if (newheight < minHeight) minHeight = newheight;
	}
	/*
	std::cout << minHeight << std::endl;*/
	for (int i = 0; i < step / 2 - 1; i++) {

		tm->triangles[index].positions[0] = float3(xPos[step - 2][2 * i], minHeight, zPos[step - 2][2 * i]);
		tm->triangles[index].positions[1] = float3(xPos[step - 2][2 * (i + 1)], height[step - 2][2 * (i + 1)], zPos[step - 2][2 * (i + 1)]);
		tm->triangles[index].positions[2] = float3(xPos[step - 2][2 * i], height[step - 2][2 * i], zPos[step - 2][2 * i]);
		tm->triangles[index + 1].positions[0] = float3(xPos[step - 2][2 * i], minHeight, zPos[step - 2][2 * i]);
		tm->triangles[index + 1].positions[1] = float3(xPos[step - 2][2 * (i + 1)], minHeight, zPos[step - 2][2 * (i + 1)]);
		tm->triangles[index + 1].positions[2] = float3(xPos[step - 2][2 * (i + 1)], height[step - 2][2 * (i + 1)], zPos[step - 2][2 * (i + 1)]);

		index += 2;
	}
	std::cout << index << std::endl;
	tm->triangles[index].positions[0] = float3(xPos[step - 2][0], minHeight, zPos[step - 2][0]);
	tm->triangles[index].positions[1] = float3(xPos[step - 2][0], origin[1], zPos[step - 2][0]);
	tm->triangles[index].positions[2] = float3(xPos[step - 2][step - 2], minHeight, zPos[step - 2][step - 2]);
	tm->triangles[index + 1].positions[0] = float3(xPos[step - 2][step - 2], origin[1], zPos[step - 2][step - 2]);
	tm->triangles[index + 1].positions[1] = float3(xPos[step - 2][0], origin[1], zPos[step - 2][0]);
	tm->triangles[index + 1].positions[2] = float3(xPos[step - 2][step - 2], minHeight, zPos[step - 2][step - 2]);
	index += 2;
	std::cout << index << std::endl;

	//front
	minHeight = height[0][0];
	std::cout << minHeight << std::endl;
	for (int i = 0; i < step / 2; i++) {
		float newheight = height[2 * i][0];
		if (newheight < minHeight) minHeight = newheight;
	}

	for (int i = 0; i < step / 2 - 1; i++) {

		tm->triangles[index].positions[0] = float3(xPos[2 * i][0], minHeight, zPos[2 * i][0]);
		tm->triangles[index].positions[1] = float3(xPos[2 * (i + 1)][0], height[2 * (i + 1)][0], zPos[2 * (i + 1)][0]);
		tm->triangles[index].positions[2] = float3(xPos[2 * i][0], height[2 * i][0], zPos[2 * i][0]);
		tm->triangles[index + 1].positions[0] = float3(xPos[2 * i][0], minHeight, zPos[2 * i][0]);
		tm->triangles[index + 1].positions[1] = float3(xPos[2 * (i + 1)][0], minHeight, zPos[2 * (i + 1)][0]);
		tm->triangles[index + 1].positions[2] = float3(xPos[2 * (i + 1)][0], height[2 * (i + 1)][0], zPos[2 * (i + 1)][0]);

		index += 2;
	}
	std::cout << index << std::endl;
	tm->triangles[index].positions[0] = float3(xPos[0][0], minHeight, zPos[0][0]);
	tm->triangles[index].positions[1] = origin;
	tm->triangles[index].positions[2] = float3(xPos[step - 2][0], minHeight, zPos[step - 2][0]);
	tm->triangles[index + 1].positions[0] = float3(xPos[step - 2][0], origin[1], zPos[step - 2][0]);
	tm->triangles[index + 1].positions[1] = origin;
	tm->triangles[index + 1].positions[2] = float3(xPos[step - 2][0], minHeight, zPos[step - 2][0]);
	index += 2;
	std::cout << index << std::endl;

	//back
	minHeight = height[0][step - 2];
	std::cout << minHeight << std::endl;
	for (int i = 0; i < step / 2; i++) {
		float newheight = height[2 * i][step - 2];
		if (newheight < minHeight) minHeight = newheight;
	}

	for (int i = 0; i < step / 2 - 1; i++) {

		tm->triangles[index].positions[0] = float3(xPos[2 * i][step - 2], minHeight, zPos[2 * i][step - 2]);
		tm->triangles[index].positions[1] = float3(xPos[2 * (i + 1)][step - 2], height[2 * (i + 1)][step - 2], zPos[2 * (i + 1)][step - 2]);
		tm->triangles[index].positions[2] = float3(xPos[2 * i][step - 2], height[2 * i][step - 2], zPos[2 * i][step - 2]);
		tm->triangles[index + 1].positions[0] = float3(xPos[2 * i][step - 2], minHeight, zPos[2 * i][step - 2]);
		tm->triangles[index + 1].positions[1] = float3(xPos[2 * (i + 1)][step - 2], minHeight, zPos[2 * (i + 1)][step - 2]);
		tm->triangles[index + 1].positions[2] = float3(xPos[2 * (i + 1)][step - 2], height[2 * (i + 1)][step - 2], zPos[2 * (i + 1)][step - 2]);

		index += 2;
	}
	std::cout << index << std::endl;
	tm->triangles[index].positions[0] = float3(xPos[0][step - 2], minHeight, zPos[0][step - 2]);
	tm->triangles[index].positions[1] = float3(xPos[0][step - 2], origin[1], zPos[0][step - 2]);
	tm->triangles[index].positions[2] = float3(xPos[step - 2][step - 2], minHeight, zPos[step - 2][step - 2]);
	tm->triangles[index + 1].positions[0] = float3(xPos[step - 2][step - 2], origin[1], zPos[step - 2][step - 2]);
	tm->triangles[index + 1].positions[1] = float3(xPos[0][step - 2], origin[1], zPos[0][step - 2]);
	tm->triangles[index + 1].positions[2] = float3(xPos[step - 2][step - 2], minHeight, zPos[step - 2][step - 2]);
	index += 2;
	std::cout << index << std::endl;
	index = tm->triangles.size();

	for (int i = 0; i < index; i++) {
		float3 e0 = tm->triangles[i].positions[1] - tm->triangles[i].positions[0];
		float3 e1 = tm->triangles[i].positions[2] - tm->triangles[i].positions[0];
		float3 n = normalize(cross(e0, e1));

		tm->triangles[i].normals[0] = n;
		tm->triangles[i].normals[1] = n;
		tm->triangles[i].normals[2] = n;

		tm->triangles[i].idMaterial = 0;

		tm->triangles[i].texcoords[0] = float2(0.0f);
		tm->triangles[i].texcoords[1] = float2(0.0f);
		tm->triangles[i].texcoords[2] = float2(0.0f);
	}



	for (int i = 0; i < step; i++) {
		delete[] xPos[i];
		delete[] zPos[i];
		delete[] freq[i];
		delete[] ampH[i];
		delete[] normal[i];
	}

	delete[] xPos;
	delete[] zPos;
	delete[] freq;
	delete[] ampH;
	delete[] normal;
	delete[] ampF;
	delete[] RoU;
	return tm;
}



// OpenGL initialization (you will not use any OpenGL/Vulkan/DirectX... APIs to render 3D objects!)
class OpenGLInit {
public:
	OpenGLInit() {
		// initialize GLFW
		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW." << std::endl;
			exit(-1);
		}

		// create a window
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		globalGLFWindow = glfwCreateWindow(globalWidth, globalHeight, "Hello World!", NULL, NULL);
		if (globalGLFWindow == NULL) {
			std::cerr << "Failed to open GLFW window." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// make OpenGL context for the window
		glfwMakeContextCurrent(globalGLFWindow);

		// initialize GLEW
		glewExperimental = true;
		if (glewInit() != GLEW_OK) {
			std::cerr << "Failed to initialize GLEW." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// set callback functions for events
		glfwSetKeyCallback(globalGLFWindow, keyFunc);
		glfwSetMouseButtonCallback(globalGLFWindow, mouseButtonFunc);
		glfwSetCursorPosCallback(globalGLFWindow, cursorPosFunc);

		// create shader
		FSDraw = glCreateProgram();
		GLuint s = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(s, 1, &PFSDrawSource, 0);
		glCompileShader(s);
		glAttachShader(FSDraw, s);
		glLinkProgram(FSDraw);

		// create texture
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &GLFrameBufferTexture);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, globalWidth, globalHeight, 0, GL_LUMINANCE, GL_FLOAT, 0);

		// initialize some OpenGL state (will not change)
		glDisable(GL_DEPTH_TEST);

		glUseProgram(FSDraw);
		glUniform1i(glGetUniformLocation(FSDraw, "input_tex"), 0);

		GLint dims[4];
		glGetIntegerv(GL_VIEWPORT, dims);
		const float BufInfo[4] = { float(dims[2]), float(dims[3]), 1.0f / float(dims[2]), 1.0f / float(dims[3]) };
		glUniform4fv(glGetUniformLocation(FSDraw, "BufInfo"), 1, BufInfo);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	virtual ~OpenGLInit() {
		glfwTerminate();
	}
};



// main window
class Window {
public:
	// put this first to make sure that the glInit's constructor is called before the one for Window
	OpenGLInit glInit;

	Window() {}
	virtual ~Window() {}

	void(*process)() = NULL;

	void start() const {
		if (globalEnableParticles) {
			globalScene.addObject(&globalParticleSystem.particlesMesh);
		}
		

		globalScene.preCalc();
		if (globalRenderType == RENDER_PHOTON) {
			globalScene.EmitPhoton();
		}
		// main loop
		while (glfwWindowShouldClose(globalGLFWindow) == GL_FALSE) {
			glfwPollEvents();
			globalViewDir = normalize(globalLookat - globalEye);
			globalRight = normalize(cross(globalViewDir, globalUp));

			if (globalEnableParticles) {
				globalParticleSystem.step();
			}

			if (globalRenderType == RENDER_RASTERIZE) {
				globalScene.Rasterize();
			} else if (globalRenderType == RENDER_RAYTRACE) {
				globalScene.Raytrace();
			} else if (globalRenderType == RENDER_IMAGE) {
				if (process) process();
			}
			else if (globalRenderType == RENDER_PHOTON) {
				globalScene.Raytrace();
			}

			if (globalRecording) {
				unsigned char* buf = new unsigned char[FrameBuffer.width * FrameBuffer.height * 4];
				int k = 0;
				for (int j = FrameBuffer.height - 1; j >= 0; j--) {
					for (int i = 0; i < FrameBuffer.width; i++) {
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).x));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).y));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).z));
						buf[k++] = 255;
					}
				}
				GifWriteFrame(&globalGIFfile, buf, globalWidth, globalHeight, globalGIFdelay);
				delete[] buf;
			}

			// drawing the frame buffer via OpenGL (you don't need to touch this)
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0][0]);
			glRecti(1, 1, -1, -1);
			glfwSwapBuffers(globalGLFWindow);
			globalFrameCount++;
			PCG32::rand();
		}
	}
};



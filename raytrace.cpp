#ifndef RAY_TRACER
#include "ray-tracer.h"
#define RAY_TRACER
#endif
#include "linalg.h"
#include "stdio.h"
using namespace linalg::aliases;
using namespace linalg::ostream_overloads;

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
	std::cout << r1 << std::endl << r2 << std::endl << r3 << std::endl;
	
	*result = { r1[3],r2[3],r3[3] };
	return true;
}

bool TriangleMesh::raytraceTriangle(HitInfo& result, const Ray& ray, const Triangle& tri, float tMin, float tMax) const {
	// ray-triangle intersection
	// fill in "result" when there is an intersection
	// return true/false if there is an intersection or not
	float3 o = ray.o;
	float3 d = ray.d;
	float3 a = tri.positions[0];
	float3 b = tri.positions[1];
	float3 c = tri.positions[2];
	float3 n = linalg::normalize(linalg::cross((b - a), (c - a)));
	float3x4 eq = { a - c, b - c, -d, o - c };
	float3 solution = { 0,0,0 };
	//if no solution to system of equations, the line is parrallel to the surface, no hits
	if (!gaussianElimination3(eq, &solution)) return false; 
	//if the distance is negative, it is not a solution: camera cannot capture what is behind it.
	float t = solution[2];
	if (t < tMin  || t > tMax) return false;
	float3 i = solution[2] * d + o;
	//now check if the point i is in the triangle.
	float3 nac = linalg::normalize(a - c);
	float3 nbc = linalg::normalize(b - c);
	float3 nab = linalg::normalize(a - b);
	float anglea = linalg::dot(nac, nab);
	float angleb = linalg::dot(nbc, -nab);
	float anglec = linalg::dot(-nac, -nbc);
	float3 nai = linalg::normalize(a - i);
	float3 nbi = linalg::normalize(b - i);
	float3 nci = linalg::normalize(c - i);
	float angleiab = linalg::dot(nai, nab);
	float angleiac = linalg::dot(nai, nac);
	if (anglea > angleiab || anglea > angleiac) return false;
	float angleiba = linalg::dot(nbi, -nab);
	float angleibc = linalg::dot(nbi, nbc);
	if (angleb > angleiba || angleb > angleibc) return false;
	result.t = t;
	result.P = solution;
	return true;
}
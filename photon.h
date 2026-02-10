#include <vector>
#include <cfloat>
#include <algorithm>

#include "linalg.h"
using namespace linalg::aliases;
using namespace linalg::ostream_overloads;

// photon data structure for photon mapping
struct Photon {
  float3 position; // 3D position the emitted photon hits
  float3 direction; // direction of the photon was going
  float3 flux; // energy/intensity of the photon
  bool caustic = false;
  friend void swap(Photon &a, Photon &b) noexcept {
    std::swap(a.position, b.position);
    std::swap(a.direction, b.direction);
    std::swap(a.flux, b.flux);
  }
};

class KDTree {
  struct Node {
    std::vector<Photon> photons;
    Node *left, *right; // subtrees
    float3 lowerBoundingCorner, upperBoundingCorner; // axis-aligned bounding box corners of the kd-tree node
    int splitingAxis; // 0 => x-axis, 1 => y-axis, 2 => z-axis
    float splitingValue; // median value to split the region of photons
    
    Node(std::vector<Photon> &photons, int start, int end, int depth, int maxPhotonsInLeaf){
      if(start >= end){
        left = nullptr;
        right = nullptr;
        return;
      }

      lowerBoundingCorner = float3(FLT_MAX); // lower-left-front corner of the axis-aligned bounding box
      for(int i = start; i < end; i++){
        if(photons[i].position.x < lowerBoundingCorner.x) lowerBoundingCorner.x = photons[i].position.x;
        if(photons[i].position.y < lowerBoundingCorner.y) lowerBoundingCorner.y = photons[i].position.y;
        if(photons[i].position.z < lowerBoundingCorner.z) lowerBoundingCorner.z = photons[i].position.z;
      }
      upperBoundingCorner = float3(-FLT_MAX); // upper-right-back corner of the axis-aligned bounding box
      for(int i = start; i < end; i++){
        if(photons[i].position.x > upperBoundingCorner.x) upperBoundingCorner.x = photons[i].position.x;
        if(photons[i].position.y > upperBoundingCorner.y) upperBoundingCorner.y = photons[i].position.y;
        if(photons[i].position.z > upperBoundingCorner.z) upperBoundingCorner.z = photons[i].position.z;
      }
      
      if(end - start <= maxPhotonsInLeaf){ // a leaf node containing a number of photons
        this->photons.insert(this->photons.begin(), photons.begin() + start, photons.begin() + end);
        left = nullptr;
        right = nullptr;
        return;
      }

      splitingAxis = depth % 3; // we alternate between splitting x, y, z-axis at each level of the kd-tree (why we must pass depth parameter)
      std::vector<float> splitingValues;
      for(int i = start; i < end; i++){
        splitingValues.push_back(photons[i].position[splitingAxis]);
      }
      std::sort(splitingValues.begin(), splitingValues.end());
      splitingValue = splitingValues[splitingValues.size()/2]; // median value
      int mid = start;
      for(int i = start; i < end; i++){
        if(photons[i].position[splitingAxis] < splitingValue){
          swap(photons[i], photons[mid]);
          mid++;
        }
      }
      left = new Node(photons, start, mid, depth + 1, maxPhotonsInLeaf);
      right = new Node(photons, mid, end, depth + 1, maxPhotonsInLeaf);
    }

    ~Node(){
      delete left;
      delete right;
    }
  };



  void recursiveNearbyPhotons(const Node *node, std::vector<Photon> &result, const float3 &point, const float radius){
    if(!node) return;
    if(point.x + radius < node->lowerBoundingCorner.x || point.x - radius > node->upperBoundingCorner.x ||
        point.y + radius < node->lowerBoundingCorner.y || point.y - radius > node->upperBoundingCorner.y ||
        point.z + radius < node->lowerBoundingCorner.z || point.z - radius > node->upperBoundingCorner.z){
      return; // the sphere does not intersect with the kd-tree node's bounding box, this subtree no longer needs to be searched
    }
    if(!node->left && !node->right){ // leaf node, check if every photon in the bounding box is within 'radius' of the point
      for(const auto &photon : node->photons){
        if(length2(point - photon.position) <= radius * radius){
          result.push_back(photon);
        }
      }
      return;
    }
    recursiveNearbyPhotons(node->left, result, point, radius);
    recursiveNearbyPhotons(node->right, result, point, radius);
  }

  Node *root; // root of the kd-tree
  int maxPhotonsInLeaf; // upper limit on the number of photons in a kd-tree leaf node
  
 public:
  

  KDTree(std::vector<Photon>& photons, int maxPhotonsInLeaf = 25): maxPhotonsInLeaf{maxPhotonsInLeaf} {
      if(photons.empty()){
      root = nullptr;
      return;
    }
    root = new Node(photons, 0, photons.size(), 0, maxPhotonsInLeaf);
  }
  
  // Returns a vector of Photons that are within 'radius' of a given point.
  std::vector<Photon> nearbyPhotons(const float3 &point, const float radius){
    std::vector<Photon> result;
    recursiveNearbyPhotons(root, result, point, radius);
    return result;
  }

  ~KDTree(){
    delete root;
  }
};

// photon tracing, photon radiance estimation, and Fresnel reflection are in Scene class

# Underwater Caustics Renderer

## Description
This C++ program renders a scene with underwater caustics and global illumination from $>=1$ light sources. Photon mapping (light transport model used) creates sharp patterns resulting from the refraction of photons across the air-water boundary. The water surface is generated from rasterized FFT-based triangle meshes. The kd-tree data structure is used to make rendering as faster and the radiance estimation of "spheres" efficient.

## Input/Output
### Input:
A scene in `.obj` format and the materials in a `.mtl` file.
### Output:
The final rendered scene in a `.png` file.

## Compilation: 
The program can be compiled by running `Build` at bottom bar in VSCode and is run through `Shift+F5`.

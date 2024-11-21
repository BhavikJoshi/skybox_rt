#include <cmath>
#include "rt_types.h"

float3 normalize(const float3& v1) {

    // Calculate the magnitude of the difference vector
    float magnitude = std::sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);

    // Check if the magnitude is not zero to avoid division by zero
    if (magnitude > 0.0f) {
        // Normalize the difference vector
        float3 normalized = {
            v1.x / magnitude,
            v1.y / magnitude,
            v1.z / magnitude
        };
        return normalized;
    } else {
        // Return a zero vector if the magnitude is zero
        return {0.0f, 0.0f, 0.0f};
    }
}


inline uint32_t RGB32FtoRGB8( float3 c )
{
	int r = (int)(min( c.x, 1.f ) * 255);
	int g = (int)(min( c.y, 1.f ) * 255);
	int b = (int)(min( c.z, 1.f ) * 255);
	return (r << 16) + (g << 8) + b;
}

float3 cross( const float3& a, const float3& b ) { 
    float3 res(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); 
    return res;
}
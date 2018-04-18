#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
using namespace optix;

struct PrimaryRayPayload {};

// Internal semantics variables
rtDeclareVariable(uint, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint, launchDim, rtLaunchDim, );
rtDeclareVariable(optix::Ray, currentRay, rtCurrentRay, );
rtDeclareVariable(PrimaryRayPayload, currentRayPayload, rtPayload, );
rtDeclareVariable(float, rayT, rtIntersectionDistance, );

// Host variables
rtDeclareVariable(rtObject, rootObject, , );
rtDeclareVariable(Matrix4x4, modelTransform, , );
rtDeclareVariable(float3, origin, , );
rtDeclareVariable(float, pointRadius, , );
rtDeclareVariable(float, depthTolerance, , );

// Input buffers
rtBuffer<float3> modelPoints;
rtBuffer<float3> scenePoints;

// Output buffers
rtBuffer<int8_t> resultBuffer;

RT_PROGRAM void shootPrimary() {
    float3 modelPoint = modelPoints[launchIndex];
    float4 transformedPoint = modelTransform * make_float4(modelPoint.x, modelPoint.y, modelPoint.z, 1.f);
    float3 targetPoint = make_float3(transformedPoint.x, transformedPoint.y, transformedPoint.z);
    
    float3 delta = targetPoint - origin;

    optix::Ray primaryRay(origin, normalize(delta), 0, 0.f, length(delta));
    PrimaryRayPayload primaryPayload;
    
    rtTrace(rootObject, primaryRay, primaryPayload);
}

RT_PROGRAM void missPrimary() {
    resultBuffer[launchIndex] = 1;
}

RT_PROGRAM void primaryAnyHit() {
    resultBuffer[launchIndex] = 0;
    rtTerminateRay();
}

RT_PROGRAM void modelIntersect(int pointIndex) {
    float3 rayOrigin = currentRay.origin;
    float3 rayDirection = currentRay.direction;
    
    float3 hitPointPos = modelPoints[pointIndex];
    float hitLambda = dot(hitPointPos - rayOrigin, rayDirection);
    if (hitLambda < 0.f) return;
    float3 hitProj = rayOrigin + hitLambda * rayDirection;
    float hitDist = length(hitProj - hitPointPos);
    
    if (hitDist > pointRadius) return;
    
    float3 targetPointPos = modelPoints[launchIndex];
    float targetLambda = dot(targetPointPos - rayOrigin, rayDirection);
    
    if (hitLambda > targetLambda - depthTolerance) return;
    
    if (rtPotentialIntersection(hitLambda)) {
        rtReportIntersection(0);
    }
}

RT_PROGRAM void sceneIntersect(int pointIndex) {
    float3 rayOrigin = currentRay.origin;
    float3 rayDirection = currentRay.direction;
    
    float3 hitPointPos = scenePoints[pointIndex];
    float hitLambda = dot(hitPointPos - rayOrigin, rayDirection);
    if (hitLambda < 0.f) return;
    float3 hitProj = rayOrigin + hitLambda * rayDirection;
    float hitDist = length(hitProj - hitPointPos);
    
    if (hitDist > pointRadius) return;
    
    float3 targetPointPosObj = modelPoints[launchIndex];
    float4 targetPointTransformed = modelTransform * make_float4(targetPointPosObj.x, targetPointPosObj.y, targetPointPosObj.z, 1.f);
    float3 targetPointPosWorld = make_float3(targetPointTransformed.x, targetPointTransformed.y, targetPointTransformed.z);
    float targetLambda = dot(targetPointPosWorld - rayOrigin, rayDirection);
    
    if (hitLambda > targetLambda - depthTolerance) return;
    
    if (rtPotentialIntersection(hitLambda)) {
        rtReportIntersection(0);
    }
}

RT_PROGRAM void modelBounds(int pointIndex, float result[6]) {
    float3 pointPos = modelPoints[pointIndex];
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = pointPos - make_float3(pointRadius, pointRadius, pointRadius);
    aabb->m_max = pointPos + make_float3(pointRadius, pointRadius, pointRadius);
}

RT_PROGRAM void sceneBounds(int pointIndex, float result[6]) {
    float3 pointPos = scenePoints[pointIndex];
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = pointPos - make_float3(pointRadius, pointRadius, pointRadius);
    aabb->m_max = pointPos + make_float3(pointRadius, pointRadius, pointRadius);
}

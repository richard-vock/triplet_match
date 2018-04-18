template <typename PointType>
raycaster<PointType>::raycaster(const PointCloud& model, const PointCloud& scene, float pointRadius, float depthTolerance) {
    m_modelSize = model.size();

    m_optixContext = optix::Context::create();
    m_optixContext->setEntryPointCount(1);
    m_optixContext->setRayTypeCount(1);
    //m_optixContext->setPrintEnabled(1); // For debugging purposes.

    std::string kernelPath = std::string(OPTIX_PREFIX) + "RayCaster.ptx";
    m_optixContext->setRayGenerationProgram(0, m_optixContext->createProgramFromPTXFile(kernelPath, "shootPrimary"));
    m_optixContext->setMissProgram(0, m_optixContext->createProgramFromPTXFile(kernelPath, "missPrimary"));

    optix::Material optixMaterial = m_optixContext->createMaterial();
    optixMaterial->setAnyHitProgram(0, m_optixContext->createProgramFromPTXFile(kernelPath, "primaryAnyHit"));

    float* pointsMap;
    unsigned int bufferPos;

    optix::Buffer optixModelPointsBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, model.size());
    pointsMap = static_cast<float*>(optixModelPointsBuffer->map());
    bufferPos = 0;
    for (const PointType& pt : model) {
        pointsMap[bufferPos++] = pt.x;
        pointsMap[bufferPos++] = pt.y;
        pointsMap[bufferPos++] = pt.z;
    }
    optixModelPointsBuffer->unmap();

    m_optixContext["modelPoints"]->set(optixModelPointsBuffer);

    optix::Buffer optixScenePointsBuffer = m_optixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, scene.size());
    pointsMap = static_cast<float*>(optixScenePointsBuffer->map());
    bufferPos = 0;
    for (const PointType& pt : scene) {
        pointsMap[bufferPos++] = pt.x;
        pointsMap[bufferPos++] = pt.y;
        pointsMap[bufferPos++] = pt.z;
    }
    optixScenePointsBuffer->unmap();

    m_optixContext["scenePoints"]->set(optixScenePointsBuffer);

    optix::Geometry modelGeometry = m_optixContext->createGeometry();
    modelGeometry->setPrimitiveCount(model.size());
    modelGeometry->setIntersectionProgram(m_optixContext->createProgramFromPTXFile(kernelPath, "modelIntersect"));
    modelGeometry->setBoundingBoxProgram(m_optixContext->createProgramFromPTXFile(kernelPath, "modelBounds"));

    optix::GeometryInstance modelGeometryInstance = m_optixContext->createGeometryInstance();
    modelGeometryInstance->setGeometry(modelGeometry);
    modelGeometryInstance->setMaterialCount(1);
    modelGeometryInstance->setMaterial(0, optixMaterial);

    optix::GeometryGroup modelGeometryGroup = m_optixContext->createGeometryGroup();
    modelGeometryGroup->setChildCount(1);
    modelGeometryGroup->setChild(0, modelGeometryInstance);
    modelGeometryGroup->setAcceleration(m_optixContext->createAcceleration("Bvh","Bvh"));

    m_modelTransform = m_optixContext->createTransform();
    m_modelTransform->setChild(modelGeometryGroup);
    const float trafo[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    m_modelTransform->setMatrix(0, trafo, 0);

    optix::Geometry sceneGeometry = m_optixContext->createGeometry();
    sceneGeometry->setPrimitiveCount(scene.size());
    sceneGeometry->setIntersectionProgram(m_optixContext->createProgramFromPTXFile(kernelPath, "sceneIntersect"));
    sceneGeometry->setBoundingBoxProgram(m_optixContext->createProgramFromPTXFile(kernelPath, "sceneBounds"));

    optix::GeometryInstance sceneGeometryInstance = m_optixContext->createGeometryInstance();
    sceneGeometryInstance->setGeometry(sceneGeometry);
    sceneGeometryInstance->setMaterialCount(1);
    sceneGeometryInstance->setMaterial(0, optixMaterial);

    optix::GeometryGroup sceneGeometryGroup = m_optixContext->createGeometryGroup();
    sceneGeometryGroup->setChildCount(1);
    sceneGeometryGroup->setChild(0, sceneGeometryInstance);
    sceneGeometryGroup->setAcceleration(m_optixContext->createAcceleration("Trbvh","Trbvh"));

    optix::Group root = m_optixContext->createGroup();
    root->setChildCount(2);
    root->setChild(0, m_modelTransform);
    root->setChild(1, sceneGeometryGroup);
    m_rootAcceleration = m_optixContext->createAcceleration("Trbvh","Trbvh");
    root->setAcceleration(m_rootAcceleration);

    m_optixContext["pointRadius"]->setFloat(pointRadius);
    m_optixContext["depthTolerance"]->setFloat(depthTolerance);

    m_optixContext["rootObject"]->set(root);
}

template <typename PointType>
raycaster<PointType>::~raycaster() {

}

template <typename PointType>
void raycaster<PointType>::cast(const Vec3f& origin, const Mat4f& modelTransform, Result& result) {
    m_optixContext["origin"]->set3fv(origin.data());
    m_optixContext["modelTransform"]->setMatrix4x4fv(true, modelTransform.data());

    m_modelTransform->setMatrix(true, modelTransform.data(), 0);
    m_rootAcceleration->markDirty();

    optix::Buffer optixResultBuffer = m_optixContext->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_BYTE, m_modelSize);
    m_optixContext["resultBuffer"]->set(optixResultBuffer);

    m_optixContext->launch(0, m_modelSize);

    result.resize(m_modelSize);
    int8_t* resultMap = (int8_t*)optixResultBuffer->map();
    std::copy(resultMap, resultMap + m_modelSize, result.begin());
    optixResultBuffer->unmap();

    optixResultBuffer->destroy();
}

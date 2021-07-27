    cl_int err = 0;
    std::unique_ptr<cl_platform_id> platforms;
    cl_device_id device_id = 0;
    cl_uint platformsCount = 0;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem buffer = NULL;
    const size_t bufferSize = sizeof(int) * 1024;
    
    cl_uint dimension = 1;
    size_t offset[3] = {0, 0, 0};
    size_t gws[3] = {bufferSize, 1, 1};
    size_t lws[3] = {4, 1, 1};
    
    err = clGetPlatformIDs(0, NULL, &platformsCount);
    platforms = std::make_unique<cl_platform_id>(platformsCount);
    err = clGetPlatformIDs(platformsCount, platforms.get(), NULL);
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
    err = clGetDeviceIDs(platforms.get()[0], deviceType, 1, &device_id, NULL);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    
    program = clCreateProgramWithSource(context, 1, &kernelStrings, 0, &err);
    err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    kernel = clCreateKernel(program, "increment", &err);
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    buffer = clCreateBuffer(context, flags, bufferSize, nullptr, &err);
    void *ptr = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0, bufferSize, 0, nullptr, nullptr, &err);
    memset(ptr, 13, bufferSize);
    err = clEnqueueUnmapMemObject(queue, buffer, ptr, 0, nullptr, nullptr);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    err = clEnqueueNDRangeKernel(queue, kernel, dimension, offset, gws, lws, 0, 0, nullptr);
    err = clFinish(queue);
    

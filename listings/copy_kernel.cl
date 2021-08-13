#ifdef FLOAT1
typedef float Type;
#endif
#ifdef FLOAT2
typedef float2 Type;
#endif

#ifdef FLOAT4
typedef float4 Type;
#endif

#ifdef FLOAT8
typedef float8 Type;
#endif

#ifdef FLOAT16
typedef float16 Type;
#endif

__kernel void readFloatType(__global Type *dst, __global Type *src){
    uint gid = get_global_id(0);
    dst[gid] = src[gid];
}

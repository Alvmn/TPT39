__kernel void vector_add(__global const float *restrict x,
                        __global const float *restrict y,
                        __global float *restrict z)
{
int i=get_global_id(0);
z[i]=x[i]+y[i];
}

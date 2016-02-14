__kernel void vector_add(__global const float *restrict x,
                        __global const float *restrict y,
                        __global float *restrict z,
                        const int N)
{

size_t tx = get_global_id(0);
size_t ty = get_global_id(1);

for(int i=0; i<N; i++) {
  z[tx*N + ty]+=x[ty* N + i]*y[i*N+tx];

}
}

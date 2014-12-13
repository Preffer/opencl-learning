__kernel void prod(__global const float *A, __global const float *B, __global float *C, const int R) {
 
    int index = get_global_id(0);
    int row = index / R;
    int col = index % R;
    float sum = 0;

    for (int i = 0; i < R; i++) {
        sum += A[row * R + i] * B[i * R + col];
    }
    C[index] = sum;
}
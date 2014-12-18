__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void prod(__read_only image2d_t A, __read_only image2d_t B, __write_only image2d_t C, const int SIZE) {
 
 	int col = get_global_id(0);
	int row = get_global_id(1);

	float4 sum = (float4)(0, 0, 0, 0);

	for (int i = 0; i < SIZE; i++) {
		float4 dataA = read_imagef(A, sampler, (int2)(i, row));
		sum += (float4)(
				dot(dataA, read_imagef(B, sampler, (int2)(i, col))),
				dot(dataA, read_imagef(B, sampler, (int2)(i, col + 1))),
				dot(dataA, read_imagef(B, sampler, (int2)(i, col + 2))),
				dot(dataA, read_imagef(B, sampler, (int2)(i, col + 3)))
			);
	}

	write_imagef(C, (int2)(col, row), sum);
}
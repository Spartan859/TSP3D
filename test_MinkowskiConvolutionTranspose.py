import pdb
import torch
import MinkowskiEngine as ME

def test_upsample_projection():
    """
    This function tests the projection field of a series of
    MinkowskiConvolutionTranspose layers, as discussed.
    It also fixes the device mismatch (CPU/GPU) runtime error.
    """
    # Set device automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Test on device: {device} ---")

    # 1. Define the upsampling layers and move them to the correct device
    upsample_st_1 = ME.MinkowskiGenerativeConvolutionTranspose(
        in_channels=1, out_channels=1, kernel_size=3, stride=2, dimension=3
    ).to(device)
    upsample_st_2 = ME.MinkowskiConvolutionTranspose(
        in_channels=1, out_channels=1, kernel_size=3, stride=2, dimension=3, bias=False
    ).to(device)
    upsample_st_4 = ME.MinkowskiConvolutionTranspose(
        in_channels=1, out_channels=1, kernel_size=3, stride=4, dimension=3, bias=False
    ).to(device)

    # 2. Create a single-point SparseTensor on the correct device
    # Use coordinates that are multiples of 8, e.g., [8, 16, 24]
    # The tensor_stride is set to [8, 8, 8]
    coords = torch.tensor([[0, 16, 32, 48]], dtype=torch.int32, device=device)
    # Use a simple non-zero feature vector
    feats = torch.tensor([[1.0]], dtype=torch.float32, device=device)

    print(f"Initial Coordinate: {coords[:, 1:]}")
    print(f"Initial Tensor Stride: [16, 16, 16]\n")

    # We need a coordinate manager. We can create a new one for this test.
    coo_manager = ME.CoordinateManager(D=3)

    single_point_st = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        tensor_stride=[16, 16, 16],
        coordinate_manager=coo_manager
    )

    print("--- Applying upsample_st_1 (stride=2, kernel=3) ---")
    upsample_st_1 = upsample_st_1(single_point_st)
    # 3. Apply the first upsampling layer (upsample_st_2)
    print("--- Applying upsample_st_2 (stride=2, kernel=3) ---")
    upsampled_2 = upsample_st_2(upsample_st_1)
    print(f"Resulting Tensor Stride: {upsampled_2.tensor_stride}")
    print(f"Number of output points: {len(upsampled_2)}")
    print(f"Min coords: {upsampled_2.C[:, 1:].min(0).values}")
    print(f"Max coords: {upsampled_2.C[:, 1:].max(0).values}\n")


    # 4. Apply the second upsampling layer (upsample_st_4)
    print("--- Applying upsample_st_4 (stride=4, kernel=3) ---")
    upsampled_4 = upsample_st_4(upsampled_2)
    print(f"Final Tensor Stride: {upsampled_4.tensor_stride}")
    print(f"Total number of final points: {len(upsampled_4)}")

    # 5. Verify the final coordinate range
    print("\n--- Verifying Final Coordinate Range ---")
    # Expected center: (8*8, 8*16, 8*24) = (64, 128, 192)
    # Expected offset range: [-5, 5]
    # Expected range: [59, 69], [123, 133], [187, 197]
    min_coords = upsampled_4.C[:, 1:].min(0).values
    max_coords = upsampled_4.C[:, 1:].max(0).values
    print(f"Final X coordinates range: [{min_coords[0]}, {max_coords[0]}]")
    print(f"Final Y coordinates range: [{min_coords[1]}, {max_coords[1]}]")
    print(f"Final Z coordinates range: [{min_coords[2]}, {max_coords[2]}]")

    # 6. Print coordinates of non-zero feature vectors
    print("\n--- Coordinates of non-zero feature vectors ---")
    final_feats = upsampled_4.F
    # For this simple test, all features will be non-zero because the kernel is initialized
    # with non-zero weights and the input feature is non-zero.
    non_zero_mask = torch.any(final_feats != 0, dim=1)
    coords_with_nonzero_feats = upsampled_4.C[non_zero_mask]
    print(f"Found {len(coords_with_nonzero_feats)} points with non-zero features.")
    # print(coords_with_nonzero_feats) # This can be very long, so we don't print all by default.
    print("Test finished.")
    pdb.set_trace()


if __name__ == "__main__":
    test_upsample_projection()

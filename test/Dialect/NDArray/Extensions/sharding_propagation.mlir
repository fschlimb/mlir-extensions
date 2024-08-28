// RUN: imex-opt %s --pass-pipeline="builtin.module(func.func(sharding-propagation))" | FileCheck %s

builtin.module {
    
mesh.mesh @mesh4(shape = 4)
// CHECK-LABEL: @test_shard_propagate_balanced
func.func @test_shard_propagate_balanced(%arg0: tensor<1024x1024xi64>) -> tensor<4x3xi64> {
    // CHECK: [[S:%.*]] = mesh.sharding @mesh4 split_axes = {{\[\[}}0]] : !mesh.sharding
    %s = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    // CHECK: mesh.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: mesh.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_sizes = [1, 1, 1, 1] : !mesh.sharding
    %1 = ndarray.subview %0[1, 0][4, 3][256, 1] : tensor<1024x1024xi64> to tensor<4x3xi64>
    return %1 : tensor<4x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_leading
func.func @test_shard_propagate_leading(%arg0: tensor<1024x1024xi64>) -> tensor<3x3xi64> {
    // CHECK: [[S:%.*]] = mesh.sharding @mesh4 split_axes = {{\[\[}}0]] : !mesh.sharding
    %s = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    // CHECK: mesh.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: mesh.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_sizes = [3, 0, 0, 0] : !mesh.sharding
    %1 = ndarray.subview %0[0, 0][3, 3][3, 1] : tensor<1024x1024xi64> to tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_mid
func.func @test_shard_propagate_mid(%arg0: tensor<1024x1024xi64>) -> tensor<3x3xi64> {
    // CHECK: [[S:%.*]] = mesh.sharding @mesh4 split_axes = {{\[\[}}0]] : !mesh.sharding
    %s = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    // CHECK: mesh.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: mesh.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_sizes = [0, 1, 2, 0] : !mesh.sharding
    %1 = ndarray.subview %0[511, 0][3, 3][1, 1] : tensor<1024x1024xi64> to tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_trailing
func.func @test_shard_propagate_trailing(%arg0: tensor<1024x1024xi64>) -> tensor<3x3xi64> {
    // CHECK: [[S:%.*]] = mesh.sharding @mesh4 split_axes = {{\[\[}}0]] : !mesh.sharding
    %s = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    // CHECK: mesh.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: mesh.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_sizes = [0, 0, 0, 3] : !mesh.sharding
    %1 = ndarray.subview %0[1000, 0][3, 3][1, 1] : tensor<1024x1024xi64> to tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
}

// CHECK-LABEL: @test_shard_propagate_gap
func.func @test_shard_propagate_gap(%arg0: tensor<1024x1024xi64>) -> tensor<3x3xi64> {
    // CHECK: [[S:%.*]] = mesh.sharding @mesh4 split_axes = {{\[\[}}0]] : !mesh.sharding
    %s = mesh.sharding @mesh4 split_axes = [[0]] : !mesh.sharding
    // CHECK: mesh.shard %arg0 to [[S]] : tensor<1024x1024xi64>
    %0 = mesh.shard %arg0 to %s : tensor<1024x1024xi64>
    // CHECK: mesh.sharding @mesh4 split_axes = {{\[\[}}0]] sharded_dims_sizes = [1, 0, 1, 1] : !mesh.sharding
    %1 = ndarray.subview %0[255, 0][3, 3][257, 1] : tensor<1024x1024xi64> to tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
}

}
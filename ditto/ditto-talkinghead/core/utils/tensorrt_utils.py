"""TensorRT Wrapper: manages TRT engine loading, buffer allocation, and inference.

Provides a unified interface for loading TensorRT engines from disk, managing
host/device buffer pairs, and running inference. Supports:
  - Dynamic-shape inputs with shape memoization (skip malloc when shapes unchanged)
  - Data-Dependent-Shape (DDS) outputs via IOutputAllocator
  - Async memcpy for overlapping compute and data transfer
  - CUDA graph capture/replay for fixed-shape workloads (zero launch overhead)

Buffer layout: self.buffer[tensor_name] = [host_buffer, device_buffer, num_bytes]
  - host_buffer: numpy array matching the tensor's dtype and shape
  - device_buffer: device pointer (int) from cudaMalloc, or None for host-only tensors
  - num_bytes: allocated size in bytes
"""

import ctypes
from collections import OrderedDict
import numpy as np
import torch
import os

# cuda-python v13+ moved to cuda.bindings; v12 uses cuda.cuda/cudart/nvrtc
try:
    from cuda import cuda, cudart, nvrtc
except ImportError:
    from cuda.bindings import driver as cuda, runtime as cudart, nvrtc

import tensorrt as trt

trt_logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(trt_logger, "")


def _get_cuda_error_name(error):
    """Get the human-readable name for a CUDA error code."""
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def check_cuda_errors(result):
    """Check a CUDA API return value and raise on error.

    Args:
        result: Tuple returned by cuda-python API calls. First element is the
            error code; remaining elements are return values.

    Returns:
        The unwrapped return value(s), or None if the call returns only a status.
    """
    if result[0].value:
        raise RuntimeError(
            f"CUDA error code={result[0].value}({_get_cuda_error_name(result[0])})"
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


# Backward-compatible aliases for external callers
checkCudaErrors = check_cuda_errors
_cudaGetErrorEnum = _get_cuda_error_name


class DynamicShapeOutputAllocator(trt.IOutputAllocator):
    """Allocator for TRT outputs whose shape depends on input data (DDS).

    TensorRT calls reallocate_output when a DDS output needs memory. The
    allocator tracks the current shape and device address so the wrapper
    can copy results back to host after inference.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shape = None
        self.num_bytes = 0
        self.address = 0

    def reallocate_output(self, tensor_name, old_address, size, alignment) -> int:
        return self._reallocate(tensor_name, old_address, size, alignment)

    def reallocate_output_async(
        self, tensor_name, old_address, size, alignment, stream
    ) -> int:
        return self._reallocate(tensor_name, old_address, size, alignment, stream)

    def notify_shape(self, tensor_name, shape):
        self.shape = shape

    def _reallocate(
        self, tensor_name, old_address, size, alignment, stream=-1
    ):
        """Reallocate device memory if needed, freeing the old buffer first."""
        if size <= self.num_bytes:
            return old_address
        if old_address != 0:
            check_cuda_errors(cudart.cudaFree(old_address))
        if stream == -1:
            address = check_cuda_errors(cudart.cudaMalloc(size))
        else:
            address = check_cuda_errors(cudart.cudaMallocAsync(size, stream))
        self.num_bytes = size
        self.address = address
        return address


# Backward-compatible alias
MyOutputAllocator = DynamicShapeOutputAllocator


class TRTWrapper:
    """TensorRT engine wrapper with buffer management and shape memoization.

    Usage::

        wrapper = TRTWrapper("model.engine")
        wrapper.setup({"input_name": input_array})
        wrapper.infer()
        output = wrapper.buffer["output_name"][0]  # host numpy array
    """

    def __init__(
        self,
        trt_file: str,
        plugin_file_list: list = [],
    ) -> None:
        # Load custom TRT plugins (e.g. GridSample3D for warp network)
        for plugin_file in plugin_file_list:
            ctypes.cdll.LoadLibrary(plugin_file)

        self.model = trt_file
        with open(trt_file, "rb") as engine_file, trt.Runtime(trt_logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        assert self.engine

        # buffer[name] = [host_array, device_ptr, num_bytes]
        self.buffer = OrderedDict()
        self.output_allocator_map = OrderedDict()
        self.context = self.engine.create_execution_context()

        # Shape memoization: skip cudaMalloc/cudaFree when input shapes unchanged
        self._cached_shapes = {}

    def setup(self, input_data: dict = {}) -> None:
        """Configure input shapes and allocate I/O buffers.

        Uses shape memoization: if input shapes match the previous call,
        skips all cudaFree/cudaMalloc operations (major performance win
        when processing frames of identical resolution).

        Args:
            input_data: Dict mapping input tensor names to numpy arrays.
        """
        # Build tensor name list once on first call
        if not hasattr(self, "tensor_name_list") or not self.tensor_name_list:
            self.tensor_name_list = [
                self.engine.get_tensor_name(idx)
                for idx in range(self.engine.num_io_tensors)
            ]
            self.num_inputs = sum(
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                for name in self.tensor_name_list
            )
            self.num_outputs = self.engine.num_io_tensors - self.num_inputs

        # Check if input shapes changed since last call
        new_shapes = {name: tuple(data.shape) for name, data in input_data.items()}
        shapes_changed = new_shapes != self._cached_shapes

        if shapes_changed:
            # Free old device buffers before reallocating
            for name, value in self.buffer.items():
                _, device_ptr, _ = value
                if (
                    device_ptr is not None
                    and device_ptr != 0
                    and name not in self.output_allocator_map
                ):
                    check_cuda_errors(cudart.cudaFree(device_ptr))
                    self.buffer[name][1] = None
                    self.buffer[name][2] = 0

        # Set input shapes/addresses on the execution context
        for name, data in input_data.items():
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                self.context.set_input_shape(name, data.shape)
            else:
                self.context.set_tensor_address(name, data.ctypes.data)

        if shapes_changed:
            # Allocate host and device buffers for all tensors
            for name in self.tensor_name_list:
                data_type = self.engine.get_tensor_dtype(name)
                runtime_shape = self.context.get_tensor_shape(name)

                if name not in self.output_allocator_map:
                    if -1 in runtime_shape:
                        # Data-Dependent-Shape output: use dynamic allocator
                        allocator = DynamicShapeOutputAllocator()
                        self.output_allocator_map[name] = allocator
                        self.context.set_output_allocator(name, allocator)
                        host_buffer = np.empty(0, dtype=trt.nptype(data_type))
                        device_ptr = None
                        num_bytes = 0
                    else:
                        num_bytes = trt.volume(runtime_shape) * data_type.itemsize
                        host_buffer = np.empty(
                            runtime_shape, dtype=trt.nptype(data_type)
                        )
                        if (
                            self.engine.get_tensor_location(name)
                            == trt.TensorLocation.DEVICE
                        ):
                            device_ptr = check_cuda_errors(cudart.cudaMalloc(num_bytes))
                        else:
                            device_ptr = None
                    self.buffer[name] = [host_buffer, device_ptr, num_bytes]

            self._cached_shapes = new_shapes

        # Update host input buffers (always needed even when shapes unchanged)
        for name, data in input_data.items():
            self.buffer[name][0] = np.ascontiguousarray(data)

        # Set tensor addresses on context (always needed — context may reset)
        for name in self.tensor_name_list:
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                if self.buffer[name][1] is not None:
                    self.context.set_tensor_address(name, self.buffer[name][1])
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.context.set_tensor_address(
                    name, self.buffer[name][0].ctypes.data
                )

    def infer(self, stream=0) -> None:
        """Run synchronous inference: H2D copy → execute → D2H copy.

        Args:
            stream: CUDA stream handle (0 = default stream).
        """
        # Copy inputs from host to device
        for name in self.tensor_name_list:
            if (
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE
            ):
                cudart.cudaMemcpy(
                    self.buffer[name][1],
                    self.buffer[name][0].ctypes.data,
                    self.buffer[name][2],
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )

        # Execute the TRT engine
        self.context.execute_async_v3(stream)

        # Handle DDS outputs: get final shape and address from allocator
        for name in self.output_allocator_map:
            allocator = self.context.get_output_allocator(name)
            runtime_shape = allocator.shape
            data_type = self.engine.get_tensor_dtype(name)
            host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
            device_ptr = allocator.address
            num_bytes = trt.volume(runtime_shape) * data_type.itemsize
            self.buffer[name] = [host_buffer, device_ptr, num_bytes]

        # Copy outputs from device to host
        for name in self.tensor_name_list:
            if (
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
                and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE
            ):
                cudart.cudaMemcpy(
                    self.buffer[name][0].ctypes.data,
                    self.buffer[name][1],
                    self.buffer[name][2],
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )

    def infer_async(self, stream=0) -> None:
        """Run asynchronous inference using cudaMemcpyAsync.

        Caller must synchronize the stream before reading outputs.

        Args:
            stream: CUDA stream handle for async operations.
        """
        # Async H2D copy
        for name in self.tensor_name_list:
            if (
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE
            ):
                cudart.cudaMemcpyAsync(
                    self.buffer[name][1],
                    self.buffer[name][0].ctypes.data,
                    self.buffer[name][2],
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream=stream,
                )

        self.context.execute_async_v3(stream)

        # Handle DDS outputs
        for name in self.output_allocator_map:
            allocator = self.context.get_output_allocator(name)
            runtime_shape = allocator.shape
            data_type = self.engine.get_tensor_dtype(name)
            host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
            device_ptr = allocator.address
            num_bytes = trt.volume(runtime_shape) * data_type.itemsize
            self.buffer[name] = [host_buffer, device_ptr, num_bytes]

        # Async D2H copy
        for name in self.tensor_name_list:
            if (
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
                and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE
            ):
                cudart.cudaMemcpyAsync(
                    self.buffer[name][0].ctypes.data,
                    self.buffer[name][1],
                    self.buffer[name][2],
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream=stream,
                )

    # ---- Fixed-shape inference with CUDA graphs ----

    def setup_fixed(self, input_data: dict) -> None:
        """One-time setup for fixed-shape repeated inference.

        Pre-allocates all I/O buffers so that subsequent calls only need to
        update input data via update_input() — no malloc/free per frame.
        After calling this, use update_input() + infer() for per-frame work,
        or capture_cuda_graph() + infer_graph() for zero-overhead replay.

        Args:
            input_data: Dict mapping input tensor names to numpy arrays
                with the shapes that will be used for all future calls.
        """
        # Run normal setup once to allocate everything
        self.setup(input_data)

        self._fixed_input_shapes = {
            name: tuple(data.shape) for name, data in input_data.items()
        }
        self._fixed = True

        # Dedicated CUDA stream for graph operations
        self._stream = check_cuda_errors(cudart.cudaStreamCreate())
        self._graph_exec = None

    def update_input(self, input_data: dict) -> None:
        """Update host input buffers without re-allocating device memory.

        Args:
            input_data: Dict mapping input names to numpy arrays. Shapes
                must match those passed to setup_fixed().
        """
        for name, data in input_data.items():
            self.buffer[name][0] = np.ascontiguousarray(data)

    def capture_cuda_graph(self) -> None:
        """Capture a CUDA graph from the H2D -> execute -> D2H sequence.

        After capture, use update_input() then infer_graph() for
        minimal-overhead inference (eliminates kernel launch overhead).

        Raises:
            RuntimeError: If graph capture fails (e.g. engine uses
                non-capturable kernels).
        """
        cuda_stream = self._stream

        # Warm-up run required before capture
        self.infer(stream=cuda_stream)

        # Begin graph capture
        check_cuda_errors(
            cudart.cudaStreamBeginCapture(
                cuda_stream,
                cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal,
            )
        )

        # H2D copies
        for name in self.tensor_name_list:
            if (
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE
            ):
                cudart.cudaMemcpyAsync(
                    self.buffer[name][1],
                    self.buffer[name][0].ctypes.data,
                    self.buffer[name][2],
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream=cuda_stream,
                )

        self.context.execute_async_v3(cuda_stream)

        # D2H copies
        for name in self.tensor_name_list:
            if (
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
                and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE
            ):
                cudart.cudaMemcpyAsync(
                    self.buffer[name][0].ctypes.data,
                    self.buffer[name][1],
                    self.buffer[name][2],
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream=cuda_stream,
                )

        # End capture — always called even if capture was invalidated
        result = cudart.cudaStreamEndCapture(cuda_stream)
        error_code = result[0]
        if error_code.value != 0:
            cudart.cudaStreamSynchronize(cuda_stream)
            raise RuntimeError(
                f"CUDA graph capture failed: {_get_cuda_error_name(error_code)}"
            )
        graph = result[1]

        self._graph_exec = check_cuda_errors(
            cudart.cudaGraphInstantiate(graph, 0)
        )
        # Free the graph template (exec copy is independent)
        cudart.cudaGraphDestroy(graph)

    def infer_graph(self) -> None:
        """Run inference by replaying the captured CUDA graph.

        Input data must have been written to host buffers via update_input()
        before calling this — the graph's H2D copies read from the same
        host addresses that were captured.
        """
        check_cuda_errors(
            cudart.cudaGraphLaunch(self._graph_exec, self._stream)
        )
        check_cuda_errors(cudart.cudaStreamSynchronize(self._stream))

    def __del__(self):
        """Clean up CUDA resources on garbage collection."""
        if hasattr(self, "_graph_exec") and self._graph_exec is not None:
            try:
                cudart.cudaGraphExecDestroy(self._graph_exec)
            except (TypeError, Exception):
                pass
        if hasattr(self, "_stream") and self._stream is not None:
            try:
                cudart.cudaStreamDestroy(self._stream)
            except (TypeError, Exception):
                pass
        if hasattr(self, "buffer") and self.buffer is not None:
            for _, device_ptr, _ in self.buffer.values():
                if (
                    device_ptr is not None
                    and device_ptr != 0
                    and cudart is not None
                ):
                    try:
                        check_cuda_errors(cudart.cudaFree(device_ptr))
                    except TypeError:
                        pass

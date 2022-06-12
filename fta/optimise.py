import torch
import torch_tensorrt

def compile_model(model, device, input_shapes=[(512, 1792//2)], input_script_path=None, output_script_path=None, output_trt_path=None):
    import torch_tensorrt
    if input_script_path:
        script_model = torch.jit.load(input_script_path, map_location=device)
    else:
        script_model = torch.jit.script(model)
    if output_script_path:
        torch.jit.save(script_model,  output_script_path)
    trt_model = torch_tensorrt.compile(
        script_model,
        inputs=[
            torch_tensorrt.Input([1, 3, shape[0], shape[1]], dtype=torch.float) for shape in input_shapes
        ],
        enabled_precisions={torch.float, torch.half},
        truncate_long_and_double=True
    )
    if output_trt_path:
        torch.jit.save(trt_model, output_trt_path)
    return model

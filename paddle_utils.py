
import paddle

############################## 相关utils函数，如下 ##############################
############################ PaConvert 自动生成的代码 ###########################

def device2str(type=None, index=None, *, device=None):
    type = device if device else type
    if isinstance(type, int):
        type = f'gpu:{type}'
    elif isinstance(type, str):
        if 'cuda' in type:
            type = type.replace('cuda', 'gpu')
        if 'cpu' in type:
            type = 'cpu'
        elif index is not None:
            type = f'{type}:{index}'
    elif isinstance(type, paddle.CPUPlace) or (type is None):
        type = 'cpu'
    elif isinstance(type, paddle.CUDAPlace):
        type = f'gpu:{type.get_device_id()}'

    return type

def _Tensor_max(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "_max", _Tensor_max)

class PaddleFlag:
    cudnn_enabled = True
    cudnn_benchmark = False
    matmul_allow_tf32 = False
    cudnn_allow_tf32 = True
    cudnn_deterministic = False

def _Tensor_split(self, split_size, dim=0):
    if isinstance(split_size, int):
        return paddle.split(self, self.shape[dim] // split_size, dim)
    else:
        return paddle.split(self, split_size, dim)

setattr(paddle.Tensor, "split", _Tensor_split)
############################## 相关utils函数，如上 ##############################


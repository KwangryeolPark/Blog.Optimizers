import tensorflow as tf
from tensorflow import keras

class Batch_Gradient_Descent(keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate = 0.01,
        name = "BGD",
        **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        #   Parameter의 데이터 타입(float32 or others)을 가지고 옵니다. 
        var_dtype = var.dtype.base_dtype
        #   Scheduler에 의해 변경된 learning_rate를 가지고 옵니다. 이는 Batch Gradient Descent와는 무관한 learning_rate scheduling을 위해 넣은 코드입니다.
        lr_t = self._decayed_lr(var_dtype)
        #   실제 Batch Gradient Descent 알고리즘을 적용합니다.
        new_var = var - lr_t * grad
        #   알고리즘으로 계산된 새로운 parameter를 업데이트합니다.
        var.assign(new_var)
  
    def _resource_apply_sparse(self, grad, var):
        #   본 class는 sparse 업데이트를 허용하지 않습니다.(구현하지 않았습니다.)
        raise NotImplementedError
  
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate")
        }
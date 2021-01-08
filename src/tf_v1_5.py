# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:42:19 2019

@author: laip2
"""

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import nn

def append_tensor_alias(tensor, alias):
  """Append an alias to the list of aliases of the tensor.
  Args:
    tensor: A `Tensor`.
    alias: String, to add to the list of aliases of the tensor.
  Returns:
    The tensor with a new alias appended to its list of aliases.
  """
  # Remove ending '/' if present.
  if alias[-1] == '/':
    alias = alias[:-1]
  if hasattr(tensor, 'aliases'):
    tensor.aliases.append(alias)
  else:
    tensor.aliases = [alias]
  return tensor

def collect_named_outputs(collections, alias, outputs):
  """Add `Tensor` outputs tagged with alias to collections.
  It is useful to collect end-points or tags for summaries. Example of usage:
  logits = collect_named_outputs('end_points', 'inception_v3/logits', logits)
  assert 'inception_v3/logits' in logits.aliases
  Args:
    collections: A collection or list of collections. If None skip collection.
    alias: String to append to the list of aliases of outputs, for example,
           'inception_v3/conv1'.
    outputs: Tensor, an output tensor to collect
  Returns:
    The outputs Tensor to allow inline call.
  """
  if collections:
    append_tensor_alias(outputs, alias)
    tf.add_to_collections(collections, outputs)
  return outputs

def get_variable_collections(variables_collections, name):
  if isinstance(variables_collections, dict):
    variable_collections = variables_collections.get(name, None)
  else:
    variable_collections = variables_collections
  return variable_collections

def layer_norm(inputs,
               center=True,
               scale=True,
               activation_fn=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               begin_norm_axis=1,
               begin_params_axis=-1,
               scope=None):
  """Adds a Layer Normalization layer.
  Based on the paper:
    "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    https://arxiv.org/abs/1607.06450.
  Can be used as a normalizer function for conv2d and fully_connected.
  Given a tensor `inputs` of rank `R`, moments are calculated and normalization
  is performed over axes `begin_norm_axis ... R - 1`.  Scaling and centering,
  if requested, is performed over axes `begin_params_axis .. R - 1`.
  By default, `begin_norm_axis = 1` and `begin_params_axis = -1`,
  meaning that normalization is performed over all but the first axis
  (the `HWC` if `inputs` is `NHWC`), while the `beta` and `gamma` trainable
  parameters are calculated for the rightmost axis (the `C` if `inputs` is
  `NHWC`).  Scaling and recentering is performed via broadcast of the
  `beta` and `gamma` parameters with the normalized tensor.
  The shapes of `beta` and `gamma` are `inputs.shape[begin_params_axis:]`,
  and this part of the inputs' shape must be fully defined.
  Args:
    inputs: A tensor having rank `R`. The normalization is performed over axes
      `begin_norm_axis ... R - 1` and centering and scaling parameters are
      calculated over `begin_params_axis ... R - 1`.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is not used. When the
      next layer is linear (also e.g. `nn.relu`), this can be disabled since the
      scaling can be done by the next layer.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    begin_norm_axis: The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: The first parameter (beta, gamma) dimension: scale and
      centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
        normalized inputs accordingly.
    scope: Optional scope for `variable_scope`.
  Returns:
    A `Tensor` representing the output of the operation, having the same
    shape and dtype as `inputs`.
  Raises:
    ValueError: If the rank of `inputs` is not known at graph build time,
      or if `inputs.shape[begin_params_axis:]` is not fully defined at
      graph build time.
  """
  with tf.variable_scope(
      scope, 'LayerNorm', [inputs], reuse=reuse) as sc:
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.shape
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if begin_norm_axis < 0:
      begin_norm_axis = inputs_rank + begin_norm_axis
    if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
      raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                       'must be < rank(inputs) (%d)' %
                       (begin_params_axis, begin_norm_axis, inputs_rank))
    params_shape = inputs_shape[begin_params_axis:]
    if not params_shape.is_fully_defined():
      raise ValueError(
          'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
          (inputs.name, begin_params_axis, inputs_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta_collections = get_variable_collections(variables_collections,
                                                        'beta')
      beta = tf.get_variable(
          'beta',
          shape=params_shape,
          dtype=dtype,
          initializer=tf.zeros_initializer(),
          collections=beta_collections,
          trainable=trainable)
    if scale:
      gamma_collections = get_variable_collections(
          variables_collections, 'gamma')
      gamma = tf.get_variable(
          'gamma',
          shape=params_shape,
          dtype=dtype,
          initializer=tf.ones_initializer(),
          collections=gamma_collections,
          trainable=trainable)
    # By default, compute the moments across all the dimensions except the one with index 0.
    norm_axes = list(range(begin_norm_axis, inputs_rank))
    mean, variance = nn.moments(inputs, norm_axes, keep_dims=True)
    # Compute layer normalization using the batch_normalization function.
    # Note that epsilon must be increased for float16 due to the limited
    # representable range.
    variance_epsilon = 1e-12 if dtype != tf.dtypes.float16 else 1e-3
    outputs = nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=variance_epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return collect_named_outputs(outputs_collections, sc.name, outputs)
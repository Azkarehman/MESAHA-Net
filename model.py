import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    ReLU,
    MaxPooling2D,
    UpSampling2D,
    Add,
    Multiply,
    Activation,
    Input,
    Concatenate
)
from tensorflow.keras.models import Model

#------------------------------------------------------------------------------
# (a) Convolutional Block
#------------------------------------------------------------------------------
# This block consists of a 3x3 Convolution, followed by a ReLU activation
# with Batch Normalization, and finally a Max Pooling layer.
def conv_block(inputs, num_filters, name):
    """
    Implements the Conv Block from the diagram.

    Args:
        inputs (tf.Tensor): The input tensor.
        num_filters (int): The number of filters for the convolutional layer.
        name (str): Name for the block.

    Returns:
        tf.Tensor: The output tensor after the block.
    """
    # 2D Convolution with a 3x3 kernel
    x = Conv2D(num_filters, (3, 3), padding='same', name=f"{name}_conv")(inputs)

    # ReLU activation and Batch Normalization
    x = ReLU(name=f"{name}_relu")(x)
    x = BatchNormalization(name=f"{name}_bn")(x)

    # Max Pooling
    pooled_output = MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool")(x)

    return x, pooled_output # Return both pre-pool and post-pool for skip connections

#------------------------------------------------------------------------------
# (b) Deconvolutional Block
#------------------------------------------------------------------------------
# This block first upsamples the input, then applies a 3x3 Convolution,
# and finishes with a ReLU activation and Batch Normalization.
def deconv_block(inputs, num_filters, name):
    """
    Implements the Deconv Block from the diagram.

    Args:
        inputs (tf.Tensor): The input tensor.
        num_filters (int): The number of filters for the convolutional layer.
        name (str): Name for the block.

    Returns:
        tf.Tensor: The output tensor after the block.
    """
    # Upsampling
    x = UpSampling2D(size=(2, 2), name=f"{name}_upsample")(inputs)

    # 2D Convolution with a 3x3 kernel
    x = Conv2D(num_filters, (3, 3), padding='same', name=f"{name}_conv")(x)

    # ReLU activation and Batch Normalization
    x = ReLU(name=f"{name}_relu")(x)
    x = BatchNormalization(name=f"{name}_bn")(x)

    return x

#------------------------------------------------------------------------------
# (c) HAGen Block
#------------------------------------------------------------------------------
# This block is a simpler version of the Conv Block, without the pooling layer.
# It consists of a 3x3 Convolution followed by ReLU and Batch Normalization.
def hagen_block(inputs, num_filters, name):
    """
    Implements the HAGen Block from the diagram.

    Args:
        inputs (tf.Tensor): The input tensor.
        num_filters (int): The number of filters for the convolutional layer.
        name (str): Name for the block.

    Returns:
        tf.Tensor: The output tensor after the block.
    """
    # 2D Convolution with a 3x3 kernel
    x = Conv2D(num_filters, (3, 3), padding='same', name=f"{name}_conv")(inputs)

    # ReLU activation and Batch Normalization
    x = ReLU(name=f"{name}_relu")(x)
    x = BatchNormalization(name=f"{name}_bn")(x)

    return x

#------------------------------------------------------------------------------
# (d) Attention Unit (Attention Gate)
#------------------------------------------------------------------------------
# This unit implements an attention mechanism, often used in architectures
# like Attention U-Net. It takes two inputs: 'x' from a skip connection and
# 'g' from the main network path. It learns to focus on salient features in 'x'.
def attention_unit(x, g, num_filters, name):
    """
    Implements the Attention Unit (Attention Gate) from the diagram.

    Args:
        x (tf.Tensor): The input tensor from the skip connection (higher resolution).
        g (tf.Tensor): The input tensor from the main path (lower resolution).
        num_filters (int): The number of intermediate filters.
        name (str): Name for the block.

    Returns:
        tf.Tensor: The output tensor, which is the attended features from x.
    """
    # Process the skip connection input 'x'
    theta_x = Conv2D(num_filters, (1, 1), strides=(1, 1), padding='same', name=f"{name}_theta_x")(x)

    # Process the gating signal 'g'
    phi_g = Conv2D(num_filters, (1, 1), strides=(1, 1), padding='same', name=f"{name}_phi_g")(g)

    # Add the processed signals
    f = Add(name=f"{name}_add")([theta_x, phi_g])
    f = Activation('relu', name=f"{name}_relu")(f)

    # Generate the attention coefficients
    psi_f = Conv2D(1, (1, 1), padding='same', name=f"{name}_psi_f")(f)
    alpha = Activation('sigmoid', name=f"{name}_sigmoid")(psi_f)

    # Apply the attention coefficients to the original skip connection input 'x'
    attended_x = Multiply(name=f"{name}_multiply")([x, alpha])

    return attended_x

#------------------------------------------------------------------------------
# Full Model from Flow Diagram
#------------------------------------------------------------------------------
def build_multi_input_unet(input_shape=(256, 256, 1), num_classes=1):
    """
    Builds the full U-Net model based on the provided flow diagram.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of output classes for segmentation.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    # Define the four inputs
    input_raw = Input(shape=input_shape, name='input_raw_slice')
    input_fwd = Input(shape=input_shape, name='input_forward_mip')
    input_bwd = Input(shape=input_shape, name='input_backward_mip')
    input_cad = Input(shape=input_shape, name='input_cad_system')

    # --- ENCODER ---
    # Three parallel encoder streams
    s1_raw, p1_raw = conv_block(input_raw, 64, name='enc1_raw')
    s2_raw, p2_raw = conv_block(p1_raw, 128, name='enc2_raw')
    s3_raw, p3_raw = conv_block(p2_raw, 256, name='enc3_raw')
    s4_raw, p4_raw = conv_block(p3_raw, 512, name='enc4_raw')

    s1_fwd, p1_fwd = conv_block(input_fwd, 64, name='enc1_fwd')
    s2_fwd, p2_fwd = conv_block(p1_fwd, 128, name='enc2_fwd')
    s3_fwd, p3_fwd = conv_block(p2_fwd, 256, name='enc3_fwd')
    s4_fwd, p4_fwd = conv_block(p3_fwd, 512, name='enc4_fwd')

    s1_bwd, p1_bwd = conv_block(input_bwd, 64, name='enc1_bwd')
    s2_bwd, p2_bwd = conv_block(p1_bwd, 128, name='enc2_bwd')
    s3_bwd, p3_bwd = conv_block(p2_bwd, 256, name='enc3_bwd')
    s4_bwd, p4_bwd = conv_block(p3_bwd, 512, name='enc4_bwd')

    # Bottleneck
    b_raw = hagen_block(p4_raw, 1024, name='bottleneck_raw')
    b_fwd = hagen_block(p4_fwd, 1024, name='bottleneck_fwd')
    b_bwd = hagen_block(p4_bwd, 1024, name='bottleneck_bwd')
    bottleneck = Concatenate(name='bottleneck_concat')([b_raw, b_fwd, b_bwd])


    # --- HA-Generator Path ---
    g1 = hagen_block(input_cad, 64, name='hagen1')
    g2 = hagen_block(g1, 128, name='hagen2')
    g3 = hagen_block(g2, 256, name='hagen3')
    g4 = hagen_block(g3, 512, name='hagen4')


    # --- DECODER ---
    # Combine skip connections at each level
    skip4 = Concatenate(name='skip4_concat')([s4_raw, s4_fwd, s4_bwd])
    skip3 = Concatenate(name='skip3_concat')([s3_raw, s3_fwd, s3_bwd])
    skip2 = Concatenate(name='skip2_concat')([s2_raw, s2_fwd, s2_bwd])
    skip1 = Concatenate(name='skip1_concat')([s1_raw, s1_fwd, s1_bwd])

    # Decoder block 1
    d1 = deconv_block(bottleneck, 512, name='dec1')
    attn1 = attention_unit(skip4, g4, 512, name='attn1')
    c1 = Concatenate(name='dec1_concat')([d1, attn1])
    dec_b1 = hagen_block(c1, 512, name='dec_block1')

    # Decoder block 2
    d2 = deconv_block(dec_b1, 256, name='dec2')
    attn2 = attention_unit(skip3, g3, 256, name='attn2')
    c2 = Concatenate(name='dec2_concat')([d2, attn2])
    dec_b2 = hagen_block(c2, 256, name='dec_block2')

    # Decoder block 3
    d3 = deconv_block(dec_b2, 128, name='dec3')
    attn3 = attention_unit(skip2, g2, 128, name='attn3')
    c3 = Concatenate(name='dec3_concat')([d3, attn3])
    dec_b3 = hagen_block(c3, 128, name='dec_block3')

    # Decoder block 4
    d4 = deconv_block(dec_b3, 64, name='dec4')
    attn4 = attention_unit(skip1, g1, 64, name='attn4')
    c4 = Concatenate(name='dec4_concat')([d4, attn4])
    dec_b4 = hagen_block(c4, 64, name='dec_block4')

    # --- OUTPUT ---
    # Final 1x1 Convolution and Sigmoid activation
    output = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid', name='final_output')(dec_b4)

    # Create the model
    model = Model(inputs=[input_raw, input_fwd, input_bwd, input_cad], outputs=output, name='Multi_Input_Attention_UNet')

    return model



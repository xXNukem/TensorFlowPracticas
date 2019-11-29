def VGG16_Without_lastPool(include_top=False, input_tensor='imagenet', input_shape=(224,224,3), pooling=None, classes=5):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
        # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)  # to 16x16

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)  # to 8x8

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)  # to 4x4

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)  # to 2x2

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.2)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.2)(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Create model.
    model = Model(img_input, x, name='vgg16Bis')
    return model

def create_vgg16WithoutPool():
  model = VGG16_Without_lastPool(include_top=True, input_tensor=None, input_shape=(224,224,3), pooling=None, classes=5)
  return model

vgg16Bis_model = create_vgg16WithoutPool()
vgg16Bis_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])

vgg16Bis_model.fit_generator(
    train_generator,
    steps_per_epoch=500,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=100,
    callbacks=[tensorboard_callback])
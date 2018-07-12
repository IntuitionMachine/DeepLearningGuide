'''Autonomous driving convolutional neural network example from p. 64 of the book.

Code based on https://github.com/OSSDC/AutonomousDrivingCookbook/tree/master/AirSimE2EDeepLearning'''

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, concatenate
from keras.optimizers import Nadam
from keras.preprocessing import image
from Generator import DriveDataGenerator



'''Runs p. 64 convolutional neural network.'''
def main():

    train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5'), 'r')

    data_generator  = DriveDataGenerator(rescale=1./255., horizontal_flip=True, brighten_range=0.4)
    train_generator = data_generator.flow\
    (train_dataset['image'], train_dataset['previous_state'], train_dataset['label'], batch_size=batch_size, zero_drop_percentage=0.95, roi=[76,135,0,255])
    eval_generator = data_generator.flow\
    (eval_dataset['image'], eval_dataset['previous_state'], eval_dataset['label'], batch_size=batch_size, zero_drop_percentage=0.95, roi=[76,135,0,255])
    [sample_batch_train_data, sample_batch_test_data] = next(train_generator)

    # above ross add

    image_input_shape = sample_batch_train_data[0].shape[1:]
    state_input_shape = sample_batch_train_data[1].shape[1:]
    activation = 'relu'

    #Create the convolutional stacks
    pic_input = Input(shape=image_input_shape)

    img_stack = Conv2D(16, (3, 3), name="convolution0", padding='same', activation=activation)(pic_input)
    img_stack = MaxPooling2D(pool_size=(2,2))(img_stack)
    img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution1')(img_stack)
    img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
    img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution2')(img_stack)
    img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
    img_stack = Flatten()(img_stack)
    img_stack = Dropout(0.2)(img_stack)

    #Inject the state input
    state_input = Input(shape=state_input_shape)
    merged = concatenate([img_stack, state_input])

    # Add a few dense layers to finish the model
    merged = Dense(64, activation=activation, name='dense0')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(10, activation=activation, name='dense2')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(1, name='output')(merged)

    adam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = Model(inputs=[pic_input, state_input], outputs=merged)
    model.compile(optimizer=adam, loss='mse')

main()

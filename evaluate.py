from model import build_multi_input_unet
from datagen import datagenrator
from utils import dice_score, sensitivity, ppv
from glob import glob


test_paths = glob('/test_Data/*.npy')


test_datagen = DataGenerator(list_IDs=list(range(len(test_paths))),paths=test_paths,to_fit=True, batch_size=15, dim=(512, 512), shuffle=True)

model = build_multi_input_unet(input_shape=(256, 256, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_score, sensitivity, ppv])

model.load_weights('model_weights.h5')

model.evaluate(test_datagen)
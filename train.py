from model import build_multi_input_unet
from datagen import datagenrator
from utils import dice_score
from glob import glob


train_paths = glob('/train_Data/*.npy')
valid_paths = glob('/valid_Data/*.npy')


train_datagen = DataGenerator(list_IDs=list(range(len(train_paths))),paths=train_paths,to_fit=True, batch_size=15, dim=(512, 512), shuffle=True)
Valid_datagen = DataGenerator(list_IDs=list(range(len(valid_paths))),paths=valid_paths,to_fit=True, batch_size=15, dim=(512, 512), shuffle=True)

model = build_multi_input_unet(input_shape=(256, 256, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_score])

hist = model.fit(train_datagen,validation_data=Valid_datagen,epochs=300)

model.save_weights('model_weights.h5')
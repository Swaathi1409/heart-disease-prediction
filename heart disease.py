import numpy as np
import pickle

loaded_model=pickle.load(open('C://Users//Admin//Pictures//SNU//First Year//2nd Semester//Foundations of Data Science//Project//trained_model.sav','rb'))

input_data=(41,0,1,130,204,0,0,172,0,1.4,2,0,2)
input_data_as_numpy_array=np.asarray(input_data)

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)

if (prediction[0]==0):
  print('Person is healthy. Does not have heart disease')
else:
  print('Person is having heart disease.')

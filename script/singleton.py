
def pobieranie_danych():
    import requests


    r = requests.get('https://raw.githubusercontent.com/datasets/covid-19/main/data/time-series-19-covid-combined.csv').text


    tab=r.split('\n')

    Country=[]
    #date Confirmed,Recovered,Deaths
    Data=[]
    rest=[]
    day=[]
    check=True
    worked=False
    for x in tab[1:-1]:
        line=x.replace(',,',',0,').split(',')
        if worked and line[1]!=was:
                check=False

        if check and line[2] == '0':   
            day.append(line[0])
            worked=True
            was=line[1]



        if line[1] not in Country:
            Country.append(line[1])
            Data.append([[int(line[-3]),int(line[-2]),int(line[-1])]])
        else:
            Data[Country.index(line[1])].append([line[0],int(line[-3]),int(line[-2]),int(line[-1])])



    head='Province/State,Country/Region,Lat,Long'
    for x in range(len(day)):
        head+=','+day[x].replace('-','/').replace('2020','20').replace('2021','21')

    linesOfTab=[]
    print(Country)
    for x in range(len(Country)):
        extnConf=',{},0,0'.format(Country)
        linesOfTab.append([',{},0,0'.format(Country[x]),',{},0,0'.format(Country[x]),',{},0,0'.format(Country[x])])
        for y in day:
            actualDay=[0,0,0]
            for cont in Data[x]:

                if y == cont[0]:
                    for n in range(3):
                        actualDay[n]+=cont[n+1]
                        

            for i in range(3):
                linesOfTab[x][i]+=',{}'.format(actualDay[i])


    typeData=['Confirmed','Recovered','Deaths']
    for i in range(3):
        with open('covid_data_{}.csv'.format(typeData[i]),'w+') as file:
            file.write(head)
            for n in linesOfTab:
                file.write('\n'+n[i])



            




            




def corona_predictions():
    import sys
    CPU = False
    BATCH_SIZE = int(12000 * 5)
    EPOCHS = 500 #3300
    MARGIN_ACCURACY = 0.9950
    MARGIN = 0.9930
    TEST_ONE_AND_MARGIN = True
    TEST_1 = True
    #FILE = "nowe\\dane_2_new4"
    FILE = "covid_data_Confirmed"
    FILE_TEST = "covid_data_Confirmed"
    NAME = "COVNET3"
    LOAD = 0

    ADD_LOCALIZATION_TO_OUTPUT = True

    LOCALIZATION = False #add localization data
    TIME_STEP = 64 #52
    DAYS_FORWARD = 14
    DEBUG_ACCURACY = True #czy ma obliczać accuracy po mojemu (łatwiej debugować, ale wolniej działa)


    SPLIT = False
    THRESHOLD = 0.2
    #FILE = "data_10.05_no_time"
    #FILE_TEST = "data_10.05_no_time"



    import numpy as np

    def isDigit(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    def loadold(nazwa:str):
        print("Loading data...")
        path = "" + nazwa + ".csv"
        try:
            f = open(path,"r")
            data = f.readlines()
            f.close()
        except:
            print("Failed to open: " + path)
            exit(1)


        inp = []
        out = []
        locations = []
        locations_count = len(data) - 1

        for i in range(1,len(data)):
            current_line = data[i].split(',')

            start = 0
            location = ""
            for j in range(len(current_line)):
                if (isDigit(current_line[j])):
                    start = j
                    break
                else:
                    if current_line[j] != '':
                        location += current_line[j] + " | "

            location = location[:len(location)-2]
            

            dane = ""
            for j in range(start,len(current_line)):
                dane += current_line[j] + ","
            dane = dane[:len(dane)-1]

            #przygotowanie train/test
            dane = dane.split(",")

            for j in range(2,len(dane)-TIME_STEP - DAYS_FORWARD):
                append_inp = []
                append_out = []
                #dodanie danych o lokalizacji
                if (LOCALIZATION):
                    append_inp.append(float(dane[0]))
                    append_inp.append(float(dane[1]))

                for z in range(j,j + TIME_STEP):
                    append_inp.append(float(dane[z]))

                append_out.append(float(dane[j + TIME_STEP + DAYS_FORWARD]))

                inp.append(append_inp)
                out.append(append_out)
                if (ADD_LOCALIZATION_TO_OUTPUT):
                    locations.append(location + "," + dane[0] + "," + dane[1])
                else:
                    locations.append(location)

        print("Converted data")

        return inp, out, locations

    def load(nazwa:str):
        print("Loading data...")
        path = "" + nazwa + ".csv"
        try:
            f = open(path,"r")
            data = f.readlines()
            f.close()
        except:
            print("Failed to open: " + path)
            exit(1)


        inp = []
        out = []
        locations = []
        locations_count = len(data) - 1

        for i in range(1,2):
            current_line = data[i].split(',')

            start = 0
            location = ""
            for j in range(len(current_line)):
                if (isDigit(current_line[j])):
                    start = j
                    break
                else:
                    if current_line[j] != '':
                        location += current_line[j] + " | "

            location = location[:len(location)-2]
            

            dane = ""
            for j in range(start,len(current_line)):
                dane += current_line[j] + ","
            dane = dane[:len(dane)-1]

            dane = dane.split(",")

            for j in range(2,len(dane)-TIME_STEP - DAYS_FORWARD):
                append_inp = []
                append_out = []
                #dodanie danych o lokalizacji
                if (LOCALIZATION):
                    append_inp.append(float(dane[0]))
                    append_inp.append(float(dane[1]))

                for z in range(j,j + TIME_STEP):
                    append_inp.append(float(dane[z]))

                append_out.append(float(dane[j + TIME_STEP + DAYS_FORWARD]))

                inp.append([append_inp])
                out.append([append_out])
                if (ADD_LOCALIZATION_TO_OUTPUT):
                    locations.append([location + "," + dane[0] + "," + dane[1]])
                else:
                    locations.append([location])
        print(len(inp))
        print(len(inp[0]))
        for i in range(2,len(data)):
            current_line = data[i].split(',')

            start = 0
            location = ""
            for j in range(len(current_line)):
                if (isDigit(current_line[j])):
                    start = j
                    break
                else:
                    if current_line[j] != '':
                        location += current_line[j] + " | "

            location = location[:len(location)-2]
            

            dane = ""
            for j in range(start,len(current_line)):
                dane += current_line[j] + ","
            dane = dane[:len(dane)-1]

            dane = dane.split(",")

            for j in range(2,len(dane)-TIME_STEP - DAYS_FORWARD):
                append_inp = []
                append_out = []
                #dodanie danych o lokalizacji
                if (LOCALIZATION):
                    append_inp.append(float(dane[0]))
                    append_inp.append(float(dane[1]))

                for z in range(j,j + TIME_STEP):
                    append_inp.append(float(dane[z]))

                append_out.append(float(dane[j + TIME_STEP + DAYS_FORWARD]))

                inp[j-2].append(append_inp)
                out[j-2].append(append_out)
                if (ADD_LOCALIZATION_TO_OUTPUT):
                    locations[j-2].append(location + "," + dane[0] + "," + dane[1])
                else:
                    locations[j-2].append(location)

        print("Converted data")
        print(len(inp))
        print(len(inp[0]))

        return inp, out, locations

    def normalize_data(out,target:bool = False):
        if target:
            maks = np.max(out, axis=0)
            print("Max:",maks)
            print(maks.shape)

            maks = np.clip(maks,a_min=1,a_max=None)

            out = out / maks

            return out, maks
        else:
            maks = out[0][0][0]

            for i in out:
                for j in i:
                    if j[0] > maks:
                        maks = j[0]

            print("Max:",maks)

            for i in range(len(out)):
                for j in range(len(out[i])):
                    out[i][j][0] = out[i][j][0] / (maks + 1.)

            return out, maks

    def denormalize_data(out, maks,target:bool = False):
        out = out * maks
        return out

    def set_seed(Seed:int):
        # Seed value
        # Apparently you may use different seed values at each stage
        seed_value= Seed

        # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
        import os
        os.environ['PYTHONHASHSEED']=str(seed_value)
        os.environ['HOROVOD_FUSION_THRESHOLD']='0'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        # 2. Set `python` built-in pseudo-random generator at a fixed value
        import random
        random.seed(seed_value)

        # 3. Set `numpy` pseudo-random generator at a fixed value
        import numpy as np
        np.random.seed(seed_value)

        # 4. Set the `tensorflow` pseudo-random generator at a fixed value
        import tensorflow as tf
        tf.random.set_seed(seed_value)
        # for later versions: 
        # tf.compat.v1.set_random_seed(seed_value)

        # 5. Configure a new global `tensorflow` session
        '''
        from keras import backend as K
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)
        '''
        #pip install tensorflow-determinism
        #from tfdeterminism import patch
        #patch()

    def Split_Data(x_train,y_train,ratio = 0.7, SEED = 2, Leak = 0.0):
        import random
        random.seed(SEED)#1
        Train_in = []
        Train_out = []
        Test_in = []
        Test_out = []

        zuzyte = []
        i = 0

        print("All data count: ",len(x_train))
        leak = 0
        while(True):
            number = random.uniform(0.0, 1.0)
            if (number > ratio):
                Test_in.append(x_train[i])
                Test_out.append(y_train[i])
                zuzyte.append(i)
            if (float(len(Test_in) / float(len(x_train)) > 1.0 - ratio)):
                break

            i += 1
            i = i % len(x_train)

        for i in range(len(x_train)):
            jest = False
            for j in range(len(zuzyte)):
                if (zuzyte[j] == i):
                    jest = True
                    break
            number = random.uniform(0.0, 1.0)           
            if (jest == True and number > Leak):
                continue
            else:
                if (number <= Leak and jest == True):
                    leak += 1
                Train_in.append(x_train[i])
                Train_out.append(y_train[i])

        print("Train data count: ",len(Train_in))
        print("Test data count: ",len(Test_in))
        print("Leak: ",leak)
        return Train_in, Train_out, Test_in, Test_out




    from keras import backend as K
    import matplotlib.pyplot as plt
    import pandas as pd
    import time
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, LayerNormalization
    from sklearn.preprocessing import MinMaxScaler

    set_seed(1)
    #tf.config.experimental.enable_mlir_graph_optimization()
    #tf.config.experimental.enable_mlir_bridge()
    tf.keras.mixed_precision.experimental.set_policy("mixed_float16")

    import msvcrt
    class haltCallback(tf.keras.callbacks.Callback):
        def __init__(self,val_x,val_y,tr_x,tr_y,epochs,patience=0):
            self.val_x = val_x
            self.val_y = val_y
            self.tr_x = tr_x
            self.tr_y = tr_y
            self.best_weights = None
            self.patience = patience
            self.epochs = epochs

        def on_train_begin(self, logs=None):
            # The number of epoch it has waited when loss is no longer minimum.
            self.wait = 0
            # The epoch the training stops at.
            self.stopped_epoch = 0
            # Initialize the best as infinity.

            self.best_loss = np.Inf #dla loss

            self.best = 0.0 #dla accuracy
            self.bestv = 0.0

        def on_epoch_end(self, epoch, logs={}):
            #Save best weights
            '''#dla loss
            current = logs.get("val_loss")
            if np.less(current, self.best):
                self.best = current
                self.wait = 0
                # Record the best weights if current results is better (less).
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                #if self.wait >= self.patience:
                if epoch == self.epochs - 1:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print("Restoring model weights from the end of the best epoch.")
                    self.model.set_weights(self.best_weights)
            '''
            #custom accuracy co parę epok
            #if (epoch % 1 == 0):
            if (DEBUG_ACCURACY):
                y_val = self.val_y
                tr_y = self.tr_y
                y_pred = np.asarray(self.model.predict(self.val_x,
                                        batch_size=BATCH_SIZE, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=32, use_multiprocessing=True))
                #y_val = y_val.reshape(-1,1)
                #y_pred = y_pred.reshape(-1,1)
                accuracy = np.count_nonzero(np.logical_and((y_pred <= y_val + y_val * THRESHOLD),(y_pred >= y_val - y_val * THRESHOLD)),axis=0)
                accuracy = np.sum(accuracy) / y_val.shape[1]
                accuracy = accuracy / y_val.shape[0]
                logs["val_accuracy"] = accuracy

                y_pred2 = np.asarray(self.model.predict(self.tr_x,
                                    batch_size=BATCH_SIZE, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=32, use_multiprocessing=True))
                #tr_y = tr_y.reshape(-1,1)
                #y_pred2 = y_pred2.reshape(-1,1)
                accuracy2 = np.count_nonzero(np.logical_and((y_pred2 <= tr_y + tr_y * THRESHOLD),(y_pred2 >= tr_y - tr_y * THRESHOLD)))
                accuracy2 = np.sum(accuracy2) / tr_y.shape[1]
                accuracy2 = accuracy2 / tr_y.shape[0]
                logs["accuracy"] = accuracy2

                #logs["accuracy"] = self.best
                #logs["loss"] = self.best_loss

                current_loss = logs["val_loss"]

                #dla accuracy
                if self.best < accuracy:
                    self.best = accuracy
                    self.best_loss = current_loss
                    self.wait = 0
                    # Record the best weights if current results is better (less).
                    self.best_weights = self.model.get_weights()
                    '''
                elif (self.best == accuracy) and (current_loss <= self.best_loss):
                    self.best_loss = current_loss
                    self.wait = 0
                    # Record the best weights if current results is better (less).
                    self.best_weights = self.model.get_weights()
                    '''
                else:
                    self.wait += 1
                    #if self.wait >= self.patience:
                    if epoch == self.epochs - 1:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                        print("Restoring model weights from the end of the best epoch.")
                        self.model.set_weights(self.best_weights)

            print(str(epoch)+"/"+str(self.epochs) + "loss: ",round(logs["loss"],6), "acc: ",round(logs["accuracy"],4)
                    ,"val_loss: ",round(logs["val_loss"],6), "val_acc: ",round(logs["val_accuracy"],4), 
                    "Best accu: ",round(self.best,4))

            #koniec treningu po q
            if (msvcrt.kbhit()): #BETA
                if (msvcrt.getch().lower() == b'q'):
                    print("\n\nKey pressed ---->  Stopping training\n\n")
                    self.model.stop_training = True
                    print("Restoring model weights from the end of the best epoch.")
                    self.model.set_weights(self.best_weights)



    #Choosing computing device
    if (CPU):
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(4) #set max threads
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        #tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])



    #Loading data
    if LOAD == 0:
        x_train, y_train, locations = load(FILE)
    else:
        x_train, y_train, labels = load2(FILE)
    if (FILE_TEST == FILE):
        x_test, y_test = x_train, y_train
    else:
        if LOAD == 0:
            x_test, y_test, locations = load(FILE_TEST)
        else:
            x_test, y_test, labels = load2(FILE_TEST)





    #convert to numpy
    x_train = np.array(x_train,dtype=float)
    x_test = np.array(x_test,dtype=float)
    y_train = np.array(y_train,dtype=float)
    y_test = np.array(y_test,dtype=float)
    print(x_train.shape)
    print(y_train.shape)

    #Data normalization
    x_train, maks = normalize_data(x_train,True)
    #
    #Normalize output data
    y_train, y_maks = normalize_data(y_train,True)
    #

    if FILE_TEST == FILE:
        x_test = x_train
        y_test = y_train
        y_maks_test = y_maks
        maks_test = maks
    else:
        x_test, maks_test = normalize_data(x_test,True)
        y_test, y_maks_test = normalize_data(y_test,True)
    #exit(1)


    #podział na test/trening
    xx = x_train
    yy = y_train
    if SPLIT:
        x_train, y_train, x_test, y_test = Split_Data(x_train,y_train,0.7,2,0.5)

    TEZD = False
    if (TEZD):
        x_train = xx
        y_train = yy





    #x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1])) #reshape beta
    #x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1])) #reshape beta

    #to jak coś
    #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1)) #reshape beta
    #x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1)) #reshape beta
    y_train = y_train.reshape(-1,y_train.shape[1])
    y_test = y_test.reshape(-1,y_test.shape[1])

    print(x_train.shape)
    print(y_train.shape)

    trainingStopCallback = haltCallback(x_test,y_test,x_train,y_train,EPOCHS)

    #exit(1)

    rows = len(x_train)
    cols = len(x_train[0])


    base_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)

    model = Sequential()

    # IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
    if (CPU):
        model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(128, activation='relu'))
        model.add(Dropout(0.1))
    else:
        model.add(Bidirectional(LSTM(1024, return_sequences=True), input_shape=(x_train.shape[1:]))) #nie dajemy return_sequences gdy przechodzimy do Dense Layer
        model.add(Dropout(0.3))

        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(Dropout(0.4))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.21))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.21))
        #model.add(Bidirectional(LSTM(64, return_sequences=True)))
        #model.add(Dropout(0.21))

        model.add(LayerNormalization()) #new

        
        #model.add(Bidirectional(LSTM(64, return_sequences=True))) #new
        #model.add(Dropout(0.21))
        #model.add(Bidirectional(LSTM(64, return_sequences=True))) #new
        #model.add(Dropout(0.21))
        model.add(Bidirectional(LSTM(64, return_sequences=True))) #new
        model.add(Dropout(0.21))

        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(Dropout(0.21))

        model.add(Bidirectional(LSTM(48, return_sequences=True)))
        model.add(Dropout(0.21))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.21))

        model.add(Bidirectional(LSTM(128, return_sequences=True))) #new
        model.add(Dropout(0.21))

        #model.add(LSTM(24, return_sequences=True)) #new
        #model.add(Dropout(0.21))

        model.add(Bidirectional(LSTM(700)))
        model.add(Dropout(0.3))




    model.add(Dense(1512, activation='tanh'))
    model.add(Dropout(0.3))


    model.add(Dense(y_train.shape[1], activation='relu'))#elu

    #opt = tf.keras.optimizers.Adam(lr=0.0003, decay=1e-6 * (EPOCHS / 8000) * 4) #decay zmniesza LR w czasie   *4 nowe  0.0005 stare
    #opt = tf.keras.optimizers.Nadam(lr=0.00030, decay=1e-6 * 4.2 * (EPOCHS / 8000))    #4
    #opt = tf.keras.optimizers.Nadam(lr=0.00050, decay=1e-4)    #80%
    opt = tf.keras.optimizers.Nadam(lr=0.001, decay=1e-4 * 2)    #0.0004



    # Compile model
    model.compile(
        #loss='mae', #52%
        loss='mean_squared_error', #74.85%
        optimizer=opt,
        metrics=['accuracy'],
        run_eagerly=False
    )


    start_time = time.perf_counter()

    checkpoints = tf.keras.callbacks.ModelCheckpoint(NAME+"_checkpoint.h5", 
            monitor='val_loss', verbose=1,
            save_best_only=False, mode='auto', period=150)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(NAME+"_bestloss.h5", 
            monitor='val_loss', verbose=1,
            save_best_only=True, mode='min', period=1)
    #trening
    history = model.fit(x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,#BATCH_SIZE
            validation_data=(x_test, y_test),
            #validation_data=(x_train, y_train),
            callbacks=[trainingStopCallback, checkpoints],
            verbose=0)


    #predicting    test



    #wypisanie czasu
    print("Training done")
    durationTh = (time.perf_counter() - start_time)
    print("Succeeded in time: [ ",end="")
    if (durationTh < 60):
        print(durationTh,"] s \n")    
    else:
        minutes = int(durationTh / 60)
        seconds = durationTh % 60
        print(str(minutes) + " min " + str(seconds),"s ] \n")

    print("\n\nTest accuracy:")
    model.evaluate(x_test, y = y_test, batch_size=BATCH_SIZE, verbose=1)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('wykresy\\accuracy_'+NAME+'.png',dpi=200)
    #plt.show()
    plt.close()


    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('wykresy\\loss_'+NAME+'.png',dpi=200)
    #plt.show()
    plt.close()







    ###########Lista źle ocenionych###########
    print("Preparing data...")
    predicted = model.predict(x_test,batch_size=BATCH_SIZE, verbose=0)

    # summarize history for loss
    plt.plot(y_test[:,0])
    plt.plot(predicted[:,0])
    plt.title('Test prediction')
    plt.ylabel('cases count')
    plt.xlabel('days')
    plt.legend(['real', 'predicted'], loc='upper left')
    plt.savefig('wykresy\\predicted_test_'+NAME+'.png',dpi=300)
    #plt.show()
    plt.close()








    dobre = 0
    lista = []
    lista.append("Location;Real;Predicted")
    for j in range(len(predicted[0])):
        for i in range(len(predicted)):
            predicted_value = str(int(predicted[i][j] * y_maks_test[j]))
            real_value = str(int(y_test[i][j] * y_maks_test[j]))
            lista.append(locations[i][j]+" ; " + real_value + " ; " + predicted_value)


    f = open("Output\\"+NAME+"_predicted.txt","w+")
    for i in lista:
        f.write(str(i) + "\n")
    f.close()


    f = open("Output\\"+NAME+"_acc.txt","w+")
    f.write("\nAccuracy: ")
    accuracy = (float(dobre) * 100.0 )/(float(len(predicted)))
    print("\n\n",accuracy,"%")
    f.write(str(accuracy))
    f.write("%")
    f.close()

    print("\a") #beep



    




def combine_results():
    def isDigit(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    class KRAJ:
        def __init__(self,name:str):
            self.name = name
            self.real = []
            self.predicted = []
        def add_day(self,real:int,predicted:int):
            self.real.append(real)
            self.predicted.append(predicted)

        def get_real(self):
            output = self.name
            for i in range(len(self.real)):
                output += "," + str(self.real[i])
            return output
        def get_predicted(self):
            output = self.name
            for i in range(len(self.predicted)):
                output += "," + str(self.predicted[i])
            return output



    f = open("covid_data_Confirmed.csv","r")
    labels = f.readline()
    f.close()

    labels = labels.split(",")

    PODPISY = 4
    ile_dni = len(labels) - PODPISY

    print("Dni w pliku wejściowym:",ile_dni)


    from os import system
    import os
    import sys

    pliki = []
    pliki.append("./output/" + "COVNET3" + "_predicted.txt")



    kraje = []
    current = -1
    temp = True
    ile_dni_predicted = 1
    for z in pliki:
        f = open(z,"r")
        dane = f.readlines()
        f.close()

        dane = dane[1:len(dane)]

        linia = dane[0].split(";")
        name = linia[0].strip()

        #print(name)
        kraje.append(KRAJ(name))
        current += 1
        kraje[current].add_day(int(linia[1]),int(linia[2]))


        iterator = 1
        while (iterator < len(dane)):
            linia = dane[iterator].split(";")


            if (linia[0].strip() != name.strip()):
                current += 1
                temp = False
                name = linia[0].strip()
                #print(name)
                kraje.append(KRAJ(name))
            elif (temp):
                ile_dni_predicted += 1
            
            kraje[current].add_day(int(linia[1]),int(linia[2]))

            iterator += 1

    print("Dni w pliku przewidywanym:",ile_dni_predicted)

    daty = labels[ile_dni + PODPISY - ile_dni_predicted:]


    first_line = ""
    for i in range(1,PODPISY):
        first_line += labels[i] + ","
    for i in range(len(daty)):
        first_line += str(int(daty[i].split("/")[1].strip())) + "/" + str(int(daty[i].split("/")[2].strip())) + "/" + str(int(daty[i].split("/")[0].strip())) + ","

    #print(first_line)


    f = open("data.csv","w+")
    f.write(first_line + "\n")
    for i in range(len(kraje)):
        dane = kraje[i].get_real().split(",")
        writee = dane[0].strip().replace("\"","") + ",0,0"
        for j in range(1,len(dane)):
            writee += "," + str(int(dane[j]))
        f.write(writee)
        f.write("\n")
    f.close()

    f = open("data_predicted.csv","w+")
    f.write(first_line + "\n")
    for i in range(len(kraje)):
        dane = kraje[i].get_predicted().split(",")
        writee = dane[0].strip().replace("\"","") + ",0,0"
        for j in range(1,len(dane)):
            writee += "," + str(int(dane[j]))
        f.write(writee)
        f.write("\n")
    f.close()



    from os import system
    #system("automatic_stabilize.exe")

    system("cp data.csv Strona\\data.csv")
    #system("cp data_predicted_stabilized.csv Strona\\data_predicted.csv")
    system("cp data_predicted.csv Strona\\data_predicted.csv")


    #system("python compare_stabilized.py")

    print("Done")










def prepare_environment():
    import os
    try:
        os.mkdir("output")
    except:
        print("Created output")
    try:
        os.mkdir("wykresy")
    except:
        print("Created wykresy")
    try:
        os.mkdir("Strona")
    except:
        print("Created Strona")






if __name__ == "__main__":
    prepare_environment()
    pobieranie_danych()
    corona_predictions()
    combine_results()
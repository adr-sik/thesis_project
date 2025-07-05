import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

for n in range(2,5):
    for features_length in [256, 512, 1024, 2048]:
        data_path = f"C:\\Users\\blob\\Desktop\\program\\data\\{n} {features_length}.pickle"
        results_dir = f"{n}_{features_length}_results"
        os.makedirs(results_dir, exist_ok=True)
        results_file = open(os.path.join(results_dir,f'{n}_{features_length}_results.txt'), 'w')

        def pick_model(shape):
            model = Sequential()
            if (shape==2048):
                model.add(Dense(units=1024, input_shape=(2048,)))
                model.add(Dense(units=512))
                model.add(Dense(units=256))
                model.add(Dense(units=64))
                model.add(Dense(units=1, activation='sigmoid'))        
            elif (shape==1024):
                model.add(Dense(units=512, input_shape=(1024,)))
                model.add(Dense(units=256))
                model.add(Dense(units=64))
                model.add(Dense(units=1, activation='sigmoid')) 
            elif (shape==512):
                model.add(Dense(units=256, input_shape=(512,)))
                model.add(Dense(units=64))
                model.add(Dense(units=1, activation='sigmoid')) 
            elif (shape==256):
                model.add(Dense(units=128, input_shape=(256,)))
                model.add(Dense(units=64))
                model.add(Dense(units=1, activation='sigmoid'))        
            return model 

        with (open(data_path, "rb")) as f:
            data = pickle.load(f)

        x = np.array([data[i][1:-1] for i in range(len(data))])
        y = np.array([data[i][-1] for i in range(len(data))])

        kf = StratifiedKFold (5, shuffle=True, random_state=42)

        all_accuracy = []
        all_precision = []
        all_recall = []
        all_f1 = []
        all_roc_auc = []
        all_val_loss = []
        all_val_acc = []

        fold = 0
        for train, test in kf.split(x, y):
            fold+=1
            print(f"Fold: {fold}")
            x_train = x[train]
            x_test = x[test]
            y_train = y[train]
            y_test = y[test]
                
            model = pick_model(features_length)
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            history = model.fit(x_train, y_train, epochs=20, batch_size=100, validation_data=(x_test, y_test), shuffle=True, verbose=0)

            y_pred_proba = model.predict(x_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            val_loss = history.history['val_loss']
            val_acc = history.history['val_accuracy']
               
            #Accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig(os.path.join(results_dir,f'{n}_{features_length}_accuracy_fold_{fold}.png'))
            plt.close()

            #Loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig(os.path.join(results_dir,f'{n}_{features_length}_loss_fold_{fold}.png'))
            plt.close()
            #ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Charactseristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(results_dir,f'{n}_{features_length}_ROCAUC_fold_{fold}.png'))
            plt.close()

            #Confusion Martix
            cm = confusion_matrix(y_test, y_pred)

            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

            cm_display.plot()
            plt.savefig(os.path.join(results_dir,f'{n}_{features_length}_matrix_fold_{fold}.png'))
            plt.close()
               
            results_file.write(f'Fold {fold}:\n')
            results_file.write(f'Accuracy: {accuracy:.4f}\n')
            results_file.write(f'Precision: {precision:.4f}\n')
            results_file.write(f'Recall: {recall:.4f}\n')
            results_file.write(f'F1-Score: {f1:.4f}\n')
            results_file.write(f'ROC AUC: {roc_auc:.4f}\n')
            results_file.write(f'Final Validation Loss: {val_loss[-1]:.4f}\n')
            results_file.write(f'Final Validation Accuracy: {val_acc[-1]:.4f}\n\n')
            
            all_accuracy.append(accuracy)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            all_roc_auc.append(roc_auc)
            all_val_loss.append(val_loss[-1])
            all_val_acc.append(val_acc[-1])

        results_file.write("Comparison\n")
        results_file.write(f'Accuracy avg:{sum(all_accuracy) / len(all_accuracy)}, max: {max(all_accuracy)}, min {min(all_accuracy)}\n')
        results_file.write(f'Precision avg:{sum(all_precision) / len(all_precision)}, max: {max(all_precision)}, min {min(all_precision)}\n')
        results_file.write(f'Recall avg: {sum(all_recall) / len(all_recall)}, max: {max(all_recall)}, min: {min(all_recall)}\n')
        results_file.write(f'F1-Score avg: {sum(all_f1) / len(all_f1)}, max: {max(all_f1)}, min: {min(all_f1)}\n')
        results_file.write(f'ROC AUC avg: {sum(all_roc_auc) / len(all_roc_auc)}, max: {max(all_roc_auc)}, min: {min(all_roc_auc)}\n')
        results_file.write(f'Final Validation Loss avg: {sum(val_loss) / len(val_loss):.4f}, max: {max(val_loss):.4f}, min: {min(val_loss):.4f}\n')
        results_file.write(f'Final Validation Accuracy avg: {sum(val_acc) / len(val_acc):.4f}, max: {max(val_acc):.4f}, min: {min(val_acc):.4f}\n')

print("Finished.")
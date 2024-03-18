# data-visualization-cheat-sheet

## Visualization of the distribution of a numerical value across multiple categories
```python
categories = dataset['category'].unique()
# Divide the categories into groups of five
data_subset_1 = dataset[dataset['category'].isin(categories[:5])]
data_subset_2 = dataset[dataset['category'].isin(categories[5:])]
# Columns of interest with numerical values
columns = ['calories', 'carbohydrate', 'sugar']
# Visualize the distribution of each interesting column across the categories 
for column in columns:
    fig, axs = plt.subplots(1,2, figsize=(12,6), sharey = True)
    sns.kdeplot(data= data_subset_1, x = column, hue ='category', ax=axs[0], fill=True)
    sns.kdeplot(data= data_subset_2, x = column, hue ='category', ax=axs[1], fill=True)
    plt.suptitle(f'Distribution of {column} by category', y= 0.95, size=16)
    plt.show()
```

<div align="center">
<img src="images/distribution of num values across multiple categories .png", alt="distribution of num values across multiple categories" height="350" width="750"></img>
</div>
<hr />


## Change the scale for better visualization
```python
# Changing protein column to a square root scale to better visualization
fig, axs = plt.subplots(1,2, figsize=(12,6), sharey = True)
sns.kdeplot(data= data_subset_1, x = np.sqrt(data_subset_1['protein']), hue ='category', ax=axs[0], fill=True)
sns.kdeplot(data= data_subset_2, x =  np.sqrt(data_subset_2['protein']), hue ='category', ax=axs[1], fill=True)
plt.suptitle('Distribution of protein by category', y= 0.95, size=16)
plt.show()
```
<div align="center">
<img src="images/change of scale.png", alt="change of scale" height="350" width="750"></img>
</div>
<hr />

## Confusion matrix
```python
from sklearn.svm import SVC
svm_model = SVC()
# Fit model
svm_model.fit(training_flatten, y_train)
# Predict
y_pred_svm = svm_model.predict(test_flatten)
# Classification report
print(classification_report(y_test, y_pred_svm))
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)
# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel('Predictions')
plt.ylabel('True values')
plt.title('Confusion Matrix of Support Vector Machine Classifier')
plt.show()
```

<div align="center">
<img src="images/Confusion matrix.png", alt="Confusion matrix" height="350" width="750"></img>
</div>
<hr />

## Plot training Results
```python
training_acc = history.history['accuracy']
training_loss = history.history['loss']
validation_acc = history.history['val_accuracy']
validation_loss = history.history['val_loss']

# Select the min loss and the max accuracy achieved in the validation
index_loss = np.argmin(validation_loss)
val_lowest = validation_loss[index_loss]
index_acc = np.argmax(validation_acc)
acc_highest = validation_acc[index_acc]

# Arrange for plotting
epochs = [i+1 for i in range(len(training_acc))]
loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

# Plot training history
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, 'r', label= 'Training loss')
plt.plot(epochs, validation_loss, 'g', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, training_acc, 'r', label= 'Training Accuracy')
plt.plot(epochs, validation_acc, 'g', label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout
plt.show()
```
<div align="center">
<img src="images/training results.png", alt="training results" height="350" width="750"></img>
</div>
<hr />

## ROC curve
```python
# Computing ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Computing area under the curve (AUC)
roc_auc = auc(fpr, tpr)

# Calculating the recall score
distance_to_top_left = np.sqrt((1 - tpr)**2 + fpr**2)
index_of_max_recall = np.argmin(distance_to_top_left)
max_recall = tpr[index_of_max_recall]
fpr_at_max_recall = fpr[index_of_max_recall]

# Plot ROC curve and recall score
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.scatter(fpr_at_max_recall, max_recall, color='red', label='Max Recall Point')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print(f'Max Recall: {max_recall:.2f} at FPR: {fpr_at_max_recall:.2f}')
```

<div align="center">
<img src="images/roc curve.png", alt="roc curve" height="350" width="750"></img>
</div>
<hr />

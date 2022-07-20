from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from prepare_images import prepare
import seaborn as sb
import pandas as pd
from cnn_util import create_classifier

from Heatmap.gradcam import apply_gradcam


train_images, train_labels, test_images, test_labels = prepare(root_directory='Dataset')

# classifier = create_classifier()
# classifier.fit(train_images, train_labels, batch_size=32, epochs=15)
# classifier.save('15_128_128_rc2.h5')
classifier = load_model('15_128_128_rc.h5')

# apply_gradcam(classifier)

prediction = classifier.predict(test_images)
winners = list(map(lambda win: list(win).index(max(win)), prediction))

cm = confusion_matrix(test_labels, winners)
sb.heatmap(cm, annot=True, fmt='', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Labels')
plt.show()

number_labels = list(range(0, 33))
report = classification_report(test_labels, winners, target_names=number_labels, output_dict=True)
sb.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
plt.show()

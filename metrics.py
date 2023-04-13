from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def metrics(Y_valid, Y_pred):
    report = classification_report(Y_valid, Y_pred, output_dict=True)
    ConfusionMatrixDisplay.from_predictions(Y_valid, Y_pred, normalize='true')
    plt.show()
    return report

"""
def calc_patch_level_metrics(self, predictions_softmax: np.array):

    Calculate the metrics for the instance (patch) level.

    predictions = np.argmax(predictions_softmax, axis=1)
    unlabeled_index = self.num_classes
    gt_classes = self.test_df['class']
    indices_of_labeled_patches = (gt_classes != str(unlabeled_index))
    gt_classes = np.array(gt_classes[indices_of_labeled_patches]).astype(np.int)
    predictions = np.array(predictions[indices_of_labeled_patches]).astype(np.int)

    ConfusionMatrixDisplay.from_predictions(gt_classes, predictions)
    plt.show()

    metrics ={}
    metrics['accuracy'] = accuracy_score(gt_classes, predictions)
    metrics['cohens_quadratic_kappa'] = cohen_kappa_score(gt_classes, predictions, weights='quadratic')
    metrics['f1_mean'] = f1_score(gt_classes, predictions, average='macro')
    f1_score_classwise = f1_score(gt_classes, predictions, average=None)
    for class_id in range(len(f1_score_classwise)):
        key = 'f1_class_id_' + str(class_id)
        metrics[key] = f1_score_classwise[class_id]

    return metrics
"""
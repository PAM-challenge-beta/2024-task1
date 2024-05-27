import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Process and evaluate detections against ground truth annotations.")
parser.add_argument('ground_truth_path', type=str, help='Path to the ground truth annotations CSV file.')
parser.add_argument('detections_path', type=str, help='Path to the detections CSV file.')
args = parser.parse_args()

# Load annotations and detections from CSV files
ground_truth = pd.read_csv(args.ground_truth_path)
detections = pd.read_csv(args.detections_path)  # Your detection csv MUST contain 'filename' and 'timestamp' fields. Additional fields will be ignored.

# Buffer. We will consider a TP to be a dete4ction that falls within a 1.5s buffer of the annotation timestamp ([timestamp - 1.5, timestamp + 1.5])
buffer = 1.5

TP = 0
FP = 0
FN = 0

# Group data by filename for easier processing
grouped_ground_truth = ground_truth.groupby('filename')
grouped_detections = detections.groupby('filename')

# Algorithm explanation: The true positive count will only be incremented 
# once per detection timestamp. For instance if multiple detection timestamps 
# aligns with only one ground truth, the true positive count will be incremented 
# only once. Similarly, if one detection timestamp aligns with multiple ground truths, 
# the true positive count will be incremented for each ground truth.
for filename, tru_group in grouped_ground_truth:
    if filename in grouped_detections.groups:
        # Get corresponding detection
        det_group = grouped_detections.get_group(filename)

        # For each ground truth, check if it is a TP or FN 
        for tru_time in tru_group['timestamp']:
            if any((tru_time - buffer <= det_group['timestamp']) & (det_group['timestamp'] <= tru_time + buffer)):
                TP += 1
            else:
                FN += 1
    else:
        # All annotations in files without detections are false negatives
        FN += len(tru_group)

# Now for the FP we have to loop through the detections and check if they do not match any ground truth
# Note that we are treating each detection independently and separetly from TP and FN. This is to ensure that any detection that matches
# a groudn truth is not counted as a FP
for filename, det_group in grouped_detections:
    if filename in grouped_ground_truth.groups:
        # Get corresponding annotations
        ann_group = grouped_ground_truth.get_group(filename)
        
        # For each detection, check if it is a FP
        for det_time in det_group['timestamp']:
            # Check if detection is not within the buffer of any ground truth
            if not any((ann_group['timestamp'] - buffer <= det_time) & (det_time <= ann_group['timestamp'] + buffer)):
                FP += 1
    else:
        # All detections in files without groudn truths are false positives
        FP += len(det_group)


# Print the metrics
print("True Positives (TP):", TP)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)

# Precision
if TP + FP == 0:
    precision = 0  # To handle the case where no positives are predicted
else:
    precision = TP / (TP + FP)

# Recall
if TP + FN == 0:
    recall = 0  # To handle the case where there are no true positives and false negatives
else:
    recall = TP / (TP + FN)

# F1 Score
if precision + recall == 0:
    f1_score = 0  # To handle the case where precision and recall are both zero
else:
    f1_score = 2 * (precision * recall) / (precision + recall)

# Print the calculated metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
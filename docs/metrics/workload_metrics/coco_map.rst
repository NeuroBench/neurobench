===================
COCO mAP
===================

Definition
----------
The correctness metric for the Event Camera Object Detection task.

OCO mAP is calculated using the intersection over union of the bounding boxes produced by the model against ground-truth boxes. Here, :math:`A` and :math:`B` refer to bounding boxes, and the intersection and union consider the overlapping area and the area covered by both boxes, respectively. The IoU is compared against 10 thresholds between 0.50 and 0.95, with a step size of 0.05.
For each threshold, precision is calculated with True Positives and False Positives determined by whether the IoU meets the threshold or not, respectively. The mAP is calculated as the averaged precision over all thresholds for each class, which is further averaged over all classes to produce the final result.

.. math::
    IoU(A,B) = \frac{|A\cap B|}{|A\cup B|}

.. math::
    Precision(TP,FP) = \frac{\sum TP}{\sum TP + FP}

Implementation Notes
--------------------
This is implemented by the Metavision SDK. If you do not have the Metavision SDK installed, it will not work.
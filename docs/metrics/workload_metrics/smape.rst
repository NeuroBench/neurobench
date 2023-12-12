===================
sMAPE
===================

Definition
----------
The correctness metric for the Chaotic Function Prediction task.

Symmetric mean absolute percentage error (sMAPE), a standard metric in forecasting, is used to measure the correctness of the model predictions :math:`\hat{y_i}` against the ground-truth :math:`y_i`, over :math:`n` data points in the test split of the time series. 
The sMAPE metric has a bounded range of :math:`[0, 200]`, thus diverging predictions (infinity or NaN) due to floating-point arithmetic have bounded error which can be used to average correctness over multiple time series instantiations.

.. math::
    sMAPE = 200 \times \frac{1}{n} \left( \sum_{i=1}^{n} \frac{|y_i - \hat{y_i}|}{(|y_i| + |\hat{y_i}|)}\right)

Implementation Notes
--------------------
# Evaluation for Master Thesis

This program is the program used for evaluating my master thesis "Extend automated processing with document analysis using AI-services" (link added when published).

The thesis is focused on creating an end-to-end method for finding document sections bases on reference documents and extracting it's signatures and stamps. Those extraced signatures are compared afterwards, to evaluate whether the signature is created by the same person. While the section detection works in 98% of the time and the signature detection is working acceptable, the signature comparison afterwards seems to be the most tricky part. A siamese network was used for evaluating the distance between them, but the area under the ROC curve achieves only 0.37.

To test it on a custom documents place the following documents into the "_input_thesis":
* request.pdf
* request_gen.pdf
* billing.pdf
* billing_gen.pdf
* reference_1.pdf



Special thanks to [Nicolas Dutly](https://github.com/Jumpst3r) for his program and provided models for [printed and handwriting segmentation](https://github.com/Jumpst3r/printed-hw-segmentation).

```
@article{Dutly2019PHTIWSAP,
  title={PHTI-WS: A Printed and Handwritten Text Identification Web Service Based on FCN and CRF Post-Processing},
  author={Nicolas Dutly and Fouad Slimane and Rolf Ingold},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={20-25}
}
```
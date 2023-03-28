# ML4SCI GSOC DeepLense Task

This GitHub repository contains three Folders, each of which focuses on a different deep learning task.

## Common Task: Multi-class Classification

The notebook in Common Task demonstrates a simple image classification task using a convolutional neural network (Transfer Learning). The dataset used in this notebook is the one provided for common task, which is a collection of strong lensing image. The notebook includes all the necessary code to load the dataset, preprocess the images, define the CNN model, train the model, and evaluate its performance.

#### Results on various Models:

| Model                                       | Epochs | Batch Size | Learning Rate | ROC_AUC   |
| :------------------------------------------ | :----- | :--------- | :------------ | :-------- |
| ResNet18                                    | 5      | 64         | 0.0004        | 0.98      | 
| DenseNet161                                 | 19     | 64         | 0.0003        | 0.98      |     
| Ensamble                                    | -      | -          | -             | 0.99      | 


## Specific Task 2 : Lens Finding 

The LensFinding folder has two notebooks focuses on a more specialized deep learning task, namely identifying gravitational lenses in astronomical images. Detecting gravitational lenses is important for understanding the structure and distribution of matter in the universe. The notebook includes all the necessary code to load the dataset, preprocess the images, define the CNN model, train the model, and evaluate its performance.

#### Results on various Models:

| Model                                       | Epochs | Batch Size | Learning Rate | ROC_AUC   |
| :------------------------------------------ | :----- | :--------- | :------------ | :-------- |
| tf_efficientnet_b2_ns                       | 5      | 64         | 0.0004        | 0.97      | 
| ResNet18                                    | 20     | 64         | 0.0004        | 0.97      |  
| tf_efficientnet_b4_ns                       | 5.     | 32         | 0.005         | 0.98      |
| Ensamble                                    | -      | -          | -             | 0.99      | 


## Specific Task 5: Exploring Transformers

The notebook in this folder demonstrates the use of a vision transformer method to build a robust and efficient model for binary classification on provided dataset.

#### Results on various Models:

| Model                                       | Epochs | Batch Size | Learning Rate | ROC_AUC   |
| :------------------------------------------ | :----- | :--------- | :------------ | :-------- |
| vit_base_patch16_224                        | 15     | 32         | 0.0003        | 0.99      | 
| vit_large_patch16_224                       | 20     | 32         | 0.00004       | 0.99      |  
| swin_base_patch4_window7_224                | 15     | 32         | 0.00005       | 0.99      |
| Ensamble                                    | -      | -          | -             | 0.99      | 


## Usage

#### 1) Setup 

Clone the Repository
```bash
$ git clone https://github.com/neerajanand321/GSOC_DeepLense.git
```
Move to the directory to access the notebooks
```bash
cd GSOC_DeepLens
```
- For **Common Task** `deep-lense.ipynb` notebook is used
- For **Lens Finding**   `deep-lense-2.ipynb` notebook is used
- For **Exploring Transformers** `dl-transformer-train-vit-base.ipynb` notebook is used

#### 2) Dataset Directory

Give the path of dataset. Eg:
```python
root_dir = '[Your Dataset Path]'
dir_no_sub = root_dir+'/no_sub/' # Path to folder having data with no substructure 
dir_sub = root_dir+'/sub/' # Path to folder having data containing substructure
```
#### 3) Hyperparameters Setting
Use `CFG` class to change the hyperparameters
```python
class CFG:
    lr = 0.0001
    batch_size = 32
    num_classes = 1
    size=[224, 224]
    target_col="target"
    epochs = 10
    seed = 42
    num_workers = 2
    transform = False
    weight_decay = 3e-5
    num_workers=2
    train=True
    debug=False
    metric_type="roc_auc"
    scheduler_type = "CosineLRScheduler"
    optimizer_type = "Adam"
    loss_type = "BCEWithLogitsLoss"
    max_grad_norm = 1000
    lr_max = 3e-4
    epochs_warmup = 1.0
    model_name = "vit_base_patch16_224"
 ```
 #### 4) Augmentation
 For augmentation use `get_transform` function
 ```python 
 def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose([
            A.Resize(*CFG.size),
            A.Rotate(limit=20),
            A.InvertImg(), 
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(*CFG.size),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
 ```
 
 #### 5) Training and Evaluation
 For training and evaluation use `Train` class
 ```python
 def main():
    
    if CFG.train: 
        # train
        train = Train(CFG) # Configuration class
        train.train_loop(train_df, val_df) # Dataframe for training and evaluation
        
 if __name__=='__main__':
    main()
```

## Dependencies

To run the notebooks, you will need to have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- NumPy
- Matplotlib
- PyTorch

You can install these dependencies using pip or conda.

